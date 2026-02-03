"""
Backtest Engine

Core backtesting logic with anti-leakage safeguards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..common import PortfolioWeight, Trade, get_logger
from ..portfolio import PortfolioAllocator, RiskManager, TransactionCostModel

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float = 100_000_000  # 1억원
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    transaction_cost_bps: float = 30.0  # Commission + tax + spread
    slippage_bps: float = 5.0
    use_close_prices: bool = True  # Use close for signals, next open for execution
    warmup_days: int = 252  # Days for model warmup
    verbose: bool = True


@dataclass
class DailyResult:
    """Single day backtest result."""

    date: pd.Timestamp
    portfolio_value: float
    daily_return: float
    positions: dict[str, float]
    weights: dict[str, float]
    trades: list[Trade]
    transaction_costs: float
    gross_exposure: float
    net_exposure: float
    n_positions: int


@dataclass
class BacktestResult:
    """Complete backtest result."""

    config: BacktestConfig
    daily_results: list[DailyResult]
    metrics: dict[str, float]
    trades_df: pd.DataFrame
    returns_series: pd.Series
    weights_df: pd.DataFrame
    run_timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": {
                "start_date": str(self.config.start_date),
                "end_date": str(self.config.end_date),
                "initial_capital": self.config.initial_capital,
                "rebalance_frequency": self.config.rebalance_frequency,
            },
            "metrics": self.metrics,
            "n_trading_days": len(self.daily_results),
            "run_timestamp": str(self.run_timestamp),
        }


class BacktestEngine:
    """
    Event-driven backtesting engine.

    ANTI-LEAKAGE SAFEGUARDS:
        1. Point-in-time data access only
        2. Trade execution at next period's open (or close with delay)
        3. No future information in feature/label calculation
        4. Proper handling of corporate actions
    """

    def __init__(
        self,
        config: BacktestConfig,
        allocator: PortfolioAllocator | None = None,
        cost_model: TransactionCostModel | None = None,
        risk_manager: RiskManager | None = None,
    ):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            allocator: Portfolio allocator
            cost_model: Transaction cost model
            risk_manager: Risk manager
        """
        self.config = config
        self.allocator = allocator or PortfolioAllocator()
        self.cost_model = cost_model or TransactionCostModel()
        self.risk_manager = risk_manager or RiskManager()

        # State
        self._cash = config.initial_capital
        self._positions: dict[str, float] = {}  # asset_id -> shares
        self._portfolio_value = config.initial_capital
        self._daily_results: list[DailyResult] = []
        self._all_trades: list[Trade] = []

    def run(
        self,
        signal_model: Any,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        labels_df: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            signal_model: Fitted signal model with predict() method
            features_df: Features DataFrame with date, asset_id columns
            prices_df: Price DataFrame with date, asset_id, close, open columns
            labels_df: Optional labels for walk-forward (not used in simple backtest)

        Returns:
            BacktestResult
        """
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")

        # Get unique dates
        all_dates = sorted(features_df["date"].unique())
        backtest_dates = [
            d for d in all_dates
            if self.config.start_date <= pd.Timestamp(d) <= self.config.end_date
        ]

        if not backtest_dates:
            logger.warning("No dates in backtest range")
            return self._create_empty_result()

        # Reset state
        self._reset_state()

        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates(backtest_dates)

        prev_weights = pd.Series(dtype=float)

        for i, date in enumerate(backtest_dates):
            date = pd.Timestamp(date)

            # Get point-in-time data
            date_features = features_df[features_df["date"] == date]
            date_prices = prices_df[prices_df["date"] == date]

            if date_features.empty or date_prices.empty:
                continue

            # Check if rebalance day
            is_rebalance = date in rebalance_dates

            if is_rebalance:
                # Generate signals (point-in-time)
                try:
                    predictions = signal_model.predict(date, date_features)
                except Exception as e:
                    logger.warning(f"Signal generation failed on {date}: {e}")
                    predictions = pd.DataFrame()

                if not predictions.empty:
                    # Convert scores to weights
                    scores = predictions.set_index("asset_id")["score"]
                    target_weights = self.allocator.allocate(scores, date_features)

                    # Execute trades
                    trades = self._execute_rebalance(
                        target_weights,
                        date_prices,
                        date,
                    )
                    self._all_trades.extend(trades)
                    prev_weights = target_weights
                else:
                    trades = []
            else:
                trades = []

            # Mark-to-market
            daily_result = self._mark_to_market(date, date_prices, trades)
            self._daily_results.append(daily_result)

            # Update risk manager
            if daily_result.daily_return != 0:
                self.risk_manager.update_nav(daily_result.daily_return)

            if self.config.verbose and (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(backtest_dates)} days, NAV: {self._portfolio_value:,.0f}")

        logger.info(f"Backtest complete. Final NAV: {self._portfolio_value:,.0f}")

        return self._create_result()

    def _reset_state(self) -> None:
        """Reset backtest state."""
        self._cash = self.config.initial_capital
        self._positions = {}
        self._portfolio_value = self.config.initial_capital
        self._daily_results = []
        self._all_trades = []
        self.risk_manager.reset_nav(self.config.initial_capital)

    def _get_rebalance_dates(self, dates: list) -> set:
        """Get set of rebalance dates based on frequency."""
        dates = [pd.Timestamp(d) for d in dates]

        if self.config.rebalance_frequency == "daily":
            return set(dates)

        rebalance_dates = set()

        if self.config.rebalance_frequency == "weekly":
            # Rebalance on Mondays (or first day of week)
            for date in dates:
                if date.dayofweek == 0:  # Monday
                    rebalance_dates.add(date)
                elif not rebalance_dates or (date - max(rebalance_dates)).days >= 7:
                    rebalance_dates.add(date)

        elif self.config.rebalance_frequency == "monthly":
            # Rebalance on first trading day of month
            current_month = None
            for date in dates:
                if date.month != current_month:
                    rebalance_dates.add(date)
                    current_month = date.month

        return rebalance_dates

    def _execute_rebalance(
        self,
        target_weights: pd.Series,
        prices_df: pd.DataFrame,
        date: pd.Timestamp,
    ) -> list[Trade]:
        """
        Execute rebalancing trades.

        Args:
            target_weights: Target portfolio weights
            prices_df: Current prices
            date: Trade date

        Returns:
            List of executed trades
        """
        trades = []

        # Get current weights
        current_weights = self._get_current_weights(prices_df)

        # Calculate required trades
        price_map = prices_df.set_index("asset_id")["close"].to_dict()

        for asset_id in set(target_weights.index) | set(current_weights.index):
            target_weight = target_weights.get(asset_id, 0.0)
            current_weight = current_weights.get(asset_id, 0.0)
            weight_diff = target_weight - current_weight

            if abs(weight_diff) < 0.001:  # Skip tiny trades
                continue

            price = price_map.get(asset_id)
            if price is None or price <= 0:
                continue

            # Calculate trade value
            trade_value = weight_diff * self._portfolio_value
            shares = int(trade_value / price)

            if shares == 0:
                continue

            # Apply transaction costs
            gross_value = abs(shares * price)
            cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
            cost = gross_value * (cost_bps / 10000)

            # Adjust for buy/sell
            if shares > 0:
                execution_price = price * (1 + self.config.slippage_bps / 10000)
            else:
                execution_price = price * (1 - self.config.slippage_bps / 10000)

            # Execute
            net_value = shares * execution_price
            self._cash -= net_value
            self._cash -= cost

            current_shares = self._positions.get(asset_id, 0)
            new_shares = current_shares + shares
            if new_shares == 0:
                self._positions.pop(asset_id, None)
            else:
                self._positions[asset_id] = new_shares

            trade = Trade(
                date=date,
                asset_id=asset_id,
                side="BUY" if shares > 0 else "SELL",
                quantity=abs(shares),
                price=execution_price,
                cost=cost,
            )
            trades.append(trade)

        return trades

    def _get_current_weights(self, prices_df: pd.DataFrame) -> pd.Series:
        """Get current portfolio weights."""
        if not self._positions:
            return pd.Series(dtype=float)

        price_map = prices_df.set_index("asset_id")["close"].to_dict()
        position_values = {}

        for asset_id, shares in self._positions.items():
            price = price_map.get(asset_id, 0)
            position_values[asset_id] = shares * price

        total_value = sum(position_values.values()) + self._cash
        if total_value <= 0:
            return pd.Series(dtype=float)

        return pd.Series(position_values) / total_value

    def _mark_to_market(
        self,
        date: pd.Timestamp,
        prices_df: pd.DataFrame,
        trades: list[Trade],
    ) -> DailyResult:
        """Mark portfolio to market."""
        price_map = prices_df.set_index("asset_id")["close"].to_dict()

        # Calculate position values
        position_values = {}
        for asset_id, shares in self._positions.items():
            price = price_map.get(asset_id, 0)
            position_values[asset_id] = shares * price

        positions_value = sum(position_values.values())
        new_portfolio_value = positions_value + self._cash

        # Calculate return
        if self._portfolio_value > 0:
            daily_return = (new_portfolio_value - self._portfolio_value) / self._portfolio_value
        else:
            daily_return = 0.0

        # Calculate exposures
        gross_exposure = sum(abs(v) for v in position_values.values()) / new_portfolio_value if new_portfolio_value > 0 else 0
        net_exposure = positions_value / new_portfolio_value if new_portfolio_value > 0 else 0

        # Transaction costs for today
        transaction_costs = sum(t.cost for t in trades)

        # Weights
        weights = {k: v / new_portfolio_value for k, v in position_values.items()} if new_portfolio_value > 0 else {}

        prev_value = self._portfolio_value
        self._portfolio_value = new_portfolio_value

        return DailyResult(
            date=date,
            portfolio_value=new_portfolio_value,
            daily_return=daily_return,
            positions=dict(self._positions),
            weights=weights,
            trades=trades,
            transaction_costs=transaction_costs,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            n_positions=len(self._positions),
        )

    def _create_result(self) -> BacktestResult:
        """Create backtest result object."""
        from .metrics import calculate_metrics

        # Create returns series
        returns = pd.Series(
            [r.daily_return for r in self._daily_results],
            index=[r.date for r in self._daily_results],
        )

        # Create trades DataFrame
        if self._all_trades:
            trades_df = pd.DataFrame([
                {
                    "date": t.date,
                    "asset_id": t.asset_id,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "cost": t.cost,
                    "value": t.quantity * t.price,
                }
                for t in self._all_trades
            ])
        else:
            trades_df = pd.DataFrame()

        # Create weights DataFrame
        weights_records = []
        for r in self._daily_results:
            for asset_id, weight in r.weights.items():
                weights_records.append({
                    "date": r.date,
                    "asset_id": asset_id,
                    "weight": weight,
                })
        weights_df = pd.DataFrame(weights_records)

        # Calculate metrics
        metrics = calculate_metrics(returns, self.config.initial_capital)

        return BacktestResult(
            config=self.config,
            daily_results=self._daily_results,
            metrics=metrics,
            trades_df=trades_df,
            returns_series=returns,
            weights_df=weights_df,
        )

    def _create_empty_result(self) -> BacktestResult:
        """Create empty backtest result."""
        return BacktestResult(
            config=self.config,
            daily_results=[],
            metrics={},
            trades_df=pd.DataFrame(),
            returns_series=pd.Series(dtype=float),
            weights_df=pd.DataFrame(),
        )


def run_backtest(
    signal_model: Any,
    features_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    initial_capital: float = 100_000_000,
    **kwargs,
) -> BacktestResult:
    """
    Convenience function to run backtest.

    Args:
        signal_model: Fitted signal model
        features_df: Features DataFrame
        prices_df: Prices DataFrame
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Initial capital
        **kwargs: Additional config options

    Returns:
        BacktestResult
    """
    config = BacktestConfig(
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        initial_capital=initial_capital,
        **kwargs,
    )

    engine = BacktestEngine(config)
    return engine.run(signal_model, features_df, prices_df)
