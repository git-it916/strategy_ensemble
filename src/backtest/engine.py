"""
Backtesting Engine

Simulates trading over historical data with realistic execution assumptions.
Supports both traditional alphas and LLM alphas (with response caching).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..alphas.base_alpha import BaseAlpha
from ..ensemble.agent import EnsembleAgent
from ..ensemble.allocator import Allocator, RiskParityAllocator
from .metrics import calculate_metrics, calculate_drawdown_series

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest results."""

    def __init__(
        self,
        returns: pd.Series,
        portfolio_values: pd.Series,
        holdings_history: list[dict],
        transaction_costs: pd.Series,
        signals_history: list[dict] | None = None,
    ):
        self.returns = returns
        self.portfolio_values = portfolio_values
        self.holdings_history = holdings_history
        self.transaction_costs = transaction_costs
        self.signals_history = signals_history
        self._metrics: dict | None = None

    @property
    def metrics(self) -> dict[str, float]:
        """Lazily compute and cache performance metrics."""
        if self._metrics is None:
            self._metrics = calculate_metrics(self.returns)
        return self._metrics

    def summary(self) -> pd.DataFrame:
        """One-row summary DataFrame."""
        m = self.metrics
        return pd.DataFrame([{
            "Total Return": f"{m['total_return']:.2%}",
            "CAGR": f"{m['cagr']:.2%}",
            "Sharpe": f"{m['sharpe_ratio']:.2f}",
            "Sortino": f"{m['sortino_ratio']:.2f}",
            "Max DD": f"{m['max_drawdown']:.2%}",
            "Win Rate": f"{m['win_rate']:.2%}",
            "Profit Factor": f"{m['profit_factor']:.2f}",
            "Trades": m["n_trades"],
            "Total Costs": f"{self.transaction_costs.sum():,.0f}",
        }])

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "metrics": self.metrics,
            "returns": self.returns.to_dict(),
            "portfolio_values": self.portfolio_values.to_dict(),
            "transaction_costs_total": self.transaction_costs.sum(),
            "n_periods": len(self.returns),
        }


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Features:
        - Simulates daily / hourly / weekly rebalancing
        - Transaction cost modeling (commission + tax, Korean market)
        - Slippage estimation
        - LLM alpha response caching
        - Comparison mode: LLM vs traditional ensemble
    """

    def __init__(
        self,
        initial_capital: float = 100_000_000,
        commission_bps: float = 1.5,
        tax_bps: float = 23.0,
        slippage_bps: float = 10.0,
        max_position_weight: float = 0.10,
        rebalance_frequency: str = "daily",
    ):
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.tax_bps = tax_bps
        self.slippage_bps = slippage_bps
        self.max_position_weight = max_position_weight
        self.rebalance_frequency = rebalance_frequency

    def run(
        self,
        ensemble: EnsembleAgent,
        allocator: Allocator,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        regime_classifier: Any | None = None,
    ) -> BacktestResult:
        """
        Run backtest over historical period.

        Timing assumption:
        - Signals are generated on rebalance date close (t)
        - Orders are executed at close (t)
        - New weights become active from next trading day open (t+1)
        - Period return is measured open-to-open between activation dates
        """
        rebalance_dates = self._get_rebalance_dates(prices, start_date, end_date)
        if len(rebalance_dates) < 2:
            logger.warning("Backtest skipped: need at least 2 rebalance dates")
            empty = pd.Series(dtype=float)
            return BacktestResult(
                returns=empty.rename("returns"),
                portfolio_values=empty.rename("portfolio_value"),
                holdings_history=[],
                transaction_costs=empty.rename("costs"),
                signals_history=[],
            )

        all_trading_dates = prices["date"].sort_values().unique()
        logger.info(
            f"Backtest: {start_date} to {end_date}, "
            f"{len(rebalance_dates) - 1} evaluated periods "
            "(close execution, next-open activation)"
        )

        # State
        portfolio_value = self.initial_capital
        current_weights: dict[str, float] = {}
        daily_returns: list[float] = []
        daily_values: list[float] = [portfolio_value]
        daily_dates: list[datetime] = []
        holdings_history: list[dict] = []
        signals_history: list[dict] = []
        daily_costs: list[float] = []

        for i in range(len(rebalance_dates) - 1):
            signal_date = rebalance_dates[i]
            next_signal_date = rebalance_dates[i + 1]
            activation_date = self._next_trading_date(all_trading_dates, signal_date)
            next_activation_date = self._next_trading_date(
                all_trading_dates, next_signal_date
            )

            if activation_date is None:
                logger.warning(
                    "Skipping period with invalid activation window: "
                    f"signal={signal_date}, next_signal={next_signal_date}, "
                    f"activation={activation_date}, next_activation={next_activation_date}"
                )
                continue
            if next_activation_date is None:
                # Natural end-of-sample boundary: nothing to evaluate beyond this point.
                break
            if activation_date >= next_activation_date:
                logger.warning(
                    "Skipping period with non-increasing activation window: "
                    f"signal={signal_date}, next_signal={next_signal_date}, "
                    f"activation={activation_date}, next_activation={next_activation_date}"
                )
                continue

            try:
                prices_until_signal = prices[prices["date"] <= signal_date]
                features_until_signal = features
                if features is not None and "date" in features.columns:
                    features_until_signal = features[
                        features["date"] <= pd.Timestamp(signal_date)
                    ]

                # Get regime if classifier available
                regime = None
                if regime_classifier is not None:
                    try:
                        regime = regime_classifier.predict_regime(
                            signal_date, prices_until_signal, features_until_signal
                        )
                    except Exception:
                        pass

                # Generate signals
                signal = ensemble.generate_signals(
                    signal_date,
                    prices_until_signal,
                    features_until_signal,
                    regime,
                )
                signals_df = signal.signals

                # Build target portfolio. Empty signals -> move to cash.
                if signals_df.empty:
                    new_weights: dict[str, float] = {}
                else:
                    target_weights = allocator.allocate(signals_df, prices_until_signal)
                    if target_weights.empty:
                        new_weights = {}
                    else:
                        new_weights = dict(
                            zip(target_weights["ticker"], target_weights["weight"])
                        )

                # Transaction costs
                cost = self._apply_transaction_costs(
                    current_weights, new_weights, portfolio_value
                )
                portfolio_value -= cost
                daily_costs.append(cost)

                # Orders filled on signal_date close, weights active from next open.
                period_return = self._calculate_period_return(
                    new_weights,
                    prices,
                    activation_date,
                    next_activation_date,
                    from_price_col="open",
                    to_price_col="open",
                )

                portfolio_value *= (1 + period_return)
                daily_returns.append(period_return)
                daily_values.append(portfolio_value)
                daily_dates.append(activation_date)

                # Record
                current_weights = new_weights
                holdings_history.append({
                    "date": signal_date,
                    "activation_date": activation_date,
                    "weights": new_weights.copy(),
                    "portfolio_value": portfolio_value,
                    "n_positions": len(new_weights),
                })
                signals_history.append({
                    "date": signal_date,
                    "activation_date": activation_date,
                    "n_signals": len(signals_df),
                    "top_signals": (
                        signals_df.nlargest(5, "score")[["ticker", "score"]]
                        .to_dict("records")
                        if not signals_df.empty else []
                    ),
                })

                if (i + 1) % 50 == 0:
                    logger.info(
                        f"  [{i + 1}/{len(rebalance_dates) - 1}] "
                        f"PV={portfolio_value:,.0f} "
                        f"Return={period_return:+.4f}"
                    )

            except Exception as e:
                logger.error(f"Backtest error at {signal_date}: {e}")
                daily_returns.append(0.0)
                daily_values.append(portfolio_value)
                daily_dates.append(activation_date)
                daily_costs.append(0.0)

        # Build result
        return_series = pd.Series(daily_returns, index=daily_dates, name="returns")
        value_series = pd.Series(
            daily_values[1:], index=daily_dates, name="portfolio_value"
        )
        cost_series = pd.Series(daily_costs, index=daily_dates, name="costs")

        result = BacktestResult(
            returns=return_series,
            portfolio_values=value_series,
            holdings_history=holdings_history,
            transaction_costs=cost_series,
            signals_history=signals_history,
        )

        logger.info(
            f"Backtest complete: "
            f"Total Return={result.metrics['total_return']:.2%}, "
            f"Sharpe={result.metrics['sharpe_ratio']:.2f}, "
            f"Max DD={result.metrics['max_drawdown']:.2%}"
        )

        return result

    def run_comparison(
        self,
        traditional_ensemble: EnsembleAgent,
        llm_ensemble: EnsembleAgent,
        allocator: Allocator,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
    ) -> dict[str, BacktestResult]:
        """Run side-by-side comparison of traditional vs LLM ensemble."""
        logger.info("=" * 60)
        logger.info("Running comparison backtest")
        logger.info("=" * 60)

        logger.info("--- Traditional Ensemble ---")
        trad_result = self.run(
            traditional_ensemble, allocator, prices, features,
            start_date, end_date,
        )

        logger.info("--- LLM Ensemble ---")
        llm_result = self.run(
            llm_ensemble, allocator, prices, features,
            start_date, end_date,
        )

        # Log comparison
        logger.info("=" * 60)
        logger.info("COMPARISON RESULTS")
        logger.info(f"  Traditional: Sharpe={trad_result.metrics['sharpe_ratio']:.2f}, "
                     f"Return={trad_result.metrics['total_return']:.2%}")
        logger.info(f"  LLM:         Sharpe={llm_result.metrics['sharpe_ratio']:.2f}, "
                     f"Return={llm_result.metrics['total_return']:.2%}")
        logger.info("=" * 60)

        return {
            "traditional": trad_result,
            "llm": llm_result,
        }

    def _get_rebalance_dates(
        self,
        prices: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> list[datetime]:
        """Get rebalance dates based on frequency."""
        all_dates = prices["date"].sort_values().unique()
        mask = (all_dates >= pd.Timestamp(start_date)) & (
            all_dates <= pd.Timestamp(end_date)
        )
        trading_dates = all_dates[mask]

        if self.rebalance_frequency == "daily":
            return list(trading_dates)
        elif self.rebalance_frequency == "weekly":
            # First trading day of each (ISO year, ISO week)
            dates_series = pd.Series(trading_dates).sort_values()
            iso = dates_series.dt.isocalendar()
            weekly = dates_series.groupby([iso.year, iso.week]).first()
            return list(weekly)
        elif self.rebalance_frequency == "monthly":
            dates_series = pd.Series(trading_dates).sort_values()
            monthly = dates_series.groupby(
                [dates_series.dt.year, dates_series.dt.month]
            ).first()
            return list(monthly)
        else:
            return list(trading_dates)

    @staticmethod
    def _next_trading_date(
        all_trading_dates: np.ndarray,
        date: datetime,
    ) -> datetime | None:
        """Return the first trading date strictly after `date`."""
        if len(all_trading_dates) == 0:
            return None
        idx = all_trading_dates.searchsorted(np.datetime64(pd.Timestamp(date)), side="right")
        if idx >= len(all_trading_dates):
            return None
        return pd.Timestamp(all_trading_dates[idx])

    def _calculate_period_return(
        self,
        holdings: dict[str, float],
        prices: pd.DataFrame,
        date_from: datetime,
        date_to: datetime,
        from_price_col: str = "close",
        to_price_col: str = "close",
    ) -> float:
        """Calculate portfolio return between two dates."""
        if not holdings:
            return 0.0

        # Get prices at from and to dates
        from_prices = prices[prices["date"] == date_from]
        to_prices = prices[prices["date"] == date_to]

        if from_prices.empty or to_prices.empty:
            return 0.0

        src_col = from_price_col if from_price_col in from_prices.columns else "close"
        dst_col = to_price_col if to_price_col in to_prices.columns else "close"
        if src_col != from_price_col or dst_col != to_price_col:
            logger.warning(
                "Price column fallback in period return: "
                f"from={from_price_col}->{src_col}, to={to_price_col}->{dst_col}"
            )

        from_map = dict(zip(from_prices["ticker"], from_prices[src_col]))
        to_map = dict(zip(to_prices["ticker"], to_prices[dst_col]))

        portfolio_return = 0.0

        for ticker, weight in holdings.items():
            if ticker in from_map and ticker in to_map:
                p0 = from_map[ticker]
                p1 = to_map[ticker]
                if p0 > 0:
                    stock_return = (p1 / p0) - 1
                    portfolio_return += weight * stock_return

        # Cash portion earns nothing
        return portfolio_return

    def _apply_transaction_costs(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
        portfolio_value: float,
    ) -> float:
        """
        Calculate transaction costs for rebalancing.

        Korean market:
        - Buy: commission (1.5 bps) + slippage (10 bps)
        - Sell: commission (1.5 bps) + tax (23 bps) + slippage (10 bps)
        """
        total_cost = 0.0

        all_tickers = set(old_weights) | set(new_weights)

        for ticker in all_tickers:
            old_w = old_weights.get(ticker, 0)
            new_w = new_weights.get(ticker, 0)
            diff = new_w - old_w

            if abs(diff) < 1e-6:
                continue

            trade_value = abs(diff) * portfolio_value

            # Slippage always applies
            slippage = trade_value * self.slippage_bps / 10000
            commission = trade_value * self.commission_bps / 10000

            if diff < 0:
                # Selling: commission + tax + slippage
                tax = trade_value * self.tax_bps / 10000
                total_cost += commission + tax + slippage
            else:
                # Buying: commission + slippage
                total_cost += commission + slippage

        return total_cost
