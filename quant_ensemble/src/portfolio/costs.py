"""
Transaction Cost Modeling

Estimate and apply realistic trading costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..common import get_logger

logger = get_logger(__name__)


@dataclass
class CostEstimate:
    """Breakdown of transaction costs."""

    commission: float
    spread: float
    market_impact: float
    total: float
    turnover: float


class TransactionCostModel:
    """
    Transaction cost model for Korean equity market.

    Cost components:
        1. Commission: Broker fee (bps)
        2. Spread: Half bid-ask spread
        3. Market impact: Price impact from trading

    Korean market specifics:
        - Securities transaction tax: 0.23% (sell side)
        - No stamp duty
        - Commission: ~0.015% online brokers
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize cost model.

        Args:
            config: Configuration with:
                - commission_bps: Commission in basis points (default: 1.5)
                - tax_bps: Transaction tax in basis points (default: 23)
                - spread_bps: Default spread in basis points (default: 10)
                - impact_coefficient: Market impact coefficient (default: 0.1)
        """
        self.config = config or {}

        # Commission (buy and sell)
        self.commission_bps = self.config.get("commission_bps", 1.5)

        # Transaction tax (sell only in Korea)
        self.tax_bps = self.config.get("tax_bps", 23.0)

        # Default spread
        self.default_spread_bps = self.config.get("spread_bps", 10.0)

        # Market impact: impact = coefficient * sqrt(participation_rate)
        self.impact_coefficient = self.config.get("impact_coefficient", 0.1)

        # ADV fraction for impact calculation
        self.participation_limit = self.config.get("participation_limit", 0.05)

    def estimate_cost(
        self,
        trade_value: float,
        is_buy: bool,
        spread_bps: float | None = None,
        adv: float | None = None,
    ) -> CostEstimate:
        """
        Estimate transaction cost for a single trade.

        Args:
            trade_value: Absolute trade value
            is_buy: True for buy, False for sell
            spread_bps: Bid-ask spread in bps (optional)
            adv: Average daily volume in value terms (optional)

        Returns:
            CostEstimate breakdown
        """
        if trade_value <= 0:
            return CostEstimate(
                commission=0.0,
                spread=0.0,
                market_impact=0.0,
                total=0.0,
                turnover=0.0,
            )

        # Commission
        commission = trade_value * (self.commission_bps / 10000)

        # Tax (sell only)
        if not is_buy:
            commission += trade_value * (self.tax_bps / 10000)

        # Spread cost (half spread)
        spread_bps_used = spread_bps if spread_bps is not None else self.default_spread_bps
        spread_cost = trade_value * (spread_bps_used / 2 / 10000)

        # Market impact
        if adv is not None and adv > 0:
            participation_rate = min(trade_value / adv, self.participation_limit)
            impact = self.impact_coefficient * np.sqrt(participation_rate) * trade_value
        else:
            # Assume small trade
            impact = 0.0

        total = commission + spread_cost + impact

        return CostEstimate(
            commission=commission,
            spread=spread_cost,
            market_impact=impact,
            total=total,
            turnover=trade_value,
        )

    def estimate_portfolio_cost(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float,
        spreads: pd.Series | None = None,
        advs: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Estimate total rebalancing cost.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            spreads: Bid-ask spreads per asset (bps)
            advs: Average daily volumes per asset

        Returns:
            Cost breakdown
        """
        # Align indices
        all_assets = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_assets, fill_value=0.0)
        target = target_weights.reindex(all_assets, fill_value=0.0)

        trades = target - current

        total_commission = 0.0
        total_spread = 0.0
        total_impact = 0.0
        total_turnover = 0.0

        for asset in trades.index:
            trade_weight = trades[asset]
            if abs(trade_weight) < 1e-6:
                continue

            trade_value = abs(trade_weight) * portfolio_value
            is_buy = trade_weight > 0

            spread_bps = spreads[asset] if spreads is not None and asset in spreads.index else None
            adv = advs[asset] if advs is not None and asset in advs.index else None

            cost = self.estimate_cost(trade_value, is_buy, spread_bps, adv)

            total_commission += cost.commission
            total_spread += cost.spread
            total_impact += cost.market_impact
            total_turnover += cost.turnover

        total_cost = total_commission + total_spread + total_impact
        cost_bps = (total_cost / portfolio_value) * 10000 if portfolio_value > 0 else 0.0
        turnover_pct = total_turnover / portfolio_value if portfolio_value > 0 else 0.0

        return {
            "total_cost": total_cost,
            "total_cost_bps": cost_bps,
            "commission": total_commission,
            "spread": total_spread,
            "market_impact": total_impact,
            "turnover": total_turnover,
            "turnover_pct": turnover_pct,
            "n_trades": (trades.abs() > 1e-6).sum(),
        }

    def adjust_weights_for_cost(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        expected_alpha: pd.Series,
        portfolio_value: float,
        spreads: pd.Series | None = None,
        advs: pd.Series | None = None,
    ) -> pd.Series:
        """
        Adjust target weights to account for transaction costs.

        Only trade if expected alpha exceeds cost.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            expected_alpha: Expected return per asset
            portfolio_value: Total portfolio value
            spreads: Bid-ask spreads
            advs: Average daily volumes

        Returns:
            Adjusted target weights
        """
        all_assets = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_assets, fill_value=0.0)
        target = target_weights.reindex(all_assets, fill_value=0.0)
        alpha = expected_alpha.reindex(all_assets, fill_value=0.0)

        adjusted = target.copy()

        for asset in all_assets:
            trade_weight = target[asset] - current[asset]

            if abs(trade_weight) < 1e-6:
                continue

            trade_value = abs(trade_weight) * portfolio_value
            is_buy = trade_weight > 0

            spread_bps = spreads[asset] if spreads is not None and asset in spreads.index else None
            adv = advs[asset] if advs is not None and asset in advs.index else None

            cost = self.estimate_cost(trade_value, is_buy, spread_bps, adv)
            cost_as_return = cost.total / trade_value if trade_value > 0 else 0.0

            # Compare expected alpha to cost
            asset_alpha = abs(alpha[asset])

            if asset_alpha < cost_as_return:
                # Don't trade if cost exceeds expected alpha
                adjusted[asset] = current[asset]

        return adjusted


class SlippageModel:
    """
    Slippage model for execution simulation.

    Models the difference between expected and actual execution price.
    """

    def __init__(
        self,
        base_slippage_bps: float = 5.0,
        volatility_multiplier: float = 0.5,
    ):
        """
        Initialize slippage model.

        Args:
            base_slippage_bps: Base slippage in basis points
            volatility_multiplier: Multiplier for volatility-based slippage
        """
        self.base_slippage_bps = base_slippage_bps
        self.volatility_multiplier = volatility_multiplier

    def estimate_slippage(
        self,
        trade_value: float,
        volatility: float = 0.0,
        participation_rate: float = 0.0,
    ) -> float:
        """
        Estimate slippage for a trade.

        Args:
            trade_value: Absolute trade value
            volatility: Asset daily volatility
            participation_rate: Trade size / ADV

        Returns:
            Estimated slippage in value terms
        """
        # Base slippage
        slippage_bps = self.base_slippage_bps

        # Add volatility component
        slippage_bps += volatility * 100 * self.volatility_multiplier

        # Add participation component
        slippage_bps += participation_rate * 100

        slippage = trade_value * (slippage_bps / 10000)

        return slippage

    def apply_slippage(
        self,
        price: float,
        is_buy: bool,
        volatility: float = 0.02,
    ) -> float:
        """
        Apply slippage to execution price.

        Args:
            price: Expected price
            is_buy: True for buy, False for sell
            volatility: Asset volatility

        Returns:
            Execution price with slippage
        """
        slippage_pct = (self.base_slippage_bps + volatility * 100 * self.volatility_multiplier) / 10000

        # Add random component
        random_factor = np.random.uniform(0.5, 1.5)
        slippage_pct *= random_factor

        if is_buy:
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)


def estimate_annual_cost(
    turnover_annual: float,
    avg_spread_bps: float = 10.0,
    commission_bps: float = 1.5,
    tax_bps: float = 23.0,
) -> float:
    """
    Quick estimate of annual transaction costs.

    Args:
        turnover_annual: Annual turnover (1.0 = 100% turnover)
        avg_spread_bps: Average spread
        commission_bps: Commission per trade
        tax_bps: Transaction tax (sell only)

    Returns:
        Estimated annual cost as percentage
    """
    # One-way cost
    buy_cost = commission_bps + avg_spread_bps / 2
    sell_cost = commission_bps + tax_bps + avg_spread_bps / 2

    # Two-way cost per unit turnover
    round_trip_cost = buy_cost + sell_cost

    # Annual cost
    annual_cost_bps = turnover_annual * round_trip_cost

    return annual_cost_bps / 100  # Convert to percentage
