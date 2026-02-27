"""
Leverage Calculator

MDD-based leverage calculation.
target_leverage = min(max_leverage, target_mdd / abs(historical_mdd))
Capped at default_leverage_cap (4x) unless overridden.
"""

from __future__ import annotations

import logging

from src.openclaw.config import EXECUTION_POLICY, ExecutionPolicy
from src.openclaw.registry.alpha_registry import AlphaEntry
from src.openclaw.registry.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class LeverageCalculator:
    """
    Calculate leverage per alpha based on historical MDD.

    Formula:
        raw = target_mdd / abs(historical_mdd)
        capped = min(raw, max_leverage)
        final = min(capped, default_leverage_cap)

    Examples (with target_mdd=0.20, max=5x, default_cap=4x):
        MDD = -5%  → 0.20/0.05 = 4.0 → 4.0x
        MDD = -10% → 0.20/0.10 = 2.0 → 2.0x
        MDD = -3%  → 0.20/0.03 = 6.67 → 5.0 → 4.0x
        MDD = -20% → 0.20/0.20 = 1.0 → 1.0x
    """

    def __init__(self, policy: ExecutionPolicy | None = None):
        self.policy = policy or EXECUTION_POLICY

    def calculate(self, historical_mdd: float) -> float:
        """
        Calculate leverage from a single MDD value.

        Args:
            historical_mdd: Historical max drawdown (negative, e.g. -0.10)

        Returns:
            Recommended leverage (float, >= 1.0)
        """
        abs_mdd = abs(historical_mdd)

        if abs_mdd < 0.01:
            # Very low MDD — cap at default
            return self.policy.default_leverage_cap

        raw = self.policy.target_mdd_for_leverage / abs_mdd

        # Apply hard cap
        capped = min(raw, self.policy.max_leverage)

        # Apply default soft cap (prefer 4x or less)
        final = min(capped, self.policy.default_leverage_cap)

        # Floor at 1x
        final = max(final, 1.0)

        return round(final, 2)

    def calculate_per_alpha(
        self,
        alpha_entries: list[AlphaEntry],
        performance_tracker: PerformanceTracker,
        lookback: int = 252,
    ) -> dict[str, float]:
        """
        Calculate leverage for each alpha based on its historical MDD.

        Args:
            alpha_entries: List of active alpha entries
            performance_tracker: Performance data source
            lookback: Days of history to consider for MDD

        Returns:
            {alpha_name: leverage}
        """
        leverages = {}

        for entry in alpha_entries:
            mdd = performance_tracker.get_rolling_mdd(
                entry.name, window=lookback
            )

            if mdd == 0.0:
                # No data yet — use backtest MDD if available
                mdd = entry.oos_mdd if entry.oos_mdd != 0 else -0.10

            leverage = self.calculate(mdd)
            leverages[entry.name] = leverage

            logger.debug(
                f"Leverage for {entry.name}: {leverage:.1f}x "
                f"(MDD={mdd:.2%})"
            )

        return leverages

    def calculate_portfolio_leverage(
        self,
        alpha_leverages: dict[str, float],
        alpha_weights: dict[str, float],
    ) -> float:
        """
        Calculate weighted average portfolio leverage.

        Args:
            alpha_leverages: {alpha_name: leverage}
            alpha_weights: {alpha_name: weight}

        Returns:
            Portfolio-level leverage.
        """
        total_weight = sum(alpha_weights.values())
        if total_weight == 0:
            return 1.0

        weighted_lev = sum(
            alpha_leverages.get(name, 1.0) * weight
            for name, weight in alpha_weights.items()
        )

        return round(weighted_lev / total_weight, 2)
