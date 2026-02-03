"""
Portfolio Constraints

Apply position limits and portfolio constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..common import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioConstraints:
    """
    Portfolio constraints specification.

    Attributes:
        max_weight_per_asset: Maximum weight for any single asset
        min_weight_per_asset: Minimum weight (below this, position is closed)
        max_leverage: Maximum gross leverage (sum of absolute weights)
        max_long_weight: Maximum total long exposure
        max_short_weight: Maximum total short exposure (absolute)
        min_liquidity_adv: Minimum ADV for including asset
        sector_limits: Maximum weight per sector
        max_turnover: Maximum daily turnover
    """

    max_weight_per_asset: float = 0.1
    min_weight_per_asset: float = 0.01
    max_leverage: float = 1.0
    max_long_weight: float = 1.0
    max_short_weight: float = 0.0
    min_liquidity_adv: float = 0.0
    sector_limits: dict[str, float] = field(default_factory=dict)
    max_turnover: float = 1.0


class ConstraintApplier:
    """
    Applies constraints to portfolio weights.
    """

    def __init__(self, constraints: PortfolioConstraints | dict[str, Any] | None = None):
        """
        Initialize constraint applier.

        Args:
            constraints: Portfolio constraints
        """
        if constraints is None:
            self.constraints = PortfolioConstraints()
        elif isinstance(constraints, dict):
            self.constraints = PortfolioConstraints(**constraints)
        else:
            self.constraints = constraints

    def apply(
        self,
        weights: pd.Series,
        current_weights: pd.Series | None = None,
        adv: pd.Series | None = None,
        sectors: pd.Series | None = None,
    ) -> pd.Series:
        """
        Apply all constraints to weights.

        Args:
            weights: Target weights
            current_weights: Current portfolio weights (for turnover)
            adv: Average daily volume per asset
            sectors: Sector mapping per asset

        Returns:
            Constrained weights
        """
        weights = weights.copy()

        # Apply individual constraints
        weights = self._apply_position_limits(weights)
        weights = self._apply_leverage_constraint(weights)
        weights = self._apply_liquidity_filter(weights, adv)
        weights = self._apply_sector_limits(weights, sectors)
        weights = self._apply_turnover_constraint(weights, current_weights)

        # Final normalization
        weights = self._normalize_weights(weights)

        return weights

    def _apply_position_limits(self, weights: pd.Series) -> pd.Series:
        """Apply per-asset position limits."""
        # Clip to max weight
        weights = weights.clip(
            lower=-self.constraints.max_weight_per_asset,
            upper=self.constraints.max_weight_per_asset,
        )

        # Zero out tiny positions
        weights[weights.abs() < self.constraints.min_weight_per_asset] = 0.0

        return weights

    def _apply_leverage_constraint(self, weights: pd.Series) -> pd.Series:
        """Apply leverage constraint."""
        gross_exposure = weights.abs().sum()

        if gross_exposure > self.constraints.max_leverage:
            # Scale down proportionally
            scale = self.constraints.max_leverage / gross_exposure
            weights = weights * scale

        # Apply long/short limits
        long_weight = weights[weights > 0].sum()
        short_weight = weights[weights < 0].sum().abs()

        if long_weight > self.constraints.max_long_weight:
            scale = self.constraints.max_long_weight / long_weight
            weights[weights > 0] *= scale

        if short_weight > self.constraints.max_short_weight:
            if self.constraints.max_short_weight > 0:
                scale = self.constraints.max_short_weight / short_weight
                weights[weights < 0] *= scale
            else:
                # No shorting allowed
                weights[weights < 0] = 0.0

        return weights

    def _apply_liquidity_filter(
        self,
        weights: pd.Series,
        adv: pd.Series | None,
    ) -> pd.Series:
        """Filter out illiquid assets."""
        if adv is None or self.constraints.min_liquidity_adv <= 0:
            return weights

        # Align indices
        common_idx = weights.index.intersection(adv.index)
        illiquid = adv.loc[common_idx] < self.constraints.min_liquidity_adv

        weights.loc[common_idx[illiquid]] = 0.0

        return weights

    def _apply_sector_limits(
        self,
        weights: pd.Series,
        sectors: pd.Series | None,
    ) -> pd.Series:
        """Apply sector weight limits."""
        if sectors is None or not self.constraints.sector_limits:
            return weights

        # Align indices
        common_idx = weights.index.intersection(sectors.index)

        for sector, limit in self.constraints.sector_limits.items():
            sector_mask = sectors.loc[common_idx] == sector
            sector_assets = common_idx[sector_mask]
            sector_weight = weights.loc[sector_assets].sum()

            if sector_weight > limit:
                # Scale down sector proportionally
                scale = limit / sector_weight
                weights.loc[sector_assets] *= scale

        return weights

    def _apply_turnover_constraint(
        self,
        weights: pd.Series,
        current_weights: pd.Series | None,
    ) -> pd.Series:
        """Apply turnover constraint."""
        if current_weights is None or self.constraints.max_turnover >= 2.0:
            return weights

        # Align indices
        all_assets = weights.index.union(current_weights.index)
        weights = weights.reindex(all_assets, fill_value=0.0)
        current_weights = current_weights.reindex(all_assets, fill_value=0.0)

        # Calculate turnover
        trade = weights - current_weights
        turnover = trade.abs().sum()

        if turnover > self.constraints.max_turnover:
            # Scale trades to meet turnover limit
            scale = self.constraints.max_turnover / turnover
            weights = current_weights + trade * scale

        return weights

    def _normalize_weights(self, weights: pd.Series) -> pd.Series:
        """Normalize weights to sum to target."""
        weights = weights[weights != 0]  # Remove zeros

        if len(weights) == 0:
            return weights

        long_weight = weights[weights > 0].sum()
        short_weight = weights[weights < 0].sum().abs()

        # For long-only, normalize to 1
        if short_weight == 0 and long_weight > 0:
            target = min(long_weight, self.constraints.max_long_weight)
            weights = weights * (target / long_weight)

        return weights

    def check_violations(
        self,
        weights: pd.Series,
        adv: pd.Series | None = None,
        sectors: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Check for constraint violations.

        Args:
            weights: Portfolio weights
            adv: Average daily volumes
            sectors: Sector mapping

        Returns:
            Dictionary of violations
        """
        violations = {}

        # Position limits
        max_pos = weights.abs().max()
        if max_pos > self.constraints.max_weight_per_asset * 1.01:
            violations["max_position"] = {
                "limit": self.constraints.max_weight_per_asset,
                "actual": max_pos,
            }

        # Leverage
        leverage = weights.abs().sum()
        if leverage > self.constraints.max_leverage * 1.01:
            violations["leverage"] = {
                "limit": self.constraints.max_leverage,
                "actual": leverage,
            }

        # Long exposure
        long_exp = weights[weights > 0].sum()
        if long_exp > self.constraints.max_long_weight * 1.01:
            violations["long_exposure"] = {
                "limit": self.constraints.max_long_weight,
                "actual": long_exp,
            }

        # Short exposure
        short_exp = weights[weights < 0].sum().abs()
        if short_exp > self.constraints.max_short_weight * 1.01:
            violations["short_exposure"] = {
                "limit": self.constraints.max_short_weight,
                "actual": short_exp,
            }

        return violations


def apply_constraints(
    weights: pd.Series,
    max_weight: float = 0.1,
    max_leverage: float = 1.0,
    long_only: bool = True,
) -> pd.Series:
    """
    Convenience function to apply basic constraints.

    Args:
        weights: Target weights
        max_weight: Maximum weight per asset
        max_leverage: Maximum leverage
        long_only: Whether long-only

    Returns:
        Constrained weights
    """
    constraints = PortfolioConstraints(
        max_weight_per_asset=max_weight,
        max_leverage=max_leverage,
        max_short_weight=0.0 if long_only else max_leverage,
    )
    applier = ConstraintApplier(constraints)
    return applier.apply(weights)
