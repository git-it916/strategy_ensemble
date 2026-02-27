"""
Weight Optimizer

Deterministic weight allocation strategies for alpha ensemble.
Supports equal-weight, risk-parity, and LLM-blended modes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.openclaw.config import ALLOCATION_POLICY, AllocationPolicy

logger = logging.getLogger(__name__)


class WeightOptimizer:
    """
    Deterministic weight allocation methods for the alpha ensemble.
    """

    def __init__(self, policy: AllocationPolicy | None = None):
        self.policy = policy or ALLOCATION_POLICY

    def equal_weight(self, alpha_names: list[str]) -> dict[str, float]:
        """Equal weight across all alphas."""
        if not alpha_names:
            return {}
        w = 1.0 / len(alpha_names)
        return {name: w for name in alpha_names}

    def risk_parity(
        self,
        alpha_returns: dict[str, pd.Series],
        lookback: int | None = None,
    ) -> dict[str, float]:
        """
        Inverse-volatility weighting across alphas.

        Alphas with lower volatility get higher weight.
        """
        lookback = lookback or self.policy.lookback_days

        if not alpha_returns:
            return {}

        vols = {}
        for name, returns in alpha_returns.items():
            recent = returns.tail(lookback)
            if len(recent) < 10:
                vols[name] = 0.3  # default vol assumption
            else:
                vol = recent.std() * np.sqrt(252)
                vols[name] = max(vol, 0.01)  # floor

        # Inverse vol weights
        inv_vols = {name: 1.0 / vol for name, vol in vols.items()}
        total = sum(inv_vols.values())

        weights = {name: iv / total for name, iv in inv_vols.items()}

        # Apply min/max constraints
        weights = self._apply_constraints(weights)

        return weights

    def blend_with_llm(
        self,
        deterministic_weights: dict[str, float],
        llm_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Blend deterministic and LLM weights.

        final = risk_parity_blend * deterministic + llm_blend * llm
        Default: 60% risk-parity + 40% LLM
        """
        if not llm_weights:
            return deterministic_weights

        all_names = set(deterministic_weights) | set(llm_weights)
        blended = {}

        rp_blend = self.policy.risk_parity_blend
        llm_blend = self.policy.llm_blend

        for name in all_names:
            det_w = deterministic_weights.get(name, 0.0)
            llm_w = llm_weights.get(name, 0.0)
            blended[name] = rp_blend * det_w + llm_blend * llm_w

        # Renormalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        # Apply constraints
        blended = self._apply_constraints(blended)

        return blended

    def _apply_constraints(
        self, weights: dict[str, float]
    ) -> dict[str, float]:
        """Apply min/max weight constraints and renormalize."""
        constrained = {}

        for name, w in weights.items():
            w = max(w, self.policy.min_weight)
            w = min(w, self.policy.max_weight)
            constrained[name] = w

        # Renormalize
        total = sum(constrained.values())
        if total > 0:
            constrained = {k: v / total for k, v in constrained.items()}

        return constrained
