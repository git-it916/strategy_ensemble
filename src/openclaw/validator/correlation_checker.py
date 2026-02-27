"""
Correlation Checker

Verify that a new alpha's returns are sufficiently uncorrelated
with existing active alphas (max correlation <= 0.3).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.openclaw.config import QUALITY_GATES

logger = logging.getLogger(__name__)


class CorrelationChecker:
    """
    Check return correlation of a new alpha against existing active alphas.

    Ensures portfolio diversification by rejecting alphas
    that are too correlated with existing ones.
    """

    def __init__(self, max_correlation: float | None = None):
        self.max_correlation = (
            max_correlation
            if max_correlation is not None
            else QUALITY_GATES.max_correlation
        )

    def check(
        self,
        new_alpha_returns: pd.Series,
        active_alpha_returns: dict[str, pd.Series],
    ) -> tuple[bool, dict[str, float]]:
        """
        Check correlation of new alpha against all active alphas.

        Args:
            new_alpha_returns: Daily returns of the new alpha
            active_alpha_returns: {alpha_name: daily_returns} for active alphas

        Returns:
            (passed: bool, correlations: {alpha_name: correlation_value})
        """
        if not active_alpha_returns:
            logger.info("No active alphas to compare — correlation check passed")
            return True, {}

        correlations = {}

        for name, existing_returns in active_alpha_returns.items():
            # Align on common dates
            combined = pd.concat(
                [new_alpha_returns.rename("new"), existing_returns.rename("existing")],
                axis=1,
            ).dropna()

            if len(combined) < 20:
                logger.warning(
                    f"Insufficient overlapping data with {name} "
                    f"({len(combined)} days), skipping"
                )
                continue

            corr = combined["new"].corr(combined["existing"])
            correlations[name] = round(float(corr), 4) if not np.isnan(corr) else 0.0

        if not correlations:
            return True, {}

        max_corr_name = max(correlations, key=lambda k: abs(correlations[k]))
        max_corr_val = abs(correlations[max_corr_name])

        passed = max_corr_val <= self.max_correlation

        log_fn = logger.info if passed else logger.warning
        log_fn(
            f"Correlation check: {'PASSED' if passed else 'FAILED'} "
            f"(max |corr| = {max_corr_val:.3f} with {max_corr_name}, "
            f"threshold = {self.max_correlation})"
        )

        return passed, correlations
