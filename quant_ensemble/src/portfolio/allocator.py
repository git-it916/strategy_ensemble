"""
Portfolio Allocator

Converts signal scores to portfolio weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..common import get_logger

logger = get_logger(__name__)


class PortfolioAllocator:
    """
    Converts signal scores to portfolio weights.

    Supports multiple allocation methods:
    - Top-K equal weighted
    - Softmax based
    - Z-score based
    - Mean-variance optimization
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize allocator.

        Args:
            config: Configuration with:
                - method: Allocation method
                - topk: Number of positions for top-k
                - max_weight: Maximum weight per asset
                - min_weight: Minimum weight per asset
                - long_only: Whether to allow short positions
        """
        self.config = config or {}
        self.method = self.config.get("method", "topk")
        self.topk = self.config.get("topk", 20)
        self.max_weight = self.config.get("max_weight", 0.1)
        self.min_weight = self.config.get("min_weight", 0.01)
        self.long_only = self.config.get("long_only", True)
        self.softmax_temperature = self.config.get("softmax_temperature", 1.0)
        self.zscore_cap = self.config.get("zscore_cap", 2.0)

    def allocate(
        self,
        scores: pd.Series | pd.DataFrame,
        prices: pd.Series | None = None,
        volumes: pd.Series | None = None,
    ) -> pd.Series:
        """
        Allocate portfolio weights based on scores.

        Args:
            scores: Signal scores (Series with asset_id index or DataFrame)
            prices: Current prices (optional, for constraints)
            volumes: Trading volumes (optional, for liquidity constraints)

        Returns:
            Portfolio weights (Series with asset_id index)
        """
        # Convert DataFrame to Series if needed
        if isinstance(scores, pd.DataFrame):
            if "score" in scores.columns:
                scores = scores.set_index("asset_id")["score"]
            else:
                scores = scores.iloc[:, 0]

        # Remove NaN scores
        scores = scores.dropna()

        if len(scores) == 0:
            return pd.Series(dtype=float)

        # Apply allocation method
        if self.method == "topk":
            weights = self.topk_weights(scores)
        elif self.method == "softmax":
            weights = self.softmax_weights(scores)
        elif self.method == "zscore":
            weights = self.zscore_weights(scores)
        elif self.method == "optimization":
            weights = self.optimization_weights(scores)
        else:
            weights = self.topk_weights(scores)

        # Apply constraints
        weights = self._apply_constraints(weights)

        return weights

    def topk_weights(self, scores: pd.Series) -> pd.Series:
        """
        Top-K equal-weighted allocation.

        Args:
            scores: Signal scores

        Returns:
            Weights with top-K assets equally weighted
        """
        k = min(self.topk, len(scores))

        if self.long_only:
            # Long only: top K assets
            top_assets = scores.nlargest(k).index
            weights = pd.Series(0.0, index=scores.index)
            weights[top_assets] = 1.0 / k
        else:
            # Long-short: top K long, bottom K short
            k_half = k // 2
            top_assets = scores.nlargest(k_half).index
            bottom_assets = scores.nsmallest(k_half).index

            weights = pd.Series(0.0, index=scores.index)
            weights[top_assets] = 1.0 / k_half / 2
            weights[bottom_assets] = -1.0 / k_half / 2

        return weights

    def softmax_weights(self, scores: pd.Series) -> pd.Series:
        """
        Softmax-based allocation.

        Args:
            scores: Signal scores

        Returns:
            Weights based on softmax of scores
        """
        # Center scores
        centered = scores - scores.mean()

        # Apply temperature
        scaled = centered / self.softmax_temperature

        # Softmax
        exp_scores = np.exp(scaled - scaled.max())  # Subtract max for numerical stability
        weights = exp_scores / exp_scores.sum()

        if self.long_only:
            weights = weights.clip(lower=0)
            weights = weights / weights.sum()

        return weights

    def zscore_weights(self, scores: pd.Series) -> pd.Series:
        """
        Z-score based allocation with capping.

        Args:
            scores: Signal scores

        Returns:
            Weights based on z-scored and capped scores
        """
        # Z-score
        mean = scores.mean()
        std = scores.std()

        if std == 0:
            return pd.Series(1.0 / len(scores), index=scores.index)

        zscores = (scores - mean) / std

        # Cap z-scores
        zscores = zscores.clip(-self.zscore_cap, self.zscore_cap)

        if self.long_only:
            # Shift to positive and normalize
            weights = zscores - zscores.min() + 0.01
            weights = weights / weights.sum()
        else:
            # Normalize to sum to 0 (market neutral)
            weights = zscores / zscores.abs().sum()

        return weights

    def optimization_weights(
        self,
        scores: pd.Series,
        cov_matrix: pd.DataFrame | None = None,
    ) -> pd.Series:
        """
        Mean-variance optimization based allocation.

        Args:
            scores: Signal scores (used as expected returns)
            cov_matrix: Covariance matrix (optional)

        Returns:
            Optimized weights
        """
        n = len(scores)

        if cov_matrix is None:
            # Use identity covariance (equal risk)
            cov_matrix = pd.DataFrame(
                np.eye(n) * 0.04,  # Assume 20% vol for each asset
                index=scores.index,
                columns=scores.index,
            )

        # Align covariance with scores
        common_idx = scores.index.intersection(cov_matrix.index)
        scores = scores.loc[common_idx]
        cov_matrix = cov_matrix.loc[common_idx, common_idx]
        n = len(scores)

        if n == 0:
            return pd.Series(dtype=float)

        # Convert to arrays
        mu = scores.values
        sigma = cov_matrix.values

        # Risk aversion parameter
        gamma = self.config.get("risk_aversion", 1.0)

        # Optimization
        def objective(w):
            ret = np.dot(w, mu)
            risk = np.dot(w, np.dot(sigma, w))
            return -(ret - gamma * risk)  # Minimize negative utility

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]  # Sum to 1

        # Bounds
        if self.long_only:
            bounds = [(0, self.max_weight) for _ in range(n)]
        else:
            bounds = [(-self.max_weight, self.max_weight) for _ in range(n)]

        # Initial guess
        w0 = np.ones(n) / n

        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = pd.Series(result.x, index=scores.index)
        else:
            logger.warning("Optimization failed, using equal weights")
            weights = pd.Series(1.0 / n, index=scores.index)

        return weights

    def _apply_constraints(self, weights: pd.Series) -> pd.Series:
        """Apply weight constraints."""
        # Apply max weight
        weights = weights.clip(upper=self.max_weight, lower=-self.max_weight)

        # Remove tiny weights
        weights[weights.abs() < self.min_weight] = 0.0

        # Renormalize
        if weights.abs().sum() > 0:
            if self.long_only:
                weights = weights / weights.sum()
            else:
                # For long-short, normalize by gross exposure
                weights = weights / weights.abs().sum()

        return weights


def allocate_portfolio(
    scores: pd.Series | pd.DataFrame,
    method: str = "topk",
    topk: int = 20,
    max_weight: float = 0.1,
    long_only: bool = True,
) -> pd.Series:
    """
    Convenience function for portfolio allocation.

    Args:
        scores: Signal scores
        method: Allocation method
        topk: Number of positions
        max_weight: Maximum weight per asset
        long_only: Whether long-only

    Returns:
        Portfolio weights
    """
    allocator = PortfolioAllocator({
        "method": method,
        "topk": topk,
        "max_weight": max_weight,
        "long_only": long_only,
    })
    return allocator.allocate(scores)
