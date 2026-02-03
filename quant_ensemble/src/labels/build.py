"""
Label Generation Module

Builds supervised learning labels from price data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..common import get_logger, load_yaml

logger = get_logger(__name__)


class LabelBuilder:
    """
    Builds labels for supervised learning.

    Supports:
    - Regression labels (forward returns)
    - Classification labels (quantile-based)
    - Ranking labels (cross-sectional ranks)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize label builder.

        Args:
            config: Label configuration dictionary
        """
        self.config = config or {}

        # Default configuration
        self.horizon_days = self.config.get("horizon_days", 21)
        self.n_quantiles = self.config.get("n_quantiles", 5)
        self.excess_return = self.config.get("excess_return", True)
        self.return_type = self.config.get("return_type", "simple")

    @classmethod
    def from_config_file(cls, config_path: str) -> "LabelBuilder":
        """Create LabelBuilder from config file."""
        config = load_yaml(config_path)
        return cls(config.get("labels", {}))

    def build_all_labels(
        self,
        prices_df: pd.DataFrame,
        benchmark_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build all label types.

        Args:
            prices_df: Price DataFrame with columns [date, asset_id, close]
            benchmark_df: Benchmark price DataFrame (optional, for excess returns)

        Returns:
            Labels DataFrame with columns [date, asset_id, y_reg, y_cls, y_rank, sample_weight, purge_group]
        """
        logger.info(f"Building labels with {self.horizon_days} day horizon")

        # Ensure datetime
        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        # Build regression label (forward return)
        forward_returns = self._calculate_forward_returns(prices_df)

        # Build excess returns if benchmark provided
        if benchmark_df is not None and self.excess_return:
            benchmark_returns = self._calculate_benchmark_returns(benchmark_df)
            forward_returns = self._subtract_benchmark(forward_returns, benchmark_returns)

        # Build all labels
        labels_df = forward_returns.copy()
        labels_df = labels_df.rename(columns={"forward_return": "y_reg"})

        # Add classification labels
        labels_df = self._add_classification_labels(labels_df)

        # Add ranking labels
        labels_df = self._add_ranking_labels(labels_df)

        # Add sample weights
        labels_df = self._add_sample_weights(labels_df)

        # Add purge group identifier
        labels_df = self._add_purge_group(labels_df)

        # Filter quality
        labels_df = self._filter_quality(labels_df)

        logger.info(f"Built labels: {labels_df.shape[0]} rows")

        return labels_df

    def _calculate_forward_returns(
        self,
        prices_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate forward returns for each asset."""
        # Sort by asset and date
        df = prices_df.sort_values(["asset_id", "date"]).copy()

        # Calculate forward return
        if self.return_type == "log":
            df["forward_return"] = (
                df.groupby("asset_id")["close"]
                .transform(lambda x: np.log(x.shift(-self.horizon_days) / x))
            )
        else:  # simple return
            df["forward_return"] = (
                df.groupby("asset_id")["close"]
                .transform(lambda x: x.shift(-self.horizon_days) / x - 1)
            )

        return df[["date", "asset_id", "forward_return"]]

    def _calculate_benchmark_returns(
        self,
        benchmark_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate benchmark forward returns."""
        df = benchmark_df.sort_values("date").copy()

        if self.return_type == "log":
            df["benchmark_return"] = np.log(df["close"].shift(-self.horizon_days) / df["close"])
        else:
            df["benchmark_return"] = df["close"].shift(-self.horizon_days) / df["close"] - 1

        return df[["date", "benchmark_return"]]

    def _subtract_benchmark(
        self,
        returns_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Subtract benchmark returns to get excess returns."""
        merged = returns_df.merge(benchmark_df, on="date", how="left")
        merged["forward_return"] = merged["forward_return"] - merged["benchmark_return"]
        return merged[["date", "asset_id", "forward_return"]]

    def _add_classification_labels(
        self,
        labels_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add classification labels based on quantiles."""
        df = labels_df.copy()

        # Calculate quantile within each date (cross-sectional)
        df["y_cls"] = df.groupby("date")["y_reg"].transform(
            lambda x: pd.qcut(
                x,
                q=self.n_quantiles,
                labels=False,
                duplicates="drop"
            ) if len(x.dropna()) >= self.n_quantiles else np.nan
        )

        # Convert top quantile to binary
        if self.config.get("top_quantile_positive", True):
            df["y_cls_binary"] = (df["y_cls"] == self.n_quantiles - 1).astype(int)
        else:
            df["y_cls_binary"] = df["y_cls"]

        return df

    def _add_ranking_labels(
        self,
        labels_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add cross-sectional ranking labels."""
        df = labels_df.copy()

        # Rank within each date (higher return = higher rank)
        df["y_rank"] = df.groupby("date")["y_reg"].transform(
            lambda x: x.rank(method="average", ascending=True)
        )

        # Normalize rank to [0, 1]
        df["y_rank_norm"] = df.groupby("date")["y_rank"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        )

        return df

    def _add_sample_weights(
        self,
        labels_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add sample weights for training."""
        df = labels_df.copy()

        weight_method = self.config.get("weighting", {}).get("method", "uniform")

        if weight_method == "uniform":
            df["sample_weight"] = 1.0

        elif weight_method == "inverse_variance":
            # Weight by inverse of return magnitude (reduce extreme influence)
            df["sample_weight"] = 1.0 / (1.0 + df["y_reg"].abs())
            df["sample_weight"] = df["sample_weight"] / df["sample_weight"].mean()

        elif weight_method == "return_magnitude":
            # Weight by return magnitude (emphasize larger moves)
            df["sample_weight"] = df["y_reg"].abs()
            df["sample_weight"] = df["sample_weight"] / df["sample_weight"].mean()

        else:
            df["sample_weight"] = 1.0

        # Clip weights
        min_weight = self.config.get("weighting", {}).get("min_weight", 0.1)
        max_weight = self.config.get("weighting", {}).get("max_weight", 10.0)
        df["sample_weight"] = df["sample_weight"].clip(min_weight, max_weight)

        return df

    def _add_purge_group(
        self,
        labels_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add purge group identifier for train/test splitting."""
        df = labels_df.copy()

        # Purge group combines date and horizon to identify overlapping samples
        df["purge_group"] = (
            df["date"].dt.strftime("%Y%m%d") + "_" +
            df["asset_id"].astype(str) + "_" +
            str(self.horizon_days)
        )

        return df

    def _filter_quality(
        self,
        labels_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter out low-quality labels."""
        df = labels_df.copy()

        initial_count = len(df)

        # Remove missing labels
        df = df.dropna(subset=["y_reg"])

        # Filter extreme returns (likely data errors or corporate actions)
        min_return = self.config.get("min_return", -0.9)
        max_return = self.config.get("max_return", 5.0)
        df = df[(df["y_reg"] >= min_return) & (df["y_reg"] <= max_return)]

        logger.info(f"Filtered labels: {initial_count} -> {len(df)} ({len(df)/initial_count*100:.1f}%)")

        return df


def build_labels(
    prices_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    horizon_days: int = 21,
    n_quantiles: int = 5,
    excess_return: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to build labels.

    Args:
        prices_df: Price DataFrame
        benchmark_df: Benchmark DataFrame (optional)
        horizon_days: Forward return horizon
        n_quantiles: Number of quantiles for classification
        excess_return: Whether to compute excess returns

    Returns:
        Labels DataFrame
    """
    config = {
        "horizon_days": horizon_days,
        "n_quantiles": n_quantiles,
        "excess_return": excess_return,
    }

    builder = LabelBuilder(config)
    return builder.build_all_labels(prices_df, benchmark_df)


def calculate_forward_returns(
    prices_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Calculate forward returns at multiple horizons.

    Args:
        prices_df: Price DataFrame with columns [date, asset_id, close]
        horizons: List of horizon days (default: [5, 10, 21, 63])

    Returns:
        DataFrame with forward returns at each horizon
    """
    if horizons is None:
        horizons = [5, 10, 21, 63]

    df = prices_df.sort_values(["asset_id", "date"]).copy()

    for horizon in horizons:
        df[f"fwd_ret_{horizon}d"] = (
            df.groupby("asset_id")["close"]
            .transform(lambda x: x.shift(-horizon) / x - 1)
        )

    return df
