"""
Feature Normalization Module

Provides cross-sectional normalization with anti-leakage guarantees.

CRITICAL: All normalization is done CROSS-SECTIONALLY (within each date).
This ensures no future information leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..common import get_logger, winsorize as winsorize_array

logger = get_logger(__name__)


class FeatureNormalizer:
    """
    Cross-sectional feature normalizer.

    ANTI-LEAKAGE GUARANTEE:
    - All statistics (mean, std, quantiles) are computed WITHIN each date
    - No information from future dates is used
    - Each date's normalization is independent
    """

    def __init__(
        self,
        method: str = "zscore",
        winsorize: bool = True,
        winsorize_limits: tuple[float, float] = (0.01, 0.99),
        handle_missing: str = "median",
        min_observations: int = 20,
    ):
        """
        Initialize normalizer.

        Args:
            method: Normalization method ('zscore', 'rank', 'minmax')
            winsorize: Whether to winsorize before normalization
            winsorize_limits: Percentile limits for winsorization
            handle_missing: How to handle missing values ('median', 'mean', 'zero', 'drop')
            min_observations: Minimum observations required for valid normalization
        """
        self.method = method
        self.winsorize = winsorize
        self.winsorize_limits = winsorize_limits
        self.handle_missing = handle_missing
        self.min_observations = min_observations

    def normalize(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        date_col: str = "date",
        asset_col: str = "asset_id",
    ) -> pd.DataFrame:
        """
        Normalize features cross-sectionally.

        Args:
            df: Features DataFrame
            feature_cols: Columns to normalize (None = all numeric except date/asset)
            date_col: Date column name
            asset_col: Asset column name

        Returns:
            Normalized DataFrame with same structure
        """
        df = df.copy()

        # Identify feature columns
        if feature_cols is None:
            exclude_cols = {date_col, asset_col}
            feature_cols = [
                c for c in df.columns
                if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
            ]

        if not feature_cols:
            logger.warning("No feature columns to normalize")
            return df

        logger.info(f"Normalizing {len(feature_cols)} features using {self.method} method")

        # Group by date and normalize
        normalized_dfs = []

        for date, group in df.groupby(date_col):
            if len(group) < self.min_observations:
                logger.debug(f"Skipping date {date}: only {len(group)} observations")
                continue

            normalized_group = group.copy()

            for col in feature_cols:
                values = group[col].values.astype(float)
                normalized_values = self._normalize_array(values)
                normalized_group[col] = normalized_values

            normalized_dfs.append(normalized_group)

        if not normalized_dfs:
            logger.warning("No dates with sufficient observations for normalization")
            return df

        result = pd.concat(normalized_dfs, ignore_index=True)
        result = result.sort_values([date_col, asset_col]).reset_index(drop=True)

        return result

    def _normalize_array(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize a single array cross-sectionally.

        Args:
            values: Input array

        Returns:
            Normalized array
        """
        # Handle missing values first
        values = values.copy()
        mask = np.isfinite(values)

        if mask.sum() < self.min_observations:
            return np.full_like(values, np.nan)

        # Winsorize if enabled
        if self.winsorize and mask.sum() > 0:
            valid_values = values[mask]
            lower = np.percentile(valid_values, self.winsorize_limits[0] * 100)
            upper = np.percentile(valid_values, self.winsorize_limits[1] * 100)
            values = np.clip(values, lower, upper)

        # Apply normalization method
        if self.method == "zscore":
            normalized = self._zscore_normalize(values, mask)
        elif self.method == "rank":
            normalized = self._rank_normalize(values, mask)
        elif self.method == "minmax":
            normalized = self._minmax_normalize(values, mask)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        # Handle remaining missing values
        normalized = self._handle_missing(normalized)

        return normalized

    def _zscore_normalize(
        self,
        values: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Z-score normalization."""
        result = np.full_like(values, np.nan)

        if mask.sum() == 0:
            return result

        valid_values = values[mask]
        mean = np.mean(valid_values)
        std = np.std(valid_values, ddof=1)

        if std == 0 or np.isnan(std):
            result[mask] = 0.0
        else:
            result[mask] = (valid_values - mean) / std

        return result

    def _rank_normalize(
        self,
        values: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Rank-based normalization (percentile ranks)."""
        result = np.full_like(values, np.nan)

        if mask.sum() == 0:
            return result

        # Calculate ranks for valid values
        valid_indices = np.where(mask)[0]
        valid_values = values[mask]

        # Rank values (1 to n)
        ranks = stats.rankdata(valid_values, method="average")
        # Convert to percentile (0 to 1)
        percentiles = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.full_like(ranks, 0.5)

        result[valid_indices] = percentiles

        return result

    def _minmax_normalize(
        self,
        values: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Min-max normalization."""
        result = np.full_like(values, np.nan)

        if mask.sum() == 0:
            return result

        valid_values = values[mask]
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)

        if max_val == min_val:
            result[mask] = 0.5
        else:
            result[mask] = (valid_values - min_val) / (max_val - min_val)

        return result

    def _handle_missing(self, values: np.ndarray) -> np.ndarray:
        """Handle remaining missing values."""
        if not np.any(np.isnan(values)):
            return values

        if self.handle_missing == "median":
            fill_value = np.nanmedian(values)
        elif self.handle_missing == "mean":
            fill_value = np.nanmean(values)
        elif self.handle_missing == "zero":
            fill_value = 0.0
        elif self.handle_missing == "drop":
            return values  # Leave as NaN
        else:
            raise ValueError(f"Unknown missing handling method: {self.handle_missing}")

        if np.isnan(fill_value):
            fill_value = 0.0

        values = np.where(np.isnan(values), fill_value, values)
        return values


def normalize_cross_sectional(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    method: str = "zscore",
    winsorize: bool = True,
    winsorize_limits: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """
    Convenience function for cross-sectional normalization.

    ANTI-LEAKAGE: Normalizes WITHIN each date, never using future data.

    Args:
        df: Features DataFrame with columns [date, asset_id, feature_1, ...]
        feature_cols: Columns to normalize (None = auto-detect)
        method: Normalization method ('zscore', 'rank', 'minmax')
        winsorize: Whether to winsorize
        winsorize_limits: Winsorization percentile limits

    Returns:
        Normalized DataFrame
    """
    normalizer = FeatureNormalizer(
        method=method,
        winsorize=winsorize,
        winsorize_limits=winsorize_limits,
    )
    return normalizer.normalize(df, feature_cols)


def normalize_by_group(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    method: str = "zscore",
) -> pd.DataFrame:
    """
    Normalize features within groups (e.g., sectors, industries).

    Args:
        df: Features DataFrame
        feature_cols: Columns to normalize
        group_col: Column to group by
        method: Normalization method

    Returns:
        Normalized DataFrame
    """
    normalizer = FeatureNormalizer(method=method)

    result_dfs = []

    for group_val, group_df in df.groupby(group_col):
        # Normalize within group, but still cross-sectionally by date
        normalized = normalizer.normalize(group_df, feature_cols)
        result_dfs.append(normalized)

    return pd.concat(result_dfs, ignore_index=True)


def standardize_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    clip_std: float = 3.0,
) -> pd.DataFrame:
    """
    Standardize and clip features to limit extreme values.

    Args:
        df: Features DataFrame
        feature_cols: Columns to standardize
        clip_std: Number of standard deviations to clip at

    Returns:
        Standardized DataFrame
    """
    df = df.copy()

    if feature_cols is None:
        exclude_cols = {"date", "asset_id"}
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

    # First normalize
    normalizer = FeatureNormalizer(method="zscore", winsorize=False)
    df = normalizer.normalize(df, feature_cols)

    # Then clip
    for col in feature_cols:
        df[col] = df[col].clip(-clip_std, clip_std)

    return df
