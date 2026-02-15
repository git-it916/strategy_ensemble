"""
Data Cleaner

Handle missing values, outliers, and data quality issues.
"""

from __future__ import annotations

from typing import Any
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clean and validate financial data.

    Features:
        - Missing value handling
        - Outlier detection and treatment
        - Corporate action adjustment
        - Data quality validation
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize cleaner.

        Args:
            config: Cleaning configuration
        """
        self.config = config or {}

        # Default settings
        self.fill_method = self.config.get("fill_method", "ffill")
        self.max_missing_pct = self.config.get("max_missing_pct", 0.3)
        self.outlier_std = self.config.get("outlier_std", 5.0)

    def clean_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean price data.

        Args:
            df: Price DataFrame with date, ticker, close columns

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        initial_rows = len(df)

        # Ensure datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Remove rows with missing essential data
        essential_cols = ["date", "ticker", "close"]
        for col in essential_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])

        # Remove zero/negative prices
        if "close" in df.columns:
            df = df[df["close"] > 0]

        # Handle missing OHLCV
        ohlcv_cols = ["open", "high", "low", "volume"]
        for col in ohlcv_cols:
            if col in df.columns:
                if col == "volume":
                    df[col] = df[col].fillna(0)
                else:
                    # Fill with close price if missing
                    df[col] = df[col].fillna(df["close"])

        # Sort
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        logger.info(f"Cleaned prices: {initial_rows} -> {len(df)} rows")

        return df

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean feature data.

        Args:
            df: Feature DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        initial_rows = len(df)

        # Remove rows with too many missing values
        feature_cols = [c for c in df.columns if c not in ["date", "ticker"]]

        if feature_cols:
            missing_pct = df[feature_cols].isna().mean(axis=1)
            df = df[missing_pct <= self.max_missing_pct]

        # Forward fill within each asset
        if "ticker" in df.columns:
            df = df.sort_values(["ticker", "date"])
            df[feature_cols] = df.groupby("ticker")[feature_cols].transform(
                lambda x: x.fillna(method="ffill").fillna(method="bfill")
            )

        # Handle remaining NaN - fill with cross-sectional median
        for col in feature_cols:
            if df[col].isna().any():
                df[col] = df.groupby("date")[col].transform(
                    lambda x: x.fillna(x.median())
                )

        # Fill any remaining with 0
        df[feature_cols] = df[feature_cols].fillna(0)

        logger.info(f"Cleaned features: {initial_rows} -> {len(df)} rows")

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        method: str = "zscore",
    ) -> pd.DataFrame:
        """
        Remove or cap outliers.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: "zscore" or "winsorize"

        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [c for c in columns if c not in ["date", "ticker"]]

        for col in columns:
            if col not in df.columns:
                continue

            if method == "zscore":
                # Remove rows with extreme z-scores
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores <= self.outlier_std]

            elif method == "winsorize":
                # Cap at percentiles
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)

        return df

    def validate_data(self, df: pd.DataFrame, data_type: str) -> dict[str, Any]:
        """
        Validate data quality.

        Args:
            df: DataFrame to validate
            data_type: "prices", "features", or "labels"

        Returns:
            Validation report
        """
        report = {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "issues": [],
        }

        # Check for required columns
        if data_type == "prices":
            required = ["date", "ticker", "close"]
        elif data_type == "features":
            required = ["date", "ticker"]
        elif data_type == "labels":
            required = ["date", "ticker", "y_reg"]
        else:
            required = []

        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            report["issues"].append(f"Missing required columns: {missing_cols}")

        # Check for missing values
        missing = df.isna().sum()
        if missing.any():
            high_missing = missing[missing > len(df) * 0.1]
            if not high_missing.empty:
                report["issues"].append(f"High missing rate columns: {high_missing.to_dict()}")

        # Check date range
        if "date" in df.columns:
            report["date_range"] = {
                "min": str(df["date"].min()),
                "max": str(df["date"].max()),
            }

        # Check for duplicates
        if "date" in df.columns and "ticker" in df.columns:
            dupes = df.duplicated(subset=["date", "ticker"]).sum()
            if dupes > 0:
                report["issues"].append(f"Duplicate date-asset pairs: {dupes}")

        # Asset coverage
        if "ticker" in df.columns:
            report["n_assets"] = df["ticker"].nunique()

        report["is_valid"] = len(report["issues"]) == 0

        return report


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to clean price data."""
    cleaner = DataCleaner()
    return cleaner.clean_prices(df)


def clean_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to clean feature data."""
    cleaner = DataCleaner()
    return cleaner.clean_features(df)


def forward_fill_panel(
    df: pd.DataFrame,
    group_col: str = "ticker",
    time_col: str = "date",
    max_gap: int = 5,
) -> pd.DataFrame:
    """
    Forward fill panel data with gap limit.

    Args:
        df: Panel DataFrame
        group_col: Group column (e.g., ticker)
        time_col: Time column
        max_gap: Maximum consecutive gaps to fill

    Returns:
        Filled DataFrame
    """
    df = df.copy()
    df = df.sort_values([group_col, time_col])

    value_cols = [c for c in df.columns if c not in [group_col, time_col]]

    for col in value_cols:
        df[col] = df.groupby(group_col)[col].transform(
            lambda x: x.fillna(method="ffill", limit=max_gap)
        )

    return df
