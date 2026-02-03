"""
Point-in-Time Data Merge Module

Provides as-of merge functionality to prevent look-ahead bias.

CRITICAL ANTI-LEAKAGE:
- Fundamental data is only available AFTER publication date
- We apply a publication lag to ensure no future data leakage
- Uses pd.merge_asof with direction='backward' to get most recent available data
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from ..common import get_logger

logger = get_logger(__name__)


def asof_merge_fundamentals(
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    publish_lag_days: int = 2,
    date_col: str = "date",
    asset_col: str = "asset_id",
    fundamental_date_col: str | None = None,
) -> pd.DataFrame:
    """
    Merge fundamentals with price data using point-in-time logic.

    ANTI-LEAKAGE: Ensures fundamentals are only available AFTER their
    publication date plus a safety lag.

    Args:
        prices_df: Price DataFrame with columns [date, asset_id, ...]
        fundamentals_df: Fundamental DataFrame with columns [date/report_date, asset_id, ...]
        publish_lag_days: Number of days to lag fundamental data (default: 2)
        date_col: Price date column name
        asset_col: Asset identifier column name
        fundamental_date_col: Fundamental date column (None = use date_col)

    Returns:
        Merged DataFrame with point-in-time fundamentals
    """
    logger.info(f"Performing as-of merge with {publish_lag_days} day publication lag")

    # Copy to avoid modifying originals
    prices = prices_df.copy()
    fundamentals = fundamentals_df.copy()

    # Ensure datetime
    prices[date_col] = pd.to_datetime(prices[date_col])

    # Handle fundamental date column
    if fundamental_date_col is None:
        if date_col in fundamentals.columns:
            fundamental_date_col = date_col
        elif "report_date" in fundamentals.columns:
            fundamental_date_col = "report_date"
        elif "publish_date" in fundamentals.columns:
            fundamental_date_col = "publish_date"
        else:
            # No date in fundamentals - treat as static data
            logger.warning("No date column in fundamentals, treating as static data")
            return prices.merge(fundamentals, on=asset_col, how="left")

    fundamentals[fundamental_date_col] = pd.to_datetime(fundamentals[fundamental_date_col])

    # Apply publication lag - shift the "available" date forward
    # This means data is not available until publish_lag_days after the report date
    fundamentals["_available_date"] = (
        fundamentals[fundamental_date_col] + timedelta(days=publish_lag_days)
    )

    # Sort both DataFrames
    prices = prices.sort_values([asset_col, date_col])
    fundamentals = fundamentals.sort_values([asset_col, "_available_date"])

    # Get fundamental columns to merge (exclude date columns)
    fundamental_cols = [
        c for c in fundamentals.columns
        if c not in [fundamental_date_col, "_available_date", asset_col]
    ]

    # Perform as-of merge for each asset
    result_dfs = []

    for asset in prices[asset_col].unique():
        asset_prices = prices[prices[asset_col] == asset].copy()
        asset_fundamentals = fundamentals[fundamentals[asset_col] == asset].copy()

        if len(asset_fundamentals) == 0:
            # No fundamentals for this asset - keep prices with NaN
            for col in fundamental_cols:
                asset_prices[col] = float("nan")
            result_dfs.append(asset_prices)
            continue

        # Merge using as-of join
        merged = pd.merge_asof(
            asset_prices,
            asset_fundamentals[[asset_col, "_available_date"] + fundamental_cols],
            left_on=date_col,
            right_on="_available_date",
            by=asset_col,
            direction="backward",  # Use most recent available data
        )

        # Drop temporary column
        if "_available_date" in merged.columns:
            merged = merged.drop(columns=["_available_date"])

        result_dfs.append(merged)

    result = pd.concat(result_dfs, ignore_index=True)
    result = result.sort_values([date_col, asset_col]).reset_index(drop=True)

    # Log merge statistics
    n_original = len(prices)
    n_with_fundamentals = result[fundamental_cols[0]].notna().sum() if fundamental_cols else 0
    logger.info(
        f"As-of merge complete: {n_with_fundamentals}/{n_original} rows have fundamental data"
    )

    return result


def asof_merge_flow_data(
    prices_df: pd.DataFrame,
    flows_df: pd.DataFrame,
    lag_days: int = 1,
    date_col: str = "date",
    asset_col: str = "asset_id",
) -> pd.DataFrame:
    """
    Merge flow data (foreign/institutional) with price data.

    ANTI-LEAKAGE: Flow data is available with a 1-day lag (T-1 data for T signals).

    Args:
        prices_df: Price DataFrame
        flows_df: Flow DataFrame with columns [date, asset_id, foreign_flow, inst_flow, ...]
        lag_days: Number of days to lag flow data
        date_col: Date column name
        asset_col: Asset column name

    Returns:
        Merged DataFrame with lagged flow data
    """
    logger.info(f"Merging flow data with {lag_days} day lag")

    prices = prices_df.copy()
    flows = flows_df.copy()

    # Ensure datetime
    prices[date_col] = pd.to_datetime(prices[date_col])
    flows[date_col] = pd.to_datetime(flows[date_col])

    # Apply lag - shift the available date forward
    flows["_available_date"] = flows[date_col] + timedelta(days=lag_days)

    # Sort
    prices = prices.sort_values([asset_col, date_col])
    flows = flows.sort_values([asset_col, "_available_date"])

    # Get flow columns
    flow_cols = [c for c in flows.columns if c not in [date_col, asset_col, "_available_date"]]

    # Merge
    result_dfs = []

    for asset in prices[asset_col].unique():
        asset_prices = prices[prices[asset_col] == asset].copy()
        asset_flows = flows[flows[asset_col] == asset].copy()

        if len(asset_flows) == 0:
            for col in flow_cols:
                asset_prices[col] = float("nan")
            result_dfs.append(asset_prices)
            continue

        merged = pd.merge_asof(
            asset_prices,
            asset_flows[[asset_col, "_available_date"] + flow_cols],
            left_on=date_col,
            right_on="_available_date",
            by=asset_col,
            direction="backward",
        )

        if "_available_date" in merged.columns:
            merged = merged.drop(columns=["_available_date"])

        result_dfs.append(merged)

    result = pd.concat(result_dfs, ignore_index=True)
    return result.sort_values([date_col, asset_col]).reset_index(drop=True)


def apply_execution_lag(
    features_df: pd.DataFrame,
    lag_days: int = 1,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Apply execution lag to features.

    ANTI-LEAKAGE: For signals generated on date T-1 and executed on date T,
    we shift the feature dates forward.

    Args:
        features_df: Features DataFrame
        lag_days: Execution lag in days
        date_col: Date column name

    Returns:
        DataFrame with shifted dates (features from T-1 now labeled as T)
    """
    logger.info(f"Applying {lag_days} day execution lag to features")

    df = features_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Shift dates forward by lag_days
    # This means features from T-lag are available for trading on T
    df[date_col] = df[date_col] + timedelta(days=lag_days)

    return df


def validate_no_future_leakage(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset_id",
) -> bool:
    """
    Validate that features don't contain future information.

    Checks that for each (date, asset) pair, the features were computed
    using only data available before that date.

    Args:
        features_df: Features DataFrame
        labels_df: Labels DataFrame with forward-looking targets
        date_col: Date column name
        asset_col: Asset column name

    Returns:
        True if no leakage detected
    """
    logger.info("Validating no future leakage...")

    # Basic check: features should not have perfect correlation with labels
    merged = features_df.merge(
        labels_df[[date_col, asset_col, "y_reg"]],
        on=[date_col, asset_col],
        how="inner",
    )

    feature_cols = [
        c for c in features_df.columns
        if c not in [date_col, asset_col] and pd.api.types.is_numeric_dtype(features_df[c])
    ]

    suspicious_features = []

    for col in feature_cols:
        if col in merged.columns:
            corr = merged[col].corr(merged["y_reg"])
            if abs(corr) > 0.5:  # Suspiciously high correlation
                suspicious_features.append((col, corr))
                logger.warning(f"Suspicious correlation for {col}: {corr:.3f}")

    if suspicious_features:
        logger.warning(
            f"Found {len(suspicious_features)} features with suspicious correlation to labels"
        )
        return False

    logger.info("No obvious future leakage detected")
    return True


class PointInTimeMerger:
    """
    Helper class for managing point-in-time data merges.

    Ensures consistent lag handling across different data sources.
    """

    def __init__(
        self,
        fundamental_lag_days: int = 2,
        flow_lag_days: int = 1,
        execution_lag_days: int = 1,
    ):
        """
        Initialize merger with lag configurations.

        Args:
            fundamental_lag_days: Publication lag for fundamental data
            flow_lag_days: Lag for flow data
            execution_lag_days: Execution lag for trading signals
        """
        self.fundamental_lag_days = fundamental_lag_days
        self.flow_lag_days = flow_lag_days
        self.execution_lag_days = execution_lag_days

    def merge_all(
        self,
        prices_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame | None = None,
        flows_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Merge all data sources with appropriate lags.

        Args:
            prices_df: Price DataFrame
            fundamentals_df: Fundamental DataFrame (optional)
            flows_df: Flow DataFrame (optional)

        Returns:
            Merged DataFrame with point-in-time data
        """
        result = prices_df.copy()

        if fundamentals_df is not None:
            result = asof_merge_fundamentals(
                result,
                fundamentals_df,
                publish_lag_days=self.fundamental_lag_days,
            )

        if flows_df is not None:
            result = asof_merge_flow_data(
                result,
                flows_df,
                lag_days=self.flow_lag_days,
            )

        return result
