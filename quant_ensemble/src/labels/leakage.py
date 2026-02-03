"""
Leakage Prevention Module

Provides purging and embargo utilities for train/test splitting.

ANTI-LEAKAGE MECHANISMS:
1. Purging: Remove training samples whose labels overlap with test period
2. Embargo: Add buffer period between train and test to prevent leakage
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from ..common import get_logger

logger = get_logger(__name__)


def apply_purging_embargo(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    dates: pd.Series,
    label_horizon_days: int,
    embargo_pct: float = 0.01,
    embargo_days: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove overlapping samples between train/test due to label horizon.

    ANTI-LEAKAGE: This function ensures that:
    1. Training samples whose forward-looking labels extend into the test period are removed
    2. An embargo period is added after the test period starts

    Args:
        train_indices: Original training indices
        test_indices: Test indices
        dates: Date series aligned with indices
        label_horizon_days: Number of days the label looks forward
        embargo_pct: Embargo period as percentage of test set size
        embargo_days: Fixed embargo days (overrides embargo_pct if provided)

    Returns:
        (purged_train_indices, test_indices) with overlapping samples removed
    """
    logger.info(
        f"Applying purging (horizon={label_horizon_days}d) and embargo "
        f"(pct={embargo_pct}, days={embargo_days})"
    )

    # Convert to datetime if needed
    dates = pd.to_datetime(dates)

    # Get date ranges
    train_dates = dates.iloc[train_indices]
    test_dates = dates.iloc[test_indices]

    test_start = test_dates.min()
    test_end = test_dates.max()

    # Calculate embargo period
    if embargo_days is None:
        n_test_days = len(test_dates.unique())
        embargo_days = max(1, int(n_test_days * embargo_pct))

    # Purging: Remove training samples whose labels overlap with test
    # A sample at date T has a label computed using data up to T + horizon
    # If T + horizon >= test_start - embargo, there's potential leakage
    purge_cutoff = test_start - pd.Timedelta(days=label_horizon_days + embargo_days)

    # Keep only training samples before the cutoff
    purged_mask = train_dates <= purge_cutoff
    purged_train_indices = train_indices[purged_mask]

    n_purged = len(train_indices) - len(purged_train_indices)
    logger.info(
        f"Purged {n_purged} training samples "
        f"({n_purged/len(train_indices)*100:.1f}%)"
    )

    return purged_train_indices, test_indices


def create_purged_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    label_horizon_days: int = 21,
    embargo_pct: float = 0.01,
    date_col: str = "date",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Create K-fold cross-validation splits with purging and embargo.

    ANTI-LEAKAGE: Each fold has samples purged to prevent label overlap.

    Args:
        df: DataFrame with date column
        n_splits: Number of folds
        label_horizon_days: Label forward-looking horizon
        embargo_pct: Embargo percentage
        date_col: Date column name

    Yields:
        (train_indices, test_indices) for each fold
    """
    logger.info(f"Creating {n_splits}-fold CV with purging/embargo")

    dates = pd.to_datetime(df[date_col])
    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)

    # Split dates into folds
    fold_size = n_dates // n_splits

    for i in range(n_splits):
        # Define test date range for this fold
        test_start_idx = i * fold_size
        test_end_idx = (i + 1) * fold_size if i < n_splits - 1 else n_dates

        test_date_start = unique_dates[test_start_idx]
        test_date_end = unique_dates[test_end_idx - 1]

        # Get indices
        test_mask = (dates >= test_date_start) & (dates <= test_date_end)
        train_mask = ~test_mask

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        # Apply purging and embargo
        purged_train, purged_test = apply_purging_embargo(
            train_indices=train_indices,
            test_indices=test_indices,
            dates=dates,
            label_horizon_days=label_horizon_days,
            embargo_pct=embargo_pct,
        )

        yield purged_train, purged_test


def create_walk_forward_splits(
    df: pd.DataFrame,
    train_period_days: int = 504,  # ~2 years
    test_period_days: int = 126,   # ~6 months
    step_days: int = 63,           # ~3 months
    label_horizon_days: int = 21,
    embargo_days: int = 5,
    date_col: str = "date",
    min_train_samples: int = 1000,
) -> Iterator[tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]]:
    """
    Create walk-forward (rolling) train/test splits with purging.

    ANTI-LEAKAGE: Each split has proper purging between train and test.

    Args:
        df: DataFrame with date column
        train_period_days: Training period in trading days
        test_period_days: Test period in trading days
        step_days: Step size between splits in trading days
        label_horizon_days: Label forward-looking horizon
        embargo_days: Embargo period in trading days
        date_col: Date column name
        min_train_samples: Minimum required training samples

    Yields:
        (train_indices, test_indices, test_start, test_end) for each split
    """
    logger.info(
        f"Creating walk-forward splits: train={train_period_days}d, "
        f"test={test_period_days}d, step={step_days}d"
    )

    dates = pd.to_datetime(df[date_col])
    unique_dates = sorted(dates.unique())

    # Calculate number of splits
    n_dates = len(unique_dates)
    total_period = train_period_days + test_period_days

    if n_dates < total_period:
        logger.warning(f"Not enough data for walk-forward: {n_dates} < {total_period}")
        return

    # Generate splits
    split_idx = 0
    start_idx = 0

    while start_idx + total_period <= n_dates:
        train_start_idx = start_idx
        train_end_idx = start_idx + train_period_days - 1
        test_start_idx = start_idx + train_period_days
        test_end_idx = min(test_start_idx + test_period_days - 1, n_dates - 1)

        train_start = unique_dates[train_start_idx]
        train_end = unique_dates[train_end_idx]
        test_start = unique_dates[test_start_idx]
        test_end = unique_dates[test_end_idx]

        # Get indices
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        # Apply purging
        purged_train, _ = apply_purging_embargo(
            train_indices=train_indices,
            test_indices=test_indices,
            dates=dates,
            label_horizon_days=label_horizon_days,
            embargo_days=embargo_days,
        )

        # Check minimum samples
        if len(purged_train) >= min_train_samples:
            logger.info(
                f"Split {split_idx}: train={len(purged_train)}, test={len(test_indices)}, "
                f"period={test_start.date()} to {test_end.date()}"
            )
            yield purged_train, test_indices, test_start, test_end
            split_idx += 1

        # Move to next split
        start_idx += step_days

    logger.info(f"Created {split_idx} walk-forward splits")


class PurgedGroupTimeSplit:
    """
    Time-series cross-validator with purging.

    Similar to sklearn's TimeSeriesSplit but with anti-leakage mechanisms.
    """

    def __init__(
        self,
        n_splits: int = 5,
        label_horizon_days: int = 21,
        embargo_pct: float = 0.01,
        embargo_days: int | None = None,
        gap_days: int = 0,
    ):
        """
        Initialize splitter.

        Args:
            n_splits: Number of splits
            label_horizon_days: Forward-looking horizon in labels
            embargo_pct: Embargo as percentage of test size
            embargo_days: Fixed embargo days
            gap_days: Additional gap between train and test
        """
        self.n_splits = n_splits
        self.label_horizon_days = label_horizon_days
        self.embargo_pct = embargo_pct
        self.embargo_days = embargo_days
        self.gap_days = gap_days

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups: pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.

        Args:
            X: Features DataFrame (must have 'date' column)
            y: Labels (unused, for sklearn compatibility)
            groups: Groups (unused, for sklearn compatibility)

        Yields:
            (train_indices, test_indices) for each split
        """
        if "date" not in X.columns:
            raise ValueError("X must have a 'date' column")

        dates = pd.to_datetime(X["date"])
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        # Calculate test size per split (increasing)
        test_size = n_dates // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Test period
            test_start_idx = (i + 1) * test_size
            test_end_idx = (i + 2) * test_size if i < self.n_splits - 1 else n_dates

            test_start = unique_dates[test_start_idx]
            test_end = unique_dates[test_end_idx - 1]

            # Train period (all data before test, with gap)
            train_end_idx = test_start_idx - self.gap_days - 1
            if train_end_idx < 0:
                continue

            train_end = unique_dates[train_end_idx]
            train_start = unique_dates[0]

            # Get indices
            train_mask = (dates >= train_start) & (dates <= train_end)
            test_mask = (dates >= test_start) & (dates <= test_end)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            # Apply purging
            purged_train, _ = apply_purging_embargo(
                train_indices=train_indices,
                test_indices=test_indices,
                dates=dates,
                label_horizon_days=self.label_horizon_days,
                embargo_pct=self.embargo_pct,
                embargo_days=self.embargo_days,
            )

            yield purged_train, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits."""
        return self.n_splits


def check_train_test_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_horizon_days: int,
    date_col: str = "date",
    asset_col: str = "asset_id",
) -> dict[str, Any]:
    """
    Check for potential leakage between train and test sets.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        label_horizon_days: Forward-looking horizon
        date_col: Date column name
        asset_col: Asset column name

    Returns:
        Dictionary with leakage analysis results
    """
    train_dates = pd.to_datetime(train_df[date_col])
    test_dates = pd.to_datetime(test_df[date_col])

    train_max = train_dates.max()
    test_min = test_dates.min()

    gap_days = (test_min - train_max).days

    results = {
        "train_date_range": (train_dates.min(), train_max),
        "test_date_range": (test_min, test_dates.max()),
        "gap_days": gap_days,
        "label_horizon_days": label_horizon_days,
        "potential_leakage": gap_days < label_horizon_days,
    }

    if results["potential_leakage"]:
        logger.warning(
            f"Potential leakage detected! Gap ({gap_days}d) < horizon ({label_horizon_days}d)"
        )
    else:
        logger.info(f"No leakage detected. Gap ({gap_days}d) >= horizon ({label_horizon_days}d)")

    return results
