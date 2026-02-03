"""
Walk-Forward Optimization

Time-series cross-validation with expanding or rolling windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from ..common import get_logger
from ..labels.leakage import create_purged_kfold
from .engine import BacktestConfig, BacktestEngine, BacktestResult
from .metrics import calculate_metrics

logger = get_logger(__name__)


@dataclass
class WalkForwardConfig:
    """Walk-forward configuration."""

    # Window settings
    train_months: int = 24  # Training window in months
    test_months: int = 3  # Test window in months
    expanding_window: bool = True  # Expand or roll training window

    # Purging/Embargo
    purge_days: int = 21  # Days to purge between train/test
    embargo_days: int = 5  # Days to embargo after train

    # Model refitting
    refit_frequency: str = "test"  # "test" = each test period, "monthly" = every month


@dataclass
class WalkForwardFold:
    """Single walk-forward fold."""

    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_result: dict[str, Any] | None = None
    test_result: BacktestResult | None = None


@dataclass
class WalkForwardResult:
    """Walk-forward optimization result."""

    config: WalkForwardConfig
    folds: list[WalkForwardFold]
    combined_returns: pd.Series
    combined_metrics: dict[str, float]
    fold_metrics: pd.DataFrame


class WalkForwardOptimizer:
    """
    Walk-forward optimization framework.

    ANTI-LEAKAGE: Ensures strict temporal separation between train and test.

    Process:
        1. Split data into train/test periods
        2. Fit model on training data
        3. Generate predictions on test data (no retraining)
        4. Slide forward and repeat
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        backtest_config: BacktestConfig | None = None,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            config: Walk-forward configuration
            backtest_config: Backtest configuration
        """
        self.config = config
        self.backtest_config = backtest_config or BacktestConfig(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2024-12-31"),
        )

    def run(
        self,
        model_factory: Callable[[], Any],
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        fit_config: dict[str, Any] | None = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            model_factory: Function that creates a new model instance
            features_df: Features DataFrame
            labels_df: Labels DataFrame
            prices_df: Prices DataFrame
            fit_config: Model fitting configuration

        Returns:
            WalkForwardResult
        """
        logger.info("Starting walk-forward optimization")

        fit_config = fit_config or {}

        # Generate folds
        folds = self._generate_folds(features_df)
        logger.info(f"Generated {len(folds)} walk-forward folds")

        all_returns = []

        for fold in folds:
            logger.info(
                f"Fold {fold.fold_idx + 1}/{len(folds)}: "
                f"Train {fold.train_start.date()} to {fold.train_end.date()}, "
                f"Test {fold.test_start.date()} to {fold.test_end.date()}"
            )

            # Filter data for this fold
            train_mask = (features_df["date"] >= fold.train_start) & (features_df["date"] <= fold.train_end)
            test_mask = (features_df["date"] >= fold.test_start) & (features_df["date"] <= fold.test_end)

            train_features = features_df[train_mask]
            train_labels = labels_df[
                (labels_df["date"] >= fold.train_start) & (labels_df["date"] <= fold.train_end)
            ]

            test_features = features_df[test_mask]
            test_prices = prices_df[
                (prices_df["date"] >= fold.test_start) & (prices_df["date"] <= fold.test_end)
            ]

            if train_features.empty or test_features.empty:
                logger.warning(f"Skipping fold {fold.fold_idx + 1} due to insufficient data")
                continue

            # Create new model instance
            model = model_factory()

            # Fit on training data
            try:
                train_result = model.fit(train_features, train_labels, fit_config)
                fold.train_result = train_result
            except Exception as e:
                logger.error(f"Training failed for fold {fold.fold_idx + 1}: {e}")
                continue

            # Run backtest on test period
            test_bt_config = BacktestConfig(
                start_date=fold.test_start,
                end_date=fold.test_end,
                initial_capital=self.backtest_config.initial_capital,
                rebalance_frequency=self.backtest_config.rebalance_frequency,
                transaction_cost_bps=self.backtest_config.transaction_cost_bps,
                slippage_bps=self.backtest_config.slippage_bps,
                verbose=False,
            )

            engine = BacktestEngine(test_bt_config)

            try:
                test_result = engine.run(model, test_features, test_prices)
                fold.test_result = test_result

                if not test_result.returns_series.empty:
                    all_returns.append(test_result.returns_series)

            except Exception as e:
                logger.error(f"Backtest failed for fold {fold.fold_idx + 1}: {e}")

        # Combine results
        if all_returns:
            combined_returns = pd.concat(all_returns).sort_index()
            # Remove duplicates (overlapping dates)
            combined_returns = combined_returns[~combined_returns.index.duplicated(keep="first")]
        else:
            combined_returns = pd.Series(dtype=float)

        combined_metrics = calculate_metrics(combined_returns, self.backtest_config.initial_capital)

        # Create fold metrics summary
        fold_metrics = self._create_fold_metrics(folds)

        logger.info(f"Walk-forward complete. Combined Sharpe: {combined_metrics.get('sharpe_ratio', 0):.2f}")

        return WalkForwardResult(
            config=self.config,
            folds=folds,
            combined_returns=combined_returns,
            combined_metrics=combined_metrics,
            fold_metrics=fold_metrics,
        )

    def _generate_folds(self, features_df: pd.DataFrame) -> list[WalkForwardFold]:
        """Generate walk-forward folds."""
        dates = sorted(features_df["date"].unique())

        if len(dates) < 100:
            logger.warning("Insufficient data for walk-forward optimization")
            return []

        min_date = pd.Timestamp(dates[0])
        max_date = pd.Timestamp(dates[-1])

        folds = []
        fold_idx = 0

        # Start first test period after initial training window
        test_start = min_date + pd.DateOffset(months=self.config.train_months)
        test_start += pd.DateOffset(days=self.config.purge_days)

        while test_start < max_date:
            test_end = test_start + pd.DateOffset(months=self.config.test_months) - pd.DateOffset(days=1)
            test_end = min(test_end, max_date)

            # Training period
            train_end = test_start - pd.DateOffset(days=self.config.purge_days)

            if self.config.expanding_window:
                train_start = min_date
            else:
                train_start = train_end - pd.DateOffset(months=self.config.train_months)
                train_start = max(train_start, min_date)

            # Adjust for embargo
            train_end -= pd.DateOffset(days=self.config.embargo_days)

            fold = WalkForwardFold(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            folds.append(fold)

            # Move to next test period
            test_start = test_end + pd.DateOffset(days=1)
            fold_idx += 1

        return folds

    def _create_fold_metrics(self, folds: list[WalkForwardFold]) -> pd.DataFrame:
        """Create summary of fold metrics."""
        records = []

        for fold in folds:
            if fold.test_result is None:
                continue

            metrics = fold.test_result.metrics

            records.append({
                "fold": fold.fold_idx + 1,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "return": metrics.get("total_return", 0),
                "sharpe": metrics.get("sharpe_ratio", 0),
                "max_dd": metrics.get("max_drawdown", 0),
                "win_rate": metrics.get("win_rate", 0),
                "n_days": metrics.get("n_days", 0),
            })

        return pd.DataFrame(records)


def create_time_series_splits(
    dates: list,
    n_splits: int = 5,
    test_size: int = 63,  # ~3 months
    gap: int = 21,  # ~1 month gap
) -> list[tuple[list, list]]:
    """
    Create time series splits for cross-validation.

    Args:
        dates: List of dates
        n_splits: Number of splits
        test_size: Test period size in days
        gap: Gap between train and test

    Returns:
        List of (train_indices, test_indices) tuples
    """
    n_samples = len(dates)
    splits = []

    # Calculate split points
    min_train_size = n_samples - (test_size + gap) * n_splits

    if min_train_size < 100:
        logger.warning("Insufficient data for requested number of splits")
        n_splits = max(1, (n_samples - 100) // (test_size + gap))

    for i in range(n_splits):
        test_end = n_samples - i * (test_size + gap)
        test_start = test_end - test_size

        train_end = test_start - gap
        train_start = 0

        if train_end < 100:
            continue

        train_idx = list(range(train_start, train_end))
        test_idx = list(range(test_start, test_end))

        splits.append((train_idx, test_idx))

    # Reverse to get chronological order
    splits = splits[::-1]

    return splits


def run_walk_forward(
    model_factory: Callable[[], Any],
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    train_months: int = 24,
    test_months: int = 3,
    **kwargs,
) -> WalkForwardResult:
    """
    Convenience function for walk-forward optimization.

    Args:
        model_factory: Function creating model instances
        features_df: Features
        labels_df: Labels
        prices_df: Prices
        train_months: Training window months
        test_months: Test window months
        **kwargs: Additional config

    Returns:
        WalkForwardResult
    """
    wf_config = WalkForwardConfig(
        train_months=train_months,
        test_months=test_months,
        **kwargs,
    )

    optimizer = WalkForwardOptimizer(wf_config)
    return optimizer.run(model_factory, features_df, labels_df, prices_df)
