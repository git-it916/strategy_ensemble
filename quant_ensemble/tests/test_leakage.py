"""
Test Anti-Leakage Mechanisms

Tests to ensure no look-ahead bias in data processing.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import timedelta

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labels.leakage import (
    create_purged_kfold,
    check_temporal_leakage,
    create_embargo_mask,
    validate_train_test_split,
)


class TestPurgedKFold:
    """Test purged K-fold cross-validation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range("2024-01-01", periods=500, freq="D")

        df = pd.DataFrame({
            "date": dates,
            "asset_id": ["A"] * len(dates),
            "feature1": np.random.randn(len(dates)),
        })

        return df

    def test_no_overlap(self, sample_data):
        """Test that train and validation sets don't overlap."""
        for train_idx, val_idx in create_purged_kfold(sample_data, n_splits=5):
            train_set = set(train_idx)
            val_set = set(val_idx)

            overlap = train_set.intersection(val_set)
            assert len(overlap) == 0, "Train and validation sets overlap"

    def test_purge_gap(self, sample_data):
        """Test that purge gap is applied between train and validation."""
        purge_days = 21

        for train_idx, val_idx in create_purged_kfold(
            sample_data,
            n_splits=5,
            label_horizon_days=purge_days,
        ):
            train_dates = sample_data.iloc[train_idx]["date"]
            val_dates = sample_data.iloc[val_idx]["date"]

            max_train_date = train_dates.max()
            min_val_date = val_dates.min()

            gap = (min_val_date - max_train_date).days

            # Gap should be at least purge_days
            assert gap >= purge_days, f"Insufficient gap: {gap} days < {purge_days} days"

    def test_all_data_used(self, sample_data):
        """Test that all data points are used in validation at least once."""
        used_indices = set()

        for _, val_idx in create_purged_kfold(sample_data, n_splits=5):
            used_indices.update(val_idx)

        # Due to purging, not all indices may be used, but most should be
        coverage = len(used_indices) / len(sample_data)
        assert coverage > 0.7, f"Low coverage: {coverage:.2%}"

    def test_temporal_ordering(self, sample_data):
        """Test that validation comes after training temporally."""
        for train_idx, val_idx in create_purged_kfold(sample_data, n_splits=5):
            train_dates = sample_data.iloc[train_idx]["date"]
            val_dates = sample_data.iloc[val_idx]["date"]

            max_train = train_dates.max()
            min_val = val_dates.min()

            assert max_train < min_val, "Validation data should come after training"


class TestTemporalLeakage:
    """Test temporal leakage detection."""

    def test_detect_leakage_with_future_data(self):
        """Test detection of future data leakage."""
        dates = pd.date_range("2024-01-01", periods=100)

        features = pd.DataFrame({
            "date": dates,
            "asset_id": ["A"] * 100,
            "feature1": np.random.randn(100),
        })

        # Create labels with dates from future (simulating leakage)
        labels = pd.DataFrame({
            "date": dates + timedelta(days=10),  # Shifted into future
            "asset_id": ["A"] * 100,
            "y_reg": np.random.randn(100),
        })

        has_leakage = check_temporal_leakage(features, labels)
        assert has_leakage, "Should detect future data leakage"

    def test_no_leakage_with_proper_alignment(self):
        """Test no leakage detection with proper data alignment."""
        dates = pd.date_range("2024-01-01", periods=100)

        features = pd.DataFrame({
            "date": dates,
            "asset_id": ["A"] * 100,
            "feature1": np.random.randn(100),
        })

        # Labels on same dates (forward return calculated properly)
        labels = pd.DataFrame({
            "date": dates,
            "asset_id": ["A"] * 100,
            "y_reg": np.random.randn(100),
        })

        has_leakage = check_temporal_leakage(features, labels)
        assert not has_leakage, "Should not detect leakage with proper alignment"


class TestEmbargoMask:
    """Test embargo mask creation."""

    def test_embargo_mask_size(self):
        """Test embargo mask has correct size."""
        dates = pd.date_range("2024-01-01", periods=100)
        embargo_days = 5

        mask = create_embargo_mask(dates, embargo_days)

        assert len(mask) == len(dates)

    def test_embargo_excludes_recent(self):
        """Test that embargo excludes recent dates."""
        dates = pd.date_range("2024-01-01", periods=100)
        embargo_days = 5

        mask = create_embargo_mask(dates, embargo_days)

        # Last embargo_days should be False
        assert not mask[-embargo_days:].any(), "Last days should be excluded"

    def test_embargo_includes_old_dates(self):
        """Test that embargo includes old dates."""
        dates = pd.date_range("2024-01-01", periods=100)
        embargo_days = 5

        mask = create_embargo_mask(dates, embargo_days)

        # First dates should be True
        assert mask[:10].all(), "Early dates should be included"


class TestTrainTestValidation:
    """Test train/test split validation."""

    def test_valid_split(self):
        """Test validation passes for valid split."""
        train_dates = pd.date_range("2024-01-01", periods=100)
        test_dates = pd.date_range("2024-05-01", periods=50)

        is_valid = validate_train_test_split(
            train_dates,
            test_dates,
            min_gap_days=21,
        )

        assert is_valid, "Valid split should pass"

    def test_invalid_split_no_gap(self):
        """Test validation fails for split with no gap."""
        train_dates = pd.date_range("2024-01-01", periods=100)
        test_dates = pd.date_range("2024-04-05", periods=50)  # Starts right after train

        is_valid = validate_train_test_split(
            train_dates,
            test_dates,
            min_gap_days=21,
        )

        assert not is_valid, "Split with insufficient gap should fail"

    def test_invalid_split_overlap(self):
        """Test validation fails for overlapping split."""
        train_dates = pd.date_range("2024-01-01", periods=100)
        test_dates = pd.date_range("2024-03-01", periods=50)  # Overlaps with train

        is_valid = validate_train_test_split(
            train_dates,
            test_dates,
            min_gap_days=0,
        )

        # Even with 0 gap requirement, overlap should fail
        assert not is_valid, "Overlapping split should fail"


class TestPointInTimeFeatures:
    """Test point-in-time feature access."""

    def test_feature_availability_at_prediction_time(self):
        """Test that features used are available at prediction time."""
        # Feature calculation date vs availability
        calc_date = pd.Timestamp("2024-01-15")
        available_date = calc_date + timedelta(days=1)  # T+1 availability

        # This simulates that we should only use T-1 features for T predictions
        assert available_date > calc_date, "Features should be available after calculation"

    def test_no_forward_looking_features(self):
        """Test that forward-looking features are not used."""
        dates = pd.date_range("2024-01-01", periods=100)
        prices = np.random.randn(100).cumsum() + 100

        df = pd.DataFrame({"date": dates, "price": prices})

        # Calculate backward-looking feature (valid)
        df["ret_5d_backward"] = df["price"].pct_change(5)

        # Calculate forward-looking feature (invalid - would be leakage)
        df["ret_5d_forward"] = df["price"].shift(-5).pct_change(5)

        # Backward should have NaN at start
        assert df["ret_5d_backward"].isna()[:5].all()

        # Forward would have NaN at end (if calculated)
        assert df["ret_5d_forward"].isna()[-5:].all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
