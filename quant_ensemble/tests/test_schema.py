"""
Test Schema Validation

Tests for data types and schema validation.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.types import (
    FeatureRow,
    LabelRow,
    SignalScore,
    PortfolioWeight,
    Trade,
    Order,
    RegimeProbability,
    validate_features_df,
    validate_labels_df,
    validate_predictions_df,
)


class TestDataTypes:
    """Test dataclass types."""

    def test_feature_row(self):
        """Test FeatureRow creation."""
        feature = FeatureRow(
            date=pd.Timestamp("2024-01-01"),
            asset_id="A000001",
            features={"ret_5d": 0.01, "vol_21d": 0.02},
        )

        assert feature.date == pd.Timestamp("2024-01-01")
        assert feature.asset_id == "A000001"
        assert feature.features["ret_5d"] == 0.01

    def test_label_row(self):
        """Test LabelRow creation."""
        label = LabelRow(
            date=pd.Timestamp("2024-01-01"),
            asset_id="A000001",
            y_reg=0.05,
            y_cls=1,
        )

        assert label.y_reg == 0.05
        assert label.y_cls == 1

    def test_signal_score(self):
        """Test SignalScore creation."""
        score = SignalScore(
            date=pd.Timestamp("2024-01-01"),
            asset_id="A000001",
            score=0.8,
            confidence=0.9,
            model_name="test_model",
        )

        assert score.score == 0.8
        assert score.confidence == 0.9

    def test_portfolio_weight(self):
        """Test PortfolioWeight creation."""
        weight = PortfolioWeight(
            date=pd.Timestamp("2024-01-01"),
            asset_id="A000001",
            weight=0.05,
            target_shares=100,
        )

        assert weight.weight == 0.05
        assert weight.target_shares == 100

    def test_trade(self):
        """Test Trade creation."""
        trade = Trade(
            date=pd.Timestamp("2024-01-01"),
            asset_id="A000001",
            side="BUY",
            quantity=100,
            price=50000.0,
            cost=75.0,
        )

        assert trade.side == "BUY"
        assert trade.quantity == 100
        assert trade.price == 50000.0

    def test_order(self):
        """Test Order creation."""
        order = Order(
            asset_id="A000001",
            side="SELL",
            quantity=50,
            price=51000.0,
            order_type="LIMIT",
        )

        assert order.side == "SELL"
        assert order.order_type == "LIMIT"

    def test_regime_probability(self):
        """Test RegimeProbability creation."""
        regime = RegimeProbability(
            date=pd.Timestamp("2024-01-01"),
            probabilities={"regime_0": 0.7, "regime_1": 0.3},
            dominant_regime=0,
        )

        assert regime.probabilities["regime_0"] == 0.7
        assert regime.dominant_regime == 0


class TestDataFrameValidation:
    """Test DataFrame validation functions."""

    def test_validate_features_df_valid(self):
        """Test valid features DataFrame."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "asset_id": [f"A00000{i}" for i in range(10)],
            "ret_5d": np.random.randn(10),
            "vol_21d": np.random.rand(10),
        })

        # Should not raise
        validate_features_df(df)

    def test_validate_features_df_missing_column(self):
        """Test features DataFrame missing required column."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            # Missing asset_id
            "ret_5d": np.random.randn(10),
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_features_df(df)

    def test_validate_labels_df_valid(self):
        """Test valid labels DataFrame."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "asset_id": [f"A00000{i}" for i in range(10)],
            "y_reg": np.random.randn(10) * 0.1,
        })

        validate_labels_df(df)

    def test_validate_labels_df_missing_column(self):
        """Test labels DataFrame missing required column."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "asset_id": [f"A00000{i}" for i in range(10)],
            # Missing y_reg
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_labels_df(df)

    def test_validate_predictions_df_valid(self):
        """Test valid predictions DataFrame."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "asset_id": [f"A00000{i}" for i in range(10)],
            "score": np.random.rand(10),
        })

        validate_predictions_df(df)

    def test_validate_predictions_df_missing_column(self):
        """Test predictions DataFrame missing required column."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            # Missing asset_id and score
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_predictions_df(df)


class TestDataTypeIntegrity:
    """Test data type integrity across conversions."""

    def test_date_conversion(self):
        """Test date type preservation."""
        date_str = "2024-01-15"
        date_ts = pd.Timestamp(date_str)
        date_dt = datetime(2024, 1, 15)

        # All should be comparable
        assert pd.Timestamp(date_str) == date_ts
        assert pd.Timestamp(date_dt) == date_ts

    def test_numeric_precision(self):
        """Test numeric precision in calculations."""
        weights = pd.Series([0.1, 0.2, 0.3, 0.4])

        # Should sum to 1.0
        assert np.isclose(weights.sum(), 1.0)

        # Should preserve precision after operations
        scaled = weights * 2
        assert np.isclose(scaled.sum(), 2.0)

    def test_dataframe_merge_integrity(self):
        """Test data integrity after merges."""
        df1 = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "asset_id": ["A"] * 5,
            "value1": [1, 2, 3, 4, 5],
        })

        df2 = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "asset_id": ["A"] * 5,
            "value2": [10, 20, 30, 40, 50],
        })

        merged = df1.merge(df2, on=["date", "asset_id"])

        assert len(merged) == 5
        assert "value1" in merged.columns
        assert "value2" in merged.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
