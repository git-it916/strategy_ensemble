"""
Pytest Configuration and Fixtures

Shared fixtures for all tests.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=252)
    tickers = [f"A{i:06d}" for i in range(1, 11)]

    records = []
    for ticker in tickers:
        prices = 50000 * np.cumprod(1 + np.random.randn(len(dates)) * 0.02)

        for i, date in enumerate(dates):
            records.append({
                "date": date,
                "asset_id": ticker,
                "PX_LAST": prices[i],
                "close": prices[i],
                "open": prices[i] * (1 + np.random.randn() * 0.005),
                "PX_HIGH": prices[i] * (1 + abs(np.random.randn()) * 0.01),
                "PX_LOW": prices[i] * (1 - abs(np.random.randn()) * 0.01),
                "PX_VOLUME": int(np.random.uniform(100000, 10000000)),
            })

    return pd.DataFrame(records)


@pytest.fixture
def sample_features(sample_prices):
    """Generate sample features from prices."""
    records = []

    for ticker in sample_prices["asset_id"].unique():
        ticker_data = sample_prices[sample_prices["asset_id"] == ticker].sort_values("date")
        closes = ticker_data["PX_LAST"].values
        dates = ticker_data["date"].values

        for i in range(21, len(ticker_data)):
            records.append({
                "date": dates[i],
                "asset_id": ticker,
                "ret_5d": (closes[i] - closes[i - 5]) / closes[i - 5],
                "ret_21d": (closes[i] - closes[i - 21]) / closes[i - 21],
                "vol_5d": np.std(np.diff(np.log(closes[i - 5:i + 1]))),
                "vol_21d": np.std(np.diff(np.log(closes[i - 21:i + 1]))),
                "rsi_14": 50 + np.random.randn() * 20,
            })

    return pd.DataFrame(records)


@pytest.fixture
def sample_labels(sample_prices):
    """Generate sample labels from prices."""
    forward_days = 21
    records = []

    for ticker in sample_prices["asset_id"].unique():
        ticker_data = sample_prices[sample_prices["asset_id"] == ticker].sort_values("date")
        closes = ticker_data["PX_LAST"].values
        dates = ticker_data["date"].values

        for i in range(21, len(ticker_data) - forward_days):
            fwd_return = (closes[i + forward_days] - closes[i]) / closes[i]

            records.append({
                "date": dates[i],
                "asset_id": ticker,
                "y_reg": fwd_return,
                "y_cls": 1 if fwd_return > 0 else 0,
            })

    return pd.DataFrame(records)


@pytest.fixture
def mock_model():
    """Create a mock signal model."""
    class MockModel:
        def __init__(self):
            self.model_name = "mock_model"
            self.is_fitted = True

        def fit(self, features_df, labels_df, config=None):
            return {"status": "fitted"}

        def predict(self, date, features_df):
            asset_ids = features_df["asset_id"].unique()
            np.random.seed(hash(str(date)) % 2**32)

            return pd.DataFrame({
                "date": date,
                "asset_id": asset_ids,
                "score": np.random.rand(len(asset_ids)),
                "confidence": np.ones(len(asset_ids)),
                "model_name": self.model_name,
            })

        def predict_batch(self, features_df):
            results = []
            for date in features_df["date"].unique():
                date_features = features_df[features_df["date"] == date]
                pred = self.predict(date, date_features)
                results.append(pred)
            return pd.concat(results, ignore_index=True)

    return MockModel()


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
