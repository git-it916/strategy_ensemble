"""
Test Backtest Engine

Tests for backtesting functionality and correctness.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import (
    BacktestConfig,
    BacktestEngine,
    calculate_metrics,
    calculate_max_drawdown,
    calculate_monthly_returns,
)
from portfolio import PortfolioAllocator, PortfolioConstraints


class MockSignalModel:
    """Mock signal model for testing."""

    def __init__(self, scores: dict[str, float] | None = None):
        self.scores = scores or {}
        self.model_name = "mock_model"
        self.is_fitted = True

    def predict(self, date: pd.Timestamp, features_df: pd.DataFrame) -> pd.DataFrame:
        """Return mock predictions."""
        asset_ids = features_df["asset_id"].unique()

        scores = []
        for asset_id in asset_ids:
            if self.scores:
                score = self.scores.get(asset_id, 0.5)
            else:
                # Random but deterministic based on date
                np.random.seed(hash(str(date) + asset_id) % 2**32)
                score = np.random.rand()

            scores.append(score)

        return pd.DataFrame({
            "date": date,
            "asset_id": asset_ids,
            "score": scores,
            "confidence": [1.0] * len(asset_ids),
            "model_name": self.model_name,
        })


class TestBacktestEngine:
    """Test BacktestEngine class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.bdate_range("2024-01-01", periods=100)
        tickers = ["A001", "A002", "A003", "A004", "A005"]

        # Generate prices
        price_records = []
        for ticker in tickers:
            np.random.seed(hash(ticker) % 2**32)
            prices = 10000 * np.cumprod(1 + np.random.randn(len(dates)) * 0.02)

            for i, date in enumerate(dates):
                price_records.append({
                    "date": date,
                    "asset_id": ticker,
                    "close": prices[i],
                    "open": prices[i] * (1 + np.random.randn() * 0.01),
                })

        prices_df = pd.DataFrame(price_records)

        # Generate features
        feature_records = []
        for ticker in tickers:
            for date in dates:
                feature_records.append({
                    "date": date,
                    "asset_id": ticker,
                    "ret_5d": np.random.randn() * 0.05,
                    "vol_21d": np.random.rand() * 0.3,
                })

        features_df = pd.DataFrame(feature_records)

        return features_df, prices_df

    def test_backtest_runs(self, sample_data):
        """Test that backtest completes without error."""
        features_df, prices_df = sample_data

        config = BacktestConfig(
            start_date=pd.Timestamp("2024-01-15"),
            end_date=pd.Timestamp("2024-04-01"),
            initial_capital=100_000_000,
        )

        engine = BacktestEngine(config)
        model = MockSignalModel()

        result = engine.run(model, features_df, prices_df)

        assert result is not None
        assert len(result.daily_results) > 0
        assert result.metrics is not None

    def test_initial_capital_preserved(self, sample_data):
        """Test that initial capital is correctly set."""
        features_df, prices_df = sample_data
        initial_capital = 50_000_000

        config = BacktestConfig(
            start_date=pd.Timestamp("2024-01-15"),
            end_date=pd.Timestamp("2024-04-01"),
            initial_capital=initial_capital,
        )

        engine = BacktestEngine(config)
        model = MockSignalModel()

        result = engine.run(model, features_df, prices_df)

        # First day should start near initial capital
        first_value = result.daily_results[0].portfolio_value
        assert abs(first_value - initial_capital) / initial_capital < 0.1

    def test_returns_series_correct_length(self, sample_data):
        """Test that returns series has correct length."""
        features_df, prices_df = sample_data

        config = BacktestConfig(
            start_date=pd.Timestamp("2024-01-15"),
            end_date=pd.Timestamp("2024-04-01"),
            initial_capital=100_000_000,
        )

        engine = BacktestEngine(config)
        model = MockSignalModel()

        result = engine.run(model, features_df, prices_df)

        assert len(result.returns_series) == len(result.daily_results)

    def test_transaction_costs_deducted(self, sample_data):
        """Test that transaction costs are properly deducted."""
        features_df, prices_df = sample_data

        config = BacktestConfig(
            start_date=pd.Timestamp("2024-01-15"),
            end_date=pd.Timestamp("2024-04-01"),
            initial_capital=100_000_000,
            transaction_cost_bps=50.0,  # High costs
        )

        engine = BacktestEngine(config)
        model = MockSignalModel()

        result = engine.run(model, features_df, prices_df)

        # Should have some trades with costs
        if not result.trades_df.empty:
            total_costs = result.daily_results[0].transaction_costs if result.daily_results else 0
            # Costs should be non-negative
            assert all(r.transaction_costs >= 0 for r in result.daily_results)


class TestMetricsCalculation:
    """Test performance metrics calculation."""

    def test_calculate_metrics_empty_returns(self):
        """Test metrics calculation with empty returns."""
        returns = pd.Series(dtype=float)
        metrics = calculate_metrics(returns)

        assert metrics == {}

    def test_calculate_metrics_positive_returns(self):
        """Test metrics with positive returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)  # Slight positive drift

        metrics = calculate_metrics(returns)

        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio is calculated correctly."""
        # Create returns with known properties
        returns = pd.Series([0.01] * 252)  # 1% daily return

        metrics = calculate_metrics(returns, risk_free_rate=0.0)

        # With constant returns, Sharpe should be high (low vol)
        assert metrics["sharpe_ratio"] > 5

    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        # Create returns with known drawdown
        returns = pd.Series([0.1, -0.2, 0.05, 0.05])

        max_dd = calculate_max_drawdown(returns)

        # Drawdown after -20% drop
        assert max_dd > 0.15

    def test_win_rate_calculation(self):
        """Test win rate is calculated correctly."""
        # 60 positive, 40 negative
        returns = pd.Series([0.01] * 60 + [-0.01] * 40)

        metrics = calculate_metrics(returns)

        assert abs(metrics["win_rate"] - 0.6) < 0.01


class TestMonthlyReturns:
    """Test monthly returns calculation."""

    def test_monthly_returns_shape(self):
        """Test monthly returns table has correct shape."""
        dates = pd.date_range("2024-01-01", periods=365)
        returns = pd.Series(np.random.randn(365) * 0.01, index=dates)

        monthly = calculate_monthly_returns(returns)

        # Should have rows for each year
        assert len(monthly) >= 1

        # Should have columns for months + yearly total
        assert "Year Total" in monthly.columns

    def test_monthly_returns_empty(self):
        """Test monthly returns with empty data."""
        returns = pd.Series(dtype=float)

        monthly = calculate_monthly_returns(returns)

        assert monthly.empty


class TestPortfolioConstraints:
    """Test portfolio constraints in backtest."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        dates = pd.bdate_range("2024-01-01", periods=50)
        tickers = [f"A{i:03d}" for i in range(20)]

        price_records = []
        for ticker in tickers:
            prices = 10000 * np.ones(len(dates))
            for i, date in enumerate(dates):
                price_records.append({
                    "date": date,
                    "asset_id": ticker,
                    "close": prices[i],
                    "open": prices[i],
                })

        prices_df = pd.DataFrame(price_records)

        feature_records = []
        for ticker in tickers:
            for date in dates:
                feature_records.append({
                    "date": date,
                    "asset_id": ticker,
                    "ret_5d": 0.0,
                })

        features_df = pd.DataFrame(feature_records)

        return features_df, prices_df

    def test_max_weight_constraint(self, sample_data):
        """Test that max weight constraint is respected."""
        features_df, prices_df = sample_data

        max_weight = 0.05  # 5% max per asset

        constraints = PortfolioConstraints(
            max_weight_per_asset=max_weight,
            max_leverage=1.0,
        )

        allocator = PortfolioAllocator(
            method="topk",
            constraints=constraints,
            config={"top_k": 20},
        )

        config = BacktestConfig(
            start_date=pd.Timestamp("2024-01-15"),
            end_date=pd.Timestamp("2024-02-15"),
            initial_capital=100_000_000,
        )

        engine = BacktestEngine(config, allocator=allocator)

        # Create model that gives high scores to few assets
        scores = {f"A{i:03d}": 1.0 if i < 3 else 0.5 for i in range(20)}
        model = MockSignalModel(scores)

        result = engine.run(model, features_df, prices_df)

        # Check weights don't exceed max
        for daily_result in result.daily_results:
            for asset_id, weight in daily_result.weights.items():
                assert weight <= max_weight + 0.01, f"Weight {weight} exceeds max {max_weight}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
