"""
Base Alpha

Abstract base class for all alpha strategies.
Every strategy must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np


@dataclass
class AlphaResult:
    """
    Result from alpha signal generation.

    Attributes:
        date: Signal date
        signals: DataFrame with asset_id, score columns
        metadata: Additional information
    """
    date: datetime
    signals: pd.DataFrame  # Must have: asset_id, score
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signals DataFrame."""
        required_cols = {"asset_id", "score"}
        if not required_cols.issubset(self.signals.columns):
            missing = required_cols - set(self.signals.columns)
            raise ValueError(f"Missing required columns: {missing}")

    @property
    def n_signals(self) -> int:
        """Number of assets with signals."""
        return len(self.signals)

    def get_top_k(self, k: int = 10) -> pd.DataFrame:
        """Get top k signals by score."""
        return self.signals.nlargest(k, "score")

    def get_bottom_k(self, k: int = 10) -> pd.DataFrame:
        """Get bottom k signals by score."""
        return self.signals.nsmallest(k, "score")


class BaseAlpha(ABC):
    """
    Base class for all alpha strategies.

    All strategies must implement:
        - fit(): Train/calibrate the strategy
        - generate_signals(): Generate trading signals

    Optional:
        - validate(): Validate strategy before use
        - get_parameters(): Get current parameters
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """
        Initialize alpha strategy.

        Args:
            name: Strategy name (unique identifier)
            config: Strategy configuration
        """
        self.name = name
        self.config = config or {}
        self.is_fitted = False
        self._fit_date: datetime | None = None
        self._performance_history: list[dict] = []

    @abstractmethod
    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Fit/train the strategy.

        Args:
            prices: Price data (date, asset_id, close, ...)
            features: Feature data (optional)
            labels: Label data for supervised strategies (optional)

        Returns:
            Fit result with metrics
        """
        pass

    @abstractmethod
    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate trading signals for a specific date.

        Args:
            date: Target date
            prices: Price data up to date (no future data!)
            features: Feature data up to date

        Returns:
            AlphaResult with signals
        """
        pass

    def validate(self) -> bool:
        """
        Validate strategy is ready for use.

        Returns:
            True if valid
        """
        return self.is_fitted

    def get_parameters(self) -> dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Parameter dictionary
        """
        return {
            "name": self.name,
            "config": self.config,
            "is_fitted": self.is_fitted,
            "fit_date": self._fit_date,
        }

    def record_performance(
        self,
        date: datetime,
        returns: float,
        hit_rate: float | None = None,
    ) -> None:
        """
        Record strategy performance for tracking.

        Args:
            date: Performance date
            returns: Strategy returns
            hit_rate: Win rate (optional)
        """
        self._performance_history.append({
            "date": date,
            "returns": returns,
            "hit_rate": hit_rate,
        })

        # Keep limited history
        if len(self._performance_history) > 252:
            self._performance_history = self._performance_history[-252:]

    def get_recent_performance(self, lookback: int = 21) -> dict[str, float]:
        """
        Get recent performance metrics.

        Args:
            lookback: Number of periods to look back

        Returns:
            Performance metrics
        """
        if not self._performance_history:
            return {"mean_return": 0, "win_rate": 0, "sharpe": 0}

        recent = self._performance_history[-lookback:]
        returns = [p["returns"] for p in recent]

        mean_ret = np.mean(returns)
        std_ret = np.std(returns) if len(returns) > 1 else 0
        win_rate = np.mean([1 if r > 0 else 0 for r in returns])
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0

        return {
            "mean_return": mean_ret,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "n_periods": len(recent),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"


class DummyAlpha(BaseAlpha):
    """
    Dummy alpha for testing.

    Generates random signals.
    """

    def __init__(self, name: str = "dummy", seed: int = 42):
        super().__init__(name)
        self.seed = seed

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """No-op fit."""
        self.is_fitted = True
        self._fit_date = datetime.now()
        return {"status": "fitted"}

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """Generate random signals."""
        # Get unique assets
        assets = prices["asset_id"].unique()

        np.random.seed(self.seed + hash(str(date)) % 1000)

        signals = pd.DataFrame({
            "asset_id": assets,
            "score": np.random.randn(len(assets)),
        })

        return AlphaResult(date=date, signals=signals)
