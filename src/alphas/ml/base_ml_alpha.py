"""
Base ML Alpha

Abstract base class for ML-based alpha strategies.
Extends BaseAlpha with model training, scaler management, and bundled persistence.
Maintains the exact same external interface as BaseAlpha so EnsembleAgent
treats rule-based and ML alphas identically.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..base_alpha import BaseAlpha, AlphaResult

logger = logging.getLogger(__name__)


class BaseMLAlpha(BaseAlpha):
    """
    Base class for ML-based alpha strategies.

    Subclasses must implement:
        - _build_model(): Create and return the ML model object
        - feature_columns (class attribute or set in __init__)

    The scaler is managed per-model (not global) to avoid
    mismatch when different ML alphas use different feature sets.
    """

    def __init__(
        self,
        name: str,
        feature_columns: list[str],
        config: dict[str, Any] | None = None,
    ):
        """
        Args:
            name: Strategy name (unique identifier)
            feature_columns: List of feature column names this model uses
            config: Model hyperparameters and settings
        """
        super().__init__(name, config)
        self.model = None
        self.scaler: StandardScaler | None = None
        self.feature_columns = feature_columns
        # May become a subset of feature_columns if allow_feature_subset is enabled.
        self.active_feature_columns = list(feature_columns)

    @abstractmethod
    def _build_model(self) -> Any:
        """
        Create and return the ML model object.

        Example:
            return XGBRegressor(n_estimators=500, max_depth=6)
        """

    # ------------------------------------------------------------------
    # BaseAlpha interface (same signature as rule-based alphas)
    # ------------------------------------------------------------------

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Train the ML model.

        Args:
            prices: Price data (unused by most ML alphas, kept for interface)
            features: Feature DataFrame with self.feature_columns
            labels: Label DataFrame with 'y_reg' column

        Returns:
            Fit result metrics
        """
        if features is None or labels is None:
            raise ValueError(
                f"{self.name}: ML alpha requires both features and labels for training"
            )

        # Align features and labels on (date, ticker)
        merged = features.merge(labels, on=["date", "ticker"], how="inner")

        # Check required columns exist
        missing = set(self.feature_columns) - set(merged.columns)
        if missing:
            allow_subset = bool(self.config.get("allow_feature_subset", False))
            if allow_subset:
                available = [c for c in self.feature_columns if c in merged.columns]
                if not available:
                    raise ValueError(
                        f"{self.name}: Missing feature columns: {missing} "
                        "(no usable fallback features)"
                    )
                self.active_feature_columns = available
                logger.warning(
                    "%s: Using feature subset (%d/%d). Missing=%s",
                    self.name,
                    len(self.active_feature_columns),
                    len(self.feature_columns),
                    sorted(missing),
                )
            else:
                raise ValueError(f"{self.name}: Missing feature columns: {missing}")
        else:
            self.active_feature_columns = list(self.feature_columns)

        X = merged[self.active_feature_columns].values
        y = merged["y_reg"].values

        # Drop rows with NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Build and train model
        self.model = self._build_model()
        self.model.fit(X_scaled, y)

        self.is_fitted = True
        self._fit_date = datetime.now()

        return {
            "status": "fitted",
            "n_samples": len(X),
            "features": self.active_feature_columns,
            "missing_features": sorted(missing) if missing else [],
        }

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate signals using the trained ML model.

        Args:
            date: Signal date
            prices: Price data up to date (kept for interface compatibility)
            features: Feature data up to date

        Returns:
            AlphaResult with model predictions as scores
        """
        if features is None or not self.is_fitted:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
                metadata={"strategy": self.name, "error": "not ready"},
            )

        # Filter up to date (no lookahead)
        feat = features[features["date"] <= pd.Timestamp(date)]

        # Get latest features per asset
        latest = feat.sort_values("date").groupby("ticker").last().reset_index()

        # Check columns
        feature_cols = getattr(self, "active_feature_columns", self.feature_columns)
        missing = set(feature_cols) - set(latest.columns)
        if missing:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
                metadata={"strategy": self.name, "error": f"missing columns: {missing}"},
            )

        X = latest[feature_cols].values

        # Handle NaN: fill with 0 for prediction (scaler expects no NaN)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X = np.nan_to_num(X, nan=0.0)

        X_scaled = self.scaler.transform(X)
        scores = self.model.predict(X_scaled)

        signals = pd.DataFrame({
            "ticker": latest["ticker"].values,
            "score": scores,
        })

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={"strategy": self.name, "features_used": feature_cols},
        )

    # ------------------------------------------------------------------
    # State persistence (model + scaler bundled in one file)
    # ------------------------------------------------------------------

    def _get_extra_state(self) -> dict[str, Any]:
        """Bundle model, scaler, and feature columns into state."""
        return {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "active_feature_columns": self.active_feature_columns,
        }

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        """Restore model, scaler, and feature columns from state."""
        self.model = state.get("model")
        self.scaler = state.get("scaler")
        self.feature_columns = state.get("feature_columns", [])
        self.active_feature_columns = state.get(
            "active_feature_columns", self.feature_columns
        )

    def save_state(self, path: Path) -> dict[str, Any]:
        """
        Save state. Overrides BaseAlpha to set type='ml'.

        Returns:
            Registry metadata with type='ml'
        """
        meta = super().save_state(path)
        meta["type"] = "ml"
        meta["features"] = self.feature_columns
        meta["active_features"] = self.active_feature_columns
        if hasattr(self.model, "get_params"):
            meta["model_params"] = self.model.get_params()
        return meta
