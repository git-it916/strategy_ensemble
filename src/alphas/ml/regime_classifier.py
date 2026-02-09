"""
Regime Classifier

RandomForest-based market regime classifier.
Classifies the market into bull / sideways / bear regimes.
Used by EnsembleAgent to dynamically adjust strategy weights.

This is NOT a per-stock alpha. It produces a single regime label per date,
which the ensemble uses via generate_signals(regime=...).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REGIME_MAP = {0: "bear", 1: "sideways", 2: "bull"}
REGIME_INV = {v: k for k, v in REGIME_MAP.items()}

# Default market-level features
REGIME_FEATURES = [
    "market_ret",
    "market_ret_5d",
    "market_ret_20d",
    "market_vol_20d",
    "cross_sectional_vol",
    "advance_decline_ratio",
    "pct_above_ma20",
    "market_breadth",
    "volume_trend",
]


class RegimeClassifier:
    """
    Market regime classifier.

    Not a BaseAlpha subclass — it has a separate interface because:
        - Input: market-level features (1 row per date, not per stock)
        - Output: single regime string, not per-stock scores

    Usage:
        classifier = RegimeClassifier()
        classifier.fit(market_features, regime_labels)
        regime = classifier.predict(today_features)  # "bull" / "bear" / "sideways"

        # Then pass to ensemble:
        ensemble.generate_signals(date, prices, features, regime=regime)
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.config = config or {}
        self.feature_columns = feature_columns or REGIME_FEATURES
        self.model: RandomForestClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.is_fitted = False

    def fit(
        self,
        market_features: pd.DataFrame,
        regime_labels: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Train the regime classifier.

        Args:
            market_features: DataFrame with date + feature columns (1 row/date)
            regime_labels: DataFrame with date, y_reg (0/1/2)

        Returns:
            Fit result metrics
        """
        merged = market_features.merge(regime_labels[["date", "y_reg"]], on="date")

        missing = set(self.feature_columns) - set(merged.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = merged[self.feature_columns].values
        y = merged["y_reg"].values

        # Drop NaN rows
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(
            n_estimators=self.config.get("n_estimators", 300),
            max_depth=self.config.get("max_depth", 8),
            min_samples_leaf=self.config.get("min_samples_leaf", 20),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Training accuracy
        train_acc = self.model.score(X_scaled, y)
        class_dist = pd.Series(y).value_counts().to_dict()

        logger.info(
            f"RegimeClassifier fitted: acc={train_acc:.3f}, "
            f"samples={len(X)}, dist={class_dist}"
        )

        return {
            "status": "fitted",
            "train_accuracy": train_acc,
            "n_samples": len(X),
            "class_distribution": class_dist,
        }

    def predict(self, market_features: pd.DataFrame, date: datetime | None = None) -> str:
        """
        Predict current market regime.

        Args:
            market_features: Market feature DataFrame
            date: Target date (uses latest if None)

        Returns:
            Regime string: "bull", "sideways", or "bear"
        """
        if not self.is_fitted:
            logger.warning("RegimeClassifier not fitted, returning 'sideways'")
            return "sideways"

        if date is not None:
            row = market_features[market_features["date"] <= pd.Timestamp(date)]
        else:
            row = market_features

        if row.empty:
            return "sideways"

        latest = row.sort_values("date").iloc[-1:]

        missing = set(self.feature_columns) - set(latest.columns)
        if missing:
            logger.warning(f"Missing features for regime: {missing}")
            return "sideways"

        X = latest[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        pred = self.model.predict(X_scaled)[0]
        regime = REGIME_MAP.get(int(pred), "sideways")

        logger.debug(f"Regime prediction for {date}: {regime}")
        return regime

    def predict_proba(self, market_features: pd.DataFrame, date: datetime | None = None) -> dict[str, float]:
        """
        Get regime probabilities.

        Returns:
            Dict like {"bear": 0.2, "sideways": 0.3, "bull": 0.5}
        """
        if not self.is_fitted:
            return {"bear": 0.33, "sideways": 0.34, "bull": 0.33}

        if date is not None:
            row = market_features[market_features["date"] <= pd.Timestamp(date)]
        else:
            row = market_features

        if row.empty:
            return {"bear": 0.33, "sideways": 0.34, "bull": 0.33}

        latest = row.sort_values("date").iloc[-1:]
        X = latest[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)[0]
        classes = self.model.classes_

        return {REGIME_MAP.get(int(c), "unknown"): float(p) for c, p in zip(classes, proba)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> dict[str, Any]:
        """Save classifier state."""
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "config": self.config,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, path)
        logger.info(f"RegimeClassifier saved -> {path}")

        return {
            "class": "RegimeClassifier",
            "module": "src.alphas.ml.regime_classifier",
            "type": "regime",
            "file": str(path),
        }

    @classmethod
    def load(cls, path: Path) -> "RegimeClassifier":
        """Load classifier from saved state."""
        state = joblib.load(path)
        instance = cls(
            feature_columns=state["feature_columns"],
            config=state["config"],
        )
        instance.model = state["model"]
        instance.scaler = state["scaler"]
        instance.is_fitted = state["is_fitted"]
        logger.info(f"RegimeClassifier loaded <- {path}")
        return instance
