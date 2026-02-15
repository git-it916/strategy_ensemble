"""
Volatility Forecast Alpha

XGBoost-based forward volatility prediction.
Used for position sizing: lower predicted vol → larger position allowed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .base_ml_alpha import BaseMLAlpha
from ..base_alpha import AlphaResult


# Default feature set - volatility-focused
VOLATILITY_FEATURES = [
    # Realized vol at multiple horizons
    "vol_5d", "vol_20d",
    "vol_of_vol",
    "vol_ratio_5_20",
    # OHLC-based vol estimators
    "parkinson_vol",
    "garman_klass_vol",
    # Return-based
    "ret_abs_ma5",
    "ret_1d", "ret_5d",
    # Range
    "range_ratio", "range_ratio_ma20",
    # Volume (vol-volume correlation)
    "volume_ratio_20d",
    # Intraday (most informative for vol prediction)
    "intraday_realized_vol",
    "intraday_vol",
    "large_bar_ratio",
]


class VolatilityForecastAlpha(BaseMLAlpha):
    """
    Predict forward realized volatility per stock.

    Unlike other alphas, this is a RISK alpha:
        - It doesn't say WHAT to buy, but HOW MUCH to hold
        - Score = -predicted_vol (negative so higher score = lower vol = safer)
        - The allocator can use this to scale positions inversely to vol

    Integration with ensemble:
        - Can be used standalone as a risk-parity signal
        - Or used by the allocator for position sizing
    """

    def __init__(
        self,
        name: str = "volatility_forecast",
        feature_columns: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        config = config or {}
        features = feature_columns or config.pop("features", None) or VOLATILITY_FEATURES

        super().__init__(name=name, feature_columns=features, config=config)

    def _build_model(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=self.config.get("n_estimators", 500),
            max_depth=self.config.get("max_depth", 5),
            learning_rate=self.config.get("learning_rate", 0.05),
            subsample=self.config.get("subsample", 0.8),
            colsample_bytree=self.config.get("colsample_bytree", 0.8),
            min_child_weight=self.config.get("min_child_weight", 20),
            reg_alpha=self.config.get("reg_alpha", 0.1),
            reg_lambda=self.config.get("reg_lambda", 1.0),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate vol-based signals.

        Overrides BaseMLAlpha to invert scores:
        score = -predicted_vol (lower vol = higher score = prefer to hold)
        """
        result = super().generate_signals(date, prices, features)

        if not result.signals.empty and "score" in result.signals.columns:
            # Invert: lower predicted vol → higher score
            result.signals["predicted_vol"] = result.signals["score"]
            result.signals["score"] = -result.signals["score"]

        result.metadata["strategy"] = self.name
        result.metadata["type"] = "risk_alpha"
        return result

    def predict_volatility(
        self,
        date: datetime,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get raw volatility predictions (not inverted).

        Useful for the allocator to directly use predicted vol for sizing.

        Args:
            date: Prediction date
            features: Feature DataFrame

        Returns:
            DataFrame with ticker, predicted_vol columns
        """
        if features is None or not self.is_fitted:
            return pd.DataFrame(columns=["ticker", "predicted_vol"])

        feat = features[features["date"] <= pd.Timestamp(date)]
        latest = feat.sort_values("date").groupby("ticker").last().reset_index()

        missing = set(self.feature_columns) - set(latest.columns)
        if missing:
            return pd.DataFrame(columns=["ticker", "predicted_vol"])

        X = latest[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Floor at small positive value (vol can't be negative)
        predictions = np.maximum(predictions, 0.01)

        return pd.DataFrame({
            "ticker": latest["ticker"].values,
            "predicted_vol": predictions,
        })
