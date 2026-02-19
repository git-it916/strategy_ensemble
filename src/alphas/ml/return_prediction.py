"""
Return Prediction Alpha

XGBoost-based forward return prediction.
Uses daily technical + intraday features to predict N-day forward returns.
"""

from __future__ import annotations

from typing import Any

from xgboost import XGBRegressor

from .base_ml_alpha import BaseMLAlpha


# Default feature set for return prediction
RETURN_FEATURES = [
    # Daily technical
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "ma_ratio_5", "ma_ratio_20", "ma_ratio_60",
    "rsi_14", "bb_pct_b",
    "macd", "macd_signal",
    "vol_5d", "vol_20d",
    "volume_ratio_20d",
    # Intraday (optional - gracefully handled if missing)
    "intraday_vol",
    "open_close_gap",
]


class ReturnPredictionAlpha(BaseMLAlpha):
    """
    Predict forward N-day stock returns using XGBoost.

    Score interpretation:
        - Positive score → expected positive return → buy signal
        - Negative score → expected negative return → avoid/short
        - Magnitude indicates conviction strength
    """

    def __init__(
        self,
        name: str = "return_prediction",
        feature_columns: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        config = config or {}
        features = feature_columns or config.pop("features", None) or RETURN_FEATURES
        # Intraday features are optional in many environments.
        config.setdefault("allow_feature_subset", True)

        super().__init__(name=name, feature_columns=features, config=config)

    def _build_model(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=self.config.get("n_estimators", 500),
            max_depth=self.config.get("max_depth", 6),
            learning_rate=self.config.get("learning_rate", 0.05),
            subsample=self.config.get("subsample", 0.8),
            colsample_bytree=self.config.get("colsample_bytree", 0.8),
            min_child_weight=self.config.get("min_child_weight", 10),
            reg_alpha=self.config.get("reg_alpha", 0.1),
            reg_lambda=self.config.get("reg_lambda", 1.0),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
