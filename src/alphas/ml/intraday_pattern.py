"""
Intraday Pattern Alpha

LightGBM-based alpha that exploits minute-bar microstructure patterns
to predict next-day open gap or close direction.
"""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMRegressor

from .base_ml_alpha import BaseMLAlpha


# Default feature set - primarily intraday features
INTRADAY_FEATURES = [
    # Intraday microstructure
    "intraday_vol",
    "bar_return_skew",
    "bar_return_kurtosis",
    "large_bar_count",
    "large_bar_ratio",
    "ret_first_30min",
    "ret_last_30min",
    "price_range_am",
    "price_range_pm",
    "vwap_deviation",
    "volume_concentration",
    "volume_profile_morning",
    "intraday_realized_vol",
    # A few daily features for context
    "ret_1d",
    "vol_20d",
    "volume_ratio_20d",
    "rsi_14",
]


class IntradayPatternAlpha(BaseMLAlpha):
    """
    Learn intraday microstructure patterns that predict short-term returns.

    Exploits patterns like:
        - Strong morning momentum → continuation into close
        - Abnormal volume concentration → informed trading
        - High intraday skew/kurtosis → regime shift signals

    Score interpretation:
        - Positive score → expected positive next-day return
        - Negative score → expected negative next-day return
    """

    def __init__(
        self,
        name: str = "intraday_pattern",
        feature_columns: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        config = config or {}
        features = feature_columns or config.pop("features", None) or INTRADAY_FEATURES

        super().__init__(name=name, feature_columns=features, config=config)

    def _build_model(self) -> LGBMRegressor:
        return LGBMRegressor(
            n_estimators=self.config.get("n_estimators", 400),
            max_depth=self.config.get("max_depth", 5),
            learning_rate=self.config.get("learning_rate", 0.05),
            subsample=self.config.get("subsample", 0.7),
            colsample_bytree=self.config.get("colsample_bytree", 0.7),
            min_child_samples=self.config.get("min_child_samples", 20),
            reg_alpha=self.config.get("reg_alpha", 0.1),
            reg_lambda=self.config.get("reg_lambda", 1.0),
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
