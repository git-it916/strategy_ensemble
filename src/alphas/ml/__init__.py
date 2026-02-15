"""
ML Alpha Strategies

Machine learning based alpha strategies.
All ML alphas inherit from BaseMLAlpha, which provides:
    - Automatic scaler management (per-model, not global)
    - Consistent save/load with model + scaler bundled together
    - Same interface as BaseAlpha (polymorphic with rule-based alphas)
"""

from .base_ml_alpha import BaseMLAlpha
from .return_prediction import ReturnPredictionAlpha
from .intraday_pattern import IntradayPatternAlpha
from .volatility_forecast import VolatilityForecastAlpha
from .regime_classifier import RegimeClassifier

__all__ = [
    "BaseMLAlpha",
    "ReturnPredictionAlpha",
    "IntradayPatternAlpha",
    "VolatilityForecastAlpha",
    "RegimeClassifier",
]
