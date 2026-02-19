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
from .volatility_forecast import VolatilityForecastAlpha
from .regime_classifier import RegimeClassifier

try:
    from .intraday_pattern import IntradayPatternAlpha
except ModuleNotFoundError as e:
    # Keep the ML package importable even when optional deps (e.g. lightgbm)
    # are not installed. IntradayPatternAlpha remains unavailable.
    if e.name != "lightgbm":
        raise
    IntradayPatternAlpha = None  # type: ignore[assignment]

__all__ = [
    "BaseMLAlpha",
    "ReturnPredictionAlpha",
    "VolatilityForecastAlpha",
    "RegimeClassifier",
]

if IntradayPatternAlpha is not None:
    __all__.append("IntradayPatternAlpha")
