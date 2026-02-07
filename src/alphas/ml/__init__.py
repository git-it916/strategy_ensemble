"""
ML Alpha Strategies

Machine learning based alpha strategies.
All ML alphas inherit from BaseMLAlpha, which provides:
    - Automatic scaler management (per-model, not global)
    - Consistent save/load with model + scaler bundled together
    - Same interface as BaseAlpha (polymorphic with rule-based alphas)
"""

from .base_ml_alpha import BaseMLAlpha

__all__ = ["BaseMLAlpha"]
