"""
Ensemble Methods Module

Provides various ensemble methods for combining signals.
"""

from .base import EnsembleModel, SimpleAverageEnsemble, WeightedAverageEnsemble
from .metalabel import MetaLabeler
from .moe import MoEEnsemble
from .stacking import StackingEnsemble

__all__ = [
    "EnsembleModel",
    "SimpleAverageEnsemble",
    "WeightedAverageEnsemble",
    "StackingEnsemble",
    "MoEEnsemble",
    "MetaLabeler",
]
