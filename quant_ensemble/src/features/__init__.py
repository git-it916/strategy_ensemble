"""
Feature Engineering Module

Provides feature construction, normalization, and point-in-time merging.
"""

from .asof_merge import (
    PointInTimeMerger,
    apply_execution_lag,
    asof_merge_flow_data,
    asof_merge_fundamentals,
    validate_no_future_leakage,
)
from .build import (
    FeatureBuilder,
    build_features,
)
from .normalize import (
    FeatureNormalizer,
    normalize_by_group,
    normalize_cross_sectional,
    standardize_features,
)

__all__ = [
    # Build
    "FeatureBuilder",
    "build_features",
    # Normalize
    "FeatureNormalizer",
    "normalize_cross_sectional",
    "normalize_by_group",
    "standardize_features",
    # As-of merge
    "asof_merge_fundamentals",
    "asof_merge_flow_data",
    "apply_execution_lag",
    "validate_no_future_leakage",
    "PointInTimeMerger",
]
