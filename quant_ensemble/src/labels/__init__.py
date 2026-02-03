"""
Label Generation Module

Provides label construction and anti-leakage utilities.
"""

from .build import (
    LabelBuilder,
    build_labels,
    calculate_forward_returns,
)
from .leakage import (
    PurgedGroupTimeSplit,
    apply_purging_embargo,
    check_train_test_leakage,
    create_purged_kfold,
    create_walk_forward_splits,
)

__all__ = [
    # Build
    "LabelBuilder",
    "build_labels",
    "calculate_forward_returns",
    # Leakage prevention
    "apply_purging_embargo",
    "create_purged_kfold",
    "create_walk_forward_splits",
    "PurgedGroupTimeSplit",
    "check_train_test_leakage",
]
