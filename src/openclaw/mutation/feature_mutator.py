"""
Feature Mutator

Feature combination changes for ML-based alphas.
Adds, removes, or swaps features and evaluates impact on performance.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.openclaw.registry.alpha_registry import AlphaEntry
from src.openclaw.validator.backtest_runner import SingleAlphaBacktestRunner
from src.openclaw.validator.quality_gates import QualityGateChecker

logger = logging.getLogger(__name__)


@dataclass
class MutationResult:
    """Result of a feature mutation."""

    alpha_name: str
    original_features: list[str]
    new_features: list[str]
    operation: str                # "add" | "remove" | "swap"
    original_sharpe: float
    new_sharpe: float
    improvement: float
    details: dict[str, Any] = field(default_factory=dict)


# Available features from FeatureEngineer
DAILY_FEATURES = [
    "ret_1d", "ret_5d", "log_ret_1d",
    "ma_ratio_5", "ma_ratio_10", "ma_ratio_20",
    "rsi_14", "bb_pct_b", "macd", "macd_signal",
    "vol_5d", "vol_20d", "vol_of_vol", "vol_ratio_5_20",
    "parkinson_vol", "garman_klass_vol",
    "volume_ratio_20d", "range_ratio", "range_ratio_ma20",
    "ret_abs_ma5",
]


class FeatureMutator:
    """
    Mutate feature sets for ML-based alphas.

    Operations:
        - Remove: drop features one at a time
        - Add: add unused features one at a time
        - Swap: replace each feature with an unused one

    Only applicable to ML alphas (BaseMLAlpha subclasses)
    that have a feature_columns attribute.
    """

    def __init__(
        self,
        backtest_runner: SingleAlphaBacktestRunner,
        quality_gates: QualityGateChecker | None = None,
        available_features: list[str] | None = None,
    ):
        self.runner = backtest_runner
        self.gates = quality_gates or QualityGateChecker()
        self.available_features = available_features or DAILY_FEATURES

    def mutate(
        self,
        alpha_entry: AlphaEntry,
        max_trials: int = 15,
    ) -> MutationResult | None:
        """
        Try feature mutations on an ML alpha.

        Tries remove, add, and swap operations.
        Returns the best mutation that improves Sharpe and passes gates.

        Args:
            alpha_entry: ML alpha entry from registry
            max_trials: Maximum number of mutations to test

        Returns:
            MutationResult if improvement found, None otherwise.
        """
        # Get current features from config
        current_features = alpha_entry.config.get("feature_columns", [])
        if not current_features:
            logger.info(
                f"No feature_columns in config for {alpha_entry.name}, "
                f"skipping feature mutation"
            )
            return None

        original_sharpe = alpha_entry.oos_sharpe
        unused_features = [
            f for f in self.available_features if f not in current_features
        ]

        # Generate mutation candidates
        candidates = self._generate_candidates(
            current_features, unused_features, max_trials
        )

        logger.info(
            f"Feature mutation for {alpha_entry.name}: "
            f"{len(candidates)} candidates "
            f"(current: {len(current_features)} features)"
        )

        best_result: MutationResult | None = None
        best_sharpe = original_sharpe

        for i, (new_features, operation, detail) in enumerate(candidates):
            try:
                # Modify alpha config with new features
                modified_config = alpha_entry.config.copy()
                modified_config["feature_columns"] = new_features

                # Create modified code (update feature_columns in source)
                from pathlib import Path
                module_path = Path(alpha_entry.module_path)
                if not module_path.exists():
                    continue

                code = module_path.read_text()
                modified_code = self._inject_features(
                    code, alpha_entry.class_name, new_features
                )

                # Run backtest
                result = self.runner.run(
                    alpha_code=modified_code,
                    class_name=alpha_entry.class_name,
                )

                oos_sharpe = result["oos_metrics"].get("sharpe_ratio", 0)

                # Check gates
                passed, failures, _ = self.gates.check_all(
                    is_metrics=result["is_metrics"],
                    oos_metrics=result["oos_metrics"],
                    daily_returns=result["daily_returns"],
                    turnover=result.get("turnover", 0),
                )

                if passed and oos_sharpe > best_sharpe:
                    best_sharpe = oos_sharpe
                    best_result = MutationResult(
                        alpha_name=alpha_entry.name,
                        original_features=current_features,
                        new_features=new_features,
                        operation=operation,
                        original_sharpe=original_sharpe,
                        new_sharpe=oos_sharpe,
                        improvement=oos_sharpe - original_sharpe,
                        details=detail,
                    )

                    logger.info(
                        f"  [{i+1}/{len(candidates)}] "
                        f"Improvement: Sharpe={oos_sharpe:.2f} "
                        f"({operation}: {detail})"
                    )

            except Exception as e:
                logger.warning(
                    f"  [{i+1}/{len(candidates)}] Mutation failed: {e}"
                )

        if best_result:
            logger.info(
                f"Feature mutation for {alpha_entry.name}: "
                f"Sharpe {original_sharpe:.2f} → {best_sharpe:.2f} "
                f"({best_result.operation})"
            )
        else:
            logger.info(
                f"No feature improvement found for {alpha_entry.name}"
            )

        return best_result

    @staticmethod
    def _generate_candidates(
        current: list[str],
        unused: list[str],
        max_trials: int,
    ) -> list[tuple[list[str], str, dict]]:
        """
        Generate feature mutation candidates.

        Returns list of (new_features, operation, detail) tuples.
        """
        candidates = []

        # 1. Remove: drop each feature one at a time
        if len(current) > 2:  # need at least 2 features
            for feat in current:
                new = [f for f in current if f != feat]
                candidates.append(
                    (new, "remove", {"removed": feat})
                )

        # 2. Add: add each unused feature
        for feat in unused:
            new = current + [feat]
            candidates.append(
                (new, "add", {"added": feat})
            )

        # 3. Swap: replace each feature with an unused one
        for old_feat in current:
            for new_feat in random.sample(unused, min(3, len(unused))):
                new = [new_feat if f == old_feat else f for f in current]
                candidates.append(
                    (new, "swap", {"removed": old_feat, "added": new_feat})
                )

        # Limit trials
        if len(candidates) > max_trials:
            random.shuffle(candidates)
            candidates = candidates[:max_trials]

        return candidates

    @staticmethod
    def _inject_features(
        code: str,
        class_name: str,
        features: list[str],
    ) -> str:
        """Modify alpha source to use new feature_columns list."""
        import re

        # Replace feature_columns assignment
        pattern = r"(feature_columns\s*=\s*)\[[\s\S]*?\]"
        replacement = f"\\1{repr(features)}"
        modified = re.sub(pattern, replacement, code, count=1)

        return modified
