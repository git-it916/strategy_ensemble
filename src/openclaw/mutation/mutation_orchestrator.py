"""
Mutation Orchestrator

Coordinates parameter sweep and feature mutation.
Runs sweep first (cheaper), then feature mutation on the improved version.
"""

from __future__ import annotations

import logging
from typing import Any

from src.openclaw.mutation.feature_mutator import FeatureMutator, MutationResult
from src.openclaw.mutation.parameter_sweeper import ParameterSweeper, SweepResult
from src.openclaw.registry.alpha_registry import AlphaEntry, AlphaRegistry
from src.openclaw.validator.backtest_runner import SingleAlphaBacktestRunner
from src.openclaw.validator.quality_gates import QualityGateChecker

logger = logging.getLogger(__name__)


class MutationOrchestrator:
    """
    Coordinates both mutation strategies:
    1. Parameter sweep (all alphas, runs first)
    2. Feature mutation (ML alphas only, runs on improved version)

    If either produces improvement, creates a variant alpha
    for approval via Telegram.
    """

    def __init__(
        self,
        backtest_runner: SingleAlphaBacktestRunner,
        registry: AlphaRegistry,
        quality_gates: QualityGateChecker | None = None,
        notifier=None,
    ):
        self.runner = backtest_runner
        self.registry = registry
        self.gates = quality_gates or QualityGateChecker()
        self.notifier = notifier

        self.param_sweeper = ParameterSweeper(backtest_runner, self.gates)
        self.feature_mutator = FeatureMutator(backtest_runner, self.gates)

    def run_mutation_cycle(
        self,
        target_alphas: list[str] | None = None,
        param_max_trials: int = 20,
        feature_max_trials: int = 15,
    ) -> list[dict[str, Any]]:
        """
        Run mutation cycle on active alphas.

        Flow per alpha:
        1. Parameter sweep (cheaper, faster)
        2. If improvement found → feature mutation on improved version
        3. If no improvement → feature mutation on original
        4. Best variant registered as pending for approval

        Args:
            target_alphas: Specific alpha names to mutate (default: all active)
            param_max_trials: Max trials for parameter sweep
            feature_max_trials: Max trials for feature mutation

        Returns:
            List of mutation result dicts.
        """
        entries = self.registry.get_active()
        if target_alphas:
            entries = [e for e in entries if e.name in target_alphas]

        if not entries:
            logger.info("No alphas to mutate")
            return []

        results = []

        for entry in entries:
            logger.info(f"Mutating: {entry.name}")

            try:
                result = self._mutate_alpha(
                    entry, param_max_trials, feature_max_trials
                )
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Mutation failed for {entry.name}: {e}")
                results.append({
                    "alpha_name": entry.name,
                    "status": "error",
                    "error": str(e),
                })

        logger.info(
            f"Mutation cycle complete: "
            f"{len(results)} results from {len(entries)} alphas"
        )

        return results

    def _mutate_alpha(
        self,
        entry: AlphaEntry,
        param_max_trials: int,
        feature_max_trials: int,
    ) -> dict[str, Any] | None:
        """
        Run both mutation strategies on a single alpha.

        Returns mutation result dict if improvement found.
        """
        sweep_result: SweepResult | None = None
        feature_result: MutationResult | None = None

        # Step 1: Parameter sweep (always, for all alpha types)
        try:
            sweep_result = self.param_sweeper.sweep(
                entry, max_trials=param_max_trials
            )
        except Exception as e:
            logger.warning(f"Parameter sweep failed for {entry.name}: {e}")

        # Step 2: Feature mutation (ML alphas only)
        is_ml = entry.config.get("type") == "ml" or entry.config.get("feature_columns")
        if is_ml:
            try:
                # If param sweep found improvement, update entry temporarily
                if sweep_result:
                    modified_entry = AlphaEntry(
                        name=entry.name,
                        class_name=entry.class_name,
                        module_path=entry.module_path,
                        oos_sharpe=sweep_result.best_sharpe,
                        config={**entry.config, **sweep_result.best_params},
                    )
                    feature_result = self.feature_mutator.mutate(
                        modified_entry, max_trials=feature_max_trials
                    )
                else:
                    feature_result = self.feature_mutator.mutate(
                        entry, max_trials=feature_max_trials
                    )
            except Exception as e:
                logger.warning(f"Feature mutation failed for {entry.name}: {e}")

        # Determine best variant
        best = self._pick_best(entry, sweep_result, feature_result)
        if not best:
            return None

        # Register variant
        variant_name = f"{entry.name}_v{self.registry.total_count + 1}"

        variant_entry = AlphaEntry(
            name=variant_name,
            class_name=entry.class_name,
            module_path=entry.module_path,
            status="pending",
            source_url=entry.source_url,
            hypothesis=f"Mutated from {entry.name}: {best['type']}",
            oos_sharpe=best["sharpe"],
            oos_mdd=entry.oos_mdd,
            config=best.get("config", entry.config),
        )

        self.registry.add(variant_entry)

        # Notify
        if self.notifier:
            try:
                self.notifier.send_message(
                    f"<b>Mutation Found: {variant_name}</b>\n\n"
                    f"Based on: {entry.name}\n"
                    f"Type: {best['type']}\n"
                    f"Sharpe: {entry.oos_sharpe:.2f} → {best['sharpe']:.2f} "
                    f"(+{best['improvement']:.2f})\n\n"
                    f"Use /approve {variant_name} to start paper trading"
                )
            except Exception as e:
                logger.warning(f"Failed to notify mutation: {e}")

        return {
            "alpha_name": entry.name,
            "variant_name": variant_name,
            "status": "pending_approval",
            "type": best["type"],
            "original_sharpe": entry.oos_sharpe,
            "new_sharpe": best["sharpe"],
            "improvement": best["improvement"],
        }

    @staticmethod
    def _pick_best(
        entry: AlphaEntry,
        sweep: SweepResult | None,
        feature: MutationResult | None,
    ) -> dict[str, Any] | None:
        """Pick the best mutation result between sweep and feature."""
        candidates = []

        if sweep and sweep.improvement > 0:
            candidates.append({
                "type": "parameter_sweep",
                "sharpe": sweep.best_sharpe,
                "improvement": sweep.improvement,
                "config": {**entry.config, **sweep.best_params},
            })

        if feature and feature.improvement > 0:
            candidates.append({
                "type": "feature_mutation",
                "sharpe": feature.new_sharpe,
                "improvement": feature.improvement,
                "config": {
                    **entry.config,
                    "feature_columns": feature.new_features,
                },
            })

        if not candidates:
            return None

        # Return the one with higher Sharpe
        return max(candidates, key=lambda c: c["sharpe"])
