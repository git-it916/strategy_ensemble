"""
Parameter Sweeper

Grid/random parameter search on existing alphas.
Tests parameter variations and returns the best config
that passes quality gates.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.openclaw.config import QUALITY_GATES
from src.openclaw.registry.alpha_registry import AlphaEntry
from src.openclaw.validator.backtest_runner import SingleAlphaBacktestRunner
from src.openclaw.validator.quality_gates import QualityGateChecker

logger = logging.getLogger(__name__)


@dataclass
class SweepResult:
    """Result of a parameter sweep."""

    alpha_name: str
    original_params: dict[str, Any]
    best_params: dict[str, Any]
    original_sharpe: float
    best_sharpe: float
    improvement: float
    all_results: list[dict] = field(default_factory=list)


# Default parameter grids for common alpha types
DEFAULT_PARAM_GRIDS = {
    "lookback_days": [2, 3, 4, 5],
    "skip_days": [0, 1, 2],
    "threshold": [1.0, 1.5, 2.0, 2.5],
    "rsi_period": [7, 10, 14],
    "oversold": [20, 25, 30],
    "overbought": [70, 75, 80],
}


class ParameterSweeper:
    """
    Grid or random search over alpha parameters.

    Works with both rule-based and ML alphas.
    Tests each parameter combination via backtest and quality gates.
    """

    def __init__(
        self,
        backtest_runner: SingleAlphaBacktestRunner,
        quality_gates: QualityGateChecker | None = None,
    ):
        self.runner = backtest_runner
        self.gates = quality_gates or QualityGateChecker()

    def sweep(
        self,
        alpha_entry: AlphaEntry,
        param_grid: dict[str, list] | None = None,
        mode: str = "random",        # "grid" | "random"
        max_trials: int = 20,
    ) -> SweepResult | None:
        """
        Sweep parameters for an existing alpha.

        Args:
            alpha_entry: Alpha to sweep
            param_grid: Override parameter grid (default: inferred)
            mode: "grid" for exhaustive, "random" for random sampling
            max_trials: Maximum number of parameter combinations to test

        Returns:
            SweepResult with best params, or None if no improvement found.
        """
        # Load alpha source code
        module_path = Path(alpha_entry.module_path)
        if not module_path.exists():
            logger.error(f"Alpha module not found: {module_path}")
            return None

        code = module_path.read_text()

        # Get parameter grid
        grid = param_grid or self._infer_param_grid(alpha_entry)
        if not grid:
            logger.info(f"No sweepable params for {alpha_entry.name}")
            return None

        # Generate parameter combinations
        combinations = self._generate_combinations(grid, mode, max_trials)

        logger.info(
            f"Sweeping {alpha_entry.name}: {len(combinations)} combinations"
        )

        # Baseline: run with original params
        original_params = alpha_entry.config.copy()
        original_sharpe = alpha_entry.oos_sharpe

        best_params = original_params
        best_sharpe = original_sharpe
        all_results = []

        for i, params in enumerate(combinations):
            try:
                # Modify alpha code to use new params
                modified_code = self._inject_params(
                    code, alpha_entry.class_name, params
                )

                # Run backtest
                result = self.runner.run(
                    alpha_code=modified_code,
                    class_name=alpha_entry.class_name,
                )

                oos_sharpe = result["oos_metrics"].get("sharpe_ratio", 0)
                oos_mdd = result["oos_metrics"].get("max_drawdown", 0)

                trial_result = {
                    "trial": i,
                    "params": params,
                    "oos_sharpe": oos_sharpe,
                    "oos_mdd": oos_mdd,
                    "turnover": result.get("turnover", 0),
                }
                all_results.append(trial_result)

                # Check quality gates
                passed, failures, _ = self.gates.check_all(
                    is_metrics=result["is_metrics"],
                    oos_metrics=result["oos_metrics"],
                    daily_returns=result["daily_returns"],
                    turnover=result.get("turnover", 0),
                )

                if passed and oos_sharpe > best_sharpe:
                    best_sharpe = oos_sharpe
                    best_params = params
                    logger.info(
                        f"  [{i+1}/{len(combinations)}] "
                        f"New best: Sharpe={oos_sharpe:.2f} params={params}"
                    )

            except Exception as e:
                logger.warning(
                    f"  [{i+1}/{len(combinations)}] Trial failed: {e}"
                )
                all_results.append({
                    "trial": i, "params": params, "error": str(e)
                })

        improvement = best_sharpe - original_sharpe

        if improvement <= 0:
            logger.info(
                f"No improvement found for {alpha_entry.name} "
                f"(original Sharpe={original_sharpe:.2f})"
            )
            return None

        result = SweepResult(
            alpha_name=alpha_entry.name,
            original_params=original_params,
            best_params=best_params,
            original_sharpe=original_sharpe,
            best_sharpe=best_sharpe,
            improvement=improvement,
            all_results=all_results,
        )

        logger.info(
            f"Sweep complete for {alpha_entry.name}: "
            f"Sharpe {original_sharpe:.2f} → {best_sharpe:.2f} "
            f"(+{improvement:.2f})"
        )

        return result

    @staticmethod
    def _infer_param_grid(entry: AlphaEntry) -> dict[str, list]:
        """Infer reasonable parameter ranges from current config."""
        grid = {}
        config = entry.config

        for param_name, default_values in DEFAULT_PARAM_GRIDS.items():
            if param_name in config:
                current = config[param_name]
                # Generate a range around the current value
                if isinstance(current, int):
                    values = sorted(set(
                        [max(1, current - 2), max(1, current - 1),
                         current, current + 1, current + 2]
                    ))
                    grid[param_name] = values
                elif isinstance(current, float):
                    values = sorted(set(
                        [max(0.1, current * 0.5), current * 0.75,
                         current, current * 1.25, current * 1.5]
                    ))
                    grid[param_name] = [round(v, 3) for v in values]

        return grid

    @staticmethod
    def _generate_combinations(
        grid: dict[str, list],
        mode: str,
        max_trials: int,
    ) -> list[dict]:
        """Generate parameter combinations."""
        import itertools

        keys = list(grid.keys())
        values = list(grid.values())

        if mode == "grid":
            all_combos = [
                dict(zip(keys, combo))
                for combo in itertools.product(*values)
            ]
            if len(all_combos) > max_trials:
                random.shuffle(all_combos)
                return all_combos[:max_trials]
            return all_combos

        else:  # random
            combos = []
            for _ in range(max_trials):
                combo = {k: random.choice(v) for k, v in zip(keys, values)}
                combos.append(combo)
            return combos

    @staticmethod
    def _inject_params(
        code: str,
        class_name: str,
        params: dict[str, Any],
    ) -> str:
        """
        Modify alpha source code to use new parameter values.

        Strategy: Replace default values in __init__ signature.
        """
        modified = code
        for param_name, value in params.items():
            # Replace default value in __init__
            # Pattern: param_name: type = old_value or param_name=old_value
            import re
            pattern = rf"({param_name}\s*(?::\s*\w+)?\s*=\s*)([^,\)]+)"
            replacement = rf"\g<1>{repr(value)}"
            modified = re.sub(pattern, replacement, modified, count=1)

        return modified
