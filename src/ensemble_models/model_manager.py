"""
Model Manager

Unified model saving, loading, versioning, and archival.
Replaces raw pickle with structured persistence via joblib + registry.yaml.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import yaml

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the full lifecycle of trained models.

    Responsibilities:
        - Save/load individual strategies and ensemble state
        - Maintain registry.yaml with metadata, versioning, and module paths
        - Archive previous model snapshots to history/
        - Dynamic class resolution via importlib (no hard-coded map)
    """

    def __init__(self, models_dir: Path):
        """
        Args:
            models_dir: Root models directory (e.g. project/models/)
        """
        self.models_dir = Path(models_dir)
        self.weights_dir = self.models_dir / "weights"
        self.history_dir = self.models_dir / "history"
        self.registry_path = self.models_dir / "registry.yaml"

        # Ensure directories exist
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_ensemble(
        self,
        ensemble,
        fit_results: dict[str, Any],
        data_meta: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save the full ensemble: individual strategies + ensemble state + registry.

        Args:
            ensemble: Trained EnsembleAgent instance
            fit_results: Results from ensemble.fit()
            data_meta: Metadata about training data (dates, row count, hash)

        Returns:
            Path to the saved registry.yaml
        """
        # 1. Save individual strategies
        strategy_metas = {}
        for name, strategy in ensemble.strategies.items():
            path = self.weights_dir / f"{name}.joblib"
            meta = strategy.save_state(path)
            strategy_metas[name] = meta
            logger.info(f"Saved strategy: {name} -> {path}")

        # 2. Save ensemble state (weights, score_board, performance history)
        ensemble_path = self.weights_dir / "ensemble.joblib"
        ensemble_state = ensemble.get_state()
        joblib.dump(ensemble_state, ensemble_path)
        logger.info(f"Saved ensemble state -> {ensemble_path}")

        # 3. Build and write registry
        registry = {
            "version": self._next_version(),
            "trained_at": datetime.now().isoformat(),
            "data": data_meta or {},
            "strategies": strategy_metas,
            "ensemble": {
                "config": ensemble.config,
                "weights": ensemble.get_weights(),
                "file": str(ensemble_path),
            },
            "fit_results": {
                k: str(v) for k, v in fit_results.items()
            },
        }

        with open(self.registry_path, "w", encoding="utf-8") as f:
            yaml.dump(registry, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Saved registry -> {self.registry_path}")

        # 4. Archive snapshot
        self._archive_snapshot()

        return self.registry_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_ensemble(self):
        """
        Load the full ensemble from registry.yaml.

        Returns:
            Fully restored EnsembleAgent with all state
        """
        from ..ensemble.agent import EnsembleAgent

        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Registry not found at {self.registry_path}. "
                "Run training first (2_train_ensemble.py)."
            )

        with open(self.registry_path, encoding="utf-8") as f:
            registry = yaml.safe_load(f)

        # 1. Load individual strategies
        strategies = []
        for name, meta in registry["strategies"].items():
            cls = self._resolve_class(meta["class"], meta["module"])
            strategy = cls.load_state(Path(meta["file"]))
            strategies.append(strategy)
            logger.info(f"Loaded strategy: {name} ({meta['class']})")

        # 2. Create EnsembleAgent with loaded strategies
        ensemble_config = registry["ensemble"].get("config", {})
        ensemble = EnsembleAgent(strategies=strategies, config=ensemble_config)

        # 3. Restore full ensemble state (score_board, weights, performance)
        ensemble_path = Path(registry["ensemble"]["file"])
        if ensemble_path.exists():
            ensemble_state = joblib.load(ensemble_path)
            ensemble.restore_state(ensemble_state)
            logger.info("Restored ensemble state (score_board, weights, performance)")

        return ensemble

    # ------------------------------------------------------------------
    # Registry info
    # ------------------------------------------------------------------

    def get_registry(self) -> dict[str, Any] | None:
        """Load and return the current registry, or None if not found."""
        if not self.registry_path.exists():
            return None
        with open(self.registry_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_version(self) -> str | None:
        """Get the current model version."""
        registry = self.get_registry()
        return registry["version"] if registry else None

    def list_history(self) -> list[str]:
        """List all archived model snapshots."""
        if not self.history_dir.exists():
            return []
        return sorted(
            [d.name for d in self.history_dir.iterdir() if d.is_dir()],
            reverse=True,
        )

    def load_from_history(self, snapshot_name: str):
        """
        Load an ensemble from a historical snapshot.

        Args:
            snapshot_name: Name of the snapshot directory (e.g. "20260208_093000")

        Returns:
            Restored EnsembleAgent
        """
        snapshot_dir = self.history_dir / snapshot_name
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_dir}")

        # Create a temporary ModelManager pointing at the snapshot
        snapshot_manager = ModelManager(snapshot_dir)
        return snapshot_manager.load_ensemble()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_class(self, class_name: str, module_path: str):
        """
        Dynamically import a class by module path and class name.

        Args:
            class_name: e.g. "RSIReversalAlpha"
            module_path: e.g. "src.alphas.technical.rsi_reversal"

        Returns:
            The class object
        """
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _next_version(self) -> str:
        """Auto-increment patch version from registry."""
        registry = self.get_registry()
        if registry and "version" in registry:
            parts = registry["version"].split(".")
            if len(parts) == 3:
                return f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
        return "1.0.0"

    def _archive_snapshot(self) -> None:
        """Copy current weights + registry to history/ with timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = self.history_dir / ts

        try:
            # Copy weights
            if self.weights_dir.exists():
                shutil.copytree(self.weights_dir, snapshot_dir / "weights")

            # Copy registry
            if self.registry_path.exists():
                (snapshot_dir).mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.registry_path, snapshot_dir / "registry.yaml")

            logger.info(f"Archived snapshot -> {snapshot_dir}")
        except Exception as e:
            logger.warning(f"Failed to archive snapshot: {e}")

    @staticmethod
    def compute_data_hash(df) -> str:
        """Compute a short hash of a DataFrame for data lineage tracking."""
        raw = f"{len(df)}_{df.columns.tolist()}_{df.shape}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
