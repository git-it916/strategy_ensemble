"""
Experiment Tracker

JSONL-based deduplication and history tracking for research sessions.
Prevents redundant alpha generation and maintains an audit trail.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.openclaw.config import EXPERIMENT_TRACKER_PATH
from src.openclaw.researcher.idea_parser import AlphaSpec

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Track all alpha research experiments to prevent duplicates
    and maintain history.

    Data stored as JSONL (one JSON object per line).
    """

    def __init__(self, tracker_path: Path | None = None):
        self._path = tracker_path or EXPERIMENT_TRACKER_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seen_hashes: set[str] = set()
        self._load_hashes()

    def is_duplicate(self, spec: AlphaSpec) -> bool:
        """
        Check if a similar alpha was already researched.

        Uses hash of (name + description) to detect duplicates.
        """
        h = self._compute_hash(spec)
        return h in self._seen_hashes

    def record_experiment(
        self,
        spec: AlphaSpec,
        validation_result: dict[str, Any],
        status: str,              # "passed" | "failed" | "rejected" | "error"
        code_hash: str = "",
    ) -> None:
        """
        Record an experiment result.

        Args:
            spec: Alpha specification
            validation_result: Backtest results and gate checks
            status: Outcome of the experiment
            code_hash: SHA256 hash of generated code
        """
        h = self._compute_hash(spec)
        self._seen_hashes.add(h)

        record = {
            "timestamp": datetime.now().isoformat(),
            "hash": h,
            "status": status,
            "name": spec.name,
            "description": spec.description[:200],
            "hypothesis": spec.hypothesis[:200],
            "source_url": spec.source_url,
            "source_title": spec.source_title,
            "expected_style": spec.expected_style,
            "code_hash": code_hash,
            "validation": self._sanitize_validation(validation_result),
        }

        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
            logger.info(
                f"Recorded experiment: {spec.name} → {status}"
            )
        except Exception as e:
            logger.error(f"Failed to record experiment: {e}")

    def get_history(self, limit: int = 50) -> list[dict]:
        """Return recent experiment records."""
        records = self._load_all()
        return records[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        records = self._load_all()

        if not records:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "rejected": 0,
                "error": 0,
                "pass_rate": 0.0,
            }

        total = len(records)
        by_status = {}
        for r in records:
            s = r.get("status", "unknown")
            by_status[s] = by_status.get(s, 0) + 1

        passed = by_status.get("passed", 0)

        return {
            "total": total,
            "passed": passed,
            "failed": by_status.get("failed", 0),
            "rejected": by_status.get("rejected", 0),
            "error": by_status.get("error", 0),
            "pass_rate": passed / total if total > 0 else 0.0,
        }

    def _load_hashes(self) -> None:
        """Load all seen hashes from the tracker file."""
        if not self._path.exists():
            return

        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        h = record.get("hash", "")
                        if h:
                            self._seen_hashes.add(h)
        except Exception as e:
            logger.error(f"Failed to load experiment tracker: {e}")

    def _load_all(self) -> list[dict]:
        """Load all records from the tracker file."""
        if not self._path.exists():
            return []

        records = []
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to load experiments: {e}")

        return records

    @staticmethod
    def _compute_hash(spec: AlphaSpec) -> str:
        """Compute a deduplication hash from spec name + description."""
        content = f"{spec.name}::{spec.description[:100]}".lower()
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _sanitize_validation(result: dict[str, Any]) -> dict[str, Any]:
        """Extract serializable fields from validation result."""
        sanitized = {}
        for key in ("is_metrics", "oos_metrics", "turnover"):
            if key in result:
                val = result[key]
                if isinstance(val, dict):
                    sanitized[key] = {
                        k: round(v, 6) if isinstance(v, float) else v
                        for k, v in val.items()
                    }
                else:
                    sanitized[key] = val
        return sanitized
