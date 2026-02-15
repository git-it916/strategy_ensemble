"""
Reasoning Logger

Structured JSON logging for all LLM decisions.
Every LLM call produces a timestamped reasoning record
written to logs/reasoning/YYYY-MM-DD/.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReasoningLogger:
    """
    Log structured JSON reasoning records for all LLM decisions.

    Records are:
        - Written to individual JSON files in logs/reasoning/YYYY-MM-DD/
        - Accumulated in memory for the current session
        - Queryable by date, model, task type
    """

    def __init__(self, log_dir: Path | str | None = None):
        if log_dir is None:
            from config.settings import LOGS_DIR
            log_dir = LOGS_DIR / "reasoning"

        self.log_dir = Path(log_dir)
        self._session_logs: list[dict[str, Any]] = []

    def log(
        self,
        model: str,
        task: str,
        reasoning: dict[str, Any],
        signals: list[dict] | None = None,
        confidence: float | None = None,
        latency_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Record a reasoning entry.

        Returns:
            The complete log entry dict
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "task": task,
            "reasoning": reasoning,
        }

        if signals is not None:
            entry["signals"] = signals
        if confidence is not None:
            entry["confidence"] = confidence
        if latency_ms is not None:
            entry["latency_ms"] = latency_ms
        if metadata is not None:
            entry["metadata"] = metadata

        # Store in memory
        self._session_logs.append(entry)

        # Write to disk
        self._write_to_file(entry)

        logger.info(
            f"Reasoning logged: model={model} task={task} "
            f"latency={latency_ms}ms confidence={confidence}"
        )

        return entry

    def _write_to_file(self, entry: dict) -> None:
        """Write entry to date-partitioned JSON file."""
        try:
            now = datetime.now()
            day_dir = self.log_dir / now.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{now.strftime('%H%M%S_%f')}_{entry['task']}.json"
            filepath = day_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to write reasoning log: {e}")

    def get_session_logs(self) -> list[dict]:
        """Get all logs from current session."""
        return list(self._session_logs)

    def get_logs_by_date(self, date: datetime | str) -> list[dict]:
        """Load logs for a specific date from disk."""
        if isinstance(date, str):
            date_str = date
        else:
            date_str = date.strftime("%Y-%m-%d")

        day_dir = self.log_dir / date_str
        if not day_dir.exists():
            return []

        logs = []
        for f in sorted(day_dir.glob("*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    logs.append(json.load(fh))
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")

        return logs

    def get_logs_by_task(self, task: str) -> list[dict]:
        """Filter session logs by task type."""
        return [log for log in self._session_logs if log.get("task") == task]

    def summary(self) -> dict[str, Any]:
        """Get summary statistics for current session."""
        if not self._session_logs:
            return {
                "total_calls": 0,
                "by_model": {},
                "by_task": {},
                "avg_latency_ms": 0,
                "avg_confidence": 0,
            }

        by_model: dict[str, int] = {}
        by_task: dict[str, int] = {}
        latencies = []
        confidences = []

        for log in self._session_logs:
            model = log.get("model", "unknown")
            task = log.get("task", "unknown")
            by_model[model] = by_model.get(model, 0) + 1
            by_task[task] = by_task.get(task, 0) + 1

            if log.get("latency_ms") is not None:
                latencies.append(log["latency_ms"])
            if log.get("confidence") is not None:
                confidences.append(log["confidence"])

        return {
            "total_calls": len(self._session_logs),
            "by_model": by_model,
            "by_task": by_task,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        }
