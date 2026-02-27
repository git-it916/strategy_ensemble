"""
Alpha Registry

YAML-backed catalog of all openclaw alphas.
Tracks status, source, performance metrics, and configuration.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.openclaw.config import REGISTRY_PATH

logger = logging.getLogger(__name__)


@dataclass
class AlphaEntry:
    """Single alpha entry in the registry."""

    name: str
    class_name: str
    module_path: str                  # relative path to .py file
    status: str = "pending"           # pending | paper | live | paused | killed
    added_date: str = ""
    source_url: str = ""
    source_title: str = ""
    hypothesis: str = ""
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    oos_mdd: float = 0.0
    oos_ic: float = 0.0
    current_leverage: float = 1.0
    current_weight: float = 0.0
    code_hash: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    paper_start_date: str = ""
    live_start_date: str = ""
    killed_date: str = ""
    kill_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlphaEntry:
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


class AlphaRegistry:
    """
    YAML-backed registry for managing openclaw alphas.

    Provides add/remove/pause/activate/query operations
    with automatic persistence to disk.
    """

    def __init__(self, registry_path: Path | None = None):
        self._path = registry_path or REGISTRY_PATH
        self._entries: dict[str, AlphaEntry] = {}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def add(self, entry: AlphaEntry) -> None:
        """Add a new alpha to the registry."""
        if not entry.added_date:
            entry.added_date = datetime.now().isoformat()
        self._entries[entry.name] = entry
        self.save()
        logger.info(f"Registered alpha: {entry.name} (status={entry.status})")

    def remove(self, name: str) -> AlphaEntry | None:
        """Remove an alpha from the registry entirely."""
        entry = self._entries.pop(name, None)
        if entry:
            self.save()
            logger.info(f"Removed alpha: {name}")
        return entry

    def kill(self, name: str, reason: str = "") -> AlphaEntry | None:
        """Mark an alpha as killed (keeps in registry for history)."""
        entry = self._entries.get(name)
        if entry:
            entry.status = "killed"
            entry.killed_date = datetime.now().isoformat()
            entry.kill_reason = reason
            self.save()
            logger.warning(f"Killed alpha: {name} reason={reason}")
        return entry

    def pause(self, name: str) -> AlphaEntry | None:
        """Pause an alpha (temporary deactivation)."""
        entry = self._entries.get(name)
        if entry and entry.status in ("paper", "live"):
            entry.status = "paused"
            self.save()
            logger.info(f"Paused alpha: {name}")
        return entry

    def activate(self, name: str, status: str = "live") -> AlphaEntry | None:
        """Activate an alpha (paper or live)."""
        entry = self._entries.get(name)
        if entry:
            entry.status = status
            if status == "paper" and not entry.paper_start_date:
                entry.paper_start_date = datetime.now().isoformat()
            elif status == "live" and not entry.live_start_date:
                entry.live_start_date = datetime.now().isoformat()
            self.save()
            logger.info(f"Activated alpha: {name} → {status}")
        return entry

    def get(self, name: str) -> AlphaEntry | None:
        """Get a specific alpha entry."""
        return self._entries.get(name)

    def get_active(self) -> list[AlphaEntry]:
        """Get all live alphas."""
        return [e for e in self._entries.values() if e.status == "live"]

    def get_paper(self) -> list[AlphaEntry]:
        """Get all paper-trading alphas."""
        return [e for e in self._entries.values() if e.status == "paper"]

    def get_pending(self) -> list[AlphaEntry]:
        """Get all pending (awaiting approval) alphas."""
        return [e for e in self._entries.values() if e.status == "pending"]

    def get_all(self) -> list[AlphaEntry]:
        """Get all entries regardless of status."""
        return list(self._entries.values())

    def get_by_status(self, status: str) -> list[AlphaEntry]:
        """Get entries by status."""
        return [e for e in self._entries.values() if e.status == status]

    @property
    def active_count(self) -> int:
        return len(self.get_active())

    @property
    def total_count(self) -> int:
        return len(self._entries)

    def update_weight(self, name: str, weight: float) -> None:
        """Update an alpha's current ensemble weight."""
        entry = self._entries.get(name)
        if entry:
            entry.current_weight = weight

    def update_leverage(self, name: str, leverage: float) -> None:
        """Update an alpha's current leverage."""
        entry = self._entries.get(name)
        if entry:
            entry.current_leverage = leverage

    def save(self) -> None:
        """Persist registry to YAML."""
        data = {
            "version": "1.0",
            "updated": datetime.now().isoformat(),
            "alphas": {
                name: entry.to_dict()
                for name, entry in self._entries.items()
            },
        }
        with open(self._path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def load(self) -> None:
        """Load registry from YAML."""
        if not self._path.exists():
            self._entries = {}
            return

        try:
            with open(self._path) as f:
                data = yaml.safe_load(f) or {}

            alphas = data.get("alphas", {})
            self._entries = {
                name: AlphaEntry.from_dict(entry_data)
                for name, entry_data in alphas.items()
            }
            logger.info(
                f"Loaded {len(self._entries)} alphas from registry"
            )
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self._entries = {}

    @staticmethod
    def compute_code_hash(code: str) -> str:
        """Compute SHA256 hash of alpha source code."""
        return hashlib.sha256(code.encode()).hexdigest()[:16]
