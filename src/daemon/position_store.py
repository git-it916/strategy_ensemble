"""
Position Store

Persists position metadata (entry price, SL/TP levels, reasoning)
to a JSON file so it survives daemon restarts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ManagedPosition:
    """A position with entry price, SL/TP levels, and reasoning."""
    ticker: str
    side: str                  # "LONG" or "SHORT"
    entry_price: float
    entry_time: str            # ISO format string
    target_weight: float
    stop_loss_price: float     # Absolute price level
    take_profit_price: float   # Absolute price level
    stop_loss_pct: float       # Original percentage from Sonnet
    take_profit_pct: float     # Original percentage from Sonnet
    reasoning: str
    status: str = "active"     # active, stopped_out, took_profit, closed


class PositionStore:
    """
    Persist and manage position metadata with SL/TP levels.

    Stores to a JSON file for crash recovery.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.positions: dict[str, ManagedPosition] = {}
        self._load()

    def upsert(
        self,
        ticker: str,
        side: str,
        entry_price: float,
        weight: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        reasoning: str = "",
    ) -> ManagedPosition:
        """Create or update a managed position after order execution."""
        if side == "LONG":
            # LONG: SL below entry, TP above entry
            # stop_loss_pct is negative (e.g., -0.05) → sl_price < entry
            # take_profit_pct is positive (e.g., +0.10) → tp_price > entry
            sl_price = entry_price * (1 + stop_loss_pct)
            tp_price = entry_price * (1 + take_profit_pct)
        else:  # SHORT
            # SHORT: SL above entry, TP below entry (directions inverted)
            # stop_loss_pct = -0.05 → 1 - (-0.05) = 1.05 → sl_price > entry ✓
            # take_profit_pct = +0.10 → 1 - 0.10 = 0.90 → tp_price < entry ✓
            sl_price = entry_price * (1 - stop_loss_pct)
            tp_price = entry_price * (1 - take_profit_pct)

        pos = ManagedPosition(
            ticker=ticker,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now().isoformat(),
            target_weight=weight,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            reasoning=reasoning,
        )
        self.positions[ticker] = pos
        self._save()
        logger.info(
            f"Position stored: {ticker} {side} @ {entry_price:.2f} "
            f"SL={sl_price:.2f} TP={tp_price:.2f}"
        )
        return pos

    def remove(self, ticker: str, reason: str = "closed") -> None:
        """Remove a position (mark as closed)."""
        if ticker in self.positions:
            self.positions[ticker].status = reason
            logger.info(f"Position removed: {ticker} ({reason})")
            self._save()
            del self.positions[ticker]
            self._save()

    def get_active(self) -> list[ManagedPosition]:
        """Get all active positions."""
        return [p for p in self.positions.values() if p.status == "active"]

    def get_recently_closed(self, minutes: int = 30) -> list[ManagedPosition]:
        """Get positions closed in the last N minutes (for Sonnet context)."""
        # Currently not tracking closed positions separately.
        # Could be extended with a history list.
        return []

    def _load(self) -> None:
        """Load positions from JSON file."""
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for ticker, pos_data in data.items():
                self.positions[ticker] = ManagedPosition(**pos_data)
            logger.info(f"Loaded {len(self.positions)} managed positions")
        except Exception as e:
            logger.warning(f"Failed to load position store: {e}")

    def _save(self) -> None:
        """Save positions to JSON file atomically (write tmp → rename)."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {ticker: asdict(pos) for ticker, pos in self.positions.items()}
            tmp_path = self.path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(self.path)
        except Exception as e:
            logger.error(f"Failed to save position store: {e}")
