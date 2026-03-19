"""
PositionStore — 포지션 상태 영속화.

JSON 파일 기반. 봇 재시작 시 복구.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import LOGS_DIR
from src.engine.decision_engine import Position

logger = logging.getLogger(__name__)

STORE_PATH = LOGS_DIR / "position_store.json"


class PositionStore:
    """포지션 상태 관리."""

    def __init__(self):
        self._position: Position | None = None
        self._trade_history: list[dict] = []
        self._load()

    @property
    def current(self) -> Position | None:
        return self._position

    def open(self, position: Position) -> None:
        """포지션 기록."""
        self._position = position
        self._save()
        logger.info(f"Position stored: {position.symbol} {position.direction}")

    def close(self, reason: str, exit_price: float, pnl_usdt: float) -> None:
        """포지션 청산 기록."""
        if self._position:
            self._trade_history.append({
                "symbol": self._position.symbol,
                "direction": self._position.direction,
                "entry_price": self._position.entry_price,
                "exit_price": exit_price,
                "entry_time": self._position.entry_time.isoformat(),
                "exit_time": datetime.utcnow().isoformat(),
                "reason": reason,
                "pnl_usdt": pnl_usdt,
            })
            # 거래 기록 파일에 추가
            self._append_trade_log(self._trade_history[-1])
        self._position = None
        self._save()

    def _save(self) -> None:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if self._position:
            data["position"] = {
                "symbol": self._position.symbol,
                "direction": self._position.direction,
                "entry_price": self._position.entry_price,
                "entry_time": self._position.entry_time.isoformat(),
                "sl_price": self._position.sl_price,
                "tp_price": self._position.tp_price,
                "trailing_active": self._position.trailing_active,
                "peak_pnl": self._position.peak_pnl,
                "entry_score": self._position.entry_score,
                "fade_since": self._position.fade_since.isoformat() if self._position.fade_since else None,
                "weak_since": self._position.weak_since.isoformat() if self._position.weak_since else None,
            }
        with open(STORE_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        if not STORE_PATH.exists():
            return
        try:
            with open(STORE_PATH) as f:
                data = json.load(f)
            pos_data = data.get("position")
            if pos_data:
                self._position = Position(
                    symbol=pos_data["symbol"],
                    direction=pos_data["direction"],
                    entry_price=pos_data["entry_price"],
                    entry_time=datetime.fromisoformat(pos_data["entry_time"]),
                    sl_price=pos_data["sl_price"],
                    tp_price=pos_data["tp_price"],
                    trailing_active=pos_data.get("trailing_active", False),
                    peak_pnl=pos_data.get("peak_pnl", 0.0),
                    entry_score=pos_data.get("entry_score", 0.0),
                    fade_since=datetime.fromisoformat(pos_data["fade_since"]) if pos_data.get("fade_since") else None,
                    weak_since=datetime.fromisoformat(pos_data["weak_since"]) if pos_data.get("weak_since") else None,
                )
                logger.info(f"Position restored: {self._position.symbol}")
        except Exception as e:
            logger.error(f"Position store load failed: {e}")

    @staticmethod
    def _append_trade_log(trade: dict) -> None:
        trade_dir = LOGS_DIR / "trades"
        trade_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        path = trade_dir / f"{today}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(trade) + "\n")
