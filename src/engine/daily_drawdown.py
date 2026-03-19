"""
DailyDrawdownGuard — CLAUDE.md 섹션 10 기준.

당일 실현 손익 누적 추적. -5% 도달 시 전 포지션 청산 + 자정까지 진입 차단.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from config.settings import DAILY_MAX_LOSS_PCT

logger = logging.getLogger(__name__)


class DailyDrawdownGuard:
    """일일 손실 한도 관리."""

    def __init__(self):
        self._daily_pnl: float = 0.0
        self._start_balance: float = 0.0
        self._blocked: bool = False
        self._reset_date: str = ""

    def set_balance(self, balance: float) -> None:
        """당일 시작 잔고 설정."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._reset_date:
            self._daily_pnl = 0.0
            self._blocked = False
            self._start_balance = balance
            self._reset_date = today
            logger.info(f"Daily drawdown reset: balance=${balance:.2f}")

    def record_pnl(self, pnl_usdt: float) -> None:
        """실현 손익 기록."""
        self._daily_pnl += pnl_usdt
        if self._start_balance > 0:
            pct = self._daily_pnl / self._start_balance
            if pct <= DAILY_MAX_LOSS_PCT:
                self._blocked = True
                logger.warning(
                    f"DAILY LOSS LIMIT: {pct:.2%} (limit={DAILY_MAX_LOSS_PCT:.2%})"
                )

    def is_blocked(self) -> bool:
        """거래 차단 여부."""
        # 자정 리셋 체크
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._reset_date:
            self._daily_pnl = 0.0
            self._blocked = False
            self._reset_date = today
        return self._blocked

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def daily_pnl_pct(self) -> float:
        if self._start_balance <= 0:
            return 0.0
        return self._daily_pnl / self._start_balance
