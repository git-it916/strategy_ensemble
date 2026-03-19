"""
SLTPMonitor — CLAUDE.md 섹션 7 기준.

5초마다 실행. 하드 SL → 트레일링 스탑 → 하드 TP 순서로 체크.
청산 시 거래 기록 + 텔레그램 알림 발송.
궤적 로그: logs/sltp/YYYY-MM-DD.jsonl (peak 갱신/이벤트 시 기록)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from config.settings import LOGS_DIR, TRAILING_ACTIVATION_PCT, TRAILING_DISTANCE_PCT
from src.engine.decision_engine import Position

logger = logging.getLogger(__name__)

# 궤적 로그 heartbeat 간격 (초)
_SLTP_LOG_HEARTBEAT_SEC = 60


def _log_sltp_event(pos: Position, price: float, pnl_pct: float, event: str) -> None:
    """SL/TP 궤적을 logs/sltp/YYYY-MM-DD.jsonl에 기록."""
    try:
        sdir = LOGS_DIR / "sltp"
        sdir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        path = sdir / f"{now.strftime('%Y-%m-%d')}.jsonl"
        entry = {
            "t": now.isoformat(),
            "symbol": pos.symbol,
            "event": event,
            "price": round(price, 4),
            "pnl_pct": round(pnl_pct * 100, 2),
            "peak_pnl_pct": round(pos.peak_pnl * 100, 2),
            "sl_price": round(pos.sl_price, 4),
            "tp_price": round(pos.tp_price, 4),
            "sl_distance_pct": round(abs(price - pos.sl_price) / price * 100, 2) if price > 0 else 0,
            "trailing_active": pos.trailing_active,
        }
        with open(path, "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception:
        pass  # 로깅 실패가 매매를 막아선 안 됨


class SLTPMonitor:
    """5초 간격 SL/TP/트레일링 모니터."""

    def __init__(self):
        self._closing = False  # 레이스 컨디션 방지 플래그
        self._last_sltp_log_time: float = 0  # heartbeat용 타임스탬프

    async def run_loop(self, executor, telegram, get_position, get_price, store=None, engine=None, drawdown=None):
        """
        별도 asyncio 태스크로 실행.

        Args:
            executor: BinanceExecutor
            telegram: TelegramNotifier
            get_position: () -> Position | None
            get_price: (symbol) -> float
            store: PositionStore (거래 기록 + peak_pnl 영속화용)
        """
        while True:
            try:
                pos = get_position()
                if pos is not None and not self._closing:
                    price = await get_price(pos.symbol)
                    if price > 0:
                        # peak_pnl 갱신 (트레일링 미활성이어도 추적)
                        pnl_pct = self._calc_pnl_pct(pos, price)
                        old_peak = pos.peak_pnl
                        old_trailing = pos.trailing_active

                        reason = self.check(pos, price)

                        # peak_pnl이 바뀌었으면 영속화
                        if store and (pos.peak_pnl != old_peak or pos.trailing_active != old_trailing):
                            store._save()

                        # 궤적 로그: peak 갱신 or 트레일링 활성화 or heartbeat
                        import time as _time
                        now_mono = _time.monotonic()
                        if pos.peak_pnl != old_peak:
                            _log_sltp_event(pos, price, pnl_pct, "UPDATE")
                            self._last_sltp_log_time = now_mono
                        elif pos.trailing_active and not old_trailing:
                            _log_sltp_event(pos, price, pnl_pct, "TRAILING_ACTIVATED")
                            self._last_sltp_log_time = now_mono
                        elif now_mono - self._last_sltp_log_time >= _SLTP_LOG_HEARTBEAT_SEC:
                            _log_sltp_event(pos, price, pnl_pct, "HEARTBEAT")
                            self._last_sltp_log_time = now_mono

                        if reason:
                            _log_sltp_event(pos, price, pnl_pct, reason)
                            self._closing = True  # 메인 루프와 동시 청산 방지
                            try:
                                logger.info(
                                    f"SLTP trigger: {pos.symbol} {pos.direction} "
                                    f"reason={reason} price=${price:,.4f} pnl={pnl_pct:+.2%}"
                                )

                                # 청산 전 포지션 정보 저장
                                closing_symbol = pos.symbol
                                closing_direction = pos.direction
                                closing_entry_price = pos.entry_price
                                closing_entry_time = pos.entry_time

                                close_pnl = await executor.close_position(pos, reason)

                                # 청산 실패시 store 건드리지 않음
                                if close_pnl == 0.0 and reason == "SL":
                                    logger.error(f"SLTP close may have failed for {closing_symbol}")

                                # cooldown 기록 (SL/TP 후 즉시 재진입 방지)
                                if engine:
                                    engine.record_exit(closing_symbol, reason)

                                # drawdown 기록
                                if drawdown:
                                    try:
                                        balance = await executor.get_balance()
                                        from config.settings import BALANCE_USAGE_RATIO, LEVERAGE
                                        pnl_usdt = balance * BALANCE_USAGE_RATIO * LEVERAGE * pnl_pct
                                        drawdown.record_pnl(pnl_usdt)
                                    except Exception:
                                        pass

                                now_utc = datetime.now(timezone.utc)
                                hold_sec = (now_utc - closing_entry_time).total_seconds() if closing_entry_time.tzinfo else 0

                                # 거래 기록
                                if store:
                                    from src.engine.position_store import PositionStore
                                    trade_entry = {
                                        "symbol": closing_symbol,
                                        "direction": closing_direction,
                                        "entry_price": closing_entry_price,
                                        "exit_price": price,
                                        "entry_time": closing_entry_time.isoformat() if hasattr(closing_entry_time, 'isoformat') else str(closing_entry_time),
                                        "exit_time": now_utc.isoformat(),
                                        "reason": reason,
                                        "pnl_pct": round(pnl_pct * 100, 2),
                                        "trajectory": {
                                            "peak_pnl_pct": round(pos.peak_pnl * 100, 2),
                                            "trough_pnl_pct": round(min(0.0, pnl_pct) * 100, 2),
                                            "trailing_activated": pos.trailing_active,
                                        },
                                    }
                                    PositionStore._append_trade_log(trade_entry)
                                    store._position = None
                                    store._save()

                                # 텔레그램 알림
                                if telegram:
                                    try:
                                        hold_min = hold_sec / 60 if hold_sec else 0
                                        telegram.send_exit(
                                            coin=closing_symbol,
                                            direction=closing_direction,
                                            entry_price=closing_entry_price,
                                            exit_price=price,
                                            reason=reason,
                                            pnl_usdt=0,
                                            pnl_pct=pnl_pct * 100,
                                            hold_duration_min=hold_min,
                                        )
                                    except Exception as e:
                                        logger.error(f"SLTP telegram failed: {e}")
                            finally:
                                self._closing = False

            except Exception as e:
                logger.error(f"SLTP monitor error: {e}")

            await asyncio.sleep(5)

    def check(self, position: Position, current_price: float) -> Optional[str]:
        """
        SL/TP/트레일링 체크.

        Returns:
            "SL" | "TP" | "TRAILING" | None
        """
        pnl_pct = self._calc_pnl_pct(position, current_price)

        # 1. 하드 SL
        if position.direction == "LONG":
            if current_price <= position.sl_price:
                return "SL"
        else:
            if current_price >= position.sl_price:
                return "SL"

        # 2. 하드 TP (트레일링 미활성 시에만)
        if not position.trailing_active:
            if position.direction == "LONG":
                if current_price >= position.tp_price:
                    return "TP"
            else:
                if current_price <= position.tp_price:
                    return "TP"

        # 3. 트레일링 스탑
        if pnl_pct >= TRAILING_ACTIVATION_PCT:
            position.trailing_active = True
            position.peak_pnl = max(position.peak_pnl, pnl_pct)

            drawdown = position.peak_pnl - pnl_pct
            if drawdown >= TRAILING_DISTANCE_PCT:
                return "TRAILING"

        return None

    @staticmethod
    def _calc_pnl_pct(pos: Position, price: float) -> float:
        if pos.entry_price <= 0:
            return 0.0
        if pos.direction == "LONG":
            return (price - pos.entry_price) / pos.entry_price
        else:
            return (pos.entry_price - price) / pos.entry_price
