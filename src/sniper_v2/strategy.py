"""
Sniper V2 — Strategy Engine

EMA(18/40) cross + EMA(200) trend + RSI(14) 55/40
+ 3개 방어 필터 (VolFilter, ADX, Cooldown)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import pandas as pd

from src.sniper_v2.config import (
    ADX_MIN,
    COOLDOWN_BARS,
    RSI_BEAR,
    RSI_BULL,
    SL_ATR_MULT,
    TP1_RR,
    TP2_RR,
    TP3_RR,
    USE_TRAILING,
    VOL_FILTER,
    WARMUP_BARS,
)
from src.sniper_v2.indicators import compute_all


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Signal:
    direction: Direction
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    risk: float
    rsi: float
    adx: float
    atr: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ActiveTrade:
    direction: Direction
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    trail_price: float
    risk: float
    entry_time: datetime
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False

    def current_stop(self) -> float:
        if USE_TRAILING:
            return self.trail_price
        return self.sl_price


class SniperV2:
    """
    Sniper V2 전략 엔진.

    진입: EMA(18/40) cross + close vs EMA(200) + RSI 55/40
    필터: VolFilter + ADX + Cooldown
    청산: ATR×1.5 SL + TP 1.5R/2.75R/4R + Trailing
    """

    def __init__(self):
        self._last_direction: int = 0   # anti-duplicate
        self._cooldown: int = 0         # SL 후 대기 카운터
        self.active_trade: Optional[ActiveTrade] = None

    def compute(self, df: pd.DataFrame) -> Optional[Signal]:
        """
        메인 시그널 생성.

        Args:
            df: OHLCV DataFrame (최소 WARMUP_BARS 이상)

        Returns:
            Signal or None
        """
        if len(df) < WARMUP_BARS:
            return None

        df = compute_all(df.copy())
        row = df.iloc[-1]

        # NaN 체크
        for col in ("ema_fast", "ema_slow", "ema_trend", "rsi", "atr", "adx"):
            if pd.isna(row[col]):
                return None

        # --- Cooldown ---
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        # --- Volatility Filter ---
        atr_val = float(row["atr"])
        atr_sma = float(row["atr_sma"]) if not pd.isna(row["atr_sma"]) else 0
        if atr_sma > 0 and atr_val / atr_sma > VOL_FILTER:
            return None

        # --- ADX Filter ---
        adx_val = float(row["adx"])
        if adx_val < ADX_MIN:
            return None

        # --- Entry conditions ---
        cross_up = bool(row["ema_bull_cross"])
        cross_down = bool(row["ema_bear_cross"])
        close = float(row["close"])
        ema_trend = float(row["ema_trend"])
        rsi_val = float(row["rsi"])

        long_sig = cross_up and close > ema_trend and rsi_val > RSI_BULL
        short_sig = cross_down and close < ema_trend and rsi_val < RSI_BEAR

        # Anti-duplicate
        if long_sig and self._last_direction == 1:
            long_sig = False
        if short_sig and self._last_direction == -1:
            short_sig = False
        if long_sig and short_sig:
            short_sig = False

        if not long_sig and not short_sig:
            return None

        # --- SL/TP 계산 ---
        entry = close
        is_long = long_sig

        sl = entry - atr_val * SL_ATR_MULT if is_long else entry + atr_val * SL_ATR_MULT
        risk = abs(entry - sl)

        # 최소 거리 보장
        if risk < atr_val * 0.3:
            risk = atr_val * 0.5
            sl = entry - risk if is_long else entry + risk

        if is_long:
            self._last_direction = 1
            return Signal(
                direction=Direction.LONG,
                entry_price=entry,
                sl_price=sl,
                tp1_price=entry + risk * TP1_RR,
                tp2_price=entry + risk * TP2_RR,
                tp3_price=entry + risk * TP3_RR,
                risk=risk,
                rsi=rsi_val, adx=adx_val, atr=atr_val,
            )
        else:
            self._last_direction = -1
            return Signal(
                direction=Direction.SHORT,
                entry_price=entry,
                sl_price=sl,
                tp1_price=entry - risk * TP1_RR,
                tp2_price=entry - risk * TP2_RR,
                tp3_price=entry - risk * TP3_RR,
                risk=risk,
                rsi=rsi_val, adx=adx_val, atr=atr_val,
            )

    # ------------------------------------------------------------------
    # Position Management
    # ------------------------------------------------------------------

    def open_trade(self, signal: Signal) -> ActiveTrade:
        self.active_trade = ActiveTrade(
            direction=signal.direction,
            entry_price=signal.entry_price,
            sl_price=signal.sl_price,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
            tp3_price=signal.tp3_price,
            trail_price=signal.sl_price,
            risk=signal.risk,
            entry_time=signal.timestamp,
        )
        return self.active_trade

    def check_exit(self, current_price: float) -> Optional[str]:
        """TP/SL/Trailing 체크. 종료 사유 반환 or None."""
        trade = self.active_trade
        if trade is None:
            return None

        is_long = trade.direction == Direction.LONG

        if is_long:
            if current_price >= trade.tp3_price and not trade.tp3_hit:
                trade.tp3_hit = True
                if USE_TRAILING:
                    trade.trail_price = trade.tp2_price

            if current_price >= trade.tp2_price and not trade.tp2_hit:
                trade.tp2_hit = True
                if USE_TRAILING:
                    trade.trail_price = trade.tp1_price

            if current_price >= trade.tp1_price and not trade.tp1_hit:
                trade.tp1_hit = True
                if USE_TRAILING:
                    trade.trail_price = trade.entry_price

            stop = trade.current_stop()
            if current_price <= stop:
                reason = f"TRAIL_TP{'3' if trade.tp3_hit else '2' if trade.tp2_hit else '1'}_HIT" if trade.tp1_hit else "SL"
                self._on_exit(reason)
                return reason

        else:  # SHORT
            if current_price <= trade.tp3_price and not trade.tp3_hit:
                trade.tp3_hit = True
                if USE_TRAILING:
                    trade.trail_price = trade.tp2_price

            if current_price <= trade.tp2_price and not trade.tp2_hit:
                trade.tp2_hit = True
                if USE_TRAILING:
                    trade.trail_price = trade.tp1_price

            if current_price <= trade.tp1_price and not trade.tp1_hit:
                trade.tp1_hit = True
                if USE_TRAILING:
                    trade.trail_price = trade.entry_price

            stop = trade.current_stop()
            if current_price >= stop:
                reason = f"TRAIL_TP{'3' if trade.tp3_hit else '2' if trade.tp2_hit else '1'}_HIT" if trade.tp1_hit else "SL"
                self._on_exit(reason)
                return reason

        return None

    def check_reverse_signal(self, signal: Signal) -> bool:
        if self.active_trade is None:
            return False
        return signal.direction != self.active_trade.direction

    def close_trade(self) -> Optional[ActiveTrade]:
        trade = self.active_trade
        self.active_trade = None
        return trade

    def _on_exit(self, reason: str):
        """종료 시 cooldown 설정."""
        if reason == "SL":
            self._cooldown = COOLDOWN_BARS
        self.active_trade = None
