"""
Sniper V2 — Strategy Engine (Multi-Symbol)

심볼별 SymbolConfig로 파라미터화.
BTC: EMA(18/40) cross + EMA(200) + RSI 55/40 + ATR SL
SOL: EMA(8/40) cross + EMA(150) + RSI 65/35 + SWING SL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.sniper_v2.config import SymbolConfig
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
    # 진입 사유 메타데이터 (텔레그램/로그용)
    reason_text: str = ""


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
    peak_r: float = 0.0        # 최고 미실현 수익 (R 단위)

    def current_stop(self, use_trailing: bool) -> float:
        if use_trailing:
            return self.trail_price
        return self.sl_price


class SniperV2:
    """
    Sniper V2 전략 엔진.

    SymbolConfig로 심볼별 파라미터 주입.
    """

    def __init__(self, cfg: SymbolConfig):
        self.cfg = cfg
        self._last_direction: int = 0
        self._cooldown: int = 0
        self.active_trade: Optional[ActiveTrade] = None

    def compute(self, df: pd.DataFrame) -> Optional[Signal]:
        cfg = self.cfg

        if len(df) < cfg.warmup_bars:
            return None

        df = compute_all(df.copy(), cfg)
        row = df.iloc[-1]

        for col in ("ema_fast", "ema_slow", "ema_trend", "rsi", "atr", "adx"):
            if pd.isna(row[col]):
                return None

        # Cooldown
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        # Volatility filter
        atr_val = float(row["atr"])
        atr_sma = float(row["atr_sma"]) if not pd.isna(row["atr_sma"]) else 0
        if atr_sma > 0 and atr_val / atr_sma > cfg.vol_filter:
            return None

        # ADX filter
        adx_val = float(row["adx"])
        if adx_val < cfg.adx_min:
            return None

        # Entry conditions
        cross_up = bool(row["ema_bull_cross"])
        cross_down = bool(row["ema_bear_cross"])
        close = float(row["close"])
        ema_trend = float(row["ema_trend"])
        rsi_val = float(row["rsi"])

        long_sig = cross_up and close > ema_trend and rsi_val > cfg.rsi_bull
        short_sig = cross_down and close < ema_trend and rsi_val < cfg.rsi_bear

        # Anti-duplicate
        if long_sig and self._last_direction == 1:
            long_sig = False
        if short_sig and self._last_direction == -1:
            short_sig = False
        if long_sig and short_sig:
            short_sig = False

        if not long_sig and not short_sig:
            return None

        # SL/TP
        entry = close
        is_long = long_sig

        if cfg.sl_method == "SWING" and "swing_low" in df.columns:
            if is_long:
                sw = float(row["swing_low"]) if not pd.isna(row.get("swing_low")) else entry - atr_val * 1.5
                sl = sw
            else:
                sw = float(row["swing_high"]) if not pd.isna(row.get("swing_high")) else entry + atr_val * 1.5
                sl = sw
        else:
            sl = entry - atr_val * cfg.sl_atr_mult if is_long else entry + atr_val * cfg.sl_atr_mult

        risk = abs(entry - sl)
        if risk < atr_val * 0.3:
            risk = atr_val * 0.5
            sl = entry - risk if is_long else entry + risk

        # 진입 사유 생성
        ema_f = float(row["ema_fast"])
        ema_s = float(row["ema_slow"])
        sl_desc = f"SWING_{cfg.swing_lookback}" if cfg.sl_method == "SWING" else f"ATR×{cfg.sl_atr_mult}"
        reason = (
            f"EMA {cfg.ema_fast}/{cfg.ema_slow} {'골든'}크로스" if is_long else f"EMA {cfg.ema_fast}/{cfg.ema_slow} 데드크로스"
        ) + (
            f"\nEMA fast={ema_f:.1f} > slow={ema_s:.1f}" if is_long else f"\nEMA fast={ema_f:.1f} < slow={ema_s:.1f}"
        ) + (
            f"\n추세: close={close:.2f} {'>' if is_long else '<'} EMA{cfg.ema_trend}={ema_trend:.2f}"
        ) + (
            f"\nRSI={rsi_val:.1f} {'>' if is_long else '<'} {cfg.rsi_bull if is_long else cfg.rsi_bear}"
            f" | ADX={adx_val:.1f}"
        ) + (
            f"\nSL: {sl_desc} ({abs(entry-sl)/entry*100:.2f}%)"
        )

        if is_long:
            self._last_direction = 1
            return Signal(
                direction=Direction.LONG, entry_price=entry, sl_price=sl,
                tp1_price=entry + risk * cfg.tp1_rr,
                tp2_price=entry + risk * cfg.tp2_rr,
                tp3_price=entry + risk * cfg.tp3_rr,
                risk=risk, rsi=rsi_val, adx=adx_val, atr=atr_val,
                reason_text=reason,
            )
        else:
            self._last_direction = -1
            return Signal(
                direction=Direction.SHORT, entry_price=entry, sl_price=sl,
                tp1_price=entry - risk * cfg.tp1_rr,
                tp2_price=entry - risk * cfg.tp2_rr,
                tp3_price=entry - risk * cfg.tp3_rr,
                risk=risk, rsi=rsi_val, adx=adx_val, atr=atr_val,
                reason_text=reason,
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
        trade = self.active_trade
        if trade is None:
            return None

        cfg = self.cfg
        is_long = trade.direction == Direction.LONG

        # --- 수익보호: 미실현 수익 추적 (R 단위) ---
        if trade.risk > 0:
            if is_long:
                cur_r = (current_price - trade.entry_price) / trade.risk
            else:
                cur_r = (trade.entry_price - current_price) / trade.risk
            trade.peak_r = max(trade.peak_r, cur_r)

            if cfg.profit_protect and trade.peak_r >= cfg.pp_trigger:
                if cur_r <= cfg.pp_exit:
                    reason = f"PP_{trade.peak_r:.1f}R"
                    return reason

        # --- TP/SL/Trailing ---
        if is_long:
            if current_price >= trade.tp3_price and not trade.tp3_hit:
                trade.tp3_hit = True
                if cfg.use_trailing:
                    trade.trail_price = trade.tp2_price
            if current_price >= trade.tp2_price and not trade.tp2_hit:
                trade.tp2_hit = True
                if cfg.use_trailing:
                    trade.trail_price = trade.tp1_price
            if current_price >= trade.tp1_price and not trade.tp1_hit:
                trade.tp1_hit = True
                if cfg.use_trailing:
                    trade.trail_price = trade.entry_price

            stop = trade.current_stop(cfg.use_trailing)
            if current_price <= stop:
                reason = f"TRAIL_TP{'3' if trade.tp3_hit else '2' if trade.tp2_hit else '1'}_HIT" if trade.tp1_hit else "SL"

                return reason
        else:
            if current_price <= trade.tp3_price and not trade.tp3_hit:
                trade.tp3_hit = True
                if cfg.use_trailing:
                    trade.trail_price = trade.tp2_price
            if current_price <= trade.tp2_price and not trade.tp2_hit:
                trade.tp2_hit = True
                if cfg.use_trailing:
                    trade.trail_price = trade.tp1_price
            if current_price <= trade.tp1_price and not trade.tp1_hit:
                trade.tp1_hit = True
                if cfg.use_trailing:
                    trade.trail_price = trade.entry_price

            stop = trade.current_stop(cfg.use_trailing)
            if current_price >= stop:
                reason = f"TRAIL_TP{'3' if trade.tp3_hit else '2' if trade.tp2_hit else '1'}_HIT" if trade.tp1_hit else "SL"

                return reason

        return None

    def check_reverse_signal(self, signal: Signal) -> bool:
        if self.active_trade is None:
            return False
        return signal.direction != self.active_trade.direction

    def close_trade(self, reason: str = "") -> Optional[ActiveTrade]:
        """포지션 정리. 유일하게 active_trade를 None으로 만드는 메서드."""
        trade = self.active_trade
        self.active_trade = None
        if reason == "SL":
            self._cooldown = self.cfg.cooldown_bars
        return trade

