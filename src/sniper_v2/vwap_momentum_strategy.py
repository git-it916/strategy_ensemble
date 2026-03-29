"""
SOL VWAP Momentum Sniper — alpha_factory 리서치 검증 전략.

EMA 크로스가 아닌 VWAP z-score 모멘텀 기반 진입.
15분봉을 1시간봉으로 리샘플하여 VWAP 계산 후 시그널 생성.

검증 결과 (Walk-Forward CV, 26 folds, 5년):
  OOS Sharpe: 35.51, IC(4h): 0.132, MaxDD: -7.55%
  파라미터 안정성: 0.048

인터페이스: SniperV2/FundingContrarianSniper와 동일
  compute(df) → Signal | None
  open_trade(signal) → ActiveTrade
  check_exit(price) → str | None
  check_reverse_signal(signal) → bool
  close_trade(reason) → ActiveTrade | None
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.sniper_v2.config import SymbolConfig
from src.sniper_v2.strategy import ActiveTrade, Direction, Signal


@dataclass(frozen=True)
class VWAPMomentumConfig:
    """VWAP 모멘텀 전략 파라미터 (WF-CV 검증값)."""
    vwap_hours: int = 6            # 6시간 rolling VWAP
    entry_z: float = 1.3           # z-score 진입 기준 (fold 평균)
    hold_bars: int = 4             # 15분봉 기준 4봉 = 1시간 홀드
    vol_confirm_ratio: float = 0.8 # 거래량 확인 비율
    # SL/TP (ATR 기반)
    sl_atr_mult: float = 2.0
    tp1_rr: float = 1.5
    tp2_rr: float = 3.0
    tp3_rr: float = 5.0


class VWAPMomentumSniper:
    """
    SOL VWAP 모멘텀 전략.

    1시간봉 VWAP 대비 z-score로 모멘텀 방향 판단:
    - 가격 > VWAP (z > entry_z) → 롱 (모멘텀 지속)
    - 가격 < VWAP (z < -entry_z) → 숏 (하락 모멘텀)
    """

    def __init__(self, cfg: SymbolConfig, vwap_cfg: VWAPMomentumConfig | None = None):
        self.cfg = cfg
        self.vcfg = vwap_cfg or VWAPMomentumConfig()
        self._last_direction: int = 0
        self._cooldown: int = 0
        self._hold_remaining: int = 0
        self._last_signal_dir: int = 0
        self.active_trade: Optional[ActiveTrade] = None

    def compute(self, df: pd.DataFrame) -> Optional[Signal]:
        """5분봉 DataFrame → 1시간 VWAP 기반 시그널 생성 (리서치와 동일)."""
        cfg = self.cfg
        vcfg = self.vcfg

        # 최소 데이터 (6시간 = 72 × 5분봉 + 여유)
        min_bars = vcfg.vwap_hours * 12 + 20
        if len(df) < min_bars:
            return None

        # 쿨다운
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        # 5분봉 → 1시간봉 리샘플 (리서치와 동일)
        df_1h = df.resample("1h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

        if len(df_1h) < vcfg.vwap_hours + 2:
            return None

        # 직전 완성 봉까지만 사용 (마지막 봉은 미완성일 수 있음)
        # df_1h[-1]이 아직 진행 중인 봉일 수 있으므로 [-2]까지를 완성 봉으로 취급
        completed = df_1h.iloc[:-1]
        if len(completed) < vcfg.vwap_hours:
            return None

        # VWAP 계산 (직전 완성된 N시간 봉)
        recent = completed.tail(vcfg.vwap_hours)
        closes = recent["close"].values
        volumes = recent["volume"].values
        total_vol = volumes.sum()

        if total_vol <= 0:
            return None

        vwap = float((closes * volumes).sum() / total_vol)
        last_completed_price = float(completed["close"].iloc[-1])
        std = float(np.std(closes, ddof=1))

        if std < 1e-8 or vwap == 0:
            return None

        z = (last_completed_price - vwap) / std

        # 진입 판단
        if abs(z) < vcfg.entry_z:
            return None

        # 거래량 확인
        if len(volumes) >= 2:
            if volumes[-1] < np.mean(volumes[:-1]) * vcfg.vol_confirm_ratio:
                return None

        is_long = z > 0
        strength = float(np.tanh(abs(z) / 2.0))

        # Anti-duplicate
        direction_int = 1 if is_long else -1
        if direction_int == self._last_direction:
            return None

        # ATR 계산 (15분봉에서)
        atr_val = self._compute_atr(df, period=14)
        if atr_val <= 0:
            return None

        # RSI 계산 (메타데이터용)
        rsi_val = self._compute_rsi(df, period=14)
        adx_val = 0.0  # VWAP 전략에서는 ADX 불필요

        # SL/TP
        entry = current_price
        risk = atr_val * vcfg.sl_atr_mult
        if is_long:
            sl = entry - risk
            tp1 = entry + risk * vcfg.tp1_rr
            tp2 = entry + risk * vcfg.tp2_rr
            tp3 = entry + risk * vcfg.tp3_rr
        else:
            sl = entry + risk
            tp1 = entry - risk * vcfg.tp1_rr
            tp2 = entry - risk * vcfg.tp2_rr
            tp3 = entry - risk * vcfg.tp3_rr

        self._last_direction = direction_int
        self._hold_remaining = vcfg.hold_bars

        reason = (
            f"VWAP 모멘텀 {'롱' if is_long else '숏'}"
            f"\nVWAP({vcfg.vwap_hours}h)={vwap:.2f} | z={z:.2f} (>{vcfg.entry_z})"
            f"\n강도: {strength:.2f} | RSI={rsi_val:.1f}"
            f"\nSL: ATR×{vcfg.sl_atr_mult} ({abs(entry-sl)/entry*100:.2f}%)"
        )

        return Signal(
            direction=Direction.LONG if is_long else Direction.SHORT,
            entry_price=entry,
            sl_price=sl,
            tp1_price=tp1,
            tp2_price=tp2,
            tp3_price=tp3,
            risk=risk,
            rsi=rsi_val,
            adx=adx_val,
            atr=atr_val,
            reason_text=reason,
        )

    # ------------------------------------------------------------------
    # Position Management (SniperV2와 동일 인터페이스)
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

        # 수익보호
        if trade.risk > 0:
            cur_r = ((current_price - trade.entry_price) / trade.risk if is_long
                     else (trade.entry_price - current_price) / trade.risk)
            trade.peak_r = max(trade.peak_r, cur_r)
            if cfg.profit_protect and trade.peak_r >= cfg.pp_trigger:
                if cur_r <= cfg.pp_exit:
                    return f"PP_{trade.peak_r:.1f}R"

        # TP/SL/Trailing
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
            stop = trade.trail_price if cfg.use_trailing else trade.sl_price
            if current_price <= stop:
                tp_label = "3" if trade.tp3_hit else "2" if trade.tp2_hit else "1"
                return f"TRAIL_TP{tp_label}_HIT" if trade.tp1_hit else "SL"
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
            stop = trade.trail_price if cfg.use_trailing else trade.sl_price
            if current_price >= stop:
                tp_label = "3" if trade.tp3_hit else "2" if trade.tp2_hit else "1"
                return f"TRAIL_TP{tp_label}_HIT" if trade.tp1_hit else "SL"

        return None

    def check_reverse_signal(self, signal: Signal) -> bool:
        if self.active_trade is None:
            return False
        return signal.direction != self.active_trade.direction

    def close_trade(self, reason: str = "") -> Optional[ActiveTrade]:
        trade = self.active_trade
        self.active_trade = None
        if reason == "SL":
            self._cooldown = self.cfg.cooldown_bars
        return trade

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        # EWM ATR
        alpha = 1.0 / period
        atr = tr[0]
        for i in range(1, len(tr)):
            atr = atr * (1 - alpha) + tr[i] * alpha
        return float(atr)

    @staticmethod
    def _compute_rsi(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 50.0
        close = df["close"].values
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        alpha = 1.0 / period
        avg_gain = gain[:period].mean()
        avg_loss = loss[:period].mean()
        for i in range(period, len(gain)):
            avg_gain = avg_gain * (1 - alpha) + gain[i] * alpha
            avg_loss = avg_loss * (1 - alpha) + loss[i] * alpha
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))
