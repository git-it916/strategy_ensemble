"""
Sniper V2 — Funding Contrarian Strategy (XRP)

alpha_factory WF-CV 검증 기반:
  OOS Sharpe: 3.41, OOS MaxDD: -9.79%, Param Stability CV: 0.148

EMA 크로스오버가 아닌 펀딩비 z-score 기반 역추세 전략.
SniperV2와 동일한 인터페이스 (compute → Signal, check_exit, open_trade, close_trade).
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
class FundingConfig:
    """펀딩 역추세 전용 파라미터 (60개월 그리드서치 최적화)."""
    funding_lookback: int = 75          # 8h 간격 기준 (~25일)
    z_threshold: float = 1.125
    price_confirm_z: float = 0.8        # 0.575→0.8: 가격 컨펌 강화 (PF 0.81→1.41)
    # 가격 z-score 계산용 (15m 봉 기준)
    price_z_periods: tuple = (16, 48, 96)  # 4h, 12h, 24h in 15m bars
    # SL/TP (ATR 기반) — TP 확대 (2.0/3.5/6.0)
    sl_atr_mult: float = 2.0
    tp1_rr: float = 2.0
    tp2_rr: float = 3.5
    tp3_rr: float = 6.0


class FundingContrarianSniper:
    """
    펀딩비 역추세 전략 엔진.

    SniperV2와 동일 인터페이스:
      compute(df, funding_df) → Signal | None
      open_trade(signal) → ActiveTrade
      check_exit(price) → str | None
      check_reverse_signal(signal) → bool
      close_trade(reason) → ActiveTrade | None
    """

    def __init__(self, cfg: SymbolConfig, fcfg: FundingConfig | None = None):
        self.cfg = cfg
        self.fcfg = fcfg or FundingConfig()
        self._cooldown: int = 0
        self._last_direction: int = 0
        self.active_trade: Optional[ActiveTrade] = None

    def compute(
        self,
        df: pd.DataFrame,
        funding_df: pd.DataFrame | None = None,
    ) -> Optional[Signal]:
        """
        펀딩비 + 가격 z-score 기반 시그널 생성.

        Args:
            df: 15m OHLCV DataFrame (index=timestamp)
            funding_df: 펀딩비 DataFrame (columns: timestamp, fundingRate)
        """
        cfg = self.cfg
        fcfg = self.fcfg

        if funding_df is None or len(funding_df) < 30:
            return None
        if len(df) < cfg.warmup_bars:
            return None

        # Cooldown
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        # ── 1. 펀딩비 z-score ──
        if "fundingRate" in funding_df.columns:
            rates = funding_df["fundingRate"].values
        else:
            return None

        n = min(len(rates), fcfg.funding_lookback)
        window = rates[-n:]
        fr_mean = float(np.mean(window))
        fr_std = float(np.std(window))
        if fr_std < 1e-10:
            return None
        fr_z = (float(rates[-1]) - fr_mean) / fr_std

        # ── 2. 가격 z-score (다중 타임프레임) ──
        closes = df["close"].values
        z_scores = []
        for period in fcfg.price_z_periods:
            if len(closes) < period + 20:
                continue
            ret = closes[-1] / closes[-period] - 1
            lookback = min(len(closes) - 1, period * 4)
            rets = np.diff(closes[-lookback:]) / closes[-lookback:-1]
            vol = float(np.std(rets)) * np.sqrt(period)
            if vol > 1e-10:
                z_scores.append(ret / vol)

        if not z_scores:
            return None
        price_z = float(np.mean(z_scores))

        # ── 3. 시그널 판단 ──
        zt = fcfg.z_threshold
        pcz = fcfg.price_confirm_z
        is_long = False
        is_short = False
        strength = 0.0

        # 극단 양의 펀딩 + 가격 과열 → 숏
        if fr_z > zt and price_z > pcz:
            is_short = True
            strength = min(fr_z / (zt * 2), 1.0)
        # 극단 음의 펀딩 + 가격 과냉 → 롱
        elif fr_z < -zt and price_z < -pcz:
            is_long = True
            strength = min(-fr_z / (zt * 2), 1.0)
        # 펀딩만 매우 극단적
        elif fr_z > zt * 1.5:
            is_short = True
            strength = min(fr_z / (zt * 3), 0.5)
        elif fr_z < -zt * 1.5:
            is_long = True
            strength = min(-fr_z / (zt * 3), 0.5)

        if not is_long and not is_short:
            return None

        # 시그널 강도 필터 (너무 약하면 무시)
        if strength < 0.3:
            return None

        # Anti-duplicate
        if is_long and self._last_direction == 1:
            return None
        if is_short and self._last_direction == -1:
            return None

        # ── 4. SL/TP 계산 (ATR 기반) ──
        high, low, close_s = df["high"], df["low"], df["close"]
        prev_close = close_s.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_val = float(tr.ewm(alpha=1/14, min_periods=14).mean().iloc[-1])

        if np.isnan(atr_val) or atr_val <= 0:
            return None

        entry = float(closes[-1])
        risk = atr_val * fcfg.sl_atr_mult

        # ADX (선택적 — 강한 추세 필터)
        adx_val = 0.0  # 펀딩 전략은 ADX 불요, placeholder

        # RSI 계산 (메타데이터용)
        delta = close_s.diff()
        gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(span=14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_val = float((100 - 100 / (1 + rs)).iloc[-1])

        # 진입 사유 생성
        last_rate = float(rates[-1])
        if fr_z > zt and price_z > pcz:
            trigger = "펀딩 과열 + 가격 과열 → 역추세"
        elif fr_z < -zt and price_z < -pcz:
            trigger = "펀딩 과냉 + 가격 과냉 → 역추세"
        elif abs(fr_z) > zt * 1.5:
            trigger = "펀딩 극단값 단독 트리거"
        else:
            trigger = "복합 시그널"

        reason = (
            f"{trigger}"
            f"\n펀딩: {last_rate:+.6f} (z={fr_z:+.2f}, 임계={zt:.3f})"
            f"\n가격Z: {price_z:+.2f} (임계={pcz:.3f})"
            f"\n강도: {strength:.0%} | RSI={rsi_val:.1f}"
            f"\nSL: ATR×{fcfg.sl_atr_mult} ({risk/entry*100:.2f}%)"
        )

        if is_long:
            self._last_direction = 1
            sl = entry - risk
            return Signal(
                direction=Direction.LONG,
                entry_price=entry, sl_price=sl,
                tp1_price=entry + risk * fcfg.tp1_rr,
                tp2_price=entry + risk * fcfg.tp2_rr,
                tp3_price=entry + risk * fcfg.tp3_rr,
                risk=risk, rsi=rsi_val, adx=adx_val, atr=atr_val,
                reason_text=reason,
            )
        else:
            self._last_direction = -1
            sl = entry + risk
            return Signal(
                direction=Direction.SHORT,
                entry_price=entry, sl_price=sl,
                tp1_price=entry - risk * fcfg.tp1_rr,
                tp2_price=entry - risk * fcfg.tp2_rr,
                tp3_price=entry - risk * fcfg.tp3_rr,
                risk=risk, rsi=rsi_val, adx=adx_val, atr=atr_val,
                reason_text=reason,
            )

    # ------------------------------------------------------------------
    # Position Management (SniperV2와 동일)
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
            cur_r = ((current_price - trade.entry_price) / trade.risk
                     if is_long else
                     (trade.entry_price - current_price) / trade.risk)
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
