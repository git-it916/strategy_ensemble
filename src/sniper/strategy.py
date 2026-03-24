"""
Precision Sniper — Strategy Core

Pine Script 원본의 Confluence Scoring + Entry Logic을 Python으로 구현.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.sniper.config import (
    ADX_STRONG,
    ATR_LEN,
    MIN_CONFLUENCE_SCORE,
    RSI_OB,
    RSI_OS,
    SL_ATR_MULT,
    SWING_LOOKBACK,
    TP1_RR,
    TP2_RR,
    TP3_RR,
    USE_STRUCTURE_SL,
    USE_TRAILING,
    WARMUP_BARS,
)
from src.sniper.indicators import compute_all, swing_high, swing_low

logger = logging.getLogger(__name__)


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, Enum):
    ACTIVE = "ACTIVE"
    TP1_HIT = "TP1_HIT"
    TP2_HIT = "TP2_HIT"
    TP3_HIT = "TP3_HIT"
    STOPPED = "STOPPED"
    CLOSED = "CLOSED"


@dataclass
class Signal:
    """진입 시그널."""
    direction: Direction
    score: float
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    risk: float               # |entry - sl|
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict = field(default_factory=dict)


@dataclass
class ActiveTrade:
    """활성 트레이드 상태."""
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
    peak_price: float = 0.0
    trough_price: float = float("inf")
    entry_score: float = 0.0

    @property
    def status(self) -> TradeStatus:
        if self.tp3_hit:
            return TradeStatus.TP3_HIT
        if self.tp2_hit:
            return TradeStatus.TP2_HIT
        if self.tp1_hit:
            return TradeStatus.TP1_HIT
        return TradeStatus.ACTIVE

    def current_stop(self) -> float:
        """현재 유효한 스탑 가격."""
        if USE_TRAILING:
            return self.trail_price
        return self.sl_price


class PrecisionSniper:
    """
    Precision Sniper 전략 엔진.

    매 캔들 완성 시 compute()를 호출하면:
    1. 지표 계산
    2. Confluence scoring
    3. 진입 시그널 생성
    4. 포지션 관리 (TP/SL/trailing)
    """

    def __init__(self):
        self._last_direction: int = 0    # anti-duplicate: 1=long, -1=short, 0=none
        self.active_trade: Optional[ActiveTrade] = None

    # ------------------------------------------------------------------
    # Confluence Scoring
    # ------------------------------------------------------------------

    def _bull_score(self, row: pd.Series, htf_bias: int) -> float:
        """Bull confluence score (max 10.0)."""
        s = 0.0
        s += 1.0 if row["ema_fast"] > row["ema_slow"] else 0.0
        s += 1.0 if row["close"] > row["ema_trend"] else 0.0
        s += 1.0 if 50 < row["rsi"] < RSI_OB else 0.0
        s += 1.0 if row["macd_hist"] > 0 else 0.0
        s += 1.0 if row["macd_line"] > row["macd_signal"] else 0.0
        s += 1.0 if row["close"] > row["vwap"] else 0.0
        s += 1.0 if row["vol_above_avg"] else 0.0
        s += 1.0 if row["adx"] > ADX_STRONG and row["di_plus"] > row["di_minus"] else 0.0
        s += 1.5 if htf_bias == 1 else 0.0
        s += 0.5 if row["close"] > row["ema_fast"] else 0.0
        return s

    def _bear_score(self, row: pd.Series, htf_bias: int) -> float:
        """Bear confluence score (max 10.0)."""
        s = 0.0
        s += 1.0 if row["ema_fast"] < row["ema_slow"] else 0.0
        s += 1.0 if row["close"] < row["ema_trend"] else 0.0
        s += 1.0 if RSI_OS < row["rsi"] < 50 else 0.0
        s += 1.0 if row["macd_hist"] < 0 else 0.0
        s += 1.0 if row["macd_line"] < row["macd_signal"] else 0.0
        s += 1.0 if row["close"] < row["vwap"] else 0.0
        s += 1.0 if row["vol_above_avg"] else 0.0
        s += 1.0 if row["adx"] > ADX_STRONG and row["di_minus"] > row["di_plus"] else 0.0
        s += 1.5 if htf_bias == -1 else 0.0
        s += 0.5 if row["close"] < row["ema_fast"] else 0.0
        return s

    # ------------------------------------------------------------------
    # SL Calculation
    # ------------------------------------------------------------------

    def _calc_sl(self, is_long: bool, entry: float, atr_val: float, df: pd.DataFrame) -> float:
        """ATR 기반 + 구조적 SL 계산."""
        atr_sl = atr_val * SL_ATR_MULT
        atr_stop = entry - atr_sl if is_long else entry + atr_sl

        if not USE_STRUCTURE_SL:
            return atr_stop

        if is_long:
            sw_low = swing_low(df["low"], SWING_LOOKBACK)
            struct_stop = sw_low - atr_val * 0.2
            final_stop = max(atr_stop, struct_stop)
        else:
            sw_high = swing_high(df["high"], SWING_LOOKBACK)
            struct_stop = sw_high + atr_val * 0.2
            final_stop = min(atr_stop, struct_stop)

        # 최소 거리 보장
        min_dist = atr_val * 0.5
        if abs(entry - final_stop) < min_dist:
            final_stop = entry - min_dist if is_long else entry + min_dist

        return final_stop

    # ------------------------------------------------------------------
    # HTF Bias
    # ------------------------------------------------------------------

    def compute_htf_bias(self, htf_df: pd.DataFrame) -> int:
        """상위 타임프레임 트렌드 바이어스."""
        if htf_df is None or len(htf_df) < 30:
            return 0

        from src.sniper.config import EMA_FAST_LEN, EMA_SLOW_LEN
        from src.sniper.indicators import ema

        c = htf_df["close"]
        htf_fast = ema(c, EMA_FAST_LEN)
        htf_slow = ema(c, EMA_SLOW_LEN)

        # [1] = 직전 완성된 봉 (non-repainting)
        fast_val = float(htf_fast.iloc[-2]) if len(htf_fast) >= 2 else float(htf_fast.iloc[-1])
        slow_val = float(htf_slow.iloc[-2]) if len(htf_slow) >= 2 else float(htf_slow.iloc[-1])

        if fast_val > slow_val:
            return 1
        elif fast_val < slow_val:
            return -1
        return 0

    # ------------------------------------------------------------------
    # Signal Generation
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame, htf_df: pd.DataFrame = None) -> Optional[Signal]:
        """
        메인 시그널 생성.

        Args:
            df: OHLCV DataFrame (5m 기본)
            htf_df: 상위 타임프레임 OHLCV (1h)

        Returns:
            Signal or None
        """
        if len(df) < WARMUP_BARS:
            return None

        # 지표 계산
        df = compute_all(df.copy())
        row = df.iloc[-1]

        # NaN 체크
        if pd.isna(row["ema_fast"]) or pd.isna(row["atr"]) or pd.isna(row["rsi"]):
            return None

        htf_bias = self.compute_htf_bias(htf_df)

        # Confluence scoring
        bull_sc = self._bull_score(row, htf_bias)
        bear_sc = self._bear_score(row, htf_bias)

        # EMA cross
        ema_bull_cross = bool(row["ema_bull_cross"])
        ema_bear_cross = bool(row["ema_bear_cross"])

        # Momentum confirmation
        bull_momentum = row["close"] > row["ema_fast"] and row["close"] > row["ema_slow"]
        bear_momentum = row["close"] < row["ema_fast"] and row["close"] < row["ema_slow"]

        # RSI filter
        rsi_not_ob = row["rsi"] < RSI_OB
        rsi_not_os = row["rsi"] > RSI_OS

        # Composite entry
        raw_buy = ema_bull_cross and bull_momentum and rsi_not_ob and bull_sc >= MIN_CONFLUENCE_SCORE
        raw_sell = ema_bear_cross and bear_momentum and rsi_not_os and bear_sc >= MIN_CONFLUENCE_SCORE

        # Anti-duplicate
        buy_cond = raw_buy and self._last_direction != 1
        sell_cond = raw_sell and self._last_direction != -1

        # Mutual exclusion
        if buy_cond and sell_cond:
            sell_cond = False

        if not buy_cond and not sell_cond:
            return None

        # --- 시그널 생성 ---
        entry = float(row["close"])
        atr_val = float(row["atr"])

        if buy_cond:
            self._last_direction = 1
            sl = self._calc_sl(True, entry, atr_val, df)
            risk = abs(entry - sl)
            return Signal(
                direction=Direction.LONG,
                score=bull_sc,
                entry_price=entry,
                sl_price=sl,
                tp1_price=entry + risk * TP1_RR,
                tp2_price=entry + risk * TP2_RR,
                tp3_price=entry + risk * TP3_RR,
                risk=risk,
                details={
                    "bull_score": bull_sc,
                    "bear_score": bear_sc,
                    "htf_bias": htf_bias,
                    "rsi": float(row["rsi"]),
                    "adx": float(row["adx"]),
                    "atr": atr_val,
                    "vol_regime": "High" if row["vol_ratio"] > 1.3 else "Low" if row["vol_ratio"] < 0.7 else "Normal",
                },
            )

        if sell_cond:
            self._last_direction = -1
            sl = self._calc_sl(False, entry, atr_val, df)
            risk = abs(entry - sl)
            return Signal(
                direction=Direction.SHORT,
                score=bear_sc,
                entry_price=entry,
                sl_price=sl,
                tp1_price=entry - risk * TP1_RR,
                tp2_price=entry - risk * TP2_RR,
                tp3_price=entry - risk * TP3_RR,
                risk=risk,
                details={
                    "bull_score": bull_sc,
                    "bear_score": bear_sc,
                    "htf_bias": htf_bias,
                    "rsi": float(row["rsi"]),
                    "adx": float(row["adx"]),
                    "atr": atr_val,
                    "vol_regime": "High" if row["vol_ratio"] > 1.3 else "Low" if row["vol_ratio"] < 0.7 else "Normal",
                },
            )

        return None

    # ------------------------------------------------------------------
    # Position Management (TP/SL/Trailing)
    # ------------------------------------------------------------------

    def open_trade(self, signal: Signal) -> ActiveTrade:
        """시그널 기반 트레이드 오픈."""
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
            peak_price=signal.entry_price,
            trough_price=signal.entry_price,
            entry_score=signal.score,
        )
        return self.active_trade

    def check_exit(self, current_price: float) -> Optional[str]:
        """
        현재가 기준 TP/SL/트레일링 체크.

        Returns: 종료 사유 문자열 or None
        """
        trade = self.active_trade
        if trade is None:
            return None

        is_long = trade.direction == Direction.LONG

        # Peak/trough 업데이트
        if is_long:
            trade.peak_price = max(trade.peak_price, current_price)
            trade.trough_price = min(trade.trough_price, current_price)
        else:
            trade.peak_price = min(trade.peak_price, current_price)
            trade.trough_price = max(trade.trough_price, current_price)

        # --- TP 체크 ---
        if is_long:
            if current_price >= trade.tp3_price and not trade.tp3_hit:
                trade.tp3_hit = True
                if USE_TRAILING:
                    trade.trail_price = trade.tp2_price
                logger.info(f"TP3 HIT @ {current_price:.2f}")

            if current_price >= trade.tp2_price and not trade.tp2_hit:
                trade.tp2_hit = True
                if USE_TRAILING:
                    trade.trail_price = trade.tp1_price
                logger.info(f"TP2 HIT @ {current_price:.2f}")

            if current_price >= trade.tp1_price and not trade.tp1_hit:
                trade.tp1_hit = True
                if USE_TRAILING:
                    trade.trail_price = trade.entry_price + trade.risk * 0.3  # 30% 수익 확보
                logger.info(f"TP1 HIT @ {current_price:.2f} — trailing to entry+30%risk")

            # SL / Trailing stop 체크
            stop = trade.current_stop()
            if current_price <= stop:
                reason = f"TRAIL_{trade.status.value}" if trade.tp1_hit else "SL"
                self._close_trade()
                return reason

        else:  # SHORT
            if current_price <= trade.tp3_price and not trade.tp3_hit:
                trade.tp3_hit = True
                if USE_TRAILING:
                    trade.trail_price = trade.tp2_price
                logger.info(f"TP3 HIT @ {current_price:.2f}")

            if current_price <= trade.tp2_price and not trade.tp2_hit:
                trade.tp2_hit = True
                if USE_TRAILING:
                    trade.trail_price = trade.tp1_price
                logger.info(f"TP2 HIT @ {current_price:.2f}")

            if current_price <= trade.tp1_price and not trade.tp1_hit:
                trade.tp1_hit = True
                if USE_TRAILING:
                    trade.trail_price = trade.entry_price - trade.risk * 0.3  # 30% 수익 확보
                logger.info(f"TP1 HIT @ {current_price:.2f} — trailing to entry-30%risk")

            stop = trade.current_stop()
            if current_price >= stop:
                reason = f"TRAIL_{trade.status.value}" if trade.tp1_hit else "SL"
                self._close_trade()
                return reason

        return None

    def check_reverse_signal(self, signal: Signal) -> bool:
        """반대 방향 시그널이면 현재 포지션 청산 필요."""
        if self.active_trade is None:
            return False
        return signal.direction != self.active_trade.direction

    def _close_trade(self):
        """트레이드 종료 정리."""
        self.active_trade = None

    def close_trade(self) -> Optional[ActiveTrade]:
        """외부에서 호출 가능한 청산."""
        trade = self.active_trade
        self.active_trade = None
        return trade
