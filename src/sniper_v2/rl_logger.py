"""
Sniper V2 — RL-Ready State Logger.

매 15분봉마다 시장 상태(state)를 기록하고,
진입/청산 시 전체 (state, action, reward) 튜플을 남김.

로그 구조:
  logs/rl/states/{symbol}/YYYY-MM-DD.jsonl  — 매 캔들 상태
  logs/rl/trades/{symbol}/YYYY-MM-DD.jsonl  — 거래 이벤트
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.sniper_v2.config import SymbolConfig
from src.sniper_v2.rl_config import RLConfig, RL_CONFIG
from src.sniper_v2.strategy import ActiveTrade, Direction, Signal


class RLStateLogger:
    """RL 학습용 상태 로거."""

    def __init__(
        self,
        symbol: str,
        strategy: str,
        cfg: SymbolConfig,
        rl_cfg: RLConfig = RL_CONFIG,
    ):
        self.symbol = symbol
        self.short_sym = symbol.split("/")[0].lower()
        self.strategy = strategy
        self.cfg = cfg
        self.rl_cfg = rl_cfg
        self._prev_unrealized_r: float = 0.0  # 이전 캔들 미실현 R (step reward용)
        self._max_adverse_r: float = 0.0       # 최대 역행 R (MAE)
        self._hold_bars: int = 0

    # ══════════════════════════════════════════════════════════
    # State Snapshot
    # ══════════════════════════════════════════════════════════

    def snapshot_state(
        self,
        df: pd.DataFrame,
        position: Optional[ActiveTrade],
        account: dict,
        funding_data: Optional[dict] = None,
    ) -> dict:
        """전체 시장 상태 스냅샷 생성."""
        row = df.iloc[-1]
        close = float(row["close"])

        # OHLCV
        ohlcv = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": close,
            "volume": float(row["volume"]),
        }

        # 지표 (있으면)
        indicators = {}
        for col in ("ema_fast", "ema_slow", "ema_trend", "rsi", "adx",
                     "atr", "atr_sma", "swing_low", "swing_high",
                     "ema_bull_cross", "ema_bear_cross"):
            if col in df.columns and not pd.isna(row.get(col, float("nan"))):
                val = row[col]
                indicators[col] = bool(val) if col.startswith("ema_b") else float(val)

        # 최근 N봉 수익률 + 거래량 비율
        n = self.rl_cfg.recent_candles
        closes = df["close"].values
        volumes = df["volume"].values
        recent_returns = []
        recent_vol_ratio = []
        if len(closes) > n:
            for i in range(1, n + 1):
                ret = (closes[-i] / closes[-i - 1] - 1) if closes[-i - 1] != 0 else 0
                recent_returns.append(round(ret, 6))
            vol_sma = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
            for i in range(n):
                vr = float(volumes[-i - 1]) / vol_sma if vol_sma > 0 else 1.0
                recent_vol_ratio.append(round(vr, 4))
            recent_returns.reverse()
            recent_vol_ratio.reverse()

        # 포지션 상태
        pos_state = self._position_state(position, close)

        state = {
            "ohlcv_last": ohlcv,
            "indicators": indicators,
            "recent_returns": recent_returns,
            "recent_vol_ratio": recent_vol_ratio,
            "position": pos_state,
            "account": account,
        }

        if funding_data:
            state["funding"] = funding_data

        # Feature vector
        state["features"] = self.compute_features(state)

        return state

    def _position_state(self, position: Optional[ActiveTrade], close: float) -> dict:
        """포지션 상태를 dict로."""
        if position is None:
            return {
                "is_active": False, "direction": None,
                "entry_price": None, "sl_price": None,
                "tp1": None, "tp2": None, "tp3": None,
                "trail_price": None, "peak_r": 0.0,
                "tp1_hit": False, "tp2_hit": False, "tp3_hit": False,
                "hold_bars": 0, "unrealized_r": 0.0,
            }

        is_long = position.direction == Direction.LONG
        if position.risk > 0:
            unrealized_r = ((close - position.entry_price) / position.risk if is_long
                            else (position.entry_price - close) / position.risk)
        else:
            unrealized_r = 0.0

        return {
            "is_active": True,
            "direction": position.direction.value,
            "entry_price": position.entry_price,
            "sl_price": position.sl_price,
            "tp1": position.tp1_price, "tp2": position.tp2_price, "tp3": position.tp3_price,
            "trail_price": position.trail_price,
            "peak_r": round(position.peak_r, 4),
            "tp1_hit": position.tp1_hit, "tp2_hit": position.tp2_hit, "tp3_hit": position.tp3_hit,
            "hold_bars": self._hold_bars,
            "unrealized_r": round(unrealized_r, 4),
        }

    # ══════════════════════════════════════════════════════════
    # Feature Vector
    # ══════════════════════════════════════════════════════════

    def compute_features(self, state: dict) -> list[float]:
        """고정 길이 정규화 feature vector 생성."""
        ind = state.get("indicators", {})
        pos = state.get("position", {})
        close = state["ohlcv_last"]["close"]

        atr = ind.get("atr", 1.0)
        if atr <= 0 or math.isnan(atr):
            atr = 1.0

        features = []

        # 0: EMA spread / ATR
        ema_f = ind.get("ema_fast", close)
        ema_s = ind.get("ema_slow", close)
        features.append(_safe((ema_f - ema_s) / atr))

        # 1: (close - ema_trend) / atr
        ema_t = ind.get("ema_trend", close)
        features.append(_safe((close - ema_t) / atr))

        # 2: (close - ema_fast) / atr
        features.append(_safe((close - ema_f) / atr))

        # 3: rsi / 100
        features.append(_safe(ind.get("rsi", 50) / 100))

        # 4: adx / 100
        features.append(_safe(ind.get("adx", 0) / 100))

        # 5: atr / close (상대 변동성)
        features.append(_safe(atr / close))

        # 6: atr / atr_sma - 1 (변동성 레짐)
        atr_sma = ind.get("atr_sma", atr)
        features.append(_safe(atr / atr_sma - 1 if atr_sma > 0 else 0))

        # 7: (close - swing_low) / atr
        sw_lo = ind.get("swing_low", close)
        features.append(_safe((close - sw_lo) / atr))

        # 8: (swing_high - close) / atr
        sw_hi = ind.get("swing_high", close)
        features.append(_safe((sw_hi - close) / atr))

        # 9-10: EMA cross flags
        features.append(1.0 if ind.get("ema_bull_cross", False) else 0.0)
        features.append(1.0 if ind.get("ema_bear_cross", False) else 0.0)

        # 11-15: 최근 5봉 수익률
        rets = state.get("recent_returns", [0] * 5)
        features.extend([_safe(r) for r in _pad(rets, 5)])

        # 16-20: 최근 5봉 거래량 비율
        vols = state.get("recent_vol_ratio", [1] * 5)
        features.extend([_safe(v) for v in _pad(vols, 5)])

        # 21: is_in_position
        features.append(1.0 if pos.get("is_active", False) else 0.0)

        # 22: position_direction (-1, 0, 1)
        d = pos.get("direction")
        features.append(1.0 if d == "LONG" else (-1.0 if d == "SHORT" else 0.0))

        # 23: unrealized_r
        features.append(_safe(pos.get("unrealized_r", 0.0)))

        # 24: hold_bars / 96 (1일 정규화)
        features.append(_safe(pos.get("hold_bars", 0) / 96))

        # 25-30: 펀딩 전략 추가 피처 (XRP)
        if self.strategy == "funding_contrarian":
            fd = state.get("funding", {})
            features.append(_safe(fd.get("funding_z", 0.0)))
            features.append(_safe(fd.get("price_z_4h", 0.0)))
            features.append(_safe(fd.get("price_z_12h", 0.0)))
            features.append(_safe(fd.get("price_z_24h", 0.0)))
            features.append(_safe(fd.get("signal_strength", 0.0)))
            features.append(_safe(fd.get("latest_rate", 0.0) * 10000))

        return features

    # ══════════════════════════════════════════════════════════
    # 로깅 메서드
    # ══════════════════════════════════════════════════════════

    def log_candle(
        self,
        df: pd.DataFrame,
        position: Optional[ActiveTrade],
        account: dict,
        signal_generated: bool,
        hold_reason: str = "",
        funding_data: Optional[dict] = None,
    ):
        """매 15분봉 상태 로그 (HOLD 결정 포함)."""
        if position is not None:
            self._hold_bars += 1

        state = self.snapshot_state(df, position, account, funding_data)

        # Step reward (포지션 보유 중 미실현 PnL 변화)
        cur_r = state["position"].get("unrealized_r", 0.0)
        step_reward = {
            "unrealized_pnl_delta": round(cur_r - self._prev_unrealized_r, 6),
        }
        self._prev_unrealized_r = cur_r

        # MAE 추적
        if position is not None and cur_r < self._max_adverse_r:
            self._max_adverse_r = cur_r

        data = {
            "v": self.rl_cfg.log_version,
            "event": "CANDLE",
            "symbol": self.symbol,
            "strategy": self.strategy,
            "ts": datetime.now(timezone.utc).isoformat(),
            "candle_ts": str(df.index[-1]),
            "state": state,
            "action": {
                "type": "HOLD",
                "signal_generated": signal_generated,
                "hold_reason": hold_reason,
            },
            "step_reward": step_reward,
        }
        self._write(data, "states")

    def log_entry(
        self,
        df: pd.DataFrame,
        signal: Signal,
        position: ActiveTrade,
        account: dict,
        funding_data: Optional[dict] = None,
    ):
        """진입 이벤트 로그."""
        self._hold_bars = 0
        self._max_adverse_r = 0.0
        self._prev_unrealized_r = 0.0

        state = self.snapshot_state(df, position, account, funding_data)

        trade_id = f"{self.short_sym}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{signal.direction.value}"

        data = {
            "v": self.rl_cfg.log_version,
            "event": "ENTRY",
            "trade_id": trade_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "ts": datetime.now(timezone.utc).isoformat(),
            "candle_ts": str(df.index[-1]),
            "state": state,
            "action": {
                "type": f"ENTER_{signal.direction.value}",
                "entry_price": signal.entry_price,
                "sl_price": signal.sl_price,
                "tp1_price": signal.tp1_price,
                "tp2_price": signal.tp2_price,
                "tp3_price": signal.tp3_price,
                "risk": signal.risk,
                "reason_text": signal.reason_text,
            },
        }
        self._write(data, "trades")

    def log_exit(
        self,
        df: pd.DataFrame,
        trade: ActiveTrade,
        exit_price: float,
        reason: str,
        pnl_pct: float,
        account: dict,
        funding_data: Optional[dict] = None,
    ):
        """청산 이벤트 로그 (reward 포함)."""
        state = self.snapshot_state(df, None, account, funding_data)

        is_long = trade.direction == Direction.LONG
        if trade.risk > 0:
            pnl_r = ((exit_price - trade.entry_price) / trade.risk if is_long
                      else (trade.entry_price - exit_price) / trade.risk)
        else:
            pnl_r = 0.0

        hold_min = self._hold_bars * 15  # 15분봉 기준

        trade_id = f"{self.short_sym}_{trade.entry_time.strftime('%Y%m%d_%H%M%S') if hasattr(trade.entry_time, 'strftime') else 'unknown'}_{trade.direction.value}"

        data = {
            "v": self.rl_cfg.log_version,
            "event": "EXIT",
            "trade_id": trade_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "ts": datetime.now(timezone.utc).isoformat(),
            "candle_ts": str(df.index[-1]),
            "state": state,
            "action": {
                "type": "EXIT",
                "exit_price": exit_price,
                "exit_reason": reason,
            },
            "reward": {
                "pnl_pct": round(pnl_pct, 4),
                "pnl_r": round(pnl_r, 4),
                "hold_bars": self._hold_bars,
                "hold_min": hold_min,
                "peak_r": round(trade.peak_r, 4),
                "max_adverse_r": round(self._max_adverse_r, 4),
                "tp1_hit": trade.tp1_hit,
                "tp2_hit": trade.tp2_hit,
                "tp3_hit": trade.tp3_hit,
                "entry_price": trade.entry_price,
                "direction": trade.direction.value,
            },
        }
        self._write(data, "trades")

        # 상태 리셋
        self._hold_bars = 0
        self._max_adverse_r = 0.0
        self._prev_unrealized_r = 0.0

    # ══════════════════════════════════════════════════════════
    # 내부
    # ══════════════════════════════════════════════════════════

    def _write(self, data: dict, log_type: str):
        """JSONL 파일에 기록."""
        base = Path(self.rl_cfg.log_base_dir) / log_type / self.short_sym
        base.mkdir(parents=True, exist_ok=True)
        fpath = base / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
        with open(fpath, "a") as f:
            f.write(json.dumps(data, default=str, ensure_ascii=False) + "\n")


def _safe(val: float) -> float:
    """NaN/Inf를 0으로."""
    if isinstance(val, (int, float)) and not (math.isnan(val) or math.isinf(val)):
        return round(float(val), 6)
    return 0.0


def _pad(lst: list, n: int, fill: float = 0.0) -> list:
    """리스트를 n 길이로 패딩."""
    if len(lst) >= n:
        return lst[:n]
    return lst + [fill] * (n - len(lst))
