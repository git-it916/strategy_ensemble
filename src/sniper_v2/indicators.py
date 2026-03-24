"""
Sniper V2 — Indicator Calculations

EMA, RSI, ATR, ADX 기반. NaN-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sniper_v2.config import (
    ADX_LEN,
    ATR_LEN,
    EMA_FAST,
    EMA_SLOW,
    EMA_TREND,
    RSI_LEN,
    VOL_FILTER_SMA,
)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = RSI_LEN) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_LEN) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_LEN) -> pd.Series:
    """ADX 값만 반환 (DI+/DI- 방향은 EMA cross로 대체)."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    atr_val = atr(high, low, close, period)
    atr_safe = atr_val.replace(0, np.nan)

    di_plus = 100 * ema(plus_dm, period) / atr_safe
    di_minus = 100 * ema(minus_dm, period) / atr_safe
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return ema(dx, period)


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """모든 지표를 DataFrame에 추가."""
    c, h, l = df["close"], df["high"], df["low"]

    df["ema_fast"] = ema(c, EMA_FAST)
    df["ema_slow"] = ema(c, EMA_SLOW)
    df["ema_trend"] = ema(c, EMA_TREND)
    df["rsi"] = rsi(c)
    df["atr"] = atr(h, l, c)
    df["atr_sma"] = df["atr"].rolling(VOL_FILTER_SMA, min_periods=10).mean()
    df["adx"] = adx(h, l, c)

    # EMA cross detection
    df["ema_bull_cross"] = (
        (df["ema_fast"] > df["ema_slow"]) &
        (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
    )
    df["ema_bear_cross"] = (
        (df["ema_fast"] < df["ema_slow"]) &
        (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
    )

    return df
