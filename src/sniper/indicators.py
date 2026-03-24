"""
Precision Sniper — Indicator Calculations

Pine Script 원본과 동일한 지표 계산.
pandas/numpy 기반, NaN-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sniper.config import (
    ADX_LEN,
    ATR_LEN,
    EMA_FAST_LEN,
    EMA_SLOW_LEN,
    EMA_TREND_LEN,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_LEN,
    SWING_LOOKBACK,
    VOL_SMA_LEN,
)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = RSI_LEN) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = ema(close, MACD_FAST)
    ema_slow = ema(close, MACD_SLOW)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, MACD_SIGNAL)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_LEN) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def adx_dmi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_LEN) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    ADX, DI+, DI-.

    Returns: (di_plus, di_minus, adx)
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)

    # +DM이 -DM보다 작으면 0, 반대도 마찬가지
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    atr_val = atr(high, low, close, period)

    di_plus = 100 * ema(plus_dm, period) / atr_val.replace(0, np.nan)
    di_minus = 100 * ema(minus_dm, period) / atr_val.replace(0, np.nan)

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_val = ema(dx, period)

    return di_plus, di_minus, adx_val


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Rolling VWAP (일중).

    Pine Script의 ta.vwap(hlc3) 근사.
    실시간이 아니므로 rolling 20봉 VWAP 사용.
    """
    hlc3 = (high + low + close) / 3
    cum_vol = volume.rolling(20, min_periods=1).sum()
    cum_pv = (hlc3 * volume).rolling(20, min_periods=1).sum()
    return cum_pv / cum_vol.replace(0, np.nan)


def swing_low(low: pd.Series, lookback: int = SWING_LOOKBACK) -> float:
    """최근 N봉 중 최저가."""
    return float(low.iloc[-lookback:].min())


def swing_high(high: pd.Series, lookback: int = SWING_LOOKBACK) -> float:
    """최근 N봉 중 최고가."""
    return float(high.iloc[-lookback:].max())


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    모든 지표를 한번에 계산해서 DataFrame에 추가.

    입력: OHLCV DataFrame (open, high, low, close, volume)
    출력: 지표 컬럼이 추가된 DataFrame
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # EMAs
    df["ema_fast"] = ema(c, EMA_FAST_LEN)
    df["ema_slow"] = ema(c, EMA_SLOW_LEN)
    df["ema_trend"] = ema(c, EMA_TREND_LEN)

    # RSI
    df["rsi"] = rsi(c)

    # MACD
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(c)

    # ATR
    df["atr"] = atr(h, l, c)

    # ATR SMA (변동성 레짐)
    df["atr_sma"] = df["atr"].rolling(42, min_periods=10).mean()
    df["vol_ratio"] = df["atr"] / df["atr_sma"].replace(0, np.nan)

    # ADX / DMI
    df["di_plus"], df["di_minus"], df["adx"] = adx_dmi(h, l, c)

    # Volume SMA
    df["vol_sma"] = v.rolling(VOL_SMA_LEN, min_periods=5).mean()
    df["vol_above_avg"] = v > (df["vol_sma"] * 1.2)

    # VWAP
    df["vwap"] = vwap(h, l, c, v)

    # EMA cross detection
    df["ema_bull_cross"] = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
    df["ema_bear_cross"] = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))

    return df
