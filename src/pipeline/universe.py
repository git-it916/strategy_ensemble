"""
Universe filtering helpers.

Binance USDT-M Perpetual Futures universe utilities.
Symbols are in ccxt format: e.g. 'BTC/USDT:USDT'.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_symbol(symbol: str) -> str:
    """
    Normalize a Binance symbol to a consistent format.

    Accepts:
        - ccxt perpetual format: 'BTC/USDT:USDT'
        - raw Binance format:    'BTCUSDT'
        - slash format:          'BTC/USDT'

    Returns ccxt perpetual format: 'BTC/USDT:USDT'
    """
    s = str(symbol or "").strip().upper()
    if not s:
        return ""

    # Already in ccxt perpetual format
    if "/" in s and ":" in s:
        return s

    # Slash format without settle currency
    if "/" in s and ":" not in s:
        base, quote = s.split("/", 1)
        return f"{base}/{quote}:{quote}"

    # Raw Binance format 'BTCUSDT' — assume USDT quote
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}/USDT:USDT"

    return s


# Backward-compat alias
normalize_order_ticker = normalize_symbol


def build_universe_snapshot(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp | None = None,
    max_stocks: int = 50,
    min_volume: float = 0.0,
    # legacy kwargs accepted but ignored
    min_market_cap: float = 0.0,
    min_turnover: float = 0.0,
    allowed_markets=None,
) -> pd.DataFrame:
    """
    Build filtered latest universe snapshot from a Binance price frame.

    Returns columns:
        date, ticker, volume
    """
    output_cols = ["date", "ticker", "volume"]

    if prices is None or prices.empty:
        return pd.DataFrame(columns=output_cols)
    if "date" not in prices.columns or "ticker" not in prices.columns:
        logger.warning("Universe build skipped: prices missing 'date' or 'ticker'")
        return pd.DataFrame(columns=output_cols)

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker"])
    if df.empty:
        return pd.DataFrame(columns=output_cols)

    ref_date = pd.Timestamp(as_of_date) if as_of_date is not None else df["date"].max()
    snapshot = (
        df[df["date"] <= ref_date]
        .sort_values("date")
        .groupby("ticker")
        .tail(1)
        .copy()
    )
    if snapshot.empty:
        return pd.DataFrame(columns=output_cols)

    if "volume" in snapshot.columns:
        snapshot["volume"] = pd.to_numeric(snapshot["volume"], errors="coerce")
    else:
        snapshot["volume"] = float("nan")

    if min_volume > 0 and snapshot["volume"].notna().any():
        snapshot = snapshot[snapshot["volume"] >= min_volume]

    snapshot = snapshot.sort_values("volume", ascending=False, na_position="last")
    snapshot = snapshot.drop_duplicates(subset=["ticker"], keep="first")

    if max_stocks > 0:
        snapshot = snapshot.head(max_stocks)

    return snapshot[["date", "ticker", "volume"]].reset_index(drop=True)
