"""
Universe filtering helpers.

Ensures we consistently use:
- KOSPI + KOSDAQ universe
- Market-cap floor (100 billion KRW by default)
- Broker-safe 6-digit ticker codes for execution paths
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TICKER_CODE_RE = re.compile(r"(\d{6})")


def normalize_order_ticker(ticker: str) -> str:
    """Convert raw ticker text (e.g., '005930 KP') to broker-safe code ('005930')."""
    text = str(ticker or "").strip().upper()
    if not text:
        return ""

    match = _TICKER_CODE_RE.search(text)
    if match:
        return match.group(1)

    return text


def infer_market_from_ticker(ticker: str, fallback: str = "KOSPI") -> str:
    """
    Infer market from ticker suffix.

    Supported examples:
    - 005930 KP / 005930 KS -> KOSPI
    - 035720 KQ -> KOSDAQ
    """
    text = str(ticker or "").strip().upper()
    if not text:
        return fallback

    if (
        text.endswith(" KQ")
        or text.endswith("KQ")
        or " KQ " in text
        or "KQ EQUITY" in text
        or "KOSDAQ" in text
    ):
        return "KOSDAQ"

    if (
        text.endswith(" KP")
        or text.endswith("KP")
        or text.endswith(" KS")
        or text.endswith("KS")
        or " KP " in text
        or " KS " in text
        or "KS EQUITY" in text
        or "KOSPI" in text
    ):
        return "KOSPI"

    return fallback


def _normalize_market_label(raw_market: object, ticker: str) -> str:
    """Normalize market labels to 'KOSPI' or 'KOSDAQ' when possible."""
    text = str(raw_market or "").strip().upper()
    if not text or text == "NAN":
        return infer_market_from_ticker(ticker)

    if "KOSDAQ" in text or text in {"KQ"}:
        return "KOSDAQ"
    if "KOSPI" in text or text in {"KS", "KP", "KRX"}:
        return "KOSPI"

    return infer_market_from_ticker(ticker)


def _safe_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def _resolve_market_cap_threshold(
    configured_min_market_cap: float,
    market_caps: pd.Series,
) -> float:
    """
    Resolve market-cap threshold scale.

    `configured_min_market_cap` is generally set in KRW (e.g., 1e11 for 1000억).
    Some vendor data stores market_cap in million KRW, so we auto-convert only when
    the observed scale strongly suggests that unit.
    """
    min_cap = float(configured_min_market_cap or 0.0)
    if min_cap <= 0:
        return 0.0

    caps = _safe_numeric(market_caps).dropna()
    caps = caps[caps > 0]
    if caps.empty:
        return min_cap

    p95 = float(caps.quantile(0.95))
    if min_cap >= 1e10 and p95 < (min_cap / 1_000):
        # Convert KRW threshold -> million-KRW threshold.
        return min_cap / 1_000_000

    return min_cap


def build_universe_snapshot(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp | None = None,
    max_stocks: int = 100,
    min_market_cap: float = 0.0,
    min_turnover: float = 0.0,
    min_volume: float = 0.0,
    allowed_markets: Iterable[str] = ("KOSPI", "KOSDAQ"),
) -> pd.DataFrame:
    """
    Build filtered latest universe snapshot from a price frame.

    Returns columns:
        date, ticker_data, ticker_order, market, market_cap, turnover, volume
    """
    output_cols = [
        "date",
        "ticker_data",
        "ticker_order",
        "market",
        "market_cap",
        "turnover",
        "volume",
    ]
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
    snapshot = df[df["date"] <= ref_date].sort_values("date").groupby("ticker").tail(1).copy()
    if snapshot.empty:
        return pd.DataFrame(columns=output_cols)

    snapshot["ticker_data"] = snapshot["ticker"].astype(str)
    snapshot["ticker_order"] = snapshot["ticker_data"].map(normalize_order_ticker)
    snapshot["market"] = [
        _normalize_market_label(snapshot.at[idx, "market"] if "market" in snapshot.columns else None, snapshot.at[idx, "ticker_data"])
        for idx in snapshot.index
    ]

    market_cap = _safe_numeric(snapshot["market_cap"] if "market_cap" in snapshot.columns else None)
    market_cap_fund = _safe_numeric(snapshot["market_cap_fund"] if "market_cap_fund" in snapshot.columns else None)
    if market_cap.empty:
        snapshot["market_cap"] = market_cap_fund
    else:
        snapshot["market_cap"] = market_cap
        if not market_cap_fund.empty:
            snapshot["market_cap"] = snapshot["market_cap"].fillna(market_cap_fund)

    snapshot["turnover"] = _safe_numeric(snapshot["turnover"] if "turnover" in snapshot.columns else None)
    snapshot["volume"] = _safe_numeric(snapshot["volume"] if "volume" in snapshot.columns else None)

    allowed = {str(m).upper() for m in allowed_markets}
    snapshot = snapshot[snapshot["market"].str.upper().isin(allowed)]

    if min_market_cap and min_market_cap > 0:
        cap_threshold = _resolve_market_cap_threshold(min_market_cap, snapshot["market_cap"])
        if snapshot["market_cap"].notna().any():
            snapshot = snapshot[snapshot["market_cap"] >= cap_threshold]
        else:
            logger.warning("Universe filter: market_cap missing, skipping min_market_cap filter")

    if min_turnover and min_turnover > 0 and snapshot["turnover"].notna().any():
        snapshot = snapshot[snapshot["turnover"] >= float(min_turnover)]

    if min_volume and min_volume > 0 and snapshot["volume"].notna().any():
        snapshot = snapshot[snapshot["volume"] >= float(min_volume)]

    snapshot = snapshot.sort_values(
        ["market_cap", "turnover", "volume"],
        ascending=False,
        na_position="last",
    )
    snapshot = snapshot.drop_duplicates(subset=["ticker_order"], keep="first")

    if max_stocks > 0:
        snapshot = snapshot.head(max_stocks)

    return snapshot[output_cols].reset_index(drop=True)
