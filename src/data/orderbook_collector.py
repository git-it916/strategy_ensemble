"""
Orderbook Collector

Periodically snapshots orderbooks for top tickers and maintains
a rolling 30-minute in-memory history.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timezone

from src.data.data_bundle import OrderbookSnapshot

logger = logging.getLogger(__name__)

# Maximum history duration per ticker (in seconds)
_HISTORY_MAX_AGE_S = 1800  # 30 minutes


class OrderbookCollector:
    """
    Collects and stores orderbook snapshots for alpha consumption.

    Usage:
        collector = OrderbookCollector(binance_api)
        collector.snapshot(["ETH/USDT:USDT", "SOL/USDT:USDT"])
        bundle.orderbook = collector.get_latest()
        bundle.orderbook_history = collector.get_history()
    """

    def __init__(self, binance_api, levels: int = 20):
        """
        Args:
            binance_api: BinanceApi instance (ccxt-based).
            levels: Number of orderbook levels to fetch (default 20).
        """
        self._api = binance_api
        self._levels = levels
        # ticker → deque of OrderbookSnapshot (newest last)
        self._history: dict[str, deque[OrderbookSnapshot]] = {}

    def snapshot(self, tickers: list[str]) -> dict[str, OrderbookSnapshot]:
        """
        Fetch orderbook snapshots for given tickers.

        Returns:
            {ticker: OrderbookSnapshot} for successfully fetched tickers.
        """
        result: dict[str, OrderbookSnapshot] = {}
        now = datetime.now(timezone.utc)

        for ticker in tickers:
            try:
                book = self._api._exchange.fetch_order_book(
                    ticker, limit=self._levels
                )
                bids = book.get("bids", []) or []
                asks = book.get("asks", []) or []
                if not bids or not asks:
                    continue

                snap = self._build_snapshot(ticker, now, bids, asks)
                result[ticker] = snap

                # Append to rolling history
                if ticker not in self._history:
                    self._history[ticker] = deque()
                self._history[ticker].append(snap)

                time.sleep(self._api.rate_limit_sleep)

            except Exception as e:
                logger.debug(f"Orderbook fetch failed for {ticker}: {e}")

        # Prune old entries
        self._prune_history(now)

        return result

    def get_latest(self) -> dict[str, OrderbookSnapshot]:
        """Return the most recent snapshot per ticker."""
        latest: dict[str, OrderbookSnapshot] = {}
        for ticker, dq in self._history.items():
            if dq:
                latest[ticker] = dq[-1]
        return latest

    def get_history(self) -> dict[str, list[OrderbookSnapshot]]:
        """Return full rolling history per ticker."""
        return {t: list(dq) for t, dq in self._history.items()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_snapshot(
        ticker: str,
        ts: datetime,
        bids: list,
        asks: list,
    ) -> OrderbookSnapshot:
        """Parse raw ccxt orderbook into an OrderbookSnapshot."""
        bids_parsed = [(float(b[0]), float(b[1])) for b in bids]
        asks_parsed = [(float(a[0]), float(a[1])) for a in asks]

        best_bid = bids_parsed[0][0]
        best_ask = asks_parsed[0][0]
        mid = (best_bid + best_ask) / 2.0

        spread_bps = ((best_ask - best_bid) / mid) * 10_000 if mid > 0 else 0.0

        # Compute depth in USDT per level bucket
        def _depth_usdt(levels: list[tuple[float, float]]) -> float:
            return sum(p * q for p, q in levels)

        def _bucket_depth(levels: list[tuple[float, float]], start: int, end: int) -> float:
            return sum(p * q for p, q in levels[start:end])

        bid_depth = _depth_usdt(bids_parsed)
        ask_depth = _depth_usdt(asks_parsed)
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0

        # Level-bucket imbalances
        imbalance_levels: dict[str, float] = {}
        for label, s, e in [("1-5", 0, 5), ("5-10", 5, 10), ("10-20", 10, 20)]:
            b = _bucket_depth(bids_parsed, s, min(e, len(bids_parsed)))
            a = _bucket_depth(asks_parsed, s, min(e, len(asks_parsed)))
            t = b + a
            imbalance_levels[label] = (b - a) / t if t > 0 else 0.0

        return OrderbookSnapshot(
            ticker=ticker,
            timestamp=ts,
            bids=bids_parsed,
            asks=asks_parsed,
            mid_price=mid,
            spread_bps=spread_bps,
            bid_depth_usdt=bid_depth,
            ask_depth_usdt=ask_depth,
            imbalance=imbalance,
            imbalance_levels=imbalance_levels,
        )

    def _prune_history(self, now: datetime) -> None:
        """Remove entries older than _HISTORY_MAX_AGE_S."""
        cutoff_ts = now.timestamp() - _HISTORY_MAX_AGE_S
        for ticker in list(self._history.keys()):
            dq = self._history[ticker]
            while dq and dq[0].timestamp.timestamp() < cutoff_ts:
                dq.popleft()
            if not dq:
                del self._history[ticker]
