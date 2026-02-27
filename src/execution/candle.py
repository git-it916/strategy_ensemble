"""
Candle Aggregator

Aggregate real-time ticks into candle bars.
Generic data classes and aggregator reusable across exchanges.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RealtimeTick:
    """Single real-time tick."""
    stock_code: str
    price: float
    volume: float
    timestamp: datetime
    change: float = 0.0
    change_rate: float = 0.0
    ask_price: float = 0.0
    bid_price: float = 0.0
    cumul_volume: float = 0.0


@dataclass
class CandleBar:
    """Aggregated candle bar."""
    stock_code: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    start_time: datetime
    end_time: datetime
    tick_count: int = 0


# =============================================================================
# Candle Aggregator
# =============================================================================

class CandleAggregator:
    """
    Aggregate real-time ticks into candle bars.
    Thread-safe via lock.
    """

    def __init__(self, interval_minutes: int = 60):
        self.interval_minutes = interval_minutes
        self._current_candles: dict[str, CandleBar] = {}
        self._completed_candles: dict[str, list[CandleBar]] = defaultdict(list)
        self._lock = threading.Lock()
        self._on_candle_complete: Callable[[CandleBar], None] | None = None

    def set_on_complete(self, callback: Callable[[CandleBar], None]) -> None:
        """Set callback for when a candle completes."""
        self._on_candle_complete = callback

    def add_tick(self, tick: RealtimeTick) -> CandleBar | None:
        """
        Add a tick to the aggregator.

        Returns completed CandleBar if the tick triggers a candle close,
        otherwise None.
        """
        with self._lock:
            code = tick.stock_code
            candle_start = self._get_candle_start(tick.timestamp)

            # Extract kline OHLC if carried via tick fields
            # (ask_price=high, bid_price=low, cumul_volume=open from kline)
            tick_open = tick.cumul_volume if tick.cumul_volume > 0 else tick.price
            tick_high = tick.ask_price if tick.ask_price > 0 else tick.price
            tick_low = tick.bid_price if tick.bid_price > 0 else tick.price

            if code not in self._current_candles:
                # New candle
                self._current_candles[code] = CandleBar(
                    stock_code=code,
                    open=tick_open,
                    high=tick_high,
                    low=tick_low,
                    close=tick.price,
                    volume=tick.volume,
                    start_time=candle_start,
                    end_time=candle_start + timedelta(minutes=self.interval_minutes),
                    tick_count=1,
                )
                return None

            current = self._current_candles[code]

            # Check if we've moved to a new candle period
            if candle_start > current.start_time:
                completed = self._finalize_candle(code)
                # Start new candle
                self._current_candles[code] = CandleBar(
                    stock_code=code,
                    open=tick_open,
                    high=tick_high,
                    low=tick_low,
                    close=tick.price,
                    volume=tick.volume,
                    start_time=candle_start,
                    end_time=candle_start + timedelta(minutes=self.interval_minutes),
                    tick_count=1,
                )
                return completed

            # Update existing candle using kline OHLC
            current.high = max(current.high, tick_high)
            current.low = min(current.low, tick_low)
            current.close = tick.price
            current.volume += tick.volume
            current.tick_count += 1

            return None

    def _get_candle_start(self, dt: datetime) -> datetime:
        """Get the candle period start time for a given datetime."""
        minutes = dt.hour * 60 + dt.minute
        period_start = (minutes // self.interval_minutes) * self.interval_minutes
        return dt.replace(
            hour=period_start // 60,
            minute=period_start % 60,
            second=0,
            microsecond=0,
        )

    def _finalize_candle(self, stock_code: str) -> CandleBar:
        """Finalize and store a completed candle."""
        candle = self._current_candles[stock_code]
        self._completed_candles[stock_code].append(candle)

        # Trigger callback
        if self._on_candle_complete:
            try:
                self._on_candle_complete(candle)
            except Exception as e:
                logger.error(f"Candle callback error: {e}")

        return candle

    def get_latest_candles(
        self, stock_code: str, n: int = 10
    ) -> list[CandleBar]:
        """Thread-safe access to completed candles."""
        with self._lock:
            candles = self._completed_candles.get(stock_code, [])
            return candles[-n:]

    def get_all_completed(self) -> dict[str, list[CandleBar]]:
        """Get all completed candles (copy)."""
        with self._lock:
            return {k: list(v) for k, v in self._completed_candles.items()}

    def to_dataframe(self, stock_code: str | None = None) -> pd.DataFrame:
        """Convert completed candles to DataFrame."""
        with self._lock:
            if stock_code:
                candles = self._completed_candles.get(stock_code, [])
            else:
                candles = [
                    c
                    for cs in self._completed_candles.values()
                    for c in cs
                ]

        if not candles:
            return pd.DataFrame(
                columns=[
                    "stock_code", "open", "high", "low", "close",
                    "volume", "start_time", "end_time", "tick_count",
                ]
            )

        return pd.DataFrame([
            {
                "stock_code": c.stock_code,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "tick_count": c.tick_count,
            }
            for c in candles
        ])
