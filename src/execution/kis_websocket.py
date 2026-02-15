"""
KIS WebSocket Client

한국투자증권 실시간 시세 WebSocket 클라이언트.
체결가(H0STCNT0) / 호가(H0STASP0) 실시간 수신 및 시간봉 집계.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KIS_WS_URL = "ws://ops.koreainvestment.com:31000"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RealtimeTick:
    """Single real-time tick from KIS WebSocket."""
    stock_code: str
    price: int
    volume: int
    timestamp: datetime
    change: int = 0
    change_rate: float = 0.0
    ask_price: int = 0
    bid_price: int = 0
    cumul_volume: int = 0


@dataclass
class CandleBar:
    """Aggregated candle bar."""
    stock_code: str
    open: int
    high: int
    low: int
    close: int
    volume: int
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

            if code not in self._current_candles:
                # New candle
                self._current_candles[code] = CandleBar(
                    stock_code=code,
                    open=tick.price,
                    high=tick.price,
                    low=tick.price,
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
                    open=tick.price,
                    high=tick.price,
                    low=tick.price,
                    close=tick.price,
                    volume=tick.volume,
                    start_time=candle_start,
                    end_time=candle_start + timedelta(minutes=self.interval_minutes),
                    tick_count=1,
                )
                return completed

            # Update existing candle
            current.high = max(current.high, tick.price)
            current.low = min(current.low, tick.price)
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


# =============================================================================
# KIS WebSocket Client
# =============================================================================

class KISWebSocket:
    """
    KIS real-time WebSocket client.

    Features:
        - Approval key acquisition (REST)
        - WebSocket connection with PINGPONG heartbeat
        - Real-time 체결가 (H0STCNT0) and 호가 (H0STASP0) subscription
        - CandleAggregator integration
        - Background thread operation
        - Auto-reconnect on disconnect
    """

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        is_paper: bool = True,
        candle_interval: int = 60,
        base_url: str | None = None,
    ):
        self.app_key = app_key
        self.app_secret = app_secret
        self.is_paper = is_paper
        self.ws_url = KIS_WS_URL

        # REST base URL for approval key
        if base_url:
            self._rest_base = base_url
        elif is_paper:
            self._rest_base = "https://openapivts.koreainvestment.com:29443"
        else:
            self._rest_base = "https://openapi.koreainvestment.com:9443"

        self._approval_key: str | None = None
        self._ws: Any = None  # websockets.WebSocketClientProtocol
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._subscriptions: set[tuple[str, str]] = set()  # (stock_code, tr_id)

        self.aggregator = CandleAggregator(interval_minutes=candle_interval)
        self._tick_callbacks: list[Callable[[RealtimeTick], None]] = []

    # --- Public API ---

    def start(self) -> None:
        """Start WebSocket in background thread."""
        if self._running:
            logger.warning("WebSocket already running")
            return

        # Get approval key first
        self._approval_key = self._get_approval_key()
        logger.info("Got WebSocket approval key")

        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="kis-websocket",
            daemon=True,
        )
        self._thread.start()
        logger.info("KIS WebSocket started")

    def stop(self) -> None:
        """Gracefully stop WebSocket."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("KIS WebSocket stopped")

    def subscribe_price(self, stock_code: str) -> None:
        """Subscribe to 체결가 (execution prices)."""
        self._queue_subscribe(stock_code, "H0STCNT0")

    def subscribe_orderbook(self, stock_code: str) -> None:
        """Subscribe to 호가 (orderbook)."""
        self._queue_subscribe(stock_code, "H0STASP0")

    def unsubscribe(self, stock_code: str, tr_id: str = "H0STCNT0") -> None:
        """Unsubscribe from a stream."""
        self._queue_subscribe(stock_code, tr_id, unsub=True)

    def add_tick_callback(
        self, callback: Callable[[RealtimeTick], None]
    ) -> None:
        """Register a tick callback."""
        self._tick_callbacks.append(callback)

    def set_candle_callback(
        self, callback: Callable[[CandleBar], None]
    ) -> None:
        """Set callback for hourly candle completion."""
        self.aggregator.set_on_complete(callback)

    def get_latest_candles(
        self, stock_code: str, n: int = 10
    ) -> list[CandleBar]:
        """Thread-safe access to completed candles."""
        return self.aggregator.get_latest_candles(stock_code, n)

    @property
    def is_connected(self) -> bool:
        return self._running and self._ws is not None

    # --- Internal ---

    def _get_approval_key(self) -> str:
        """Get WebSocket approval key via REST API."""
        url = f"{self._rest_base}/oauth2/Approval"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "secretkey": self.app_secret,
        }

        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        key = data.get("approval_key")
        if not key:
            raise ValueError(f"Failed to get approval_key: {data}")

        return key

    def _run_event_loop(self) -> None:
        """Run async event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception as e:
            logger.error(f"WebSocket event loop error: {e}")
        finally:
            self._loop.close()

    async def _connect(self) -> None:
        """Main async connection loop with auto-reconnect."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed: pip install websockets")
            return

        reconnect_delay = 5.0
        max_reconnect = 10
        attempt = 0

        while self._running and attempt < max_reconnect:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    attempt = 0
                    logger.info(f"WebSocket connected to {self.ws_url}")

                    # Re-subscribe after reconnect
                    for stock_code, tr_id in list(self._subscriptions):
                        await self._send_subscribe(ws, stock_code, tr_id)

                    # Message loop
                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)

            except Exception as e:
                attempt += 1
                logger.error(
                    f"WebSocket error (attempt {attempt}): {e}"
                )
                self._ws = None
                if self._running and attempt < max_reconnect:
                    await asyncio.sleep(reconnect_delay)

        logger.info("WebSocket connection loop ended")

    async def _handle_message(self, message: str | bytes) -> None:
        """Parse KIS WebSocket message."""
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")

        # PINGPONG heartbeat
        if "PINGPONG" in message:
            if self._ws:
                await self._ws.send("PONG")
            return

        # KIS messages are | delimited
        # Format: encrypted_flag|tr_id|data_count|data
        parts = message.split("|")
        if len(parts) < 4:
            return

        tr_id = parts[1]

        if tr_id == "H0STCNT0":
            # 체결가 (execution price)
            self._handle_execution(parts[3])
        elif tr_id == "H0STASP0":
            # 호가 (orderbook)
            self._handle_orderbook(parts[3])

    def _handle_execution(self, data: str) -> None:
        """Parse execution price data and create tick."""
        # KIS 체결가 fields are ^ delimited
        fields = data.split("^")
        if len(fields) < 20:
            return

        try:
            tick = RealtimeTick(
                stock_code=fields[0],           # 종목코드
                timestamp=datetime.now(),
                price=int(fields[2]),            # 체결가
                change=int(fields[4]),           # 전일대비
                change_rate=float(fields[5]),    # 전일대비율
                volume=int(fields[12]),          # 체결거래량
                cumul_volume=int(fields[13]),    # 누적거래량
                ask_price=int(fields[6]) if fields[6] else 0,   # 매도호가
                bid_price=int(fields[7]) if fields[7] else 0,   # 매수호가
            )

            # Feed to candle aggregator
            completed = self.aggregator.add_tick(tick)

            # Notify tick callbacks
            for cb in self._tick_callbacks:
                try:
                    cb(tick)
                except Exception as e:
                    logger.error(f"Tick callback error: {e}")

            if completed:
                logger.info(
                    f"Candle completed: {completed.stock_code} "
                    f"O={completed.open} H={completed.high} "
                    f"L={completed.low} C={completed.close} "
                    f"V={completed.volume}"
                )

        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse execution data: {e}")

    def _handle_orderbook(self, data: str) -> None:
        """Parse orderbook data (호가)."""
        # Log for monitoring; detailed parsing can be added as needed
        fields = data.split("^")
        if len(fields) >= 3:
            logger.debug(f"Orderbook: {fields[0]} ask={fields[3]} bid={fields[13]}")

    def _queue_subscribe(
        self, stock_code: str, tr_id: str, unsub: bool = False
    ) -> None:
        """Queue a subscription request."""
        if unsub:
            self._subscriptions.discard((stock_code, tr_id))
        else:
            self._subscriptions.add((stock_code, tr_id))

        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(
                self._send_subscribe(self._ws, stock_code, tr_id, unsub),
                self._loop,
            )

    async def _send_subscribe(
        self,
        ws: Any,
        stock_code: str,
        tr_id: str,
        unsub: bool = False,
    ) -> None:
        """Send subscription/unsubscription message."""
        msg = json.dumps({
            "header": {
                "approval_key": self._approval_key,
                "custtype": "P",
                "tr_type": "2" if unsub else "1",
                "content-type": "utf-8",
            },
            "body": {
                "input": {
                    "tr_id": tr_id,
                    "tr_key": stock_code,
                },
            },
        })

        await ws.send(msg)
        action = "Unsubscribed" if unsub else "Subscribed"
        logger.info(f"{action}: {stock_code} ({tr_id})")
