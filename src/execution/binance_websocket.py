"""
Binance USDT-M Futures WebSocket Client

Binance Futures kline/candle 실시간 WebSocket 클라이언트.
기존 KISWebSocket 구조를 유지하며 CandleAggregator를 재사용.

Stream URL: wss://fstream.binance.com/stream
Subscribe format: <symbol>@kline_<interval>  (e.g. btcusdt@kline_1m)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Callable

from .candle import CandleAggregator, CandleBar, RealtimeTick

logger = logging.getLogger(__name__)

BINANCE_FUTURES_WS_URL = "wss://fstream.binance.com/stream"


class BinanceWebSocket:
    """
    Binance Futures real-time WebSocket client.

    Features:
        - Multi-symbol kline subscription via combined stream
        - CandleAggregator integration
        - Background thread operation
        - Auto-reconnect on disconnect
        - Tick callbacks for downstream consumers
    """

    def __init__(
        self,
        candle_interval: str = "1m",
        ws_url: str = BINANCE_FUTURES_WS_URL,
    ):
        """
        Args:
            candle_interval: Kline interval string (e.g. '1m', '5m', '15m', '1h')
            ws_url: WebSocket base URL
        """
        self.candle_interval = candle_interval
        self.ws_url = ws_url

        # Derive integer minutes for CandleAggregator
        self._interval_minutes = self._parse_interval_minutes(candle_interval)
        self.aggregator = CandleAggregator(interval_minutes=self._interval_minutes)

        self._subscriptions: set[str] = set()  # Binance symbols (lowercase, e.g. 'btcusdt')
        self._tick_callbacks: list[Callable[[RealtimeTick], None]] = []

        self._ws = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start WebSocket in background daemon thread."""
        if self._running:
            logger.warning("BinanceWebSocket already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="binance-websocket",
            daemon=True,
        )
        self._thread.start()
        logger.info("BinanceWebSocket started")

    def stop(self) -> None:
        """Gracefully stop WebSocket."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("BinanceWebSocket stopped")

    def subscribe(self, symbol: str) -> None:
        """
        Subscribe to kline stream for a symbol.

        Args:
            symbol: ccxt-style symbol ('BTC/USDT:USDT') or raw ('BTCUSDT')
        """
        normalized = self._normalize_symbol(symbol)
        self._subscriptions.add(normalized)

        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(
                self._send_subscribe(self._ws, [normalized]),
                self._loop,
            )

    def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from a symbol's kline stream."""
        normalized = self._normalize_symbol(symbol)
        self._subscriptions.discard(normalized)

        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(
                self._send_unsubscribe(self._ws, [normalized]),
                self._loop,
            )

    def add_tick_callback(self, callback: Callable[[RealtimeTick], None]) -> None:
        """Register callback for each incoming tick."""
        self._tick_callbacks.append(callback)

    def set_candle_callback(self, callback: Callable[[CandleBar], None]) -> None:
        """Set callback for completed candle bars."""
        self.aggregator.set_on_complete(callback)

    def get_latest_candles(self, symbol: str, n: int = 10) -> list[CandleBar]:
        """Thread-safe access to completed candles."""
        normalized = self._normalize_symbol(symbol)
        return self.aggregator.get_latest_candles(normalized, n)

    @property
    def is_connected(self) -> bool:
        return self._running and self._ws is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        """Run asyncio event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception as e:
            logger.error(f"BinanceWebSocket event loop error: {e}")
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
            # Build stream URL with all subscribed symbols
            streams = self._build_stream_path()
            url = f"{self.ws_url}?streams={streams}" if streams else self.ws_url

            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    attempt = 0
                    logger.info(f"Binance WebSocket connected: {len(self._subscriptions)} streams")

                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)

            except Exception as e:
                attempt += 1
                logger.error(f"WebSocket error (attempt {attempt}): {e}")
                self._ws = None
                if self._running and attempt < max_reconnect:
                    await asyncio.sleep(reconnect_delay)

        logger.info("BinanceWebSocket connection loop ended")

    def _build_stream_path(self) -> str:
        """Build combined stream path string."""
        streams = [
            f"{sym}@kline_{self.candle_interval}"
            for sym in self._subscriptions
        ]
        return "/".join(streams)

    async def _handle_message(self, message: str | bytes) -> None:
        """Parse Binance combined stream message."""
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        # Combined stream wraps data under 'data' key
        payload = data.get("data", data)

        event_type = payload.get("e")
        if event_type == "kline":
            self._handle_kline(payload)

    def _handle_kline(self, payload: dict) -> None:
        """Parse kline payload and create RealtimeTick."""
        kline = payload.get("k", {})
        if not kline:
            return

        try:
            symbol = payload.get("s", "")  # e.g. 'BTCUSDT'
            close_price = float(kline.get("c", 0))
            volume = float(kline.get("v", 0))
            ts_ms = int(kline.get("t", 0))

            # Use OHLCV from kline, not just close price
            open_price = float(kline.get("o", close_price))
            high_price = float(kline.get("h", close_price))
            low_price = float(kline.get("l", close_price))

            tick = RealtimeTick(
                stock_code=symbol,
                price=close_price,
                volume=volume,
                timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                change=0.0,
                change_rate=0.0,
                ask_price=high_price,   # Carry kline high in ask_price
                bid_price=low_price,    # Carry kline low in bid_price
                cumul_volume=open_price,  # Carry kline open in cumul_volume (reused field)
            )

            # Feed aggregator
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

        except (ValueError, KeyError) as e:
            logger.debug(f"Failed to parse kline: {e}")

    async def _send_subscribe(self, ws, symbols: list[str]) -> None:
        """Send SUBSCRIBE request for symbols."""
        streams = [f"{sym}@kline_{self.candle_interval}" for sym in symbols]
        msg = json.dumps({
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1,
        })
        await ws.send(msg)
        logger.info(f"Subscribed: {streams}")

    async def _send_unsubscribe(self, ws, symbols: list[str]) -> None:
        """Send UNSUBSCRIBE request for symbols."""
        streams = [f"{sym}@kline_{self.candle_interval}" for sym in symbols]
        msg = json.dumps({
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": 2,
        })
        await ws.send(msg)
        logger.info(f"Unsubscribed: {streams}")

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """
        Convert symbol to lowercase Binance stream format.

        Examples:
            'BTC/USDT:USDT' → 'btcusdt'
            'BTCUSDT'       → 'btcusdt'
            'BTC/USDT'      → 'btcusdt'
        """
        s = symbol.upper()
        # Remove ccxt perpetual suffix ':USDT'
        if ":" in s:
            s = s.split(":")[0]
        # Remove slash
        s = s.replace("/", "")
        return s.lower()

    @staticmethod
    def _parse_interval_minutes(interval: str) -> int:
        """Convert interval string to integer minutes."""
        mapping = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
            "12h": 720, "1d": 1440,
        }
        return mapping.get(interval, 1)
