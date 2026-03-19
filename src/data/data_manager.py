"""
DataManager — CLAUDE.md 섹션 3-1 기준.

바이낸스 ccxt async로 데이터 수집/갱신.
심볼별 Dict로 저장, to_bundle()로 DataBundle 생성.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List

import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd

from config.settings import (
    BINANCE_RATE_LIMIT_SLEEP,
    BLACKLIST,
    COIN_WHITELIST,
    MIN_24H_VOLUME_USDT,
    UNIVERSE_SIZE,
)
from src.data.data_bundle import DataBundle, OIData, OrderbookSnapshot

logger = logging.getLogger(__name__)


class DataManager:
    """
    비동기 데이터 수집/관리.

    수집 주기:
        - 일봉(1d) + 펀딩: 4시간
        - 1시간봉(1h) + 5분봉(5m): 5분
        - 오더북: 60초
        - OI + L/S 비율: 5분
        - 유니버스: 1시간
    """

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self._exchange = ccxt_async.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
                "recvWindow": 60000,
            },
            "enableRateLimit": True,
        })

        # 심볼별 데이터 저장
        self._ohlcv_1d: Dict[str, pd.DataFrame] = {}
        self._ohlcv_1h: Dict[str, pd.DataFrame] = {}
        self._ohlcv_5m: Dict[str, pd.DataFrame] = {}
        self._funding: Dict[str, pd.DataFrame] = {}
        self._orderbook_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=30))
        self._oi: Dict[str, OIData] = {}
        self._ls_ratio: Dict[str, float] = {}

        self.universe: List[str] = []

        # 타임스탬프
        self._last_daily_refresh: datetime | None = None
        self._last_universe_refresh: datetime | None = None

    async def close(self):
        """거래소 연결 종료."""
        await self._exchange.close()

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    async def refresh_universe(self) -> List[str]:
        """거래량 상위 N개 심볼 선별. CLAUDE.md 섹션 3-3."""
        try:
            self._sync_time()
            markets = await self._exchange.load_markets()
            tickers = await self._exchange.fetch_tickers()

            candidates = []
            for symbol, info in tickers.items():
                if not symbol.endswith("/USDT:USDT"):
                    continue
                base = symbol.split("/")[0]
                # 화이트리스트에 없으면 제외 (펌프 코인 방지)
                if base not in COIN_WHITELIST:
                    continue
                if base in BLACKLIST:
                    continue
                vol_24h = float(info.get("quoteVolume", 0) or 0)
                if vol_24h < MIN_24H_VOLUME_USDT:
                    continue
                candidates.append((symbol, vol_24h))

            candidates.sort(key=lambda x: x[1], reverse=True)
            self.universe = [s for s, _ in candidates[:UNIVERSE_SIZE]]
            self._last_universe_refresh = datetime.now(timezone.utc)
            logger.info(f"Universe refreshed: {len(self.universe)} symbols")
            return self.universe

        except Exception as e:
            logger.error(f"Universe refresh failed: {e}")
            return self.universe

    # ------------------------------------------------------------------
    # Initial fetch
    # ------------------------------------------------------------------

    def _sync_time(self):
        """시간 동기화는 30초 백그라운드 태스크에서 처리. 여기선 no-op."""
        pass

    async def initial_fetch(self):
        """최초 데이터 수집 (시작 시 1회)."""
        logger.info("Initial data fetch...")
        self._sync_time()
        await self.refresh_universe()
        await self._fetch_ohlcv_all("1d", 300, self._ohlcv_1d)
        await self._fetch_ohlcv_all("1h", 168, self._ohlcv_1h)   # 7일
        await self._fetch_ohlcv_all("5m", 1440, self._ohlcv_5m)  # 5일
        await self._fetch_funding_all()
        await self._fetch_oi_all()
        await self._fetch_ls_ratio_all()
        self._last_daily_refresh = datetime.now(timezone.utc)
        logger.info("Initial fetch complete")

    # ------------------------------------------------------------------
    # Periodic refresh
    # ------------------------------------------------------------------

    async def refresh_intraday(self):
        """5분마다: 1h + 5m 갱신 + OI + L/S."""
        self._sync_time()
        await self._fetch_ohlcv_all("1h", 168, self._ohlcv_1h)
        await self._fetch_ohlcv_all("5m", 1440, self._ohlcv_5m)
        await self._fetch_oi_all()
        await self._fetch_ls_ratio_all()

    async def refresh_daily(self):
        """4시간마다: 일봉 + 펀딩."""
        self._sync_time()
        await self._fetch_ohlcv_all("1d", 300, self._ohlcv_1d)
        await self._fetch_funding_all()
        self._last_daily_refresh = datetime.now(timezone.utc)

    async def refresh_orderbooks(self):
        """60초마다: 유니버스 전체 오더북 스냅샷."""
        for symbol in self.universe:
            try:
                book = await self._exchange.fetch_order_book(symbol, limit=20)
                snap = self._parse_orderbook(book)
                self._orderbook_history[symbol].append(snap)
            except Exception as e:
                logger.debug(f"Orderbook fetch failed {symbol}: {e}")

    # ------------------------------------------------------------------
    # to_bundle
    # ------------------------------------------------------------------

    def to_bundle(self) -> DataBundle:
        """현재 데이터로 DataBundle 생성."""
        ob_snaps = {
            s: list(dq) for s, dq in self._orderbook_history.items()
        }
        return DataBundle(
            ohlcv_1d=dict(self._ohlcv_1d),
            ohlcv_1h=dict(self._ohlcv_1h),
            ohlcv_5m=dict(self._ohlcv_5m),
            funding_rates=dict(self._funding),
            orderbook_snapshots=ob_snaps,
            open_interest=dict(self._oi),
            long_short_ratio=dict(self._ls_ratio),
            universe=list(self.universe),
            timestamp=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Internal fetchers
    # ------------------------------------------------------------------

    async def _fetch_ohlcv_all(
        self, timeframe: str, limit: int, store: Dict[str, pd.DataFrame],
    ):
        """유니버스 전체 OHLCV 수집."""
        for symbol in self.universe:
            try:
                raw = await self._exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, limit=limit,
                )
                if raw:
                    df = pd.DataFrame(
                        raw, columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    store[symbol] = df
            except Exception as e:
                logger.debug(f"OHLCV {timeframe} fetch failed {symbol}: {e}")

    async def _fetch_funding_all(self):
        """유니버스 전체 펀딩비율 수집 (90일 ≈ 270개 8h 구간)."""
        for symbol in self.universe:
            try:
                raw = await self._exchange.fetch_funding_rate_history(
                    symbol, limit=270,
                )
                if raw:
                    rows = [
                        {"timestamp": r["timestamp"], "fundingRate": r["fundingRate"]}
                        for r in raw if r.get("fundingRate") is not None
                    ]
                    df = pd.DataFrame(rows)
                    if not df.empty:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                        self._funding[symbol] = df
            except Exception as e:
                logger.debug(f"Funding fetch failed {symbol}: {e}")

    async def _fetch_oi_all(self):
        """유니버스 전체 OI 수집."""
        for symbol in self.universe:
            try:
                # ccxt에서 OI 직접 지원 안 할 수 있으므로 fapiPublic 사용
                base = symbol.replace("/USDT:USDT", "USDT")
                resp = await self._exchange.fapiPublicGetOpenInterest({"symbol": base})
                current_oi = float(resp.get("openInterest", 0))
                # 24h 전 OI는 근사: 저장된 이전값 사용
                prev = self._oi.get(symbol)
                oi_24h_ago = prev.current_oi if prev else current_oi
                self._oi[symbol] = OIData(
                    current_oi=current_oi,
                    oi_24h_ago=oi_24h_ago,
                )
            except Exception as e:
                logger.debug(f"OI fetch failed {symbol}: {e}")

    async def _fetch_ls_ratio_all(self):
        """유니버스 전체 롱/숏 비율."""
        import aiohttp
        for symbol in self.universe:
            try:
                base = symbol.replace("/USDT:USDT", "USDT")
                url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={base}&period=5m&limit=1"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data and len(data) > 0:
                                ratio = float(data[0].get("longAccount", 0.5))
                                self._ls_ratio[symbol] = ratio
            except Exception as e:
                logger.debug(f"LS ratio fetch failed {symbol}: {e}")

    @staticmethod
    def _parse_orderbook(book: dict) -> OrderbookSnapshot:
        """ccxt 오더북 → OrderbookSnapshot."""
        bids = [(float(b[0]), float(b[1])) for b in (book.get("bids") or [])]
        asks = [(float(a[0]), float(a[1])) for a in (book.get("asks") or [])]

        mid = 0.0
        spread_bps = 0.0
        if bids and asks:
            best_bid, best_ask = bids[0][0], asks[0][0]
            mid = (best_bid + best_ask) / 2
            spread_bps = ((best_ask - best_bid) / mid * 10000) if mid > 0 else 0

        bid_depth = sum(p * q for p, q in bids)
        ask_depth = sum(p * q for p, q in asks)

        return OrderbookSnapshot(
            timestamp=datetime.now(timezone.utc),
            bids=bids,
            asks=asks,
            mid_price=mid,
            spread_bps=spread_bps,
            bid_depth_usdt=bid_depth,
            ask_depth_usdt=ask_depth,
        )
