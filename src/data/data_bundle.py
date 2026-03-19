"""
DataBundle — CLAUDE.md 섹션 3-2 기준.

모든 알파에게 전달되는 불변 데이터 컨테이너.
각 필드는 Dict[str, ...] 형태로 심볼별 데이터를 담는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


@dataclass
class OrderbookSnapshot:
    """단일 오더북 스냅샷."""

    timestamp: datetime
    bids: List[tuple[float, float]]  # (price, qty) — 상위 20레벨
    asks: List[tuple[float, float]]
    mid_price: float = 0.0
    spread_bps: float = 0.0
    bid_depth_usdt: float = 0.0
    ask_depth_usdt: float = 0.0


@dataclass
class OIData:
    """미결제약정(OI) 데이터."""

    current_oi: float = 0.0
    oi_24h_ago: float = 0.0

    @property
    def change_pct(self) -> float:
        if self.oi_24h_ago <= 0:
            return 0.0
        return (self.current_oi - self.oi_24h_ago) / self.oi_24h_ago * 100


@dataclass(frozen=True)
class DataBundle:
    """
    불변 데이터 번들.

    모든 OHLCV 필드: Dict[str, pd.DataFrame]
        key = 심볼 (예: "ETH/USDT:USDT")
        value = DataFrame[columns: timestamp, open, high, low, close, volume]

    funding_rates: Dict[str, pd.DataFrame]
        columns: [timestamp, fundingRate]

    orderbook_snapshots: Dict[str, List[OrderbookSnapshot]]
        최근 30분 스냅샷 히스토리

    open_interest: Dict[str, OIData]
    long_short_ratio: Dict[str, float]  # 0.0~1.0 (롱 비율)
    universe: List[str]
    timestamp: datetime
    """

    ohlcv_1d: Dict[str, pd.DataFrame] = field(default_factory=dict)
    ohlcv_1h: Dict[str, pd.DataFrame] = field(default_factory=dict)
    ohlcv_5m: Dict[str, pd.DataFrame] = field(default_factory=dict)
    funding_rates: Dict[str, pd.DataFrame] = field(default_factory=dict)
    orderbook_snapshots: Dict[str, List[OrderbookSnapshot]] = field(default_factory=dict)
    open_interest: Dict[str, OIData] = field(default_factory=dict)
    long_short_ratio: Dict[str, float] = field(default_factory=dict)
    universe: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def has_ohlcv_1d(self, symbol: str) -> bool:
        return symbol in self.ohlcv_1d and not self.ohlcv_1d[symbol].empty

    def has_ohlcv_1h(self, symbol: str) -> bool:
        return symbol in self.ohlcv_1h and not self.ohlcv_1h[symbol].empty

    def has_ohlcv_5m(self, symbol: str) -> bool:
        return symbol in self.ohlcv_5m and not self.ohlcv_5m[symbol].empty

    def has_funding(self, symbol: str) -> bool:
        return symbol in self.funding_rates and not self.funding_rates[symbol].empty

    def has_orderbook(self, symbol: str) -> bool:
        return symbol in self.orderbook_snapshots and len(self.orderbook_snapshots[symbol]) > 0

    def has_oi(self, symbol: str) -> bool:
        return symbol in self.open_interest
