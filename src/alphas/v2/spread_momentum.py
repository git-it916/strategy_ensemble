"""Alpha 9: SpreadMomentum — CLAUDE.md 섹션 4-2."""

from __future__ import annotations

import numpy as np

from src.alphas.base_alpha_v2 import AlphaSignal, BaseAlphaV2
from src.data.data_bundle import DataBundle


class SpreadMomentum(BaseAlphaV2):

    def __init__(self):
        super().__init__(
            name="SpreadMomentum",
            weight=0.03,
            category="micro",
            required_data=["orderbook_snapshots", "ohlcv_5m"],
        )

    async def compute(self, symbol: str, data: DataBundle) -> AlphaSignal:
        if not data.has_orderbook(symbol):
            return AlphaSignal()

        snapshots = data.orderbook_snapshots[symbol]
        if len(snapshots) < 10:
            return AlphaSignal()

        # 1. 스프레드 변화
        current_spread = snapshots[-1].spread_bps
        past_spread = snapshots[-10].spread_bps  # ~10분 전
        if past_spread <= 0:
            return AlphaSignal()

        spread_change = (current_spread - past_spread) / past_spread

        # 2. 가격 방향
        price_dir = 0.0
        if data.has_ohlcv_5m(symbol):
            df5 = data.ohlcv_5m[symbol]
            if len(df5) >= 2:
                ret = (df5["close"].iloc[-1] / df5["close"].iloc[-2]) - 1
                price_dir = float(np.sign(ret))
        elif len(snapshots) >= 2:
            if snapshots[-10].mid_price > 0:
                price_dir = float(np.sign(
                    snapshots[-1].mid_price / snapshots[-10].mid_price - 1
                ))

        # 3. 시그널
        score = float(-spread_change * price_dir)
        score = float(np.clip(score, -1.0, 1.0))

        return AlphaSignal(
            score=score,
            confidence=0.3,
            metadata={"spread_change": spread_change, "price_dir": price_dir},
        )
