"""Alpha 4: IntradayVWAPV2 — CLAUDE.md 섹션 4-2."""

from __future__ import annotations

import numpy as np

from src.alphas.base_alpha_v2 import AlphaSignal, BaseAlphaV2
from src.data.data_bundle import DataBundle


class IntradayVWAPV2(BaseAlphaV2):

    def __init__(self):
        super().__init__(
            name="IntradayVWAPV2",
            weight=0.10,
            category="mean_reversion",
            required_data=["ohlcv_1h"],
        )

    async def compute(self, symbol: str, data: DataBundle) -> AlphaSignal:
        if not data.has_ohlcv_1h(symbol):
            return AlphaSignal()

        df = data.ohlcv_1h[symbol]
        if len(df) < 24:
            return AlphaSignal()

        recent = df.tail(24)
        closes = recent["close"].values
        volumes = recent["volume"].values

        total_vol = volumes.sum()
        if total_vol == 0:
            return AlphaSignal()

        # 1. VWAP
        vwap = float((closes * volumes).sum() / total_vol)
        current_price = float(closes[-1])
        if vwap == 0:
            return AlphaSignal()

        # 2. Z-score
        std_24h = float(np.std(closes, ddof=1))
        if std_24h < 1e-8:
            return AlphaSignal()

        z = (current_price - vwap) / std_24h

        # 3. 시그널 — mean reversion (VWAP 이탈시 복귀 방향)
        if abs(z) < 0.8:
            score = 0.0
        elif z > 0:
            score = float(-np.tanh((z - 0.8) / 1.5))
        else:
            score = float(np.tanh((-z - 0.8) / 1.5))

        # 4. confidence: z 크기에 비례 + 추세 필터
        confidence = min(abs(z) / 3.0, 0.8)  # z=1.5→0.5, z=2.4→0.8
        confidence = max(confidence, 0.2)     # 최소 0.2
        if len(closes) >= 4:
            ret_4h = (closes[-1] / closes[-4]) - 1
            if abs(ret_4h) > 0.02:
                confidence *= 0.5

        return AlphaSignal(
            score=score,
            confidence=confidence,
            metadata={"vwap": vwap, "z": z},
        )
