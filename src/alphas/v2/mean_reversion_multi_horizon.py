"""Alpha 7: MeanReversionMultiHorizon — CLAUDE.md 섹션 4-2."""

from __future__ import annotations

import numpy as np

from src.alphas.base_alpha_v2 import AlphaSignal, BaseAlphaV2
from src.data.data_bundle import DataBundle


class MeanReversionMultiHorizon(BaseAlphaV2):

    def __init__(self):
        super().__init__(
            name="MeanReversionMultiHorizon",
            weight=0.08,
            category="mean_reversion",
            required_data=["ohlcv_1d"],
        )

    async def compute(self, symbol: str, data: DataBundle) -> AlphaSignal:
        if not data.has_ohlcv_1d(symbol):
            return AlphaSignal()

        df = data.ohlcv_1d[symbol]
        if len(df) < 121:
            return AlphaSignal()

        closes = df["close"].values

        # 1. 세 가지 Z-score
        z_short = self._zscore(closes, sma_period=3, std_period=20)
        z_mid = self._zscore(closes, sma_period=5, std_period=60)
        z_long = self._zscore(closes, sma_period=10, std_period=120)

        valid = [z for z in [z_short, z_mid, z_long] if z is not None]
        if not valid:
            return AlphaSignal()

        # 2. 평균 Z-score
        z_avg = float(np.mean(valid))

        # 3. score: Z 양수(과매수) → 숏
        score = float(-np.tanh(z_avg / 2))

        # 4. 합의 보너스 — 일봉 기반(4h 갱신) → 기본 낮게
        confidence = 0.4
        if len(valid) == 3:
            signs = [np.sign(z) for z in valid]
            if signs[0] == signs[1] == signs[2]:
                confidence = 0.55  # 3구간 합의시 올림

        # 5. 볼린저 밴드 확인
        if len(closes) >= 20:
            sma_20 = float(np.mean(closes[-20:]))
            std_20 = float(np.std(closes[-20:], ddof=1))
            if std_20 > 0:
                upper = sma_20 + 2 * std_20
                lower = sma_20 - 2 * std_20
                if closes[-1] > upper or closes[-1] < lower:
                    confidence = min(confidence + 0.15, 0.7)  # 볼밴 이탈시 추가

        return AlphaSignal(
            score=score,
            confidence=confidence,
            metadata={"z_short": z_short, "z_mid": z_mid, "z_long": z_long},
        )

    @staticmethod
    def _zscore(closes: np.ndarray, sma_period: int, std_period: int) -> float | None:
        if len(closes) < max(sma_period, std_period):
            return None
        current = float(closes[-1])
        sma = float(np.mean(closes[-sma_period:]))
        std = float(np.std(closes[-std_period:], ddof=1))
        if std < 1e-8:
            return None
        return (current - sma) / std
