"""Alpha 5: IntradayRSIV2 — CLAUDE.md 섹션 4-2."""

from __future__ import annotations

import numpy as np

from src.alphas.base_alpha_v2 import AlphaSignal, BaseAlphaV2
from src.data.data_bundle import DataBundle


class IntradayRSIV2(BaseAlphaV2):

    def __init__(self):
        super().__init__(
            name="IntradayRSIV2",
            weight=0.08,
            category="mean_reversion",
            required_data=["ohlcv_1h"],
        )

    async def compute(self, symbol: str, data: DataBundle) -> AlphaSignal:
        if not data.has_ohlcv_1h(symbol):
            return AlphaSignal()

        df = data.ohlcv_1h[symbol]
        if len(df) < 15:
            return AlphaSignal()

        closes = df["close"].values

        # Wilder RSI (14기간)
        rsi = self._compute_rsi(closes, 14)
        if rsi is None:
            return AlphaSignal()

        # 시그널
        if rsi > 70:
            score = -(rsi - 70) / 30  # 최대 -1.0
        elif rsi < 30:
            score = (30 - rsi) / 30   # 최대 +1.0
        else:
            score = (50 - rsi) / 100  # 약한 시그널

        confidence = min(abs(score) * 1.5, 1.0)

        return AlphaSignal(
            score=float(score),
            confidence=confidence,
            metadata={"rsi": rsi},
        )

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int) -> float | None:
        if len(closes) < period + 1:
            return None
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
