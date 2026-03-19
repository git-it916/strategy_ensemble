"""Alpha 10: VolatilityRegime — CLAUDE.md 섹션 4-2.

방향성 시그널 없음 (score=0 고정).
confidence 필드로 다른 알파의 스케일링에 사용.
"""

from __future__ import annotations

import numpy as np

from src.alphas.base_alpha_v2 import AlphaSignal, BaseAlphaV2
from src.data.data_bundle import DataBundle


class VolatilityRegime(BaseAlphaV2):

    def __init__(self):
        super().__init__(
            name="VolatilityRegime",
            weight=0.03,
            category="auxiliary",
            required_data=["ohlcv_1d", "ohlcv_1h"],
        )

    async def compute(self, symbol: str, data: DataBundle) -> AlphaSignal:
        # 아무 심볼이나 하나로 시장 전체 변동성 측정 (BTC 우선)
        ref_symbol = symbol
        for s in data.universe:
            if "BTC" in s:
                ref_symbol = s
                break

        if not data.has_ohlcv_1d(ref_symbol):
            return AlphaSignal(score=0.0, confidence=1.0)

        df = data.ohlcv_1d[ref_symbol]
        if len(df) < 90:
            return AlphaSignal(score=0.0, confidence=1.0)

        closes = df["close"].values

        # 1. 20일 실현 변동성
        daily_rets = np.diff(closes[-21:]) / closes[-21:-1]
        vol_20d = float(np.std(daily_rets, ddof=1) * np.sqrt(365))

        # 2. 최근 90일 롤링 변동성으로 백분위 계산
        rolling_vols = []
        tail = closes[-90:] if len(closes) > 90 else closes
        for i in range(20, len(tail)):
            r = np.diff(tail[i-20:i+1]) / tail[i-20:i]
            rolling_vols.append(float(np.std(r, ddof=1) * np.sqrt(365)))

        if not rolling_vols:
            return AlphaSignal(score=0.0, confidence=1.0)

        percentile = float(np.mean([1 for v in rolling_vols if vol_20d > v]) / len(rolling_vols) * 100)

        # 3. 레짐 판단
        if percentile < 70:
            regime_confidence = 1.0
        elif percentile < 90:
            regime_confidence = 0.7
        else:
            regime_confidence = 0.4

        # 4. 6시간 급등 감지
        if data.has_ohlcv_1h(ref_symbol):
            hdf = data.ohlcv_1h[ref_symbol]
            if len(hdf) >= 6:
                hourly_rets = np.diff(hdf["close"].values[-7:]) / hdf["close"].values[-7:-1]
                vol_6h = float(np.std(hourly_rets, ddof=1) * np.sqrt(8760))
                if vol_6h > vol_20d * 2:
                    regime_confidence *= 0.5

        regime_confidence = max(0.3, min(1.0, regime_confidence))

        return AlphaSignal(
            score=0.0,  # 방향성 없음
            confidence=regime_confidence,
            metadata={"vol_20d": vol_20d, "percentile": percentile},
        )
