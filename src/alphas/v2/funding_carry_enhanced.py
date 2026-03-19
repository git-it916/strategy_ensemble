"""Alpha 2: FundingCarryEnhanced — CLAUDE.md 섹션 4-2."""

from __future__ import annotations

import numpy as np

from src.alphas.base_alpha_v2 import AlphaSignal, BaseAlphaV2
from src.data.data_bundle import DataBundle


class FundingCarryEnhanced(BaseAlphaV2):

    def __init__(self):
        super().__init__(
            name="FundingCarryEnhanced",
            weight=0.18,
            category="carry",
            required_data=["funding_rates", "open_interest", "ohlcv_1d"],
        )

    async def compute(self, symbol: str, data: DataBundle) -> AlphaSignal:
        if not data.has_funding(symbol):
            return AlphaSignal()

        fdf = data.funding_rates[symbol]
        if len(fdf) < 21:  # 최소 7일치
            return AlphaSignal()

        rates = fdf["fundingRate"].values

        # 1. 14일 평균 (42개 8h 구간)
        avg_funding = float(np.mean(rates[-42:])) if len(rates) >= 42 else float(np.mean(rates))

        # 2. 기본 점수: 펀딩 음수 → 롱 유리
        # tanh로 부드러운 포화 — avg_funding=±0.0003에서 ±0.71, ±0.0005에서 ±0.88
        base_score = float(np.tanh(-avg_funding * 2500))

        # confidence: 펀딩은 8h마다 변동, 일봉 4h 갱신 → 기본 낮게
        confidence = 0.5

        # 3. 펀딩 속도 — 포화 전에 적용해서 실제 효과 발생
        velocity_bonus = 0.0
        if len(rates) >= 42:
            recent_7d = float(np.mean(rates[-21:]))
            prev_7d = float(np.mean(rates[-42:-21]))
            velocity = recent_7d - prev_7d
            if np.sign(velocity) == np.sign(-avg_funding):
                # 가속 중이면 보너스 (최대 ±0.15)
                velocity_bonus = float(np.tanh(abs(velocity) * 10000)) * 0.15 * np.sign(base_score)
        base_score = float(np.clip(base_score + velocity_bonus, -1.0, 1.0))

        # 4. 군중 할인
        if data.has_oi(symbol):
            oi = data.open_interest[symbol]
            if oi.change_pct > 50 and abs(avg_funding) > 0.0005:
                confidence *= 0.5

        # 5. 추세 확인
        if data.has_ohlcv_1d(symbol):
            ddf = data.ohlcv_1d[symbol]
            if len(ddf) >= 3:
                ret_3d = (ddf["close"].iloc[-1] / ddf["close"].iloc[-3]) - 1
                if np.sign(ret_3d) == np.sign(base_score) and abs(ret_3d) > 0.03:
                    confidence *= 0.6

        base_score = float(np.clip(base_score, -1.0, 1.0))

        return AlphaSignal(
            score=base_score,
            confidence=confidence,
            metadata={"avg_funding": avg_funding},
        )
