"""Alpha 6: DerivativesSentiment — CLAUDE.md 섹션 4-2."""

from __future__ import annotations

import numpy as np

from src.alphas.base_alpha_v2 import AlphaSignal, BaseAlphaV2
from src.data.data_bundle import DataBundle


class DerivativesSentiment(BaseAlphaV2):

    def __init__(self):
        super().__init__(
            name="DerivativesSentiment",
            weight=0.08,
            category="carry",
            required_data=["open_interest", "funding_rates", "long_short_ratio"],
        )

    async def compute(self, symbol: str, data: DataBundle) -> AlphaSignal:
        if not data.has_oi(symbol):
            return AlphaSignal()

        oi = data.open_interest[symbol]
        oi_change = oi.change_pct

        # 현재 펀딩
        funding_rate = 0.0
        if data.has_funding(symbol):
            fdf = data.funding_rates[symbol]
            if not fdf.empty:
                funding_rate = float(fdf["fundingRate"].iloc[-1])

        # 1. 스퀴즈 점수 (OI 2%+ & 펀딩 불일치 → 포지션 쏠림 감지)
        # 기존 OI 5% 임계값이 너무 높아 61%가 침묵 → 2%로 하향
        squeeze_score = 0.0
        if oi_change > 2 and funding_rate < -0.00003:
            # 숏 과열 → 롱 squeeze 기대
            squeeze_score = min(1.0, oi_change / 15 + abs(funding_rate) * 3000)
        elif oi_change > 2 and funding_rate > 0.00003:
            # 롱 과열 → 숏 squeeze 기대
            squeeze_score = -min(1.0, oi_change / 15 + abs(funding_rate) * 3000)

        # 2. 롱/숏 비율 — contrarian (군중의 반대)
        # ls > 0.5 = 롱 과잉 → 숏 유리, ls < 0.5 = 숏 과잉 → 롱 유리
        ls_adj = 0.0
        ls = data.long_short_ratio.get(symbol, 0.5)
        if ls > 0.52:
            ls_adj = float(-np.clip((ls - 0.52) / 0.18, 0, 1) * 0.5)
        elif ls < 0.48:
            ls_adj = float(np.clip((0.48 - ls) / 0.18, 0, 1) * 0.5)

        # 3. 합산
        score = squeeze_score * 0.6 + ls_adj * 0.4
        score = float(np.clip(score, -1.0, 1.0))

        confidence = 0.6 if abs(score) > 0.05 else 0.3

        return AlphaSignal(
            score=score,
            confidence=confidence,
            metadata={"oi_change": oi_change, "funding": funding_rate, "ls_ratio": ls},
        )
