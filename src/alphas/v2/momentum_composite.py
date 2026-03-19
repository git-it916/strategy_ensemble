"""Alpha 3: MomentumComposite — CLAUDE.md 섹션 4-2."""

from __future__ import annotations

import numpy as np

from src.alphas.base_alpha_v2 import AlphaSignal, BaseAlphaV2
from src.data.data_bundle import DataBundle


class MomentumComposite(BaseAlphaV2):

    def __init__(self):
        super().__init__(
            name="MomentumComposite",
            weight=0.15,
            category="momentum",
            required_data=["ohlcv_1d"],
        )

    async def compute(self, symbol: str, data: DataBundle) -> AlphaSignal:
        if not data.has_ohlcv_1d(symbol):
            return AlphaSignal()

        df = data.ohlcv_1d[symbol]
        if len(df) < 21:
            return AlphaSignal()

        closes = df["close"].values

        # 1. 절대 모멘텀: 20일 수익률 zscore
        ret_20d = (closes[-1] / closes[-21]) - 1 if closes[-21] != 0 else 0
        daily_rets = np.diff(closes[-21:]) / closes[-21:-1]
        vol_20d = float(np.std(daily_rets, ddof=1)) if len(daily_rets) > 1 else 0.03
        vol_20d = max(vol_20d, 0.001)
        abs_mom = float(np.tanh(ret_20d / vol_20d))

        # 2. 상대 모멘텀: universe 내 순위
        all_rets = {}
        for sym in data.universe:
            if data.has_ohlcv_1d(sym):
                sdf = data.ohlcv_1d[sym]
                if len(sdf) >= 21 and sdf["close"].iloc[-21] != 0:
                    all_rets[sym] = (sdf["close"].iloc[-1] / sdf["close"].iloc[-21]) - 1

        if len(all_rets) >= 3 and symbol in all_rets:
            sorted_syms = sorted(all_rets, key=lambda s: all_rets[s])
            rank = sorted_syms.index(symbol)
            n = len(sorted_syms)
            rank_score = (2 * rank / (n - 1) - 1) if n > 1 else 0.0  # [-1, +1] 대칭
        else:
            rank_score = 0.0

        # 3. 단기 모멘텀 (5일) — abs_mom(20일)과 독립적인 시간대
        if len(closes) >= 6 and closes[-6] != 0:
            ret_5d = (closes[-1] / closes[-6]) - 1
            short_rets = np.diff(closes[-6:]) / closes[-6:-1]
            vol_5d = float(np.std(short_rets, ddof=1)) if len(short_rets) > 1 else vol_20d
            vol_5d = max(vol_5d, 0.001)
            mom_5d = float(np.tanh(ret_5d / vol_5d))
        else:
            mom_5d = 0.0

        # 4. 합산 (×0.5 스케일 정규화 — 다른 알파와 magnitude 맞춤)
        score = (0.4 * abs_mom + 0.3 * rank_score + 0.3 * mom_5d) * 0.5
        score = float(np.clip(score, -1.0, 1.0))

        # confidence: 일봉 기반(4h 갱신) → 기본 낮게, 구성요소 합의시 올림
        confidence = 0.35
        if np.sign(abs_mom) == np.sign(rank_score) == np.sign(mom_5d):
            confidence = 0.50  # 3요소 방향 일치
        # 20일과 5일 방향이 다르면 추세 전환 가능성 → 낮게
        if np.sign(abs_mom) != np.sign(mom_5d) and abs(mom_5d) > 0.3:
            confidence = 0.25

        return AlphaSignal(
            score=score,
            confidence=confidence,
            metadata={"abs_mom": abs_mom, "rank_score": rank_score, "mom_5d": mom_5d},
        )
