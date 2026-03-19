"""Alpha 1: MomentumMultiScale — CLAUDE.md 섹션 4-2."""

from __future__ import annotations

import numpy as np

from src.alphas.base_alpha_v2 import AlphaSignal, BaseAlphaV2
from src.data.data_bundle import DataBundle


class MomentumMultiScale(BaseAlphaV2):

    _EMA_ALPHA = 0.5  # 노이즈 감쇠용 (stddev 0.276 → ~0.14)

    def __init__(self):
        super().__init__(
            name="MomentumMultiScale",
            weight=0.22,
            category="momentum",
            required_data=["ohlcv_5m"],
        )
        self._prev_scores: dict[str, float] = {}  # 심볼별 EMA 상태

    async def compute(self, symbol: str, data: DataBundle) -> AlphaSignal:
        if not data.has_ohlcv_5m(symbol):
            return AlphaSignal()

        df = data.ohlcv_5m[symbol]
        if len(df) < 37:
            return AlphaSignal()

        closes = df["close"].values
        volumes = df["volume"].values

        # 1. 멀티스케일 수익률
        ret_30m = (closes[-1] / closes[-7]) - 1 if closes[-7] != 0 else 0    # 6봉
        ret_90m = (closes[-1] / closes[-19]) - 1 if closes[-19] != 0 else 0  # 18봉
        ret_180m = (closes[-1] / closes[-37]) - 1 if closes[-37] != 0 else 0 # 36봉

        weighted_return = 0.4 * ret_30m + 0.35 * ret_90m + 0.25 * ret_180m

        # 2. 변동성 정규화 (5분봉 288개 = 1일, 근사로 ::288 사용)
        bars_per_day = 288  # 24h × 60min / 5min
        daily_closes = closes[::bars_per_day] if len(closes) > bars_per_day * 2 else closes[::48]  # 48봉=4시간 폴백
        if len(daily_closes) > 2:
            rets = np.diff(daily_closes) / daily_closes[:-1]
            vol_5d = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.03
        else:
            vol_5d = 0.03
        vol_5d = max(vol_5d, 0.001)

        score = float(np.tanh(weighted_return / vol_5d)) * 0.5  # 스케일 정규화

        # 3. 거래량 확인
        confidence = 0.8
        if len(volumes) >= 60:
            recent_vol = float(np.mean(volumes[-6:]))
            baseline_vol = float(np.mean(volumes))
            if baseline_vol > 0:
                vol_ratio = recent_vol / baseline_vol
                if vol_ratio < 0.5:
                    confidence *= 0.5
                elif vol_ratio > 2.0:
                    confidence = min(confidence * 1.2, 1.0)

        # 4. 가속도
        accel = ret_30m - ret_90m
        if np.sign(accel) == np.sign(score):
            score += 0.1 * accel

        score = float(np.clip(score, -1.0, 1.0))

        # 알파 내부 EMA — 사이클간 급변동 감쇠
        prev = self._prev_scores.get(symbol, score)
        smoothed = self._EMA_ALPHA * score + (1 - self._EMA_ALPHA) * prev
        self._prev_scores[symbol] = smoothed

        return AlphaSignal(
            score=smoothed,
            confidence=confidence,
            metadata={"ret_30m": ret_30m, "ret_90m": ret_90m, "ret_180m": ret_180m, "raw": score},
        )
