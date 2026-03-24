"""Alpha 1: MomentumMultiScale — CLAUDE.md 섹션 4-2."""

from __future__ import annotations

import numpy as np

from config.settings import VOL_FILTER_BASELINE_BARS
from src.alphas.base_alpha_v2 import AlphaSignal, BaseAlphaV2
from src.data.data_bundle import DataBundle


class MomentumMultiScale(BaseAlphaV2):

    _EMA_ALPHA = 0.5  # 노이즈 감쇠용 (stddev 0.276 → ~0.14)

    def __init__(self):
        super().__init__(
            name="MomentumMultiScale",
            weight=0.22,
            category="momentum",
            required_data=["ohlcv_5m", "ohlcv_1d"],
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

        # 2. 변동성 정규화 — 일봉 데이터에서 직접 가져옴 (안정적)
        vol_5d = 0.03  # 기본값
        if data.has_ohlcv_1d(symbol):
            ddf = data.ohlcv_1d[symbol]
            if len(ddf) >= 6:
                daily_closes = ddf["close"].values[-6:]
                daily_rets = np.diff(daily_closes) / daily_closes[:-1]
                vol_5d = float(np.std(daily_rets, ddof=1)) if len(daily_rets) > 1 else 0.03
        vol_5d = max(vol_5d, 0.001)

        score = float(np.tanh(weighted_return / vol_5d))  # [-1, 1] 통일 스케일

        # 3. 거래량 확인
        confidence = 0.8
        if len(volumes) >= 12:
            recent_vol = float(np.mean(volumes[-6:]))
            baseline_window = min(len(volumes), VOL_FILTER_BASELINE_BARS)
            baseline_vol = float(np.mean(volumes[-baseline_window:]))
            if baseline_vol > 0:
                vol_ratio = recent_vol / baseline_vol
                if vol_ratio < 0.5:
                    confidence *= 0.5
                elif vol_ratio > 2.0:
                    confidence = min(confidence * 1.2, 1.0)

        # 4. 가속도 보너스
        accel = ret_30m - ret_90m
        if np.sign(accel) == np.sign(score):
            score += 0.15 * accel

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
