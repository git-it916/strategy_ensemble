"""Alpha 8: OrderbookImbalance — CLAUDE.md 섹션 4-2."""

from __future__ import annotations

import numpy as np

from src.alphas.base_alpha_v2 import AlphaSignal, BaseAlphaV2
from src.data.data_bundle import DataBundle


class OrderbookImbalance(BaseAlphaV2):

    _EMA_ALPHA = 0.5  # 노이즈 감쇠용 (stddev 0.211 → ~0.10)

    def __init__(self):
        super().__init__(
            name="OrderbookImbalance",
            weight=0.05,
            category="micro",
            required_data=["orderbook_snapshots"],
        )
        self._prev_scores: dict[str, float] = {}  # 심볼별 EMA 상태

    async def compute(self, symbol: str, data: DataBundle) -> AlphaSignal:
        if not data.has_orderbook(symbol):
            return AlphaSignal()

        snapshots = data.orderbook_snapshots[symbol]
        if not snapshots:
            return AlphaSignal()

        # 최신 스냅샷 기준
        latest = snapshots[-1]
        bids = latest.bids
        asks = latest.asks

        if not bids or not asks:
            return AlphaSignal()

        # 1. 3구간 불균형
        imb_near = self._bucket_imbalance(bids, asks, 0, 5)
        imb_mid = self._bucket_imbalance(bids, asks, 5, 10)
        imb_far = self._bucket_imbalance(bids, asks, 10, 20)

        raw = 0.5 * imb_near + 0.3 * imb_mid + 0.2 * imb_far

        # 2. 히스토리 지수 가중 평균 (최신에 높은 가중)
        if len(snapshots) >= 3:
            # 최근 5개만 사용 (5분 히스토리), 지수 가중
            recent_snaps = snapshots[-5:]
            ema_raw = None
            decay = 0.6  # 이전 값 40% 유지
            for snap in recent_snaps:
                if snap.bids and snap.asks:
                    n = self._bucket_imbalance(snap.bids, snap.asks, 0, 5)
                    m = self._bucket_imbalance(snap.bids, snap.asks, 5, 10)
                    f = self._bucket_imbalance(snap.bids, snap.asks, 10, 20)
                    snap_raw = 0.5 * n + 0.3 * m + 0.2 * f
                    if ema_raw is None:
                        ema_raw = snap_raw
                    else:
                        ema_raw = (1 - decay) * ema_raw + decay * snap_raw
            if ema_raw is not None:
                raw = ema_raw

        # 3. score — EMA 스무딩은 히스토리에서 이미 처리, 이중 스무딩 제거
        score = float(np.tanh(raw * 3))

        # 심볼별 EMA — 사이클 간 급변동만 감쇠 (가벼운 스무딩)
        prev = self._prev_scores.get(symbol, score)
        smoothed = 0.7 * score + 0.3 * prev  # 새 값 70%, 이전 30%
        self._prev_scores[symbol] = smoothed

        # confidence: 불균형 크기에 비례 (노이즈 많은 데이터 → 강한 신호만 신뢰)
        confidence = min(abs(smoothed) * 1.5, 0.7)
        confidence = max(confidence, 0.2)

        return AlphaSignal(
            score=smoothed,
            confidence=confidence,
            metadata={"imb_near": imb_near, "imb_mid": imb_mid, "imb_far": imb_far, "raw": score},
        )

    @staticmethod
    def _bucket_imbalance(
        bids: list, asks: list, start: int, end: int,
    ) -> float:
        b_vol = sum(p * q for p, q in bids[start:min(end, len(bids))])
        a_vol = sum(p * q for p, q in asks[start:min(end, len(asks))])
        total = b_vol + a_vol
        return (b_vol - a_vol) / total if total > 0 else 0.0
