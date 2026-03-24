"""Alpha 8: OrderbookImbalance — CLAUDE.md 섹션 4-2.

Contrarian 방식 (IC 검증: 부호 반전 시 IC 양수):
  - bid 과잉(imb > 0) → 대형 매수벽 뒤 매도 예상 → 숏 시그널
  - ask 과잉(imb < 0) → 대형 매도벽 뒤 매수 예상 → 롱 시그널
  + 일관성 확인: 최근 5개 스냅샷 방향 일관시만 발동
  + 모멘텀 확인: 가격 방향과 반대일 때만 발동 (역추세 확인)
"""

from __future__ import annotations

import numpy as np

from src.alphas.base_alpha_v2 import AlphaSignal, BaseAlphaV2
from src.data.data_bundle import DataBundle


class OrderbookImbalance(BaseAlphaV2):

    def __init__(self):
        super().__init__(
            name="OrderbookImbalance",
            weight=0.08,
            category="contrarian",
            required_data=["orderbook_snapshots", "ohlcv_5m"],
        )
        self._prev_scores: dict[str, float] = {}

    async def compute(self, symbol: str, data: DataBundle) -> AlphaSignal:
        if not data.has_orderbook(symbol):
            return AlphaSignal()

        snapshots = data.orderbook_snapshots[symbol]
        if len(snapshots) < 3:
            return AlphaSignal()

        # ── 1. 현재 불균형 ──
        latest = snapshots[-1]
        if not latest.bids or not latest.asks:
            return AlphaSignal()

        imb_near = self._bucket_imbalance(latest.bids, latest.asks, 0, 5)
        imb_mid = self._bucket_imbalance(latest.bids, latest.asks, 5, 10)
        imb_far = self._bucket_imbalance(latest.bids, latest.asks, 10, 20)
        current_raw = 0.5 * imb_near + 0.3 * imb_mid + 0.2 * imb_far

        # ── 2. 일관성 체크: 최근 5개 스냅샷 방향 ──
        recent_snaps = snapshots[-5:]
        directions = []
        for snap in recent_snaps:
            if snap.bids and snap.asks:
                n = self._bucket_imbalance(snap.bids, snap.asks, 0, 5)
                m = self._bucket_imbalance(snap.bids, snap.asks, 5, 10)
                f = self._bucket_imbalance(snap.bids, snap.asks, 10, 20)
                snap_raw = 0.5 * n + 0.3 * m + 0.2 * f
                directions.append(np.sign(snap_raw))

        # 방향 일관성: 최근 스냅샷 중 같은 방향 비율
        if len(directions) >= 3:
            dominant_dir = np.sign(current_raw)
            agree_count = sum(1 for d in directions if d == dominant_dir)
            consistency = agree_count / len(directions)  # 0.0 ~ 1.0
        else:
            consistency = 0.0

        # 일관성 60% 미만이면 침묵 (방향이 왔다갔다)
        if consistency < 0.6:
            self._prev_scores[symbol] = 0.0
            return AlphaSignal(
                score=0.0, confidence=0.0,
                metadata={"imb_near": imb_near, "imb_mid": imb_mid,
                          "imb_far": imb_far, "consistency": consistency},
            )

        # ── 3. 모멘텀 확인: contrarian — 가격 방향과 OB 불균형이 같을 때만 ──
        # bid 과잉 + 가격 상승 중 = 매수벽 뒤에서 대량 매도 준비 → 숏 시그널
        price_momentum = 0.0
        if data.has_ohlcv_5m(symbol):
            df5 = data.ohlcv_5m[symbol]
            if len(df5) >= 7:
                closes = df5["close"].values
                price_momentum = (closes[-1] / closes[-7]) - 1 if closes[-7] != 0 else 0

        ob_direction = np.sign(current_raw)
        mom_direction = np.sign(price_momentum)

        # contrarian: OB와 모멘텀이 같은 방향(과열)일 때만 반전 시그널
        # 모멘텀 중립이거나 OB와 반대면 침묵
        if abs(price_momentum) < 0.002 or (mom_direction != 0 and ob_direction != mom_direction):
            self._prev_scores[symbol] = 0.0
            return AlphaSignal(
                score=0.0, confidence=0.0,
                metadata={"imb_near": imb_near, "imb_mid": imb_mid,
                          "imb_far": imb_far, "consistency": consistency,
                          "price_mom": price_momentum},
            )

        # ── 4. 스코어 계산 (부호 반전: bid 과잉 → 숏) ──
        score = float(-np.tanh(current_raw * 2))  # contrarian: 반전

        # 사이클 간 EMA
        prev = self._prev_scores.get(symbol, score)
        smoothed = 0.7 * score + 0.3 * prev
        self._prev_scores[symbol] = smoothed

        # confidence: 일관성 × 불균형 크기 (상한 0.6)
        confidence = consistency * min(abs(smoothed), 0.8)
        confidence = min(confidence, 0.6)
        confidence = max(confidence, 0.1)

        return AlphaSignal(
            score=smoothed,
            confidence=confidence,
            metadata={"imb_near": imb_near, "imb_mid": imb_mid,
                      "imb_far": imb_far, "consistency": consistency,
                      "price_mom": price_momentum},
        )

    @staticmethod
    def _bucket_imbalance(
        bids: list, asks: list, start: int, end: int,
    ) -> float:
        b_vol = sum(p * q for p, q in bids[start:min(end, len(bids))])
        a_vol = sum(p * q for p, q in asks[start:min(end, len(asks))])
        total = b_vol + a_vol
        return (b_vol - a_vol) / total if total > 0 else 0.0
