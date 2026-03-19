"""
SignalAggregator — CLAUDE.md 섹션 5 기준.

가중합 모드 (기본) + Stacking 모드 (60일 이후 자동 전환).
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from config.settings import ALPHA_WEIGHTS
from src.alphas.base_alpha_v2 import AlphaSignal

logger = logging.getLogger(__name__)


class SignalAggregator:
    """
    앙상블: 10개 알파 시그널 → 심볼별 최종 스코어.

    Mode 1 (가중합): score × confidence × weight → 정규화 → vol_regime → EMA 스무딩
    Mode 2 (Stacking): StackingMetaModel.predict() → 60일 이후 자동 전환
    """

    EMA_ALPHA = 0.4  # 스무딩 계수 (0.4 = 새 값 40%, 이전 60%)

    def __init__(self):
        self._stacking = None
        self._prev_scores: Dict[str, float] = {}  # EMA 스무딩용 이전 스코어
        self._raw_scores: Dict[str, float] = {}   # EMA 전 원본 스코어 (로깅용)

    def aggregate(
        self,
        signals: Dict[str, Dict[str, AlphaSignal]],
    ) -> Dict[str, float]:
        """
        앙상블 스코어 계산.

        Args:
            signals: {symbol: {alpha_name: AlphaSignal}}

        Returns:
            {symbol: ensemble_score}  — -1.0 ~ +1.0
        """
        # 항상 가중합 사용 (Stacking 비활성화)
        return self._aggregate_weighted(signals)

    def _aggregate_weighted(
        self,
        signals: Dict[str, Dict[str, AlphaSignal]],
    ) -> Dict[str, float]:
        """CLAUDE.md 섹션 5-1 가중합."""
        result: Dict[str, float] = {}

        for symbol, alpha_signals in signals.items():
            raw = 0.0
            total_weight = 0.0

            # VolatilityRegime confidence 추출
            vol_conf = 1.0
            vol_sig = alpha_signals.get("VolatilityRegime")
            if vol_sig:
                vol_conf = vol_sig.confidence

            for alpha_name, sig in alpha_signals.items():
                if alpha_name == "VolatilityRegime":
                    continue
                w = ALPHA_WEIGHTS.get(alpha_name, 0)
                if w <= 0 or sig.confidence <= 0:
                    continue
                if np.isnan(sig.score) or np.isnan(sig.confidence):
                    continue
                raw += sig.score * sig.confidence * w
                total_weight += w  # confidence는 분자(score)에만 적용, 분모에서 제거

            if total_weight > 0:
                normalized = raw / total_weight
            else:
                normalized = 0.0

            # vol_conf 최소 0.5 — 한 알파가 전체 시그널을 죽이는 것 방지
            vol_conf_capped = max(vol_conf, 0.5)
            final = normalized * vol_conf_capped
            if np.isnan(final):
                final = 0.0

            # 원본 스코어 저장 (로깅/백테스트용)
            self._raw_scores[symbol] = float(np.clip(final, -1.0, 1.0))

            # 앙상블 레벨 스무딩 없음 — 노이즈 알파는 개별 EMA 적용
            result[symbol] = float(np.clip(final, -1.0, 1.0))

        return result

    def _aggregate_stacking(
        self,
        signals: Dict[str, Dict[str, AlphaSignal]],
    ) -> Dict[str, float]:
        """Stacking 메타모델 사용."""
        result: Dict[str, float] = {}

        for symbol, alpha_signals in signals.items():
            features = {}
            for alpha_name, sig in alpha_signals.items():
                features[f"{alpha_name}_score"] = sig.score
                features[f"{alpha_name}_conf"] = sig.confidence

            pred = self._stacking.predict_one(features)
            result[symbol] = float(np.clip(pred, -1.0, 1.0))

        return result

    def set_stacking(self, model) -> None:
        """Stacking 모델 설정."""
        self._stacking = model
        logger.info("Stacking model loaded into aggregator")
