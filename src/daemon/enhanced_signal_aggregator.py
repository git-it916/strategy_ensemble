"""
Enhanced Signal Aggregator

3-stage signal aggregation:
    1. Confidence-weighted per-alpha scores (no double-weighting)
    2. Category-based combination with intra-category correlation dampening
    3. Regime multipliers + vol confidence + signal decay

Weight flow (피드백 #1 해결):
    - 개별 알파 가중치(alpha_weights)만으로 1차 가중합 수행
    - 카테고리 가중치(CATEGORY_WEIGHTS)는 카테고리 간 상대적 조정에만 사용
    - 이중 곱셈 방지: raw_score × alpha_weight → category_sum → regime 조정
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EnhancedAggregatedSignal:
    """Result of enhanced signal aggregation."""

    scores: dict[str, float]
    effective_scores: dict[str, float]
    contributions: dict[str, dict[str, float]]
    weights_used: dict[str, float]
    raw_scores: dict[str, dict[str, float]]
    confidence: dict[str, float] = field(default_factory=dict)
    agreement: dict[str, tuple[int, int, str]] = field(default_factory=dict)
    vol_confidence: float = 1.0
    # Stacking meta-model 결과
    stacking_scores: dict[str, float] = field(default_factory=dict)
    stacking_contributions: dict[str, dict[str, float]] = field(default_factory=dict)
    ensemble_method: str = "weighted_sum"  # "weighted_sum" or "stacking"


# 카테고리 비율 — 카테고리 간 균형 조정에만 사용
# 개별 알파 weight와 이중으로 곱하지 않음
CATEGORY_TARGET_RATIO = {
    "momentum": 0.37,
    "mean_reversion": 0.25,
    "carry": 0.26,
    "microstructure": 0.12,
}


class EnhancedSignalAggregator:
    """
    Signal aggregator with dual mode:
        1. Weighted sum (fallback / cold start)
        2. Stacking meta-model (학습 데이터 충분 시)

    Stacking이 학습되면 자동 전환, 미학습 시 가중합 fallback.
    두 결과를 모두 계산해서 Sonnet에 함께 전달.
    """

    def __init__(self):
        self._ic_history: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        self._dynamic_adjustments: dict[str, float] = {}
        self._ic_tracking_enabled = False
        self._last_signal_times: dict[str, datetime] = {}
        # Stacking ensemble
        self._stacking = None

    def aggregate(
        self,
        alpha_signals: dict[str, pd.DataFrame],
        alpha_weights: dict[str, float],
        alpha_metadata: dict[str, dict] | None = None,
        alpha_categories: dict[str, str] | None = None,
        regime_multipliers: dict[str, float] | None = None,
        vol_confidence: float = 1.0,
        min_abs_score: float = 0.01,
        signal_timestamp: datetime | None = None,
    ) -> EnhancedAggregatedSignal:
        if alpha_metadata is None:
            alpha_metadata = {}
        if alpha_categories is None:
            alpha_categories = {}
        if regime_multipliers is None:
            regime_multipliers = {}
        if signal_timestamp is None:
            signal_timestamp = datetime.now(timezone.utc)

        # ==============================================================
        # Stage 1: 개별 알파 가중 (alpha_weight × confidence × dyn_adj)
        # 카테고리 weight는 여기서 곱하지 않음 (이중 가중치 방지)
        # ==============================================================
        # {category: {ticker: [(alpha_name, contribution)]}}
        category_contribs: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        raw_scores: dict[str, dict[str, float]] = defaultdict(dict)
        all_contributions: dict[str, dict[str, float]] = defaultdict(dict)

        for alpha_name, signals_df in alpha_signals.items():
            weight = alpha_weights.get(alpha_name, 0.0)
            if weight == 0 or signals_df.empty:
                continue

            meta = alpha_metadata.get(alpha_name, {})
            confidence = meta.get("confidence", 1.0)
            decay_hours = meta.get("signal_decay_hours", 24.0)
            category = alpha_categories.get(alpha_name, "momentum")
            dyn_adj = self._dynamic_adjustments.get(alpha_name, 1.0)

            # Signal decay: 시그널이 마지막 갱신 이후 얼마나 지났는지
            # (피드백 #4 해결)
            decay_factor = 1.0
            last_time = self._last_signal_times.get(alpha_name)
            if last_time and decay_hours > 0:
                elapsed_hours = (
                    signal_timestamp - last_time
                ).total_seconds() / 3600
                # 지수 감쇠: half-life = decay_hours
                decay_factor = float(np.exp(-0.693 * elapsed_hours / decay_hours))
                decay_factor = max(0.1, decay_factor)

            self._last_signal_times[alpha_name] = signal_timestamp

            effective_weight = weight * confidence * dyn_adj * decay_factor

            for _, row in signals_df.iterrows():
                ticker = row["ticker"]
                raw_score = float(row.get("score", 0))
                if math.isnan(raw_score):
                    continue
                contribution = raw_score * effective_weight
                category_contribs[category][ticker].append(
                    (alpha_name, contribution)
                )
                raw_scores[ticker][alpha_name] = raw_score
                all_contributions[ticker][alpha_name] = contribution

        # ==============================================================
        # Stage 2: 카테고리 기반 통합 + 카테고리 내 상관 감쇄
        # (피드백 #1, #7 해결)
        #
        # 수식:
        #   cat_sum = Σ(contribution_i) for i in category
        #   n_same_sign = 같은 방향 알파 수
        #   correlation_penalty = 1 / sqrt(n_same_sign)  (분산효과 보정)
        #   adjusted_cat_sum = cat_sum × correlation_penalty
        #
        # 그 다음 카테고리 간에는 단순 합산 (이미 alpha_weight로 비율 반영됨)
        # category_target_ratio는 레짐 조정 시에만 사용
        # ==============================================================
        combined: dict[str, float] = {}
        ticker_category_sums: dict[str, dict[str, float]] = defaultdict(dict)

        for category, ticker_alphas in category_contribs.items():
            for ticker, alpha_contribs in ticker_alphas.items():
                cat_sum = sum(c for _, c in alpha_contribs)

                # 카테고리 내 상관 감쇄: 같은 방향 알파가 많을수록 패널티
                if len(alpha_contribs) > 1:
                    signs = [np.sign(c) for _, c in alpha_contribs if c != 0]
                    if signs:
                        dominant_count = max(
                            sum(1 for s in signs if s > 0),
                            sum(1 for s in signs if s < 0),
                        )
                        # sqrt(n) 보정: 2개 동방향 → 0.71, 3개 → 0.58
                        corr_penalty = 1.0 / np.sqrt(max(dominant_count, 1))
                        # 혼합 방향이면 패널티 안 줌 (이미 상쇄됨)
                        agreement_ratio = dominant_count / len(signs)
                        # agreement 80% 이상일 때만 패널티 적용
                        if agreement_ratio >= 0.8:
                            cat_sum *= corr_penalty

                ticker_category_sums[ticker][category] = cat_sum

        # 카테고리 합산: 단순 합산 (alpha_weight에 이미 비율 반영)
        for ticker, cat_sums in ticker_category_sums.items():
            combined[ticker] = sum(cat_sums.values())

        # ==============================================================
        # Stage 3: 레짐 조정 + vol confidence
        # (피드백 #5 해결: vol_confidence는 최종 스코어에 일괄 적용)
        # ==============================================================
        effective: dict[str, float] = {}
        confidence_map: dict[str, float] = {}
        agreement_map: dict[str, tuple[int, int, str]] = {}

        for ticker, score in combined.items():
            eff_score = score

            # 레짐 멀티플라이어: 카테고리별 조정을 가중평균
            if regime_multipliers:
                cats = ticker_category_sums.get(ticker, {})
                if cats:
                    total_abs = sum(abs(v) for v in cats.values()) or 1
                    regime_factor = sum(
                        regime_multipliers.get(cat, 1.0) * abs(cat_val) / total_abs
                        for cat, cat_val in cats.items()
                    )
                    eff_score *= regime_factor

            # Vol confidence: 최종 스코어에 일괄 적용
            eff_score *= vol_confidence

            effective[ticker] = eff_score

            # Per-ticker confidence
            ticker_raw = raw_scores.get(ticker, {})
            confidences = []
            for alpha_name in ticker_raw:
                meta = alpha_metadata.get(alpha_name, {})
                confidences.append(meta.get("confidence", 1.0))
            avg_conf = float(np.mean(confidences)) if confidences else 0.5
            confidence_map[ticker] = avg_conf * vol_confidence

            # Signal agreement
            n_long = sum(1 for s in ticker_raw.values() if s > 0.05)
            n_short = sum(1 for s in ticker_raw.values() if s < -0.05)
            n_neutral = sum(1 for s in ticker_raw.values() if abs(s) <= 0.05)
            n_total = len(ticker_raw)
            if n_long > n_short:
                direction = "LONG"
                n_agree = n_long
            elif n_short > n_long:
                direction = "SHORT"
                n_agree = n_short
            else:
                direction = "NEUTRAL"
                n_agree = n_neutral
            agreement_map[ticker] = (n_agree, n_total, direction)

        # Filter tiny scores
        if min_abs_score > 0:
            combined = {k: v for k, v in combined.items() if abs(v) >= min_abs_score}
            effective = {k: v for k, v in effective.items() if k in combined}

        # ==============================================================
        # Stacking meta-model (학습된 경우)
        # 가중합과 병렬로 계산 → Sonnet에 둘 다 전달
        # ==============================================================
        stacking_scores: dict[str, float] = {}
        stacking_contribs: dict[str, dict[str, float]] = {}
        ensemble_method = "weighted_sum"

        if self._stacking is not None and self._stacking.is_fitted:
            ensemble_method = "stacking"
            # 현재 시그널을 stacking 입력 형태로 변환
            ticker_signals: dict[str, dict[str, float]] = {}
            for ticker in set(raw_scores.keys()) | set(combined.keys()):
                ticker_signals[ticker] = raw_scores.get(ticker, {})

            # 시장 피처 (regime_multipliers에서 추출)
            mkt_feat = {}
            if regime_multipliers:
                mkt_feat["regime_bull"] = regime_multipliers.get("momentum", 1.0) - 1.0
                mkt_feat["regime_bear"] = regime_multipliers.get("carry", 1.0) - 1.0

            detail = self._stacking.predict_with_detail(
                ticker_signals, mkt_feat
            )
            for ticker, info in detail.items():
                stacking_scores[ticker] = info["score"]
                stacking_contribs[ticker] = info["contributions"]

            # Stacking이 primary → effective_scores를 stacking으로 교체
            for ticker in stacking_scores:
                effective[ticker] = stacking_scores[ticker] * vol_confidence

            # 시그널 버퍼 기록 (재학습용)
            self._stacking.record_signals(
                signal_timestamp, ticker_signals, mkt_feat
            )

        return EnhancedAggregatedSignal(
            scores=combined,
            effective_scores=effective,
            contributions=dict(all_contributions),
            weights_used={
                k: v for k, v in alpha_weights.items()
                if k in alpha_signals and v > 0
            },
            raw_scores=dict(raw_scores),
            confidence=confidence_map,
            agreement=agreement_map,
            vol_confidence=vol_confidence,
            stacking_scores=stacking_scores,
            stacking_contributions=stacking_contribs,
            ensemble_method=ensemble_method,
        )

    # ------------------------------------------------------------------
    # Dynamic weight adjustment (피드백 #2 해결)
    # - lookback 30일 (14 → 60 observations at 2x/day)
    # - "지속" = 최근 30일 평균
    # - 재배분: 감소분을 나머지에 비례 배분
    # ------------------------------------------------------------------

    def init_stacking(self, alpha_names: list[str], **kwargs) -> None:
        """StackingEnsemble 초기화."""
        from src.ensemble.stacking import StackingEnsemble
        self._stacking = StackingEnsemble(alpha_names=alpha_names, **kwargs)
        logger.info(f"Stacking initialized: {len(alpha_names)} features")

    def load_stacking(self, path=None) -> bool:
        """저장된 stacking 모델 로드. 없으면 False."""
        from src.ensemble.stacking import StackingEnsemble
        try:
            self._stacking = StackingEnsemble.load(path)
            return True
        except (FileNotFoundError, Exception) as e:
            logger.info(f"Stacking model not found (will use weighted sum): {e}")
            return False

    def retrain_stacking_if_needed(self, prices) -> None:
        """30일 경과 시 라이브 버퍼로 stacking 재학습."""
        if self._stacking is None:
            return
        if self._stacking.should_retrain():
            result = self._stacking.fit_from_live_buffer(prices)
            logger.info(f"Stacking retrain: {result}")
            if result.get("status") == "fitted":
                self._stacking.save()

    def enable_ic_tracking(self) -> None:
        self._ic_tracking_enabled = True

    def record_ic(self, alpha_name: str, ic: float) -> None:
        """Record an IC observation for an alpha."""
        now = datetime.now(timezone.utc)
        self._ic_history[alpha_name].append((now, ic))
        # Keep last 60 observations (~30 days at 2x/day)
        if len(self._ic_history[alpha_name]) > 60:
            self._ic_history[alpha_name] = self._ic_history[alpha_name][-60:]
        self._update_dynamic_adjustments()

    def _update_dynamic_adjustments(self) -> None:
        if not self._ic_tracking_enabled:
            return

        raw_adjustments: dict[str, float] = {}

        for alpha_name, ic_records in self._ic_history.items():
            # 최소 20 observations 필요 (~10일)
            if len(ic_records) < 20:
                raw_adjustments[alpha_name] = 1.0
                continue

            ics = [ic for _, ic in ic_records[-30:]]
            avg_ic = np.mean(ics)

            if avg_ic < -0.02:
                raw_adjustments[alpha_name] = 0.5
            elif avg_ic > 0.03:
                raw_adjustments[alpha_name] = min(1.25, 1.0 + avg_ic * 5)
            else:
                raw_adjustments[alpha_name] = 1.0

        # 재배분: 총 가중치 합이 보존되도록 정규화
        if raw_adjustments:
            total_adj = sum(raw_adjustments.values())
            n = len(raw_adjustments)
            if total_adj > 0 and n > 0:
                # 목표: 평균 adjustment = 1.0
                scale = n / total_adj
                for name in raw_adjustments:
                    raw_adjustments[name] *= scale

        self._dynamic_adjustments = raw_adjustments

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def scores_to_target_weights(
        scores: dict[str, float],
        max_positions: int = 1,
        max_weight: float = 0.95,
    ) -> dict[str, float]:
        if not scores:
            return {}
        ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
        top = ranked[:max_positions]
        weights = {}
        for ticker, score in top:
            w = max_weight if score > 0 else -max_weight
            weights[ticker] = w
        return weights
