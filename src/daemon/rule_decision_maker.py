"""
Rule-Based Decision Maker

Sonnet AI를 대체하는 순수 규칙 기반 의사결정.
알파 앙상블 스코어만으로 진입/청산/홀드를 결정.

결정 로직:
    1. |effective_score| >= entry_threshold → 진입 (LONG or SHORT)
    2. 기존 포지션과 반대 |score| >= reverse_threshold → 청산
    3. 그 외 → HOLD

SL/TP 설정:
    - 변동성 비례: 최근 20일 실현 변동성에 따라 SL/TP 폭 조정
    - SL: -2% ~ -8%, TP: +3% ~ +15% 범위 내
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from config.settings import SONNET_DECISION, TRADING

logger = logging.getLogger(__name__)


@dataclass
class PositionDecision:
    """단일 포지션 결정."""
    ticker: str
    action: str          # LONG, SHORT, CLOSE, HOLD
    weight: float        # 0.0 ~ 0.95
    stop_loss_pct: float # -0.02 ~ -0.08
    take_profit_pct: float  # 0.03 ~ 0.15
    reasoning: str


@dataclass
class DecisionResult:
    """리밸런스 사이클 결정 결과."""
    positions: list[PositionDecision]
    market_assessment: str
    risk_note: str


class RuleDecisionMaker:
    """
    규칙 기반 의사결정 — Sonnet 없이 앙상블 스코어로 직접 매매.

    규칙:
        - 포지션 없을 때: |score| >= entry_threshold → 최고 점수 티커에 진입
        - 포지션 있을 때: 반대 방향 |score| >= reverse_threshold → 청산
        - SL/TP: 변동성 비례 자동 설정
    """

    def __init__(self):
        self.entry_threshold = SONNET_DECISION.get("entry_score_threshold", 0.22)
        self.reverse_threshold = SONNET_DECISION.get("reverse_score_threshold", 0.165)
        self.max_positions = TRADING.get("max_positions", 1)
        self.default_weight = 0.95  # 단일 포지션, 올인

        # SL/TP 범위
        self.tightest_sl = SONNET_DECISION.get("tightest_stop_loss_pct", -0.02)
        self.loosest_sl = SONNET_DECISION.get("loosest_stop_loss_pct", -0.08)
        self.smallest_tp = SONNET_DECISION.get("smallest_take_profit_pct", 0.03)
        self.largest_tp = SONNET_DECISION.get("largest_take_profit_pct", 0.15)

    def make_decision(
        self,
        effective_scores: dict[str, float],
        contributions: dict[str, dict[str, float]],
        current_positions: pd.DataFrame | None,
        prices: pd.DataFrame | None = None,
        managed_positions: list | None = None,
        cooldown_tickers: set[str] | None = None,
    ) -> DecisionResult:
        """
        규칙 기반 매매 결정.

        Args:
            effective_scores: {ticker: 앙상블 스코어}
            contributions: {ticker: {alpha: contribution}}
            current_positions: 현재 바이낸스 포지션 DataFrame
            prices: 일봉 데이터 (변동성 계산용)
            managed_positions: 관리 중인 포지션 리스트
            cooldown_tickers: 쿨다운 중인 티커 set

        Returns:
            DecisionResult with positions list.
        """
        if cooldown_tickers is None:
            cooldown_tickers = set()

        positions: list[PositionDecision] = []

        # 현재 보유 중인 포지션 파악
        held: dict[str, str] = {}  # {ticker: side}
        if current_positions is not None and not current_positions.empty:
            for _, row in current_positions.iterrows():
                ticker = str(row.get("ticker", ""))
                side = str(row.get("side", "")).upper()
                if ticker and side in ("LONG", "SHORT"):
                    held[ticker] = side

        # 스코어 기준 정렬
        ranked = sorted(
            effective_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        # ── 기존 포지션 관리 ──
        for ticker, side in held.items():
            score = effective_scores.get(ticker, 0)
            contribs = contributions.get(ticker, {})

            # 반대 방향 시그널이 충분히 강하면 → CLOSE
            if side == "LONG" and score < -self.reverse_threshold:
                positions.append(PositionDecision(
                    ticker=ticker,
                    action="CLOSE",
                    weight=0.0,
                    stop_loss_pct=0.0,
                    take_profit_pct=0.0,
                    reasoning=f"반대 시그널 {score:+.3f} < -{self.reverse_threshold}",
                ))
            elif side == "SHORT" and score > self.reverse_threshold:
                positions.append(PositionDecision(
                    ticker=ticker,
                    action="CLOSE",
                    weight=0.0,
                    stop_loss_pct=0.0,
                    take_profit_pct=0.0,
                    reasoning=f"반대 시그널 {score:+.3f} > +{self.reverse_threshold}",
                ))
            else:
                # HOLD
                positions.append(PositionDecision(
                    ticker=ticker,
                    action="HOLD",
                    weight=self.default_weight,
                    stop_loss_pct=0.0,
                    take_profit_pct=0.0,
                    reasoning=f"유지 (score={score:+.3f}, side={side})",
                ))

        # ── 신규 진입 판단 ──
        n_active = sum(1 for p in positions if p.action not in ("CLOSE",))
        if n_active < self.max_positions:
            for ticker, score in ranked:
                if ticker in held:
                    continue
                if ticker in cooldown_tickers:
                    continue
                if abs(score) < self.entry_threshold:
                    break  # 정렬되어 있으므로 이후도 다 threshold 미만

                direction = "LONG" if score > 0 else "SHORT"
                sl_pct, tp_pct = self._compute_sltp(ticker, direction, prices)
                contribs = contributions.get(ticker, {})

                # 주요 기여 알파 요약
                top_contribs = sorted(
                    contribs.items(), key=lambda x: abs(x[1]), reverse=True
                )[:3]
                reason_parts = [f"{name}={val:+.2f}" for name, val in top_contribs]
                reasoning = f"{direction} score={score:+.3f} ({', '.join(reason_parts)})"

                positions.append(PositionDecision(
                    ticker=ticker,
                    action=direction,
                    weight=self.default_weight,
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                    reasoning=reasoning,
                ))
                n_active += 1
                if n_active >= self.max_positions:
                    break

        # 시장 평가 요약
        if ranked:
            top_ticker, top_score = ranked[0]
            market_assessment = (
                f"최강 시그널: {top_ticker.replace('/USDT:USDT', '')} "
                f"{'LONG' if top_score > 0 else 'SHORT'} {top_score:+.3f}"
            )
        else:
            market_assessment = "시그널 없음"

        risk_note = ""
        if any(abs(s) > 0.5 for _, s in ranked[:3]):
            risk_note = "강한 시그널 감지 — 변동성 주의"

        return DecisionResult(
            positions=positions,
            market_assessment=market_assessment,
            risk_note=risk_note,
        )

    def _compute_sltp(
        self,
        ticker: str,
        direction: str,
        prices: pd.DataFrame | None,
    ) -> tuple[float, float]:
        """
        변동성 비례 SL/TP 설정.

        높은 변동성 → 넓은 SL/TP (빈번한 손절 방지)
        낮은 변동성 → 좁은 SL/TP (빠른 손익 확정)
        """
        # 기본값
        default_sl = SONNET_DECISION.get("default_stop_loss_pct", -0.05)
        default_tp = SONNET_DECISION.get("default_take_profit_pct", 0.10)

        if prices is None or prices.empty:
            return default_sl, default_tp

        tkr = prices[prices["ticker"] == ticker].sort_values("date")
        if len(tkr) < 20:
            return default_sl, default_tp

        closes = tkr["close"].values[-20:]
        daily_rets = np.diff(closes) / closes[:-1]
        vol_20d = float(np.std(daily_rets, ddof=1))

        if vol_20d <= 0:
            return default_sl, default_tp

        # 변동성 기준: 크립토 평균 일 변동성 ~3%
        # vol_ratio > 1: 평균보다 변동 큼, < 1: 작음
        vol_ratio = vol_20d / 0.03

        # SL: 기본 -5% × vol_ratio, 범위 제한
        sl = -0.05 * vol_ratio
        sl = max(self.loosest_sl, min(self.tightest_sl, sl))

        # TP: 기본 +10% × vol_ratio, 범위 제한
        tp = 0.10 * vol_ratio
        tp = max(self.smallest_tp, min(self.largest_tp, tp))

        return round(sl, 4), round(tp, 4)
