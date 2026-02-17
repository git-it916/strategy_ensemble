"""
Risk Manager — Step 5 of Sequential Pipeline

Position sizing, short→inverse ETF 변환, 포트폴리오 제약 적용.

LLM이 "SHORT KOSPI" 시그널을 내면 인버스 ETF 매수로 변환.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PositionOrder:
    """최종 주문 대상."""
    ticker: str
    side: str         # "BUY" or "SELL" (실제 주문 방향)
    weight: float     # 포트폴리오 비중
    score: float      # LLM 스코어
    reason: str       # LLM 판단 이유
    is_inverse: bool = False  # 인버스 ETF 여부
    original_signal: str = ""  # 원래 LLM 시그널 (long/short)


class RiskManager:
    """
    리스크 관리 + 인버스 ETF 변환.

    Responsibilities:
        1. LLM output → 포지션 사이징
        2. SHORT 시그널 → 인버스 ETF BUY로 변환
        3. 포지션 제약 적용 (max weight, max positions)
        4. 현재 포트폴리오 대비 리밸런싱 diff 계산
    """

    def __init__(
        self,
        inverse_mapping: dict[str, str] | None = None,
        max_position_weight: float = 0.1,
        max_positions: int = 20,
        max_leverage: float = 1.0,
        min_trade_value: int = 100_000,
    ):
        from config.settings import INVERSE_MAPPING
        self.inverse_mapping = inverse_mapping or INVERSE_MAPPING
        self.inverse_tickers = set(self.inverse_mapping.values())
        self.max_position_weight = max_position_weight
        self.max_positions = max_positions
        self.max_leverage = max_leverage
        self.min_trade_value = min_trade_value

    def process_llm_decisions(
        self,
        llm_signals: list[dict[str, Any]],
        current_positions: pd.DataFrame | None = None,
    ) -> list[PositionOrder]:
        """
        LLM의 JSON 결정을 실행 가능한 주문으로 변환.

        Args:
            llm_signals: LLM 출력의 signals 배열
                [{"ticker": "005930", "score": 0.7, "side": "long", "reason": "..."}]
            current_positions: 현재 보유 포지션 (optional)

        Returns:
            실행 가능한 PositionOrder 리스트
        """
        orders: list[PositionOrder] = []

        for signal in llm_signals:
            ticker = signal.get("ticker", "")
            score = float(signal.get("score", 0))
            side = signal.get("side", "long").lower()
            reason = signal.get("reason", "")

            if abs(score) < 0.1:
                continue  # 너무 약한 시그널 무시

            if side == "short" or score < 0:
                # SHORT 시그널 → 인버스 ETF BUY로 변환
                inverse_orders = self._convert_to_inverse(
                    ticker, score, reason
                )
                orders.extend(inverse_orders)
            else:
                # LONG 시그널 → 일반 BUY
                orders.append(PositionOrder(
                    ticker=ticker,
                    side="BUY",
                    weight=0.0,  # 아래서 계산
                    score=score,
                    reason=reason,
                    is_inverse=False,
                    original_signal="long",
                ))

        # 포지션 사이징
        orders = self._apply_position_sizing(orders)

        # 현재 포지션과 비교하여 청산 대상 추가
        if current_positions is not None and not current_positions.empty:
            exit_orders = self._calculate_exits(orders, current_positions)
            orders.extend(exit_orders)

        return orders

    def _convert_to_inverse(
        self, ticker: str, score: float, reason: str
    ) -> list[PositionOrder]:
        """SHORT 시그널을 인버스 ETF 매수로 변환."""
        orders = []

        # 종목 특정 숏이면 → 시장 인버스 ETF로 변환
        # 시장/인덱스 숏이면 → 해당 인버스 ETF로 변환
        for market, inverse_ticker in self.inverse_mapping.items():
            orders.append(PositionOrder(
                ticker=inverse_ticker,
                side="BUY",
                weight=0.0,
                score=abs(score),
                reason=f"[HEDGE via {market} inverse] {reason}",
                is_inverse=True,
                original_signal="short",
            ))
            break  # 기본은 KOSPI 인버스 하나만

        return orders

    def _apply_position_sizing(
        self, orders: list[PositionOrder]
    ) -> list[PositionOrder]:
        """스코어 기반 포지션 사이징."""
        if not orders:
            return orders

        # 상위 N개만
        orders.sort(key=lambda o: abs(o.score), reverse=True)
        orders = orders[:self.max_positions]

        # 스코어 비례 가중치
        total_score = sum(abs(o.score) for o in orders)
        if total_score == 0:
            return []

        for order in orders:
            raw_weight = abs(order.score) / total_score
            order.weight = min(raw_weight, self.max_position_weight)

        # 정규화
        total_weight = sum(o.weight for o in orders)
        if total_weight > 0:
            scale = min(self.max_leverage, 1.0) / total_weight
            for order in orders:
                order.weight *= scale

        return orders

    def _calculate_exits(
        self,
        new_orders: list[PositionOrder],
        current_positions: pd.DataFrame,
    ) -> list[PositionOrder]:
        """현재 보유 중이나 새 시그널에 없는 종목 청산 주문."""
        new_tickers = {o.ticker for o in new_orders}
        exits = []

        for _, pos in current_positions.iterrows():
            stock_code = pos.get("stock_code", "")
            if stock_code and stock_code not in new_tickers:
                quantity = pos.get("quantity", 0)
                if quantity > 0:
                    exits.append(PositionOrder(
                        ticker=stock_code,
                        side="SELL",
                        weight=0.0,
                        score=0.0,
                        reason="Position exit: not in new LLM signals",
                        original_signal="exit",
                    ))

        return exits

    def to_target_weights(
        self, orders: list[PositionOrder]
    ) -> pd.DataFrame:
        """PositionOrder 리스트를 OrderManager가 인식하는 DataFrame으로 변환."""
        buy_orders = [o for o in orders if o.side == "BUY" and o.weight > 0]
        if not buy_orders:
            return pd.DataFrame(columns=["ticker", "weight"])

        return pd.DataFrame([
            {"ticker": o.ticker, "weight": o.weight}
            for o in buy_orders
        ])

    def format_proposal(
        self, orders: list[PositionOrder], total_value: int = 0
    ) -> str:
        """사용자 승인용 포맷 문자열 생성."""
        if not orders:
            return "No trades proposed."

        lines = []
        buy_orders = [o for o in orders if o.side == "BUY" and o.weight > 0]
        sell_orders = [o for o in orders if o.side == "SELL"]

        if buy_orders:
            lines.append("<b>BUY</b>")
            for o in sorted(buy_orders, key=lambda x: x.score, reverse=True):
                inv_tag = " [INV]" if o.is_inverse else ""
                amount = f" (~₩{total_value * o.weight:,.0f})" if total_value else ""
                lines.append(
                    f"  {'🔵' if not o.is_inverse else '🟡'} "
                    f"{o.ticker}{inv_tag} {o.weight:.1%}{amount} "
                    f"(score: {o.score:+.2f})"
                )
                lines.append(f"    → {o.reason}")

        if sell_orders:
            lines.append("\n<b>SELL (Exit)</b>")
            for o in sell_orders:
                lines.append(f"  🔴 {o.ticker}: {o.reason}")

        return "\n".join(lines)
