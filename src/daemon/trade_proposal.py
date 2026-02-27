"""
Trade Proposal Builder

Builds rich Telegram messages that explain:
  - WHY: alpha-level contributions and reasoning
  - WHAT: position changes (new, increase, decrease, close)
  - EXIT: stop-loss, daily limit, rebalance frequency, kill rules
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PositionChange:
    """A single position change in the proposal."""
    ticker: str
    direction: str          # LONG / SHORT / CLOSE
    target_weight: float    # signed target weight
    current_weight: float   # signed current weight
    leverage: float
    top_alphas: list[tuple[str, float]]  # [(alpha_name, contribution), ...]
    is_new: bool = False


@dataclass
class TradeProposal:
    """Complete trade proposal for Telegram."""
    timestamp: datetime
    balance_usdt: float
    changes: list[PositionChange]
    alpha_weights: dict[str, float]
    alpha_descriptions: dict[str, str]
    risk_params: dict[str, float]


# Korean-friendly alpha descriptions
ALPHA_DESCRIPTIONS = {
    "cs_momentum": "횡단면 모멘텀 (11개월 수익률 상위)",
    "ts_momentum": "시계열 모멘텀 (20일 추세 추종)",
    "ts_mean_reversion": "시계열 평균회귀 (과매도 반등)",
    "pv_divergence": "가격-거래량 괴리 (거래량 급증 감지)",
    "volume_momentum": "거래량 모멘텀 (거래량 가속도)",
    "low_volatility_anomaly": "저변동성 이상현상 (안정적 코인 선호)",
    "funding_rate_carry": "펀딩비 캐리 (음수 펀딩비 롱)",
    "rsi_reversal": "RSI 역전 (과매수/과매도 신호)",
    "vol_breakout": "변동성 돌파 (볼린저밴드 기반)",
}


class TradeProposalBuilder:
    """Build trade proposal messages for Telegram."""

    def build_proposal(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        contributions: dict[str, dict[str, float]],
        alpha_weights: dict[str, float],
        leverage_per_symbol: dict[str, float],
        balance_usdt: float,
        risk_params: dict[str, float] | None = None,
    ) -> TradeProposal:
        """
        Build a TradeProposal from target vs current positions.

        Args:
            target_weights: {ticker: signed_weight} target portfolio
            current_weights: {ticker: signed_weight} current portfolio
            contributions: {ticker: {alpha: contribution}} from SignalAggregator
            alpha_weights: {alpha_name: weight}
            leverage_per_symbol: {ticker: leverage}
            balance_usdt: Account balance in USDT
            risk_params: Override risk parameters

        Returns:
            TradeProposal ready for formatting.
        """
        if risk_params is None:
            from config.settings import TRADING
            risk_params = {
                "max_drawdown": TRADING.get("max_drawdown", 0.15),
                "daily_loss_limit": TRADING.get("daily_loss_limit", 0.03),
                "rebalance_minutes": 15,
                "kill_loss_days": 5,
            }

        all_tickers = set(target_weights) | set(current_weights)
        changes = []

        for ticker in all_tickers:
            tw = target_weights.get(ticker, 0.0)
            cw = current_weights.get(ticker, 0.0)

            # Skip unchanged
            if abs(tw - cw) < 0.005:
                continue

            if tw == 0:
                direction = "CLOSE"
            elif tw > 0:
                direction = "LONG"
            else:
                direction = "SHORT"

            is_new = cw == 0 and tw != 0

            from .signal_aggregator import SignalAggregator
            top_alphas = SignalAggregator.get_top_contributors(
                contributions, ticker, top_n=3
            )

            changes.append(PositionChange(
                ticker=ticker,
                direction=direction,
                target_weight=tw,
                current_weight=cw,
                leverage=leverage_per_symbol.get(ticker, 1.0),
                top_alphas=top_alphas,
                is_new=is_new,
            ))

        # Sort: new positions first, then by absolute weight change
        changes.sort(
            key=lambda c: (not c.is_new, -abs(c.target_weight - c.current_weight))
        )

        return TradeProposal(
            timestamp=datetime.now(),
            balance_usdt=balance_usdt,
            changes=changes,
            alpha_weights=alpha_weights,
            alpha_descriptions=ALPHA_DESCRIPTIONS,
            risk_params=risk_params,
        )

    def format_telegram(
        self,
        proposal: TradeProposal,
        max_changes: int = 10,
    ) -> str:
        """
        Format TradeProposal as HTML for Telegram.

        Returns:
            HTML string (max ~4000 chars for Telegram limit).
        """
        lines = []

        # Header
        ts = proposal.timestamp.strftime("%Y-%m-%d %H:%M")
        lines.append(f"<b>━━ 거래 제안서 ━━</b>")
        lines.append(f"{ts} | 잔고: ${proposal.balance_usdt:,.0f}")
        lines.append("")

        # Alpha analysis section
        lines.append("<b>[알파 분석]</b>")
        sorted_alphas = sorted(
            proposal.alpha_weights.items(), key=lambda x: x[1], reverse=True
        )
        for alpha_name, weight in sorted_alphas[:5]:
            desc = proposal.alpha_descriptions.get(alpha_name, alpha_name)
            lines.append(f"  {alpha_name} ({weight:.0%}): {desc}")
        lines.append("")

        # Position changes
        lines.append("<b>[포지션 변경]</b>")
        shown = proposal.changes[:max_changes]
        for c in shown:
            # Direction icon
            if c.direction == "CLOSE":
                icon = "🔴"
                detail = f"청산 (현재 {c.current_weight:+.1%})"
            elif c.is_new:
                icon = "🟢"
                detail = (
                    f"{c.direction} {abs(c.target_weight):.1%} "
                    f"(신규, 레버리지 {c.leverage:.1f}x)"
                )
            else:
                icon = "🔵"
                change = c.target_weight - c.current_weight
                detail = (
                    f"{c.direction} {abs(c.target_weight):.1%} "
                    f"({change:+.1%}, 레버리지 {c.leverage:.1f}x)"
                )

            # Shorten ticker for display
            short_ticker = c.ticker.replace("/USDT:USDT", "").replace("USDT", "")
            lines.append(f"  {icon} {short_ticker} → {detail}")

            # Top alpha contributions
            if c.top_alphas:
                parts = [f"{name} {val:+.2f}" for name, val in c.top_alphas[:2]]
                lines.append(f"    주도: {', '.join(parts)}")

        if len(proposal.changes) > max_changes:
            lines.append(f"  ... 외 {len(proposal.changes) - max_changes}건")
        lines.append("")

        # Exit strategy
        rp = proposal.risk_params
        lines.append("<b>[매도 전략]</b>")
        lines.append(f"  손절: 포트폴리오 -{rp.get('max_drawdown', 0.15):.0%} 낙폭 시 전량 청산")
        lines.append(f"  일일 한도: 하루 -{rp.get('daily_loss_limit', 0.03):.0%} 도달 시 신규 진입 중단")
        lines.append(f"  리밸런스: {int(rp.get('rebalance_minutes', 15))}분마다 재평가")
        lines.append(f"  퇴출: {int(rp.get('kill_loss_days', 5))}일 연속 손실 시 해당 알파 자동 제거")

        return "\n".join(lines)

    def build_approval_keyboard(self) -> dict:
        """Build inline keyboard with APPROVE / REJECT buttons."""
        return {
            "inline_keyboard": [
                [
                    {"text": "✅ APPROVE", "callback_data": "trade_approve"},
                    {"text": "❌ REJECT", "callback_data": "trade_reject"},
                ]
            ]
        }
