"""
Slack Bot

Send trading notifications to Slack.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
import logging

import requests

logger = logging.getLogger(__name__)


class SlackNotifier:
    """
    Send trading notifications to Slack.

    Features:
        - Trade alerts
        - Daily summary
        - Error notifications
        - Position updates
    """

    def __init__(self, webhook_url: str, channel: str | None = None):
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL
            channel: Override channel (optional)
        """
        self.webhook_url = webhook_url
        self.channel = channel

    def _send_message(self, payload: dict[str, Any]) -> bool:
        """Send message to Slack."""
        if self.channel:
            payload["channel"] = self.channel

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False

    def send_text(self, text: str) -> bool:
        """Send simple text message."""
        return self._send_message({"text": text})

    def send_trade_alert(
        self,
        stock_code: str,
        stock_name: str,
        side: str,
        quantity: int,
        price: int,
        strategy: str | None = None,
    ) -> bool:
        """
        Send trade execution alert.

        Args:
            stock_code: Stock code
            stock_name: Stock name
            side: BUY or SELL
            quantity: Trade quantity
            price: Trade price
            strategy: Strategy name

        Returns:
            True if sent
        """
        emoji = "📈" if side == "BUY" else "📉"
        side_kr = "매수" if side == "BUY" else "매도"
        value = quantity * price

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {side_kr} 체결",
                    "emoji": True,
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*종목*\n{stock_name} ({stock_code})"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*수량*\n{quantity:,}주"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*가격*\n{price:,}원"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*금액*\n{value:,}원"
                    },
                ]
            },
        ]

        if strategy:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📊 전략: {strategy}"
                    }
                ]
            })

        return self._send_message({"blocks": blocks})

    def send_daily_summary(
        self,
        date: datetime,
        total_value: int,
        daily_pnl: int,
        daily_return: float,
        positions: list[dict],
        trades_count: int,
    ) -> bool:
        """
        Send daily trading summary.

        Args:
            date: Summary date
            total_value: Total portfolio value
            daily_pnl: Daily P&L
            daily_return: Daily return percentage
            positions: Current positions
            trades_count: Number of trades today

        Returns:
            True if sent
        """
        pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
        pnl_sign = "+" if daily_pnl >= 0 else ""

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📊 일일 리포트 ({date.strftime('%Y-%m-%d')})",
                    "emoji": True,
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*총 자산*\n{total_value:,}원"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*일간 손익*\n{pnl_emoji} {pnl_sign}{daily_pnl:,}원 ({pnl_sign}{daily_return:.2f}%)"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*보유 종목*\n{len(positions)}개"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*오늘 거래*\n{trades_count}건"
                    },
                ]
            },
            {"type": "divider"},
        ]

        # Top holdings
        if positions:
            top_positions = sorted(
                positions,
                key=lambda x: x.get("eval_amount", 0),
                reverse=True
            )[:5]

            holdings_text = "\n".join([
                f"• {p['name']} ({p['stock_code']}): {p['profit_rate']:+.1f}%"
                for p in top_positions
            ])

            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*📌 주요 보유종목*\n{holdings_text}"
                }
            })

        return self._send_message({"blocks": blocks})

    def send_error(self, error_msg: str, context: str | None = None) -> bool:
        """
        Send error notification.

        Args:
            error_msg: Error message
            context: Additional context

        Returns:
            True if sent
        """
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 오류 발생",
                    "emoji": True,
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"```{error_msg}```"
                }
            },
        ]

        if context:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📍 {context}"
                    }
                ]
            })

        return self._send_message({"blocks": blocks})

    def send_position_update(
        self,
        action: str,
        stock_code: str,
        stock_name: str,
        quantity: int,
        price: int,
        reason: str | None = None,
    ) -> bool:
        """
        Send position change notification.

        Args:
            action: "ENTER", "EXIT", "INCREASE", "DECREASE"
            stock_code: Stock code
            stock_name: Stock name
            quantity: New position quantity
            price: Current price
            reason: Reason for change

        Returns:
            True if sent
        """
        action_emoji = {
            "ENTER": "🆕",
            "EXIT": "🔚",
            "INCREASE": "⬆️",
            "DECREASE": "⬇️",
        }

        action_kr = {
            "ENTER": "신규 진입",
            "EXIT": "전량 청산",
            "INCREASE": "비중 증가",
            "DECREASE": "비중 축소",
        }

        emoji = action_emoji.get(action, "📝")
        action_text = action_kr.get(action, action)

        text = f"{emoji} *{action_text}*: {stock_name} ({stock_code})\n"
        text += f"수량: {quantity:,}주 | 현재가: {price:,}원"

        if reason:
            text += f"\n💡 {reason}"

        return self.send_text(text)

    def send_signal_alert(
        self,
        strategy: str,
        stocks: list[dict],
        regime: str | None = None,
    ) -> bool:
        """
        Send strategy signal alert.

        Args:
            strategy: Strategy name
            stocks: List of stocks with signals
            regime: Current market regime

        Returns:
            True if sent
        """
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🎯 {strategy} 신호",
                    "emoji": True,
                }
            },
        ]

        if regime:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📈 시장 국면: {regime}"
                    }
                ]
            })

        # Long signals
        longs = [s for s in stocks if s.get("score", 0) > 0]
        if longs:
            long_text = "\n".join([
                f"• {s['name']} ({s['stock_code']}): {s['score']:.2f}"
                for s in sorted(longs, key=lambda x: x["score"], reverse=True)[:5]
            ])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*📈 매수 후보*\n{long_text}"
                }
            })

        # Short signals (if any)
        shorts = [s for s in stocks if s.get("score", 0) < 0]
        if shorts:
            short_text = "\n".join([
                f"• {s['name']} ({s['stock_code']}): {s['score']:.2f}"
                for s in sorted(shorts, key=lambda x: x["score"])[:5]
            ])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*📉 주의 종목*\n{short_text}"
                }
            })

        return self._send_message({"blocks": blocks})
