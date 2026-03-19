"""
Telegram Bot for Trading Notifications

트레이딩 알림을 텔레그램으로 전송.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Telegram notification sender for trading alerts."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str | int,
        broadcast_chat_ids: list[str | int] | None = None,
    ):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram Bot API token
            chat_id: Primary chat ID (user or group)
            broadcast_chat_ids: Additional chat IDs to broadcast to
        """
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.broadcast_chat_ids = [
            str(cid) for cid in (broadcast_chat_ids or [])
        ]
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def _send_to_chat(self, chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to a specific chat ID."""
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message to {chat_id}: {e}")
            return False

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to primary chat + all broadcast chats."""
        ok = self._send_to_chat(self.chat_id, text, parse_mode)
        for cid in self.broadcast_chat_ids:
            self._send_to_chat(cid, text, parse_mode)
        return ok

    def send_trade_alert(
        self,
        stock_code: str,
        stock_name: str,
        side: str,
        quantity: int,
        price: float,
        strategy: str = "Ensemble",
    ) -> bool:
        """
        Send trade execution alert.

        Args:
            stock_code: Stock ticker
            stock_name: Stock name
            side: BUY or SELL
            quantity: Order quantity
            price: Execution price
            strategy: Strategy name
        """
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        side_kr = "매수" if side.upper() == "BUY" else "매도"

        message = f"""
{emoji} <b>체결 알림</b>

📌 <b>{stock_name}</b> ({stock_code})
📊 {side_kr} {quantity:,}주 @ ₩{price:,.0f}
💰 총액: ₩{quantity * price:,.0f}
🤖 전략: {strategy}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message.strip())

    def send_fill_alert(
        self,
        stock_code: str,
        stock_name: str,
        side: str,
        quantity: int,
        price: float,
    ) -> bool:
        """Send order fill notification."""
        emoji = "\U0001F7E2" if side.upper() == "BUY" else "\U0001F534"
        name = stock_name or stock_code
        if stock_name and stock_code and stock_name != stock_code:
            name = f"{stock_name} ({stock_code})"

        message = f"{emoji} {name} {quantity:,}\uC8FC \uCCB4\uACB0 @ {price:,.0f}\uC6D0"
        return self.send_message(message.strip())

    def send_order_submitted(
        self,
        stock_code: str,
        stock_name: str,
        side: str,
        quantity: int,
        order_type: str = "시장가",
    ) -> bool:
        """Send order submitted notification."""
        emoji = "📤"
        side_kr = "매수" if side.upper() == "BUY" else "매도"

        message = f"""
{emoji} <b>주문 접수</b>

📌 {stock_name} ({stock_code})
📊 {side_kr} {quantity:,}주 ({order_type})
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send_message(message.strip())

    def send_signal_alert(
        self,
        strategy: str,
        stocks: list[dict],
        regime: str | None = None,
    ) -> bool:
        """
        Send trading signal alert.

        Args:
            strategy: Strategy name
            stocks: List of {stock_code, name, score}
            regime: Current market regime
        """
        regime_emoji = {
            "bull": "📈",
            "bear": "📉",
            "sideways": "➡️",
            "volatile": "🌊",
        }.get(regime.lower() if regime else "", "📊")

        stock_lines = []
        for i, stock in enumerate(stocks[:10], 1):
            score = stock.get("score", 0)
            emoji = "🔥" if score > 0.5 else "⚡" if score > 0 else "❄️"
            stock_lines.append(
                f"{i}. {emoji} {stock['name']} ({stock['stock_code']}) - {score:.2f}"
            )

        message = f"""
{regime_emoji} <b>신호 알림</b> - {strategy}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}
{f'📊 시장 국면: {regime}' if regime else ''}

<b>Top 종목:</b>
{chr(10).join(stock_lines)}
"""
        return self.send_message(message.strip())

    def send_daily_summary(
        self,
        date: datetime,
        total_value: float,
        daily_pnl: float,
        daily_return: float,
        positions: list[dict],
        trades_count: int,
    ) -> bool:
        """Send end-of-day summary."""
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        pnl_sign = "+" if daily_pnl >= 0 else ""

        position_lines = []
        for pos in positions[:10]:
            name = pos.get("name", pos.get("stock_code", ""))
            pnl = pos.get("profit_loss", 0)
            emoji = "🟢" if pnl >= 0 else "🔴"
            position_lines.append(f"  {emoji} {name}: {pnl:+,.0f}")

        message = f"""
{pnl_emoji} <b>일일 리포트</b>

📅 {date.strftime('%Y-%m-%d')}
💰 총 평가액: ₩{total_value:,.0f}
📊 일일 손익: {pnl_sign}₩{daily_pnl:,.0f} ({pnl_sign}{daily_return:.2f}%)
🔄 거래 횟수: {trades_count}건

<b>보유 종목:</b>
{chr(10).join(position_lines) if position_lines else '  없음'}
"""
        return self.send_message(message.strip())

    def send_error(self, error: str, context: str = "") -> bool:
        """Send error notification."""
        message = f"""
🚨 <b>오류 발생</b>

📍 {context}
❌ {error}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message.strip())

    def send_startup(self) -> bool:
        """Send bot startup notification."""
        message = f"""
🤖 <b>트레이딩 봇 시작</b>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
✅ 시스템 정상 작동 중
"""
        return self.send_message(message.strip())

    def get_updates(self) -> list[dict]:
        """
        Get recent messages to the bot.
        Useful for getting chat_id.
        """
        try:
            response = requests.get(
                f"{self.base_url}/getUpdates",
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("result", [])
        except Exception as e:
            logger.error(f"Failed to get updates: {e}")
            return []

    @classmethod
    def get_chat_id_from_updates(cls, bot_token: str) -> str | None:
        """
        Helper to get chat_id from recent messages.

        Usage:
            1. Send /start to your bot in Telegram
            2. Call this method to get your chat_id
        """
        try:
            response = requests.get(
                f"https://api.telegram.org/bot{bot_token}/getUpdates",
                timeout=10,
            )
            response.raise_for_status()
            updates = response.json().get("result", [])

            if updates:
                # Get the most recent chat_id
                chat_id = updates[-1]["message"]["chat"]["id"]
                print(f"Found chat_id: {chat_id}")
                return str(chat_id)
            else:
                print("No messages found. Send /start to the bot first.")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
