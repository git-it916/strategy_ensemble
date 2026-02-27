"""
Telegram Command Handler

Handles /research, /status, /kill, /approve, /reject commands
and inline keyboard callbacks for alpha approval.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import requests

from src.openclaw.telegram.formatters import TelegramFormatter

logger = logging.getLogger(__name__)


class OpenClawTelegramHandler:
    """
    Telegram bot for OpenClaw commands and approval workflow.

    Commands:
        /research [query]  - Trigger alpha research session
        /status            - Show active alphas + performance
        /kill <name>       - Manually kill an alpha
        /approve <name>    - Approve a pending alpha
        /reject <name>     - Reject a pending alpha
        /help              - Show available commands
    """

    POLL_INTERVAL = 2
    APPROVAL_TIMEOUT = 300  # 5 minutes

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

        if not self.token or not self.chat_id:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required. "
                "Set via env vars or pass as arguments."
            )

        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self._last_update_id = self._get_last_update_id()
        self.formatter = TelegramFormatter()

        # Command handlers (registered by main.py)
        self._command_handlers: dict[str, Callable] = {}
        self._chat_handler = None

    def set_chat_handler(self, handler) -> None:
        """Set the conversational chat handler for natural language messages."""
        self._chat_handler = handler
        logger.info("Chat handler registered")

    def register_command(self, command: str, handler: Callable) -> None:
        """Register a command handler function."""
        self._command_handlers[command] = handler
        logger.info(f"Registered command: /{command}")

    # ── Telegram API Methods ──────────────────────────────────────────

    def _get(self, method: str, params: dict | None = None) -> dict:
        r = requests.get(
            f"{self.base_url}/{method}", params=params, timeout=10
        )
        r.raise_for_status()
        return r.json()

    def _post(self, method: str, data: dict) -> dict:
        r = requests.post(
            f"{self.base_url}/{method}", json=data, timeout=10
        )
        r.raise_for_status()
        return r.json()

    def _get_last_update_id(self) -> int:
        try:
            resp = self._get("getUpdates", {"limit": 1, "offset": -1})
            updates = resp.get("result", [])
            if updates:
                return updates[-1]["update_id"]
        except Exception:
            pass
        return 0

    def send_message(
        self,
        text: str,
        reply_markup: dict | None = None,
        parse_mode: str = "HTML",
    ) -> int:
        """Send message, return message_id."""
        data: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text[:4096],  # Telegram limit
            "parse_mode": parse_mode,
        }
        if reply_markup:
            data["reply_markup"] = reply_markup

        try:
            resp = self._post("sendMessage", data)
            return resp["result"]["message_id"]
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return 0

    def answer_callback(self, callback_query_id: str, text: str = "") -> None:
        """Answer inline keyboard callback."""
        try:
            self._post("answerCallbackQuery", {
                "callback_query_id": callback_query_id,
                "text": text,
            })
        except Exception as e:
            logger.warning(f"Failed to answer callback: {e}")

    # ── Command Polling ───────────────────────────────────────────────

    def poll_commands(self) -> list[dict]:
        """
        Check for new commands and callbacks. Returns list of processed events.

        Should be called periodically (e.g. every 5 seconds) from main loop.
        """
        processed = []
        offset = self._last_update_id + 1

        try:
            resp = self._get("getUpdates", {
                "offset": offset,
                "timeout": 1,
                "allowed_updates": ["message", "callback_query"],
            })
            updates = resp.get("result", [])

            for upd in updates:
                self._last_update_id = upd["update_id"]

                # Handle text commands
                msg = upd.get("message")
                if msg:
                    if str(msg.get("chat", {}).get("id", "")) != str(self.chat_id):
                        continue

                    text = (msg.get("text") or "").strip()
                    if not text:
                        continue
                    if text.startswith("/"):
                        event = self._handle_command(text)
                        if event:
                            processed.append(event)
                    elif self._chat_handler:
                        event = self._handle_text_message(text)
                        if event:
                            processed.append(event)

                # Handle callback queries (inline keyboard)
                cq = upd.get("callback_query")
                if cq:
                    if str(cq["message"]["chat"]["id"]) != str(self.chat_id):
                        continue

                    data = cq.get("data", "")
                    self.answer_callback(cq["id"], "Processing...")
                    event = self._handle_callback(data)
                    if event:
                        processed.append(event)

        except Exception as e:
            logger.warning(f"Command poll error: {e}")

        return processed

    def _handle_command(self, text: str) -> dict | None:
        """Parse and dispatch a command."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower().lstrip("/")
        args = parts[1] if len(parts) > 1 else ""

        logger.info(f"Received command: /{cmd} {args}")

        if cmd == "help":
            self.send_message(
                "<b>OpenClaw Commands</b>\n\n"
                "/research [query] - Search for new alphas\n"
                "/status - Show active alphas\n"
                "/kill &lt;name&gt; - Kill an alpha\n"
                "/approve &lt;name&gt; - Approve pending alpha\n"
                "/reject &lt;name&gt; - Reject pending alpha\n"
                "/help - Show this help"
            )
            return {"type": "command", "command": "help"}

        handler = self._command_handlers.get(cmd)
        if handler:
            try:
                handler(args)
            except Exception as e:
                logger.error(f"Command handler error for /{cmd}: {e}")
                self.send_message(
                    self.formatter.error_alert(str(e), f"/{cmd} {args}")
                )

        return {"type": "command", "command": cmd, "args": args}

    def _handle_text_message(self, text: str) -> dict | None:
        """Route natural language message to chat handler."""
        logger.info(f"Received text message: {text[:50]}")
        try:
            reply = self._chat_handler.handle(text)
            if reply:
                self.send_message(reply, parse_mode="")
        except Exception as e:
            logger.error(f"Chat handler error: {e}")
        return {"type": "chat", "text": text}

    def _handle_callback(self, data: str) -> dict | None:
        """Handle inline keyboard callback data."""
        logger.info(f"Received callback: {data}")

        if data.startswith("approve_"):
            alpha_name = data[len("approve_"):]
            handler = self._command_handlers.get("approve")
            if handler:
                try:
                    handler(alpha_name)
                except Exception as e:
                    logger.error(f"Approve callback error: {e}")
            return {"type": "callback", "action": "approve", "alpha": alpha_name}

        elif data.startswith("reject_"):
            alpha_name = data[len("reject_"):]
            handler = self._command_handlers.get("reject")
            if handler:
                try:
                    handler(alpha_name)
                except Exception as e:
                    logger.error(f"Reject callback error: {e}")
            return {"type": "callback", "action": "reject", "alpha": alpha_name}

        return {"type": "callback", "data": data}

    # ── Approval Workflow ─────────────────────────────────────────────

    def request_alpha_approval(
        self,
        alpha_name: str,
        summary_text: str,
        reply_markup: dict,
        timeout: int | None = None,
    ) -> bool:
        """
        Send approval request and wait for callback response.

        Args:
            alpha_name: Name of the alpha awaiting approval
            summary_text: HTML-formatted summary
            reply_markup: Inline keyboard markup
            timeout: Max wait seconds (default: APPROVAL_TIMEOUT)

        Returns:
            True if approved, False if rejected or timed out.
        """
        self.send_message(summary_text, reply_markup=reply_markup)
        logger.info(f"Approval request sent for: {alpha_name}")

        timeout = timeout or self.APPROVAL_TIMEOUT
        deadline = time.time() + timeout
        offset = self._last_update_id + 1

        while time.time() < deadline:
            try:
                resp = self._get("getUpdates", {
                    "offset": offset,
                    "timeout": self.POLL_INTERVAL,
                    "allowed_updates": ["callback_query"],
                })

                for upd in resp.get("result", []):
                    offset = upd["update_id"] + 1
                    self._last_update_id = upd["update_id"]

                    cq = upd.get("callback_query")
                    if not cq:
                        continue

                    if str(cq["message"]["chat"]["id"]) != str(self.chat_id):
                        continue

                    data = cq.get("data", "")
                    self.answer_callback(cq["id"])

                    if data == f"approve_{alpha_name}":
                        self.send_message(
                            f"Approved: {alpha_name} → paper trading"
                        )
                        return True
                    elif data == f"reject_{alpha_name}":
                        self.send_message(f"Rejected: {alpha_name}")
                        return False

            except Exception as e:
                logger.warning(f"Approval poll error: {e}")
                time.sleep(self.POLL_INTERVAL)

        # Timeout
        self.send_message(f"Approval timed out for {alpha_name} → rejected")
        return False
