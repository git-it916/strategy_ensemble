"""
Telegram Channel Message Collector

Telethon을 사용한 실시간 공시/뉴스 메시지 수집.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from telethon import TelegramClient, events

logger = logging.getLogger(__name__)


def parse_channels(channels_str: str) -> list:
    """쉼표로 구분된 채널 문자열을 리스트로 변환."""
    if not channels_str:
        return []
    channels = []
    for ch in channels_str.split(","):
        ch = ch.strip()
        if not ch:
            continue
        if ch.lstrip("-").isdigit():
            channels.append(int(ch))
        else:
            channels.append(ch)
    return channels


def format_message_link(chat_id: int, message_id: int) -> str:
    """메시지 링크 생성."""
    if str(chat_id).startswith("-100"):
        clean_id = str(chat_id)[4:]
    else:
        clean_id = str(abs(chat_id))
    return f"https://t.me/c/{clean_id}/{message_id}"


class TelegramNewsCollector:
    """
    텔레그램 채널에서 실시간 뉴스/공시 수집.

    Usage:
        collector = TelegramNewsCollector(
            api_id="YOUR_API_ID",
            api_hash="YOUR_API_HASH",
            channels=["channel1", "channel2"],
        )

        # 콜백 등록
        collector.on_message(my_callback)

        # 실행
        await collector.start()
    """

    def __init__(
        self,
        api_id: str,
        api_hash: str,
        channels: list[str | int],
        session_name: str = "telegram_collector",
        session_dir: str | None = None,
    ):
        """
        Initialize collector.

        Args:
            api_id: Telegram API ID
            api_hash: Telegram API Hash
            channels: List of channel usernames or IDs
            session_name: Session file name
            session_dir: Directory for session file
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.channels = channels
        self.session_name = session_name

        # Session path
        if session_dir:
            session_path = Path(session_dir) / session_name
        else:
            session_path = session_name

        self.client = TelegramClient(str(session_path), api_id, api_hash)
        self._callbacks: list[Callable] = []
        self._messages: list[dict] = []

    def on_message(self, callback: Callable[[dict], None]) -> None:
        """
        메시지 수신 콜백 등록.

        Args:
            callback: Function that receives message dict
        """
        self._callbacks.append(callback)

    async def _handle_message(self, event) -> None:
        """내부 메시지 핸들러."""
        message = event.message
        chat = await event.get_chat()

        # 채널명
        if hasattr(chat, "title"):
            chat_title = chat.title
        elif hasattr(chat, "username"):
            chat_title = f"@{chat.username}"
        else:
            chat_title = "Unknown"

        chat_id = event.chat_id

        # 메시지 데이터 구조화
        msg_data = {
            "channel": chat_title,
            "channel_id": chat_id,
            "message_id": message.id,
            "text": message.text or "",
            "timestamp": message.date,
            "link": format_message_link(chat_id, message.id),
            "has_media": message.media is not None,
        }

        # 저장
        self._messages.append(msg_data)

        # 로깅
        logger.info(f"[{chat_title}] {msg_data['text'][:100]}...")

        # 콜백 실행
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(msg_data)
                else:
                    callback(msg_data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def start(self) -> None:
        """수집 시작."""
        logger.info("Starting Telegram collector...")
        logger.info(f"Monitoring {len(self.channels)} channels")

        await self.client.start()

        me = await self.client.get_me()
        logger.info(f"Logged in as: {me.first_name} (@{me.username})")

        # 이벤트 핸들러 등록
        self.client.add_event_handler(
            self._handle_message,
            events.NewMessage(chats=self.channels),
        )

        logger.info("Listening for messages...")
        await self.client.run_until_disconnected()

    async def stop(self) -> None:
        """수집 중지."""
        await self.client.disconnect()
        logger.info("Collector stopped")

    def get_messages(self) -> list[dict]:
        """수집된 메시지 반환."""
        return self._messages.copy()

    def clear_messages(self) -> None:
        """메시지 버퍼 비우기."""
        self._messages.clear()

    def get_recent_summary(self, n: int = 20) -> list[dict]:
        """
        최근 N개 메시지를 요약 형태로 반환 (파이프라인 연동용).

        Returns:
            [{"channel": str, "text": str, "timestamp": datetime}, ...]
        """
        recent = self._messages[-n:] if self._messages else []
        return [
            {
                "channel": m.get("channel", ""),
                "text": m.get("text", ""),
                "timestamp": m.get("timestamp"),
            }
            for m in recent
            if m.get("text", "").strip()
        ]

    async def fetch_history(
        self,
        channel: str | int,
        limit: int = 100,
    ) -> list[dict]:
        """
        채널의 과거 메시지 가져오기.

        Args:
            channel: Channel username or ID
            limit: Number of messages to fetch

        Returns:
            List of message dicts
        """
        messages = []

        async with self.client:
            async for message in self.client.iter_messages(channel, limit=limit):
                chat = await message.get_chat()

                if hasattr(chat, "title"):
                    chat_title = chat.title
                else:
                    chat_title = str(channel)

                msg_data = {
                    "channel": chat_title,
                    "channel_id": message.chat_id,
                    "message_id": message.id,
                    "text": message.text or "",
                    "timestamp": message.date,
                    "link": format_message_link(message.chat_id, message.id),
                    "has_media": message.media is not None,
                }
                messages.append(msg_data)

        return messages


# Standalone runner
async def main():
    """Standalone execution."""
    from pathlib import Path
    import yaml

    # keys.yaml에서 설정 로드
    keys_path = Path(__file__).parent.parent.parent / "config" / "keys.yaml"

    if not keys_path.exists():
        print(f"keys.yaml not found at {keys_path}")
        print("Copy keys.example.yaml to keys.yaml and fill in credentials")
        return

    with open(keys_path, encoding="utf-8") as f:
        keys = yaml.safe_load(f)

    telegram_api = keys.get("telegram_api", {})
    api_id = str(telegram_api.get("api_id", ""))
    api_hash = str(telegram_api.get("api_hash", ""))
    channels_str = str(telegram_api.get("channels", ""))

    if not api_id or not api_hash or "YOUR" in api_id or "YOUR" in api_hash:
        print("Set telegram_api credentials in config/keys.yaml:")
        print("  api_id: YOUR_API_ID")
        print("  api_hash: YOUR_API_HASH")
        print("\nGet credentials from https://my.telegram.org")
        return

    channels = parse_channels(channels_str)
    if not channels:
        print("Set telegram_api.channels in config/keys.yaml (comma-separated)")
        return

    print(f"API ID: {api_id}")
    print(f"Channels: {channels}")

    collector = TelegramNewsCollector(
        api_id=api_id,
        api_hash=api_hash,
        channels=channels,
    )

    # 콘솔 출력 콜백
    def print_message(msg: dict):
        print("\n" + "=" * 60)
        print(f"📢 채널: {msg['channel']}")
        print(f"🕐 시간: {msg['timestamp']}")
        print(f"🔗 링크: {msg['link']}")
        print("-" * 60)
        print(f"📝 내용:\n{msg['text'] or '[미디어/빈 메시지]'}")
        print("=" * 60)

    collector.on_message(print_message)

    try:
        await collector.start()
    except KeyboardInterrupt:
        print("\n종료됨")
        await collector.stop()


if __name__ == "__main__":
    asyncio.run(main())
