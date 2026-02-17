"""
Approval Agent — Step 6 of Sequential Pipeline (Human-in-the-Loop)

LLM 결정을 사용자에게 보여주고 승인/거부를 대기.
텔레그램 봇의 getUpdates를 통해 사용자 응답 수신.

Flow:
    1. 제안 메시지 전송 (텔레그램)
    2. 사용자 응답 대기 (Y/N + timeout)
    3. 승인 시 → 주문 실행
    4. 거부/타임아웃 → 건너뜀
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ApprovalAgent:
    """
    텔레그램 기반 Human-in-the-Loop 승인 에이전트.

    사용자가 "Y" 또는 "y"를 입력하면 승인.
    "N", "n", 또는 타임아웃이면 거부.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str | int,
        timeout_seconds: int = 300,
        poll_interval: float = 3.0,
    ):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval

    def request_approval(
        self,
        proposal_text: str,
        regime: str | None = None,
        confidence: float | None = None,
    ) -> bool:
        """
        사용자에게 제안을 보내고 승인을 대기.

        Args:
            proposal_text: RiskManager.format_proposal()의 출력
            regime: 현재 시장 레짐
            confidence: LLM 신뢰도

        Returns:
            True if approved, False if rejected or timeout
        """
        # 1. 제안 메시지 구성
        header = f"🤖 <b>AI 투자 제안</b>\n"
        header += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        if regime:
            regime_emoji = {"bull": "📈", "bear": "📉", "sideways": "➡️"}.get(
                regime, "📊"
            )
            header += f"{regime_emoji} 시장 레짐: {regime.upper()}\n"
        if confidence is not None:
            header += f"🎯 LLM 신뢰도: {confidence:.0%}\n"

        message = f"{header}\n{proposal_text}\n\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n"
        message += "✅ <b>Y</b> = 주문 실행\n"
        message += "❌ <b>N</b> = 건너뛰기\n"
        message += f"⏱ {self.timeout_seconds}초 후 자동 건너뛰기"

        # 2. 메시지 전송
        sent = self._send_message(message)
        if not sent:
            logger.error("Failed to send approval request")
            return False

        # 3. 현재 update_id 기록 (이전 메시지 무시하기 위해)
        baseline_update_id = self._get_latest_update_id()

        # 4. 사용자 응답 대기
        logger.info(
            f"Waiting for user approval (timeout: {self.timeout_seconds}s)..."
        )
        start_time = time.time()

        while time.time() - start_time < self.timeout_seconds:
            response = self._check_for_response(baseline_update_id)

            if response is not None:
                if response:
                    self._send_message("✅ <b>승인됨</b> — 주문 실행합니다.")
                    logger.info("User APPROVED the proposal")
                    return True
                else:
                    self._send_message("❌ <b>거부됨</b> — 이번 사이클 건너뜁니다.")
                    logger.info("User REJECTED the proposal")
                    return False

            time.sleep(self.poll_interval)

        # Timeout
        self._send_message("⏱ <b>타임아웃</b> — 응답 없어 건너뜁니다.")
        logger.info("Approval request TIMED OUT")
        return False

    def send_execution_result(
        self,
        orders: list[dict[str, Any]],
        success: bool = True,
    ) -> None:
        """주문 실행 결과 전송."""
        if success:
            emoji = "✅"
            status = "주문 실행 완료"
        else:
            emoji = "⚠️"
            status = "주문 실행 실패"

        lines = [f"{emoji} <b>{status}</b>\n"]
        for o in orders[:10]:
            side_emoji = "🟢" if o.get("side") == "BUY" else "🔴"
            lines.append(
                f"  {side_emoji} {o.get('stock_code', '')} "
                f"{o.get('side', '')} {o.get('quantity', 0):,}주"
            )

        self._send_message("\n".join(lines))

    def _send_message(self, text: str) -> bool:
        """텔레그램 메시지 전송."""
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def _get_latest_update_id(self) -> int:
        """현재 최신 update_id 조회."""
        try:
            resp = requests.get(
                f"{self.base_url}/getUpdates",
                params={"limit": 1, "offset": -1},
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json().get("result", [])
            if results:
                return results[-1]["update_id"]
            return 0
        except Exception:
            return 0

    def _check_for_response(self, after_update_id: int) -> bool | None:
        """
        사용자 응답 확인.

        Returns:
            True = approved, False = rejected, None = no response yet
        """
        try:
            resp = requests.get(
                f"{self.base_url}/getUpdates",
                params={
                    "offset": after_update_id + 1,
                    "timeout": 1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json().get("result", [])

            for update in results:
                msg = update.get("message", {})
                chat_id = str(msg.get("chat", {}).get("id", ""))
                text = (msg.get("text") or "").strip().upper()

                # 올바른 채팅방에서 온 메시지만 처리
                if chat_id != self.chat_id:
                    continue

                if text in ("Y", "YES", "ㅛ", "ㅇ"):
                    return True
                elif text in ("N", "NO", "ㅜ", "ㄴ"):
                    return False

            return None

        except Exception as e:
            logger.error(f"Failed to check updates: {e}")
            return None
