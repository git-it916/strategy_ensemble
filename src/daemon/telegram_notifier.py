"""
텔레그램 알림 모듈
- 포지션 진입/청산 시 한글 알림 발송
- python-telegram-bot 또는 requests 기반 (의존성 최소화)

사용법:
    from telegram_notifier import TelegramNotifier

    notifier = TelegramNotifier(
        bot_token="YOUR_BOT_TOKEN",
        chat_id="YOUR_CHAT_ID"
    )

    # 진입 알림
    notifier.send_entry(
        coin="ETH/USDT",
        direction="LONG",
        entry_price=3000.50,
        stop_loss_price=2850.48,
        take_profit_price=3300.55,
        alpha_contributions={
            "MomentumMultiScale": 0.72,
            "FundingCarryEnhanced": 0.45,
            "IntradayRSIV2": -0.12,
        }
    )

    # 청산 알림
    notifier.send_exit(
        coin="ETH/USDT",
        direction="LONG",
        entry_price=3000.50,
        exit_price=3300.55,
        reason="TP",
        pnl_usdt=90.02,
        pnl_pct=10.0,
    )

텔레그램 봇 토큰 발급:
    1. @BotFather에게 /newbot 전송
    2. 봇 이름 설정 후 토큰 수령

채팅 ID 확인:
    1. 봇에게 아무 메시지 전송
    2. https://api.telegram.org/bot<TOKEN>/getUpdates 접속
    3. result[0].message.chat.id 확인
"""

import requests
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 한글 매핑
# ──────────────────────────────────────────────

DIRECTION_KR = {
    "LONG": "🟢 롱 (매수)",
    "SHORT": "🔴 숏 (매도)",
}

REASON_KR = {
    "SL": "🛑 손절 (Stop Loss)",
    "TP": "🎯 익절 (Take Profit)",
    "TRAILING": "📐 트레일링 스탑",
    "SIGNAL": "🔄 시그널 반전",
    "MANUAL": "✋ 수동 청산",
    "LIQUIDATION": "💀 강제 청산",
}

ALPHA_NAME_KR = {
    "MomentumMultiScale": "단기모멘텀",
    "FundingCarryEnhanced": "펀딩캐리",
    "MomentumComposite": "장기모멘텀",
    "IntradayVWAPV2": "VWAP이탈",
    "IntradayRSIV2": "RSI과열",
    "DerivativesSentiment": "파생심리",
    "MeanReversionMultiHorizon": "평균회귀",
    "OrderbookImbalance": "호가불균형",
    "SpreadMomentum": "스프레드",
    "VolatilityRegime": "변동성레짐",
}

# ──────────────────────────────────────────────
# 코인 이름 매핑 (주요 코인)
# ──────────────────────────────────────────────

COIN_NAME_KR = {
    "BTC": "비트코인",
    "ETH": "이더리움",
    "BNB": "바이낸스코인",
    "SOL": "솔라나",
    "XRP": "리플",
    "ADA": "에이다",
    "AVAX": "아발란체",
    "DOGE": "도지코인",
    "DOT": "폴카닷",
    "MATIC": "폴리곤",
    "LINK": "체인링크",
    "UNI": "유니스왑",
    "ATOM": "코스모스",
    "LTC": "라이트코인",
    "FIL": "파일코인",
    "ARB": "아비트럼",
    "OP": "옵티미즘",
    "APT": "앱토스",
    "SUI": "수이",
    "NEAR": "니어",
    "FET": "페치AI",
    "RENDER": "렌더",
    "INJ": "인젝티브",
    "TIA": "셀레스티아",
    "SEI": "세이",
    "WLD": "월드코인",
    "JUP": "주피터",
    "PEPE": "페페",
    "WIF": "위프",
    "BONK": "봉크",
}


def _parse_coin_symbol(coin: str) -> str:
    """'ETH/USDT:USDT' -> 'ETH', 'BTCUSDT' -> 'BTC' 등 다양한 포맷 처리"""
    symbol = coin.upper().split("/")[0].split(":")[0]
    for suffix in ["USDT", "BUSD", "USD", "PERP"]:
        if symbol.endswith(suffix):
            symbol = symbol[: -len(suffix)]
            break
    return symbol


def _coin_display(coin: str) -> str:
    """코인 심볼 -> '이더리움(ETH)' 형태"""
    symbol = _parse_coin_symbol(coin)
    kr_name = COIN_NAME_KR.get(symbol, symbol)
    return f"{kr_name}({symbol})"


def _format_price(price: float) -> str:
    """가격 포맷: 소수점 아래 유효숫자 보존"""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"


def _format_alpha_bar(score: float, max_width: int = 8) -> str:
    """알파 점수를 시각적 바로 표현"""
    filled = int(abs(score) * max_width)
    filled = min(filled, max_width)
    if score > 0:
        return "▓" * filled + "░" * (max_width - filled) + f" +{score:.2f}"
    elif score < 0:
        return "░" * (max_width - filled) + "▓" * filled + f" {score:.2f}"
    else:
        return "░" * max_width + "  0.00"


# ──────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────

class TelegramNotifier:
    """텔레그램 트레이딩 알림 발송기"""

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def _send(self, text: str) -> bool:
        """텔레그램 메시지 발송 (HTML 파싱 모드)"""
        if not self.enabled:
            import re
            print(re.sub(r"<[^>]+>", "", text))
            return True

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(self.api_url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("[TG] 알림 발송 완료")
                return True
            else:
                logger.error(f"[TG] 발송 실패: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"[TG] 발송 에러: {e}")
            return False

    # ──────────────────────────────────────────
    # 진입 알림
    # ──────────────────────────────────────────

    def send_entry(
        self,
        coin: str,
        direction: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        alpha_contributions: Optional[dict] = None,
        leverage: float = 3.0,
        ensemble_method: str = "weighted_sum",
        ensemble_score: float = 0.0,
    ) -> bool:
        kst = datetime.now(timezone(timedelta(hours=9)))
        dir_kr = DIRECTION_KR.get(direction.upper(), direction)
        coin_display = _coin_display(coin)

        if direction.upper() == "LONG":
            sl_pct = (stop_loss_price - entry_price) / entry_price * 100
            tp_pct = (take_profit_price - entry_price) / entry_price * 100
        else:
            sl_pct = (entry_price - stop_loss_price) / entry_price * 100 * -1
            tp_pct = (entry_price - take_profit_price) / entry_price * 100 * -1

        lines = [
            f"━━━━━━━━━━━━━━━━━━━━",
            f"📈 <b>포지션 진입</b>",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"코인: <b>{coin_display}</b>",
            f"방향: {dir_kr}",
            f"레버리지: {leverage:.0f}배",
            f"앙상블: {ensemble_method} (점수: {ensemble_score:+.3f})",
            f"",
            f"진입가: <b>{_format_price(entry_price)}</b>",
            f"손절가: {_format_price(stop_loss_price)} ({sl_pct:+.1f}%)",
            f"익절가: {_format_price(take_profit_price)} ({tp_pct:+.1f}%)",
        ]

        if alpha_contributions:
            lines.append("")
            lines.append("📊 <b>알파 기여도</b>")
            sorted_alphas = sorted(
                alpha_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            for name, score in sorted_alphas:
                if abs(score) < 0.01:
                    continue
                kr_name = ALPHA_NAME_KR.get(name, name)
                bar = _format_alpha_bar(score)
                lines.append(f"  <code>{kr_name:　<6} {bar}</code>")

        lines.append("")
        lines.append(f"⏰ {kst.strftime('%Y-%m-%d %H:%M:%S')} KST")

        return self._send("\n".join(lines))

    # ──────────────────────────────────────────
    # 청산 알림
    # ──────────────────────────────────────────

    def send_exit(
        self,
        coin: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        reason: str,
        pnl_usdt: float,
        pnl_pct: float,
        hold_duration_min: Optional[float] = None,
    ) -> bool:
        kst = datetime.now(timezone(timedelta(hours=9)))
        dir_kr = DIRECTION_KR.get(direction.upper(), direction)
        reason_kr = REASON_KR.get(reason.upper(), reason)
        coin_display = _coin_display(coin)

        if pnl_usdt > 0:
            pnl_emoji = "💰"
            result_text = "수익"
        elif pnl_usdt < 0:
            pnl_emoji = "💸"
            result_text = "손실"
        else:
            pnl_emoji = "➖"
            result_text = "본전"

        lines = [
            f"━━━━━━━━━━━━━━━━━━━━",
            f"{pnl_emoji} <b>포지션 청산</b>",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"코인: <b>{coin_display}</b>",
            f"방향: {dir_kr}",
            f"사유: {reason_kr}",
            f"",
            f"진입가: {_format_price(entry_price)}",
            f"청산가: {_format_price(exit_price)}",
            f"",
            f"<b>{result_text}: {pnl_usdt:+,.2f} USDT ({pnl_pct:+.2f}%)</b>",
        ]

        if hold_duration_min is not None:
            if hold_duration_min >= 60:
                hours = int(hold_duration_min // 60)
                mins = int(hold_duration_min % 60)
                lines.append(f"보유시간: {hours}시간 {mins}분")
            else:
                lines.append(f"보유시간: {int(hold_duration_min)}분")

        lines.append("")
        lines.append(f"⏰ {kst.strftime('%Y-%m-%d %H:%M:%S')} KST")

        return self._send("\n".join(lines))
