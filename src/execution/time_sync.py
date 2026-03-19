"""
바이낸스 서버 시간 동기화.

WSL2에서 시계가 지속적으로 밀리는 문제 해결.
매 호출 시 바이낸스 서버 시간을 기준으로 nonce를 직접 계산.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

BINANCE_TIME_URL = "https://fapi.binance.com/fapi/v1/time"

# 마지막으로 측정한 서버 시간과 로컬 monotonic 시계의 매핑
_server_time_ms: int = 0
_local_mono_at_sync: float = 0.0
_synced: bool = False


def sync_time_offset(exchange=None) -> int:
    """
    바이낸스 서버 시간을 측정하고, exchange의 nonce를 서버 시간 기준으로 고정.

    핵심: time.time() 대신 time.monotonic()으로 경과 시간을 측정하고,
    서버 시간에 더해서 nonce를 만듦. time.time()이 밀려도 영향 없음.
    """
    global _server_time_ms, _local_mono_at_sync, _synced

    try:
        mono_before = time.monotonic()
        resp = requests.get(BINANCE_TIME_URL, timeout=5)
        mono_after = time.monotonic()

        server_ms = resp.json()["serverTime"]
        latency_ms = int((mono_after - mono_before) * 1000)

        # 서버 시간 + 편도 레이턴시 보정
        _server_time_ms = server_ms + latency_ms // 2
        _local_mono_at_sync = (mono_before + mono_after) / 2
        _synced = True

        if exchange is not None:
            exchange.options["recvWindow"] = 60000
            exchange.options["adjustForTimeDifference"] = False
            # nonce와 milliseconds를 서버 시간 기준으로 고정
            exchange.nonce = _get_server_time_ms
            exchange.milliseconds = _get_server_time_ms

        # 드리프트 측정 (로깅용)
        drift = int(time.time() * 1000) - _get_server_time_ms()
        if abs(drift) > 1000:
            logger.info(f"Time sync OK (system drift={drift}ms, corrected via monotonic)")

        return drift

    except Exception as e:
        logger.error(f"Time sync failed: {e}")
        return 0


def _get_server_time_ms() -> int:
    """monotonic 시계 기반으로 현재 서버 시간 추정."""
    if not _synced:
        return int(time.time() * 1000)
    elapsed_ms = int((time.monotonic() - _local_mono_at_sync) * 1000)
    return _server_time_ms + elapsed_ms
