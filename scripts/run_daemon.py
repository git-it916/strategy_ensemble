#!/usr/bin/env python3
"""
Unified Daemon Runner

Scripts 4 + 5를 하나로 합친 통합 데몬 진입점.
전략 연구 → 백테스트 → 승인 → 실시간 매매까지 전부 자동화.

Usage:
    python scripts/run_daemon.py                    # 기본 (AI 자율 매매)
    python scripts/run_daemon.py --dry-run          # 신호만, 주문 안 넣음
    python scripts/run_daemon.py --require-approval # 매매 전 승인 필요
    python scripts/run_daemon.py --auto-research    # 자동 연구 활성화
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logging

logger = setup_logging("unified_daemon")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Daemon — OpenClaw + Trading Pipeline"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate signals and proposals only, no real orders",
    )
    parser.add_argument(
        "--require-approval", action="store_true",
        help="Require human approval before executing trades",
    )
    parser.add_argument(
        "--auto-research", action="store_true",
        help="Enable automatic 24h research sessions (uses Claude API tokens)",
    )

    args = parser.parse_args()

    from src.daemon.unified_daemon import UnifiedDaemon

    daemon = UnifiedDaemon(
        dry_run=args.dry_run,
        require_approval=args.require_approval,
        enable_research=args.auto_research,
    )

    def signal_handler(_sig, _frame):
        logger.info("Shutdown signal received")
        daemon.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        daemon.initialize()
        daemon.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
