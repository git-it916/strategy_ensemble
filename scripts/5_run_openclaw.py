#!/usr/bin/env python3
"""
OpenClaw Runner

CLI entry point for the OpenClaw autonomous alpha research & trading system.

Usage:
    python scripts/5_run_openclaw.py              # Run daemon (live)
    python scripts/5_run_openclaw.py --dry-run     # Dry run (no orders)
    python scripts/5_run_openclaw.py --research "query"  # One-off research
    python scripts/5_run_openclaw.py --status       # Print status and exit
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import LOGGING
from src.openclaw.main import OpenClawDaemon


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                project_root / "logs" / "openclaw" / "daemon.log",
                mode="a",
            ),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="OpenClaw Alpha Research & Trading")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without placing real orders"
    )
    parser.add_argument(
        "--research", type=str, default=None,
        help="Run a one-off research session with given query"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current status and exit"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    args = parser.parse_args()

    # Ensure log directory exists
    (project_root / "logs" / "openclaw").mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level)

    logger = logging.getLogger("openclaw")

    daemon = OpenClawDaemon(dry_run=args.dry_run)

    try:
        daemon.initialize()

        if args.status:
            # Print status and exit
            daemon._cmd_status("")
            return

        if args.research:
            # One-off research session
            daemon.run_research_session(query=args.research)
            return

        # Run main daemon loop
        logger.info("Starting OpenClaw daemon...")
        daemon.run()

    except KeyboardInterrupt:
        logger.info("Shutdown requested via Ctrl+C")
        daemon.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
