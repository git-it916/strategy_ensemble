"""
Logging Configuration

Setup logging for the trading system.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from .settings import LOGS_DIR, LOGGING


def setup_logging(
    name: str = "strategy_ensemble",
    level: str | None = None,
    log_file: bool = True,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        name: Logger name
        level: Log level (default from settings)
        log_file: Whether to write to file

    Returns:
        Configured logger
    """
    level = level or LOGGING["level"]

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        log_filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(
            LOGS_DIR / log_filename,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(LOGGING["format"])
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Module name

    Returns:
        Logger instance
    """
    return logging.getLogger(f"strategy_ensemble.{name}")
