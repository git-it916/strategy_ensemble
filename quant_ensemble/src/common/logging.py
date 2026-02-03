"""
Structured Logging Setup

Provides consistent logging across the entire system using loguru.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# Remove default handler
logger.remove()


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    run_id: str | None = None,
    json_logs: bool = False,
) -> str:
    """
    Setup logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (None = console only)
        run_id: Unique run identifier (auto-generated if None)
        json_logs: Whether to output JSON formatted logs

    Returns:
        The run_id used for this session
    """
    # Generate run_id if not provided
    if run_id is None:
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Console handler
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    if json_logs:
        logger.add(
            sys.stderr,
            level=level,
            serialize=True,
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=log_format,
            colorize=True,
        )

    # File handler (if log_dir specified)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        log_file = log_dir / f"run_{run_id}.log"
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="100 MB",
            retention="30 days",
            compression="gz",
        )

        # Error log file
        error_file = log_dir / f"error_{run_id}.log"
        logger.add(
            error_file,
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="50 MB",
            retention="90 days",
        )

    logger.info(f"Logging initialized with run_id: {run_id}")

    return run_id


def get_logger(name: str | None = None):
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance bound with the name
    """
    if name:
        return logger.bind(name=name)
    return logger


def log_config(config: dict[str, Any], config_name: str = "config") -> None:
    """
    Log configuration dictionary.

    Args:
        config: Configuration dictionary
        config_name: Name of the configuration
    """
    logger.info(f"{'=' * 50}")
    logger.info(f"Configuration: {config_name}")
    logger.info(f"{'=' * 50}")

    def _log_dict(d: dict, indent: int = 0):
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                _log_dict(value, indent + 1)
            else:
                logger.info(f"{prefix}{key}: {value}")

    _log_dict(config)
    logger.info(f"{'=' * 50}")


def log_metrics(metrics: dict[str, float], metrics_name: str = "metrics") -> None:
    """
    Log performance metrics.

    Args:
        metrics: Dictionary of metric name to value
        metrics_name: Name of the metrics group
    """
    logger.info(f"{'=' * 50}")
    logger.info(f"Metrics: {metrics_name}")
    logger.info(f"{'=' * 50}")

    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
        else:
            logger.info(f"  {name}: {value}")

    logger.info(f"{'=' * 50}")


def log_dataframe_info(df, df_name: str = "DataFrame") -> None:
    """
    Log DataFrame information.

    Args:
        df: pandas DataFrame
        df_name: Name of the DataFrame
    """
    logger.info(f"{'=' * 50}")
    logger.info(f"DataFrame: {df_name}")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {df.columns.tolist()}")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    if "date" in df.columns:
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    if "asset_id" in df.columns:
        logger.info(f"  Unique assets: {df['asset_id'].nunique()}")

    logger.info(f"{'=' * 50}")


class LogContext:
    """Context manager for logging blocks."""

    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.level = level
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        logger.log(self.level, f"[START] {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = datetime.now() - self.start_time
        if exc_type:
            logger.error(f"[FAILED] {self.name} - {exc_val} ({elapsed})")
        else:
            logger.log(self.level, f"[DONE] {self.name} ({elapsed})")
        return False
