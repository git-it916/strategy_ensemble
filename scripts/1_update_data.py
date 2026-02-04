#!/usr/bin/env python
"""
1. Data Update Script

매일 장 마감 후 실행하여 데이터 업데이트.

Usage:
    python scripts/1_update_data.py
    python scripts/1_update_data.py --source bloomberg
    python scripts/1_update_data.py --full-refresh
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    DUCKDB_PATH,
    UNIVERSE,
)
from config.logging_config import setup_logging
from src.database import DuckDBConnector, SchemaManager
from src.etl import DataCleaner

logger = setup_logging("data_update")


def load_keys():
    """Load API keys from config."""
    import yaml
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"

    if not keys_path.exists():
        logger.warning("keys.yaml not found, using example config")
        keys_path = Path(__file__).parent.parent / "config" / "keys.example.yaml"

    with open(keys_path) as f:
        return yaml.safe_load(f)


def update_from_bloomberg(start_date: str, end_date: str) -> dict:
    """Update data from Bloomberg."""
    logger.info("Updating from Bloomberg...")

    try:
        import blpapi
        from src.database import DuckDBConnector

        # Use the existing Bloomberg fetch script logic
        script_path = Path(__file__).parent.parent / "quant_ensemble" / "scripts" / "fetch_bloomberg_data.py"

        if script_path.exists():
            import subprocess
            result = subprocess.run([
                sys.executable,
                str(script_path),
                "--start-date", start_date,
                "--end-date", end_date,
                "--output-dir", str(PROCESSED_DATA_DIR),
            ], capture_output=True, text=True)

            if result.returncode == 0:
                return {"status": "success", "source": "bloomberg"}
            else:
                logger.error(f"Bloomberg update failed: {result.stderr}")
                return {"status": "error", "error": result.stderr}
        else:
            logger.warning("Bloomberg fetch script not found")
            return {"status": "error", "error": "Script not found"}

    except ImportError:
        logger.warning("blpapi not available")
        return {"status": "error", "error": "blpapi not installed"}


def update_database_views():
    """Update DuckDB views after data refresh."""
    logger.info("Updating database views...")

    connector = DuckDBConnector(DUCKDB_PATH)
    schema_mgr = SchemaManager(connector, DATA_DIR)

    schema_mgr.setup_views()
    summary = schema_mgr.get_data_summary()

    logger.info(f"Database summary: {summary}")

    connector.close()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Update trading data")

    parser.add_argument(
        "--source",
        choices=["bloomberg", "csv"],
        default="bloomberg",
        help="Data source",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (default: 5 days ago)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (default: today)",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Full data refresh from 2020",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Data Update Started")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 60)

    # Determine date range
    if args.full_refresh:
        start_date = "2020-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = args.start_date or (
            datetime.now() - timedelta(days=5)
        ).strftime("%Y-%m-%d")

    logger.info(f"Date range: {start_date} to {end_date}")

    # Create directories
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Update data
    if args.source == "bloomberg":
        result = update_from_bloomberg(start_date, end_date)
    else:
        logger.info("CSV source not implemented, use Bloomberg")
        result = {"status": "skipped"}

    logger.info(f"Data update result: {result}")

    # Update database views
    if result.get("status") == "success":
        db_summary = update_database_views()
        logger.info(f"Database updated: {db_summary}")

    logger.info("=" * 60)
    logger.info("Data Update Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
