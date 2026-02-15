#!/usr/bin/env python
"""
0. Prepare Prices Script

파티셔닝된 raw 데이터를 읽어서 prices.parquet를 생성합니다.

Usage:
    python scripts/0_prepare_prices.py
    python scripts/0_prepare_prices.py --start-date 2020-01-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, PROCESSED_DATA_DIR
from config.logging_config import setup_logging

logger = setup_logging("prepare_prices")


def load_market_daily(
    data_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load partitioned market_daily data.

    Args:
        data_dir: Base data directory
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Combined DataFrame
    """
    market_dir = data_dir / "market_daily"

    if not market_dir.exists():
        raise FileNotFoundError(f"Market data not found: {market_dir}")

    logger.info("Loading market_daily data...")

    # Read all partitioned data
    df = pd.read_parquet(market_dir)

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Filter by date range
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    logger.info(
        f"Loaded {len(df)} market records from {df['date'].min()} to {df['date'].max()}"
    )
    logger.info(f"Unique tickers: {df['ticker'].nunique()}")

    return df


def load_fundamental_q(
    data_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load partitioned fundamental_Q data.

    Args:
        data_dir: Base data directory
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Combined DataFrame
    """
    # Note: folder name has typo "fundametal_Q"
    fundamental_dir = data_dir / "fundametal_Q"

    if not fundamental_dir.exists():
        logger.warning(f"Fundamental data not found: {fundamental_dir}")
        return pd.DataFrame()

    logger.info("Loading fundamental_Q data...")

    # Read all partitioned data
    df = pd.read_parquet(fundamental_dir)

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Filter by date range
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    logger.info(
        f"Loaded {len(df)} fundamental records from {df['date'].min()} to {df['date'].max()}"
    )

    return df


def merge_market_and_fundamental(
    market_df: pd.DataFrame,
    fundamental_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge market daily data with fundamental quarterly data.

    Fundamental data is forward-filled to match daily frequency.

    Args:
        market_df: Daily market data
        fundamental_df: Quarterly fundamental data

    Returns:
        Merged DataFrame
    """
    if fundamental_df.empty:
        logger.warning("No fundamental data to merge")
        return market_df

    logger.info("Merging market and fundamental data...")

    # Get unique tickers
    tickers = market_df["ticker"].unique()

    merged_frames = []

    for ticker in tickers:
        # Market data for this ticker
        market_ticker = market_df[market_df["ticker"] == ticker].copy()
        market_ticker = market_ticker.sort_values("date")

        # Fundamental data for this ticker
        fundamental_ticker = fundamental_df[fundamental_df["ticker"] == ticker].copy()

        if fundamental_ticker.empty:
            merged_frames.append(market_ticker)
            continue

        fundamental_ticker = fundamental_ticker.sort_values("date")

        # Merge with forward fill
        merged = pd.merge_asof(
            market_ticker,
            fundamental_ticker,
            on="date",
            by="ticker",
            direction="backward",
            suffixes=("", "_fund"),
        )

        merged_frames.append(merged)

    result = pd.concat(merged_frames, ignore_index=True)

    logger.info(f"Merged data: {len(result)} rows")

    return result


def create_prices_parquet(
    market_df: pd.DataFrame,
    fundamental_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create prices.parquet file.

    Args:
        market_df: Market daily data
        fundamental_df: Fundamental quarterly data
        output_path: Output file path
    """
    logger.info("Creating prices.parquet...")

    # Merge data
    prices = merge_market_and_fundamental(market_df, fundamental_df)

    # Sort by date and ticker
    prices = prices.sort_values(["date", "ticker"])

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(output_path, index=False, compression="snappy")

    logger.info(f"Saved prices.parquet: {len(prices)} rows")
    logger.info(f"Date range: {prices['date'].min()} to {prices['date'].max()}")
    logger.info(f"Tickers: {prices['ticker'].nunique()}")
    logger.info(f"Columns: {list(prices.columns)}")
    logger.info(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare prices.parquet from raw data")

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Prepare Prices Started")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 60)

    # Load market data
    market_df = load_market_daily(
        DATA_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Load fundamental data
    fundamental_df = load_fundamental_q(
        DATA_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Create prices.parquet
    output_path = PROCESSED_DATA_DIR / "prices.parquet"
    create_prices_parquet(market_df, fundamental_df, output_path)

    logger.info("=" * 60)
    logger.info("Prepare Prices Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
