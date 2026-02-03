#!/usr/bin/env python
"""
ETL Pipeline Script

Extract data from Bloomberg, transform to features, load for model training.

Usage:
    python run_etl.py --config config/backtest.yaml --start-date 2020-01-01 --end-date 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

# Add src to path - need to add parent of src for package imports to work
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.parent))
sys.path.insert(0, str(src_path))

from quant_ensemble.src.common import get_logger
from quant_ensemble.src.ingestion import BloombergDataFetcher

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_etl(
    config_path: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    use_synthetic: bool = False,
) -> dict[str, Path]:
    """
    Run the ETL pipeline.

    Args:
        config_path: Path to configuration file
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory for processed data
        use_synthetic: Use synthetic data instead of Bloomberg

    Returns:
        Dictionary of output file paths
    """
    logger.info("=" * 60)
    logger.info("Starting ETL Pipeline")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    config = load_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize data fetcher
    logger.info("Initializing data fetcher...")

    if use_synthetic:
        fetcher = BloombergDataFetcher()
        fetcher._use_synthetic = True
    else:
        fetcher = BloombergDataFetcher()

    # Get universe
    universe = config.get("universe", {})
    index_ticker = universe.get("index_ticker", "KOSPI200 Index")
    n_stocks = universe.get("n_stocks", 100)

    logger.info(f"Fetching universe from {index_ticker}...")

    try:
        members = fetcher.get_index_members(index_ticker)
    except Exception as e:
        logger.warning(f"Failed to get index members: {e}")
        logger.info("Using synthetic universe...")
        members = [f"KR{i:04d}" for i in range(n_stocks)]

    tickers = members[:n_stocks]
    logger.info(f"Universe: {len(tickers)} tickers")

    # Fetch price data
    logger.info("Fetching price data...")

    price_fields = ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_VOLUME"]
    prices_df = fetcher.fetch_historical(
        tickers=tickers,
        fields=price_fields,
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(f"Fetched {len(prices_df)} price records")

    # Fetch fundamental data
    logger.info("Fetching fundamental data...")

    fundamental_fields = [
        "PE_RATIO",
        "PX_TO_BOOK_RATIO",
        "RETURN_COM_EQY",
        "TOT_DEBT_TO_TOT_EQY",
        "CUR_MKT_CAP",
    ]

    fundamentals_df = fetcher.fetch_fundamentals(
        tickers=tickers,
        fields=fundamental_fields,
    )

    logger.info(f"Fetched {len(fundamentals_df)} fundamental records")

    # Generate features
    logger.info("Generating technical features...")

    tech_features = TechnicalFeatures()
    tech_df = tech_features.generate(prices_df)

    logger.info(f"Generated {len(tech_df.columns)} technical features")

    logger.info("Generating fundamental features...")

    fund_features = FundamentalFeatures()
    fund_df = fund_features.generate(fundamentals_df)

    logger.info(f"Generated {len(fund_df.columns)} fundamental features")

    # Merge features
    logger.info("Merging features...")

    features_df = tech_df.merge(
        fund_df,
        on=["date", "asset_id"],
        how="left",
    )

    # Generate labels
    logger.info("Generating labels...")

    label_config = config.get("labels", {})
    label_gen = LabelGenerator(label_config)

    labels_df = label_gen.generate(
        prices_df,
        features_df,
    )

    logger.info(f"Generated {len(labels_df)} labels")

    # Save outputs
    logger.info("Saving outputs...")

    outputs = {}

    # Prices
    prices_path = output_path / "prices.parquet"
    prices_df.to_parquet(prices_path, index=False)
    outputs["prices"] = prices_path

    # Features
    features_path = output_path / "features.parquet"
    features_df.to_parquet(features_path, index=False)
    outputs["features"] = features_path

    # Labels
    labels_path = output_path / "labels.parquet"
    labels_df.to_parquet(labels_path, index=False)
    outputs["labels"] = labels_path

    # Fundamentals
    fundamentals_path = output_path / "fundamentals.parquet"
    fundamentals_df.to_parquet(fundamentals_path, index=False)
    outputs["fundamentals"] = fundamentals_path

    # Summary
    summary = {
        "run_date": datetime.now().isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "n_tickers": len(tickers),
        "n_price_records": len(prices_df),
        "n_feature_records": len(features_df),
        "n_label_records": len(labels_df),
        "n_features": len([c for c in features_df.columns if c not in ["date", "asset_id"]]),
        "output_files": {k: str(v) for k, v in outputs.items()},
    }

    summary_path = output_path / "etl_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    outputs["summary"] = summary_path

    logger.info("=" * 60)
    logger.info("ETL Pipeline Complete")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Total records: {len(features_df)}")
    logger.info("=" * 60)

    return outputs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ETL pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default="config/backtest.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of Bloomberg",
    )

    args = parser.parse_args()

    try:
        outputs = run_etl(
            config_path=args.config,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            use_synthetic=args.synthetic,
        )
        print(f"\nETL complete. Outputs saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"ETL failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
