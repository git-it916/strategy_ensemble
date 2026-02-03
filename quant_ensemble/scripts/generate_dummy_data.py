#!/usr/bin/env python
"""
Generate Dummy Data Script

Generate synthetic data for testing when Bloomberg is not available.

Usage:
    python generate_dummy_data.py --output-dir data/processed --n-tickers 100 --n-days 1000
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common import get_logger

logger = get_logger(__name__)


def generate_price_data(
    tickers: list[str],
    start_date: datetime,
    n_days: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic price data with realistic properties.

    Args:
        tickers: List of ticker symbols
        start_date: Start date
        n_days: Number of trading days
        seed: Random seed

    Returns:
        Price DataFrame
    """
    np.random.seed(seed)

    records = []
    dates = pd.bdate_range(start=start_date, periods=n_days)

    for ticker in tickers:
        # Random parameters for each stock
        initial_price = np.random.uniform(10000, 200000)  # Korean won
        annual_drift = np.random.uniform(-0.1, 0.2)  # -10% to +20%
        annual_vol = np.random.uniform(0.2, 0.5)  # 20% to 50%

        daily_drift = annual_drift / 252
        daily_vol = annual_vol / np.sqrt(252)

        # Generate returns with mean reversion component
        returns = np.random.normal(daily_drift, daily_vol, n_days)

        # Add some autocorrelation (momentum)
        momentum_factor = 0.1
        for i in range(1, len(returns)):
            returns[i] += momentum_factor * returns[i - 1]

        # Generate prices
        prices = initial_price * np.cumprod(1 + returns)

        # Generate OHLCV
        for i, date in enumerate(dates):
            close = prices[i]
            daily_range = close * np.random.uniform(0.01, 0.03)

            open_price = close * (1 + np.random.uniform(-0.01, 0.01))
            high = max(open_price, close) + daily_range * np.random.uniform(0.3, 1.0)
            low = min(open_price, close) - daily_range * np.random.uniform(0.3, 1.0)

            # Volume (higher on volatile days)
            base_volume = np.random.uniform(100000, 10000000)
            volume_mult = 1 + abs(returns[i]) * 10
            volume = int(base_volume * volume_mult)

            records.append({
                "date": date,
                "asset_id": ticker,
                "PX_OPEN": round(open_price, 0),
                "PX_HIGH": round(high, 0),
                "PX_LOW": round(low, 0),
                "PX_LAST": round(close, 0),
                "close": round(close, 0),
                "open": round(open_price, 0),
                "PX_VOLUME": volume,
            })

    return pd.DataFrame(records)


def generate_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features from price data.

    Args:
        prices_df: Price DataFrame

    Returns:
        Features DataFrame
    """
    feature_records = []

    for ticker in prices_df["asset_id"].unique():
        ticker_data = prices_df[prices_df["asset_id"] == ticker].sort_values("date")

        if len(ticker_data) < 60:
            continue

        closes = ticker_data["PX_LAST"].values
        volumes = ticker_data["PX_VOLUME"].values
        dates = ticker_data["date"].values

        for i in range(60, len(ticker_data)):
            window_close = closes[i - 60:i]
            window_volume = volumes[i - 60:i]

            # Technical features
            returns_5d = (closes[i] - closes[i - 5]) / closes[i - 5] if closes[i - 5] > 0 else 0
            returns_21d = (closes[i] - closes[i - 21]) / closes[i - 21] if closes[i - 21] > 0 else 0
            returns_60d = (closes[i] - closes[i - 60]) / closes[i - 60] if closes[i - 60] > 0 else 0

            vol_5d = np.std(np.diff(np.log(window_close[-6:])))
            vol_21d = np.std(np.diff(np.log(window_close[-22:])))

            sma_5 = np.mean(window_close[-5:])
            sma_21 = np.mean(window_close[-21:])
            sma_60 = np.mean(window_close)

            # RSI
            gains = np.maximum(np.diff(window_close[-15:]), 0)
            losses = np.maximum(-np.diff(window_close[-15:]), 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            rsi = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 50

            # Volume features
            volume_sma = np.mean(window_volume[-21:])
            volume_ratio = volumes[i] / volume_sma if volume_sma > 0 else 1

            # Fundamental features (simulated)
            np.random.seed(hash(ticker) % 2**32)
            pe_ratio = np.random.uniform(5, 50)
            pb_ratio = np.random.uniform(0.5, 5)
            roe = np.random.uniform(-0.1, 0.3)
            debt_ratio = np.random.uniform(0, 2)
            market_cap = closes[i] * np.random.uniform(1e6, 1e9)

            feature_records.append({
                "date": dates[i],
                "asset_id": ticker,
                # Returns
                "ret_5d": returns_5d,
                "ret_21d": returns_21d,
                "ret_60d": returns_60d,
                # Volatility
                "vol_5d": vol_5d,
                "vol_21d": vol_21d,
                # Moving averages
                "close_to_sma5": closes[i] / sma_5 - 1 if sma_5 > 0 else 0,
                "close_to_sma21": closes[i] / sma_21 - 1 if sma_21 > 0 else 0,
                "close_to_sma60": closes[i] / sma_60 - 1 if sma_60 > 0 else 0,
                "sma5_to_sma21": sma_5 / sma_21 - 1 if sma_21 > 0 else 0,
                # Technical indicators
                "rsi_14": rsi,
                "volume_ratio": volume_ratio,
                # Fundamentals
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
                "roe": roe,
                "debt_to_equity": debt_ratio,
                "market_cap": market_cap,
            })

    return pd.DataFrame(feature_records)


def generate_labels(
    prices_df: pd.DataFrame,
    forward_days: int = 21,
) -> pd.DataFrame:
    """
    Generate labels from price data.

    Args:
        prices_df: Price DataFrame
        forward_days: Forward return horizon

    Returns:
        Labels DataFrame
    """
    label_records = []

    for ticker in prices_df["asset_id"].unique():
        ticker_data = prices_df[prices_df["asset_id"] == ticker].sort_values("date")

        if len(ticker_data) < forward_days + 60:
            continue

        closes = ticker_data["PX_LAST"].values
        dates = ticker_data["date"].values

        for i in range(60, len(ticker_data) - forward_days):
            fwd_return = (closes[i + forward_days] - closes[i]) / closes[i]

            label_records.append({
                "date": dates[i],
                "asset_id": ticker,
                "y_reg": fwd_return,
                "y_cls": 1 if fwd_return > 0 else 0,
                "forward_days": forward_days,
            })

    return pd.DataFrame(label_records)


def generate_dummy_data(
    output_dir: str,
    n_tickers: int = 100,
    n_days: int = 1000,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Generate complete dummy dataset.

    Args:
        output_dir: Output directory
        n_tickers: Number of tickers
        n_days: Number of trading days
        seed: Random seed

    Returns:
        Dictionary of output file paths
    """
    logger.info("=" * 60)
    logger.info("Generating Dummy Data")
    logger.info(f"Tickers: {n_tickers}, Days: {n_days}")
    logger.info("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate tickers (Korean-style)
    tickers = [f"A{i:06d}" for i in range(1, n_tickers + 1)]

    # Start date
    start_date = datetime.now() - timedelta(days=int(n_days * 1.5))

    # Generate prices
    logger.info("Generating price data...")
    prices_df = generate_price_data(tickers, start_date, n_days, seed)
    logger.info(f"Generated {len(prices_df)} price records")

    # Generate features
    logger.info("Generating features...")
    features_df = generate_features(prices_df)
    logger.info(f"Generated {len(features_df)} feature records with {len(features_df.columns)} features")

    # Generate labels
    logger.info("Generating labels...")
    labels_df = generate_labels(prices_df)
    logger.info(f"Generated {len(labels_df)} labels")

    # Save outputs
    logger.info("Saving outputs...")

    outputs = {}

    prices_path = output_path / "prices.parquet"
    prices_df.to_parquet(prices_path, index=False)
    outputs["prices"] = prices_path

    features_path = output_path / "features.parquet"
    features_df.to_parquet(features_path, index=False)
    outputs["features"] = features_path

    labels_path = output_path / "labels.parquet"
    labels_df.to_parquet(labels_path, index=False)
    outputs["labels"] = labels_path

    # Summary
    summary = {
        "run_date": datetime.now().isoformat(),
        "n_tickers": n_tickers,
        "n_days": n_days,
        "seed": seed,
        "date_range": {
            "start": str(prices_df["date"].min()),
            "end": str(prices_df["date"].max()),
        },
        "record_counts": {
            "prices": len(prices_df),
            "features": len(features_df),
            "labels": len(labels_df),
        },
        "output_files": {k: str(v) for k, v in outputs.items()},
    }

    summary_path = output_path / "dummy_data_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    outputs["summary"] = summary_path

    logger.info("=" * 60)
    logger.info("Dummy Data Generation Complete")
    logger.info(f"Output directory: {output_path}")
    logger.info("=" * 60)

    return outputs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate dummy data for testing")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory",
    )
    parser.add_argument(
        "--n-tickers",
        type=int,
        default=100,
        help="Number of tickers to generate",
    )
    parser.add_argument(
        "--n-days",
        type=int,
        default=1000,
        help="Number of trading days",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    try:
        outputs = generate_dummy_data(
            output_dir=args.output_dir,
            n_tickers=args.n_tickers,
            n_days=args.n_days,
            seed=args.seed,
        )
        print(f"\nDummy data generated. Files saved to: {args.output_dir}")
        print(f"  - prices.parquet")
        print(f"  - features.parquet")
        print(f"  - labels.parquet")

    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
