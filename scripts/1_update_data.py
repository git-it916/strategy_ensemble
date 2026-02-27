#!/usr/bin/env python
"""
1. Data Update Script — Binance USDT-M Futures

Binance에서 거래량 상위 50개 USDT-M 무기한 선물 OHLCV + 펀딩비율 수집.

Usage:
    python scripts/1_update_data.py                    # 최근 5일 업데이트
    python scripts/1_update_data.py --full-refresh      # 300일 전체 재수집
    python scripts/1_update_data.py --build-features    # 피처/라벨도 재생성
    python scripts/1_update_data.py --days 60           # 최근 60일
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    PROCESSED_DATA_DIR,
    UNIVERSE,
)
from config.logging_config import setup_logging

logger = setup_logging("data_update")


def load_keys() -> dict | None:
    """Load API keys from config/keys.yaml."""
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"
    if not keys_path.exists():
        logger.error("keys.yaml not found! Copy from keys.example.yaml")
        return None
    with open(keys_path) as f:
        return yaml.safe_load(f)


def update_from_binance(
    api_key: str,
    api_secret: str,
    days: int = 300,
    n_symbols: int | None = None,
) -> dict:
    """
    Binance USDT-M 선물 OHLCV + 펀딩비율 수집.

    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        days: 수집 일수
        n_symbols: 거래량 기준 상위 N개 심볼 (None → settings에서)

    Returns:
        업데이트 결과 dict
    """
    from src.execution.binance_api import BinanceApi

    n = n_symbols or UNIVERSE.get("top_n_by_volume", 50)
    exclude = UNIVERSE.get("exclude_symbols", [])

    api = BinanceApi(api_key=api_key, api_secret=api_secret)

    # 1. 유니버스 결정 (거래량 상위 N개)
    logger.info(f"Fetching top {n} symbols by 24h volume...")
    symbols = api.get_top_symbols(n=n, exclude=exclude)
    logger.info(f"Universe: {len(symbols)} symbols")
    logger.info(f"  {symbols[:5]} ... {symbols[-3:]}")

    # 2. OHLCV 수집
    logger.info(f"Fetching {days}d OHLCV for {len(symbols)} symbols...")
    prices = api.get_ohlcv_batch(symbols, timeframe="1d", days=days)
    prices = prices.rename(columns={"ticker": "ticker"})  # already 'ticker'

    if prices.empty:
        logger.error("No OHLCV data fetched")
        return {"status": "error", "error": "No OHLCV data"}

    logger.info(
        f"OHLCV: {len(prices)} rows, {prices['ticker'].nunique()} symbols, "
        f"{prices['date'].min().date()} ~ {prices['date'].max().date()}"
    )

    # 3. 펀딩비율 수집 (최근 90일)
    funding_days = min(days, 90)
    logger.info(f"Fetching {funding_days}d funding rate history...")
    funding = api.get_funding_history_batch(symbols, days=funding_days)

    if not funding.empty:
        logger.info(
            f"Funding rates: {len(funding)} rows, "
            f"{funding['date'].min().date()} ~ {funding['date'].max().date()}"
        )

    # 4. 저장
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    prices_path = PROCESSED_DATA_DIR / "prices.parquet"
    prices.to_parquet(prices_path, index=False, compression="snappy")
    logger.info(f"Saved: {prices_path}")

    if not funding.empty:
        funding_path = PROCESSED_DATA_DIR / "funding_rates.parquet"
        funding.to_parquet(funding_path, index=False, compression="snappy")
        logger.info(f"Saved: {funding_path}")

    # 5. features.parquet에 펀딩비율 병합 (FundingRateCarry용)
    if not funding.empty:
        _merge_funding_into_features(prices, funding)

    return {
        "status": "success",
        "source": "binance",
        "n_records": len(prices),
        "n_symbols": prices["ticker"].nunique(),
        "date_range": f"{prices['date'].min().date()} ~ {prices['date'].max().date()}",
    }


def _merge_funding_into_features(
    prices: pd.DataFrame,
    funding: pd.DataFrame,
) -> None:
    """
    funding_rates를 prices에 left-join하여 features.parquet로 저장.

    FundingRateCarry alpha가 features['funding_rate']를 사용.
    """
    features = prices[["date", "ticker"]].copy()
    features = features.merge(
        funding[["date", "ticker", "funding_rate"]],
        on=["date", "ticker"],
        how="left",
    )
    features_path = PROCESSED_DATA_DIR / "features.parquet"
    features.to_parquet(features_path, index=False, compression="snappy")
    logger.info(f"Saved features (with funding_rate): {features_path} ({len(features)} rows)")


def build_features_and_labels() -> None:
    """피처 및 라벨 재생성 (기술적 피처)."""
    prices_path = PROCESSED_DATA_DIR / "prices.parquet"
    if not prices_path.exists():
        logger.error("prices.parquet not found!")
        return

    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])

    logger.info(f"Building technical features from {len(prices)} records...")

    try:
        from src.etl import FeatureEngineer
        fe = FeatureEngineer()
        features = fe.build_daily_features(prices)

        # Preserve funding_rate column if it exists in the previous features file
        features_path = PROCESSED_DATA_DIR / "features.parquet"
        if features_path.exists():
            try:
                prev_features = pd.read_parquet(features_path)
                if "funding_rate" in prev_features.columns:
                    funding_cols = prev_features[["date", "ticker", "funding_rate"]].drop_duplicates()
                    funding_cols["date"] = pd.to_datetime(funding_cols["date"])
                    features["date"] = pd.to_datetime(features["date"])
                    if "funding_rate" not in features.columns:
                        features = features.merge(
                            funding_cols, on=["date", "ticker"], how="left"
                        )
                        logger.info("Preserved funding_rate from previous features")
            except Exception as e:
                logger.warning(f"Could not preserve funding_rate: {e}")

        features.to_parquet(
            features_path, index=False, compression="snappy"
        )
        logger.info(f"Features saved: {len(features)} records")
    except Exception as e:
        logger.warning(f"FeatureEngineer failed (non-critical): {e}")

    try:
        from src.etl import LabelEngineer
        le = LabelEngineer()
        labels = le.build_all(prices)
        for name, label_df in labels.items():
            path = PROCESSED_DATA_DIR / f"labels_{name}.parquet"
            label_df.to_parquet(path, index=False, compression="snappy")
            logger.info(f"Labels ({name}) saved: {len(label_df)} records")
    except Exception as e:
        logger.warning(f"LabelEngineer failed (non-critical): {e}")


def main():
    parser = argparse.ArgumentParser(description="Update trading data from Binance Futures")

    parser.add_argument(
        "--full-refresh", action="store_true",
        help="Full refresh: fetch 300 days of data",
    )
    parser.add_argument(
        "--days", type=int, default=5,
        help="Number of days to fetch (default: 5)",
    )
    parser.add_argument(
        "--build-features", action="store_true",
        help="Rebuild technical features and labels after update",
    )
    parser.add_argument(
        "--n-symbols", type=int, default=None,
        help="Number of top symbols by volume (default: from settings)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Data Update Started (Binance USDT-M Futures)")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 60)

    days = 300 if args.full_refresh else args.days

    keys = load_keys()
    if keys is None:
        sys.exit(1)

    binance_keys = keys.get("binance", {})
    api_key = binance_keys.get("api_key", "")
    api_secret = binance_keys.get("api_secret", "")

    if not api_key or "YOUR" in api_key:
        logger.error("Binance API keys not configured in config/keys.yaml")
        sys.exit(1)

    result = update_from_binance(
        api_key=api_key,
        api_secret=api_secret,
        days=days,
        n_symbols=args.n_symbols,
    )
    logger.info(f"Update result: {result}")

    if args.build_features and result.get("status") == "success":
        build_features_and_labels()

    logger.info("=" * 60)
    logger.info("Data Update Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
