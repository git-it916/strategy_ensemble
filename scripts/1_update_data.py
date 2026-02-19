#!/usr/bin/env python
"""
1. Data Update Script

매일 장 마감 후 실행하여 KIS API로 데이터 업데이트.

Usage:
    python scripts/1_update_data.py                    # 최근 5일 업데이트
    python scripts/1_update_data.py --full-refresh      # 2020년부터 전체
    python scripts/1_update_data.py --start-date 2024-01-01
    python scripts/1_update_data.py --build-features    # 피처/라벨도 재생성
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    DUCKDB_PATH,
    UNIVERSE,
)
from config.logging_config import setup_logging

logger = setup_logging("data_update")


# Fallback seed universe (dynamic universe build 실패 시 사용)
KOSPI200_TICKERS = [
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "035420",  # NAVER
    "005380",  # 현대차
    "051910",  # LG화학
    "006400",  # 삼성SDI
    "035720",  # 카카오
    "068270",  # 셀트리온
    "028260",  # 삼성물산
    "012330",  # 현대모비스
    "055550",  # 신한지주
    "105560",  # KB금융
    "096770",  # SK이노베이션
    "003670",  # 포스코홀딩스
    "066570",  # LG전자
    "086790",  # 하나금융지주
    "034730",  # SK
    "017670",  # SK텔레콤
    "015760",  # 한국전력
    "032830",  # 삼성생명
    "003550",  # LG
    "018260",  # 삼성SDS
    "000270",  # 기아
    "033780",  # KT&G
    "009150",  # 삼성전기
    "316140",  # 우리금융지주
    "010950",  # S-Oil
    "090430",  # 아모레퍼시픽
    "011170",  # 롯데케미칼
    "036570",  # 엔씨소프트
    "030200",  # KT
    "000810",  # 삼성화재
    "034020",  # 두산에너빌리티
    "005490",  # POSCO
    "009540",  # 현대중공업
    "267250",  # HD현대
    "010130",  # 고려아연
    "018880",  # 한온시스템
    "047050",  # 대우건설
    "011780",  # 금호석유
    "024110",  # 기업은행
    "323410",  # 카카오뱅크
    "259960",  # 크래프톤
    "352820",  # 하이브
    "003490",  # 대한항공
    "097950",  # CJ제일제당
    "000720",  # 현대건설
    "207940",  # 삼성바이오로직스
    "034220",  # LG디스플레이
    "010140",  # 삼성중공업
    "021240",  # 코웨이
    "028050",  # 삼성엔지니어링
    "005830",  # DB손해보험
    "032640",  # LG유플러스
    "011200",  # HMM
    "006800",  # 미래에셋증권
    "139480",  # 이마트
    "161390",  # 한국타이어
    "001040",  # CJ
    "004020",  # 현대제철
    "088350",  # 한화생명
    "009830",  # 한화솔루션
    "000880",  # 한화
    "078930",  # GS
    "036460",  # 한국가스공사
    "051900",  # LG생활건강
    "004170",  # 신세계
    "010620",  # 현대미포조선
    "003410",  # 쌍용C&E
    "006360",  # GS건설
    "042660",  # 대우조선해양
    "002790",  # 아모레G
    "004990",  # 롯데지주
    "000100",  # 유한양행
    "011070",  # LG이노텍
    "271560",  # 오리온
    "008770",  # 호텔신라
    "003620",  # SK네트웍스
    "002380",  # KCC
    "004800",  # 효성
    "241560",  # 두산밥캣
    "120110",  # 코오롱인더스트리
    "329180",  # HD현대마린솔루션
    "069500",  # KODEX 200
    "114090",  # GKL
    "005940",  # NH투자증권
    "047810",  # 한국항공우주
    "010060",  # OCI홀딩스
    "071050",  # 한국금융지주
    "006280",  # 녹십자
    "128940",  # 한미약품
    "000150",  # 두산
    "012450",  # 한화에어로스페이스
    "000210",  # DL이앤씨
    "028670",  # 팬오션
    "079550",  # LIG넥스원
    "004370",  # 농심
    "005850",  # 삼성카드
    "039490",  # 키움증권
]


def load_keys() -> dict | None:
    """Load API keys from config/keys.yaml."""
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"

    if not keys_path.exists():
        logger.error("keys.yaml not found! Copy from keys.example.yaml")
        return None

    with open(keys_path) as f:
        return yaml.safe_load(f)


def update_from_kis(
    keys: dict,
    start_date: str,
    end_date: str,
    tickers: list[str] | None = None,
) -> dict:
    """
    KIS API로 일봉 데이터 업데이트.

    Args:
        keys: API keys
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        tickers: 종목코드 리스트 (None이면 기본 유니버스 사용)

    Returns:
        업데이트 결과
    """
    from src.execution import KISApi, KISAuth

    kis_keys = keys.get("kis", {})
    auth = KISAuth(
        app_key=kis_keys.get("app_key", ""),
        app_secret=kis_keys.get("app_secret", ""),
        account_number=kis_keys.get("account_number", ""),
        is_paper=kis_keys.get("is_paper", True),
    )
    api = KISApi(auth)

    if tickers is None:
        tickers = build_default_ticker_universe(100)

    # 날짜 형식 변환 (YYYY-MM-DD → YYYYMMDD)
    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")

    logger.info(f"Fetching {len(tickers)} stocks from KIS API...")
    logger.info(f"Period: {start_date} ~ {end_date}")

    def progress(i, total, ticker):
        logger.info(f"  [{i}/{total}] {ticker} fetched")

    records = api.get_daily_prices_batch(
        stock_codes=tickers,
        start_date=start_fmt,
        end_date=end_fmt,
        progress_callback=progress,
    )

    if not records:
        logger.error("No data fetched from KIS API")
        return {"status": "error", "error": "No data fetched"}

    # DataFrame 생성
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    logger.info(
        f"Fetched {len(df)} records, "
        f"{df['ticker'].nunique()} tickers, "
        f"{df['date'].min().date()} ~ {df['date'].max().date()}"
    )

    # 기존 데이터와 병합
    prices_path = PROCESSED_DATA_DIR / "prices.parquet"

    if prices_path.exists():
        existing = pd.read_parquet(prices_path)
        existing["date"] = pd.to_datetime(existing["date"])

        # 새 데이터 범위 제거 후 합치기 (겹치는 날짜 교체)
        mask = ~(
            (existing["date"] >= df["date"].min())
            & (existing["date"] <= df["date"].max())
            & (existing["ticker"].isin(df["ticker"].unique()))
        )
        existing_kept = existing[mask]

        # 기존 데이터에 있는 추가 컬럼 유지 (재무 데이터 등)
        extra_cols = [c for c in existing.columns if c not in df.columns]
        if extra_cols:
            # 기존 재무데이터를 새 데이터에 merge
            fund_data = existing[["date", "ticker"] + extra_cols].drop_duplicates(
                subset=["date", "ticker"], keep="last"
            )
            df = pd.merge(df, fund_data, on=["date", "ticker"], how="left")

        combined = pd.concat([existing_kept, df], ignore_index=True)
        combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

        logger.info(
            f"Merged: {len(existing)} existing + {len(df)} new "
            f"= {len(combined)} total"
        )
    else:
        combined = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # 저장
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(prices_path, index=False, compression="snappy")

    logger.info(f"Saved: {prices_path} ({len(combined)} records)")

    return {
        "status": "success",
        "source": "kis",
        "n_records": len(df),
        "n_tickers": df["ticker"].nunique(),
        "date_range": f"{df['date'].min().date()} ~ {df['date'].max().date()}",
    }


def build_default_ticker_universe(n_stocks: int) -> list[str]:
    """
    Build update universe from latest processed prices.

    Priority:
        1) KOSPI + KOSDAQ, 시총 1000억 이상(설정값), 거래대금 기준 상위
        2) 실패 시 fallback seed(KOSPI200_TICKERS)
    """
    from src.pipeline import build_universe_snapshot

    prices_path = PROCESSED_DATA_DIR / "prices.parquet"
    fallback = KOSPI200_TICKERS[:n_stocks]

    if not prices_path.exists():
        logger.warning("prices.parquet not found, fallback seed universe will be used")
        return fallback

    try:
        prices = pd.read_parquet(prices_path)
        prices["date"] = pd.to_datetime(prices["date"])
        universe_df = build_universe_snapshot(
            prices=prices,
            min_market_cap=UNIVERSE.get("min_market_cap", 0),
            min_turnover=UNIVERSE.get("min_volume", 0),
            max_stocks=n_stocks,
            allowed_markets=("KOSPI", "KOSDAQ"),
        )
        tickers = universe_df["ticker_order"].astype(str).tolist()
        if tickers:
            logger.info(
                f"Using dynamic universe: {len(tickers)} tickers "
                "(KOSPI/KOSDAQ + min market cap filter)"
            )
            return tickers
    except Exception as e:
        logger.warning(f"Dynamic universe build failed, fallback will be used: {e}")

    logger.warning("Fallback seed universe (KOSPI200_TICKERS) is being used")
    return fallback


def build_features_and_labels() -> None:
    """피처 및 라벨 재생성."""
    prices_path = PROCESSED_DATA_DIR / "prices.parquet"

    if not prices_path.exists():
        logger.error("prices.parquet not found!")
        return

    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])

    logger.info(f"Building features from {len(prices)} price records...")

    # Feature engineering
    from src.etl import FeatureEngineer

    fe = FeatureEngineer()
    features = fe.build_daily_features(prices)
    features.to_parquet(
        PROCESSED_DATA_DIR / "features.parquet", index=False, compression="snappy"
    )
    logger.info(f"Features saved: {len(features)} records")

    # Market features
    market_features = fe.build_market_features(prices)
    market_features.to_parquet(
        PROCESSED_DATA_DIR / "market_features.parquet",
        index=False,
        compression="snappy",
    )
    logger.info(f"Market features saved: {len(market_features)} records")

    # Labels
    from src.etl import LabelEngineer

    le = LabelEngineer()
    labels = le.build_all(prices)

    for name, label_df in labels.items():
        path = PROCESSED_DATA_DIR / f"labels_{name}.parquet"
        label_df.to_parquet(path, index=False, compression="snappy")
        logger.info(f"Labels ({name}) saved: {len(label_df)} records")

    # Generic labels (return)
    if "return" in labels:
        labels["return"].to_parquet(
            PROCESSED_DATA_DIR / "labels.parquet", index=False, compression="snappy"
        )


def update_database_views() -> None:
    """DuckDB 뷰 업데이트."""
    logger.info("Updating database views...")

    from src.database import DuckDBConnector, SchemaManager

    connector = DuckDBConnector(DUCKDB_PATH)
    schema_mgr = SchemaManager(connector, DATA_DIR)

    schema_mgr.setup_views()
    summary = schema_mgr.get_data_summary()
    logger.info(f"Database summary: {summary}")

    connector.close()


def main():
    parser = argparse.ArgumentParser(description="Update trading data via KIS API")

    parser.add_argument(
        "--start-date", type=str,
        help="Start date YYYY-MM-DD (default: 5 days ago)",
    )
    parser.add_argument(
        "--end-date", type=str,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--full-refresh", action="store_true",
        help="Full data refresh from 2020",
    )
    parser.add_argument(
        "--build-features", action="store_true",
        help="Rebuild features and labels after update",
    )
    parser.add_argument(
        "--n-stocks", type=int, default=100,
        help="Number of stocks to fetch (default: 100)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Data Update Started (KIS API)")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 60)

    # 날짜 범위
    if args.full_refresh:
        start_date = "2020-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = args.start_date or (
            datetime.now() - timedelta(days=5)
        ).strftime("%Y-%m-%d")

    logger.info(f"Date range: {start_date} to {end_date}")

    # API 키 로드
    keys = load_keys()
    if keys is None:
        sys.exit(1)

    # 데이터 업데이트 (동적 유니버스 기본)
    tickers = build_default_ticker_universe(args.n_stocks)
    result = update_from_kis(keys, start_date, end_date, tickers)
    logger.info(f"Update result: {result}")

    # 피처/라벨 재생성
    if args.build_features and result.get("status") == "success":
        build_features_and_labels()

    # DB 뷰 업데이트
    if result.get("status") == "success":
        try:
            update_database_views()
        except Exception as e:
            logger.warning(f"Database view update failed (non-critical): {e}")

    logger.info("=" * 60)
    logger.info("Data Update Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
