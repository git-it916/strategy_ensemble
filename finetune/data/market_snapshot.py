"""
Real Market Data Snapshot Store.

data/market_daily/ 파티션 parquet에서 실제 시장 데이터를 로드하여
synthetic data 생성 시 사용할 수 있는 MarketSnapshot 객체를 제공합니다.

Usage:
    from finetune.data.market_snapshot import MarketSnapshotStore
    store = MarketSnapshotStore("2020-01-01", "2025-12-31", universe_tickers)
    store.load()
    snap = store.sample_snapshot(regime="MILD_BULL")
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
MARKET_DAILY_DIR = PROJECT_ROOT / "data" / "market_daily"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class StockSnapshot:
    """단일 종목의 특정 일자 데이터."""
    ticker: str
    close: float
    open_: float
    high: float
    low: float
    volume: float
    daily_return_pct: float
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    rsi_14: Optional[float] = None
    sma_5: Optional[float] = None
    sma_10: Optional[float] = None
    sma_20: Optional[float] = None
    sma_60: Optional[float] = None
    realized_vol_20d: Optional[float] = None
    market_cap: Optional[float] = None


@dataclass
class MarketSnapshot:
    """특정 거래일의 시장 전체 스냅샷."""
    date: pd.Timestamp
    regime: str
    market_return_pct: float
    market_breadth_pct: float
    momentum_10d_pct: float
    realized_vol_10d_pct: float
    stocks: dict[str, StockSnapshot] = field(default_factory=dict)
    universe_coverage: int = 0


# =============================================================================
# MarketSnapshotStore
# =============================================================================


class MarketSnapshotStore:
    """
    market_daily parquet 데이터를 로드하고 MarketSnapshot 객체를 생성합니다.

    Args:
        start_date: 로드 시작일 (YYYY-MM-DD)
        end_date: 로드 종료일 (YYYY-MM-DD)
        universe_tickers: 유니버스 티커 리스트 (e.g., ["005930", "000660"])
        data_dir: market_daily 디렉토리 경로
    """

    def __init__(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2025-12-31",
        universe_tickers: list[str] | None = None,
        data_dir: Path = MARKET_DAILY_DIR,
    ):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.data_dir = data_dir
        self.universe_tickers = set(universe_tickers or [])
        self._snapshots: dict[pd.Timestamp, MarketSnapshot] = {}
        self._dates: list[pd.Timestamp] = []
        self._regime_index: dict[str, list[pd.Timestamp]] = {}

    def load(self) -> int:
        """데이터 로드 및 스냅샷 생성. 유효 스냅샷 수를 반환합니다."""
        raw_df = self._load_parquet_files()
        logger.info(
            f"Loaded {len(raw_df):,} rows, "
            f"{raw_df['date'].min().date()} ~ {raw_df['date'].max().date()}, "
            f"{raw_df['ticker'].nunique()} tickers"
        )

        features_df = self._compute_features(raw_df)
        market_df = self._compute_market_aggregates(features_df)
        self._build_snapshots(features_df, market_df)

        self._dates = sorted(self._snapshots.keys())

        # regime별 인덱스 구축
        self._regime_index.clear()
        for d, snap in self._snapshots.items():
            self._regime_index.setdefault(snap.regime, []).append(d)

        logger.info(f"Built {len(self._dates)} market snapshots")
        return len(self._dates)

    def sample_snapshot(self, regime: str | None = None) -> MarketSnapshot:
        """
        랜덤 스냅샷 반환. regime을 지정하면 해당 regime 날짜에서 샘플링.
        매칭되는 날짜가 없으면 전체에서 랜덤 선택.
        """
        if regime and regime in self._regime_index:
            dates = self._regime_index[regime]
            if dates:
                return self._snapshots[random.choice(dates)]

        return self._snapshots[random.choice(self._dates)]

    def get_regime_distribution(self) -> dict[str, int]:
        """전체 스냅샷의 regime 분포를 반환합니다."""
        return {r: len(dates) for r, dates in sorted(self._regime_index.items())}

    @property
    def dates(self) -> list[pd.Timestamp]:
        return self._dates

    # -----------------------------------------------------------------
    # Private: 데이터 로드
    # -----------------------------------------------------------------

    def _load_parquet_files(self) -> pd.DataFrame:
        """
        파티션 디렉토리에서 data.parquet 파일만 로드합니다.
        Zone.Identifier 파일은 무시합니다.
        """
        frames: list[pd.DataFrame] = []
        start_year = self.start_date.year
        end_year = self.end_date.year

        for year in range(start_year, end_year + 1):
            year_dir = self.data_dir / f"year={year}"
            if not year_dir.exists():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                    continue
                parquet_path = month_dir / "data.parquet"
                if not parquet_path.exists():
                    continue
                try:
                    df = pd.read_parquet(parquet_path)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df[
                        (df["date"] >= self.start_date)
                        & (df["date"] <= self.end_date)
                    ]
                    if not df.empty:
                        frames.append(df)
                except Exception as e:
                    logger.warning(f"Failed to read {parquet_path}: {e}")

        if not frames:
            raise ValueError(
                f"No data in {self.data_dir} for "
                f"{self.start_date.date()} ~ {self.end_date.date()}"
            )

        combined = pd.concat(frames, ignore_index=True)

        # ticker 정리: "005930 KP" → "005930"
        combined["ticker"] = combined["ticker"].str.replace(" KP", "", regex=False)

        # price 결측 행 제거
        combined = combined.dropna(subset=["close"])

        return combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    # -----------------------------------------------------------------
    # Private: 파생 피처 계산
    # -----------------------------------------------------------------

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """종목별 rolling 파생 피처를 계산합니다."""
        df = df.copy()
        groups = df.groupby("ticker", sort=False)

        # 일간 수익률 (%)
        df["daily_return_pct"] = groups["close"].pct_change() * 100

        # SMA-20, SMA-60 (기존 값이 NULL이면 계산으로 대체)
        if "sma_20" not in df.columns:
            df["sma_20"] = np.nan
        if "sma_60" not in df.columns:
            df["sma_60"] = np.nan

        sma20_calc = groups["close"].transform(
            lambda x: x.rolling(20, min_periods=15).mean()
        )
        sma60_calc = groups["close"].transform(
            lambda x: x.rolling(60, min_periods=45).mean()
        )
        df["sma_20"] = df["sma_20"].fillna(sma20_calc)
        df["sma_60"] = df["sma_60"].fillna(sma60_calc)

        # 20일 실현 변동성 (연율화, %)
        log_ret = np.log(df["close"] / groups["close"].shift(1))
        df["realized_vol_20d"] = df.groupby("ticker", sort=False)[
            "close"
        ].transform(
            lambda x: (
                np.log(x / x.shift(1))
                .rolling(20, min_periods=15)
                .std()
                * np.sqrt(252)
                * 100
            )
        )

        return df

    # -----------------------------------------------------------------
    # Private: 시장 전체 지표 계산
    # -----------------------------------------------------------------

    def _compute_market_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        일자별 시장 수준 집계 지표를 계산합니다:
        - 시가총액 가중 수익률
        - 시장 breadth (상승 종목 비율)
        - 10일 모멘텀 (누적 수익률)
        - 10일 실현 변동성 (연율화)
        """
        valid = df[
            df["daily_return_pct"].notna() & df["market_cap"].notna()
        ].copy()

        daily_records = []
        for date, day_df in valid.groupby("date"):
            total_cap = day_df["market_cap"].sum()
            if total_cap <= 0:
                continue

            weights = day_df["market_cap"] / total_cap
            mkt_ret = (day_df["daily_return_pct"] * weights).sum()

            n_stocks = len(day_df)
            n_advancing = (day_df["daily_return_pct"] > 0).sum()
            breadth = n_advancing / n_stocks * 100 if n_stocks > 0 else 50.0

            daily_records.append({
                "date": date,
                "market_return_pct": round(mkt_ret, 4),
                "market_breadth_pct": round(breadth, 1),
                "n_stocks": n_stocks,
            })

        market_df = pd.DataFrame(daily_records).sort_values("date")

        # 10일 모멘텀 (누적)
        market_df["momentum_10d_pct"] = (
            market_df["market_return_pct"].rolling(10, min_periods=5).sum()
        )

        # 10일 실현 변동성 (연율화)
        market_df["realized_vol_10d_pct"] = (
            market_df["market_return_pct"]
            .rolling(10, min_periods=5)
            .std()
            * np.sqrt(252)
        )

        return market_df

    # -----------------------------------------------------------------
    # Private: Regime 분류
    # -----------------------------------------------------------------

    @staticmethod
    def _classify_regime(
        market_return_pct: float,
        breadth_pct: float,
        momentum_10d_pct: float,
        vol_10d_pct: float,
    ) -> str:
        """
        시장 지표 기반 regime 분류.

        우선순위 (가장 제한적인 조건부터):
        1. BEAR
        2. SHORT_TERM_STRESS
        3. STRONG_BULL
        4. WEAKENING
        5. MILD_BULL
        6. Fallback
        """
        if (
            market_return_pct < -2
            and breadth_pct < 30
            and momentum_10d_pct < -5
        ):
            return "BEAR"
        if market_return_pct < -1.5 and vol_10d_pct > 25:
            return "SHORT_TERM_STRESS"
        if (
            market_return_pct > 1.0
            and breadth_pct > 65
            and momentum_10d_pct > 3
        ):
            return "STRONG_BULL"
        if market_return_pct < 0 and 35 <= breadth_pct <= 50:
            return "WEAKENING"
        if market_return_pct > 0 and breadth_pct > 50:
            return "MILD_BULL"

        # Fallback
        if market_return_pct < -1:
            return "SHORT_TERM_STRESS"
        if market_return_pct < 0:
            return "WEAKENING"
        return "MILD_BULL"

    # -----------------------------------------------------------------
    # Private: 스냅샷 조립
    # -----------------------------------------------------------------

    def _build_snapshots(
        self, features_df: pd.DataFrame, market_df: pd.DataFrame
    ) -> None:
        """MarketSnapshot 객체를 조립합니다."""
        market_dict = market_df.set_index("date").to_dict(orient="index")

        for date, day_df in features_df.groupby("date"):
            if date not in market_dict:
                continue

            mkt = market_dict[date]

            # rolling 지표가 충분하지 않은 날은 스킵
            mom = mkt.get("momentum_10d_pct")
            vol = mkt.get("realized_vol_10d_pct")
            if mom is None or vol is None or pd.isna(mom) or pd.isna(vol):
                continue

            regime = self._classify_regime(
                market_return_pct=mkt["market_return_pct"],
                breadth_pct=mkt["market_breadth_pct"],
                momentum_10d_pct=mom,
                vol_10d_pct=vol,
            )

            # 유니버스 종목만 StockSnapshot으로 저장 (메모리 절약)
            stocks: dict[str, StockSnapshot] = {}
            universe_count = 0

            for _, row in day_df.iterrows():
                ticker = row["ticker"]
                if self.universe_tickers and ticker not in self.universe_tickers:
                    continue

                def _val(col: str):
                    v = row.get(col)
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    return float(v)

                stocks[ticker] = StockSnapshot(
                    ticker=ticker,
                    close=float(row["close"]),
                    open_=float(row.get("open", np.nan)),
                    high=float(row.get("high", np.nan)),
                    low=float(row.get("low", np.nan)),
                    volume=float(row.get("volume", 0)),
                    daily_return_pct=float(row.get("daily_return_pct", 0)),
                    pe_ratio=_val("pe_ratio"),
                    pb_ratio=_val("pb_ratio"),
                    rsi_14=_val("rsi_14"),
                    sma_5=_val("sma_5"),
                    sma_10=_val("sma_10"),
                    sma_20=_val("sma_20"),
                    sma_60=_val("sma_60"),
                    realized_vol_20d=_val("realized_vol_20d"),
                    market_cap=_val("market_cap"),
                )
                universe_count += 1

            self._snapshots[date] = MarketSnapshot(
                date=date,
                regime=regime,
                market_return_pct=mkt["market_return_pct"],
                market_breadth_pct=mkt["market_breadth_pct"],
                momentum_10d_pct=mom,
                realized_vol_10d_pct=vol,
                stocks=stocks,
                universe_coverage=universe_count,
            )
