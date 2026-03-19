#!/usr/bin/env python3
"""
Alpha Evaluator CLI

각 알파의 시그널 품질을 히스토리컬 데이터로 평가.
IC, Information Ratio, hit rate, turnover, decay profile, 레짐별 IC 측정.

사용법:
    # v1 알파 평가 (일봉 기반 4개)
    python scripts/evaluate_alphas.py --version v1

    # v2 알파 평가 (일봉 기반 6개 — 인트라데이/오더북 알파 제외)
    python scripts/evaluate_alphas.py --version v2

    # v1 vs v2 비교
    python scripts/evaluate_alphas.py --version all

    # 평가 기간 변경 (기본 60일)
    python scripts/evaluate_alphas.py --version v2 --days 120

    # 알파 간 시그널 상관행렬 출력
    python scripts/evaluate_alphas.py --version v2 --correlation

    # 결과 CSV 저장
    python scripts/evaluate_alphas.py --version v2 --output results.csv

알파 백테스트 방법:
    1. Binance에서 300일 일봉 + 90일 펀딩비율 수집
    2. 각 평가일(t)마다:
       - t 이전 데이터만 사용하여 시그널 생성 (look-ahead 방지)
       - 시그널 score와 t+1d, t+5d, t+10d, t+20d 실현 수익률의 Spearman 상관(IC) 계산
    3. 전 평가일에 걸쳐 IC 평균, IR(IC/IC_std), hit rate(부호 일치율) 집계
    4. 시그널 turnover(변화율), autocorrelation(지속성), decay profile(IC 감쇠) 측정
    5. BTC 20일 수익률 기준 bull/bear/sideways 레짐별 IC 분리 측정

주의:
    - 인트라데이 알파(MomentumMultiScale, IntradayVWAPV2, IntradayRSIV2)는
      일봉 백테스트로 정확히 평가할 수 없으므로 제외됨
    - OrderbookImbalance, SpreadMomentum, DerivativesSentiment는
      히스토리컬 오더북/OI 데이터가 없으므로 제외됨
    - 위 알파들은 라이브에서만 IC 추적 가능
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import STRATEGIES, UNIVERSE
from src.backtest.alpha_evaluator import AlphaEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
# 데이터 수집
# ======================================================================

def load_keys() -> dict:
    """config/keys.yaml에서 API 키 로드."""
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"
    if not keys_path.exists():
        raise FileNotFoundError(
            "config/keys.yaml not found! "
            "Copy from keys.example.yaml and fill in your Binance API keys."
        )
    with open(keys_path) as f:
        return yaml.safe_load(f)


def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Binance에서 일봉 300일 + 펀딩비율 90일 수집."""
    from src.execution.binance_api import BinanceApi

    keys = load_keys()
    binance_cfg = keys.get("binance", {})
    api = BinanceApi(
        api_key=binance_cfg.get("api_key", ""),
        api_secret=binance_cfg.get("api_secret", ""),
    )

    exclude = UNIVERSE.get("exclude_symbols", [])
    universe = api.get_top_symbols(
        n=UNIVERSE.get("top_n_by_volume", 50), exclude=exclude
    )
    logger.info(f"Universe: {len(universe)} symbols")

    prices = api.get_ohlcv_batch(universe, timeframe="1d", days=300)
    logger.info(f"Daily OHLCV: {len(prices)} rows, {prices['ticker'].nunique()} symbols")

    funding = api.get_funding_history_batch(universe, days=90)
    features = None
    if not funding.empty:
        features = prices[["date", "ticker"]].copy()
        features = features.merge(
            funding[["date", "ticker", "funding_rate"]],
            on=["date", "ticker"],
            how="left",
        )
        logger.info(f"Features: {len(features)} rows (with funding_rate)")

    return prices, features


# ======================================================================
# 알파 로딩
# ======================================================================

def load_v1_alphas() -> list:
    """v1 (openclaw_1) 일봉 기반 알파 4개 로드."""
    from src.alphas.openclaw_1 import (
        CSMomentum,
        TimeSeriesMomentum,
        TimeSeriesMeanReversion,
        FundingRateCarry,
    )

    cfg_cs = STRATEGIES.get("cs_momentum", {})
    cfg_ts = STRATEGIES.get("time_series_momentum", {})
    cfg_mr = STRATEGIES.get("time_series_mean_reversion", {})
    cfg_fc = STRATEGIES.get("funding_rate_carry", {})

    return [
        CSMomentum(
            lookback_days=int(cfg_cs.get("lookback_days", 21)),
            skip_days=int(cfg_cs.get("skip_days", 3)),
        ),
        TimeSeriesMomentum(
            lookback_days=int(cfg_ts.get("lookback_days", 20)),
        ),
        TimeSeriesMeanReversion(
            signal_window=int(cfg_mr.get("signal_window", 5)),
            baseline_window=int(cfg_mr.get("baseline_window", 60)),
        ),
        FundingRateCarry(
            lookback_days=int(cfg_fc.get("lookback_days", 14)),
            abs_threshold=float(cfg_fc.get("abs_threshold", 0.0001)),
        ),
    ]


def load_v2_alphas() -> list:
    """
    v2 일봉 기반 알파 로드.

    일봉 백테스트가 가능한 알파만 포함:
      - MomentumComposite (일봉 20d)
      - FundingCarryEnhanced (일봉 + 펀딩)
      - MeanReversionMultiHorizon (일봉 다중 호라이즌)

    제외 (인트라데이/오더북 의존):
      - MomentumMultiScale (5m 데이터 필요)
      - IntradayVWAPV2, IntradayRSIV2 (1h 데이터 필요)
      - OrderbookImbalance, SpreadMomentum (실시간 오더북 필요)
      - DerivativesSentiment (실시간 OI 필요)
      - VolatilityRegime (비방향성 — IC 측정 무의미)
    """
    from src.alphas.v2 import (
        MomentumComposite,
        FundingCarryEnhanced,
        MeanReversionMultiHorizon,
    )

    return [
        MomentumComposite(
            name="MomentumComposite",
            lookback_days=20,
            skip_days=3,
        ),
        FundingCarryEnhanced(
            name="FundingCarryEnhanced",
            lookback_days=14,
            velocity_lookback=7,
        ),
        MeanReversionMultiHorizon(
            name="MeanReversionMultiHorizon",
            horizons=[(3, 20), (5, 60), (10, 120)],
        ),
    ]


# ======================================================================
# 결과 출력
# ======================================================================

def print_results(df: pd.DataFrame) -> None:
    """평가 결과 출력."""
    print("\n" + "=" * 95)
    print("ALPHA EVALUATION RESULTS")
    print("=" * 95)

    fmt = {
        "ic_1d": "{:+.4f}",
        "ic_5d": "{:+.4f}",
        "ic_10d": "{:+.4f}",
        "ir_1d": "{:+.3f}",
        "ir_5d": "{:+.3f}",
        "hit_1d": "{:.1%}",
        "hit_5d": "{:.1%}",
        "turnover": "{:.3f}",
        "autocorr": "{:.3f}",
    }

    for _, row in df.iterrows():
        print(f"\n{'─' * 50}")
        print(f"  {row['alpha']}")
        print(f"{'─' * 50}")
        for col, f in fmt.items():
            if col in row and not np.isnan(row[col]):
                label = {
                    "ic_1d": "IC (1일)",
                    "ic_5d": "IC (5일)",
                    "ic_10d": "IC (10일)",
                    "ir_1d": "IR (1일)",
                    "ir_5d": "IR (5일)",
                    "hit_1d": "Hit Rate (1일)",
                    "hit_5d": "Hit Rate (5일)",
                    "turnover": "Turnover",
                    "autocorr": "Autocorrelation",
                }.get(col, col)
                print(f"  {label:>18s}: {f.format(row[col])}")
        print(f"  {'평가일 수':>18s}: {int(row.get('n_dates', 0))}")

    # 비교 테이블
    print(f"\n{'=' * 95}")
    print("COMPARISON TABLE")
    print(f"{'=' * 95}")
    summary_cols = ["alpha", "ic_5d", "ir_5d", "hit_5d", "turnover", "n_dates"]
    available = [c for c in summary_cols if c in df.columns]
    print(df[available].to_string(index=False, float_format="%+.4f"))
    print()


def print_regime_ic(results: list) -> None:
    """레짐별 IC 출력."""
    print(f"\n{'=' * 60}")
    print("REGIME-CONDITIONAL IC (5-day forward return)")
    print(f"{'=' * 60}")
    header = f"{'Alpha':>30s} | {'Bull':>8s} | {'Bear':>8s} | {'Sideways':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        ric = r.regime_conditional_ic
        print(
            f"{r.alpha_name:>30s} | "
            f"{ric.get('bull', 0):+.4f} | "
            f"{ric.get('bear', 0):+.4f} | "
            f"{ric.get('sideways', 0):+.4f}"
        )
    print()


def print_decay_profile(results: list) -> None:
    """IC decay profile 출력."""
    print(f"\n{'=' * 70}")
    print("IC DECAY PROFILE (IC at increasing forward horizons)")
    print(f"{'=' * 70}")
    horizons = [1, 2, 3, 5, 10, 20]
    header = f"{'Alpha':>30s} | " + " | ".join(f"{h:>5d}d" for h in horizons)
    print(header)
    print("-" * len(header))
    for r in results:
        decay_dict = dict(r.ic_decay_profile)
        vals = " | ".join(f"{decay_dict.get(h, 0):+.4f}" for h in horizons)
        print(f"{r.alpha_name:>30s} | {vals}")
    print()


# ======================================================================
# 메인
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="알파 시그널 품질 평가 (IC, hit rate, decay, regime IC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/evaluate_alphas.py --version v2           # v2 알파만 평가
  python scripts/evaluate_alphas.py --version all --days 90 # 전체 비교, 90일
  python scripts/evaluate_alphas.py --version v2 --correlation  # 상관행렬 포함
        """,
    )
    parser.add_argument(
        "--version", choices=["v1", "v2", "all"], default="v2",
        help="평가할 알파 버전 (default: v2)",
    )
    parser.add_argument(
        "--days", type=int, default=60,
        help="평가일 수 (default: 60)",
    )
    parser.add_argument(
        "--correlation", action="store_true",
        help="알파 간 시그널 상관행렬 출력",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="결과 CSV 파일 저장 경로",
    )
    args = parser.parse_args()

    # ── 1. 데이터 수집 ──
    logger.info("Binance에서 히스토리컬 데이터 수집 중...")
    prices, features = fetch_data()

    # ── 2. 알파 로드 + fit ──
    alphas = []
    if args.version in ("v1", "all"):
        v1 = load_v1_alphas()
        for a in v1:
            try:
                a.fit(prices, features)
                logger.info(f"  v1 loaded: {a.name}")
            except Exception as e:
                logger.error(f"Fit failed for {a.name}: {e}")
        alphas.extend(v1)

    if args.version in ("v2", "all"):
        v2 = load_v2_alphas()
        for a in v2:
            try:
                a.fit(prices, features)
                logger.info(f"  v2 loaded: {a.name}")
            except Exception as e:
                logger.error(f"Fit failed for {a.name}: {e}")
        alphas.extend(v2)

    if not alphas:
        logger.error("로드된 알파가 없습니다!")
        return

    logger.info(f"{len(alphas)}개 알파 평가 시작...")

    # ── 3. 평가일 결정 ──
    all_dates = sorted(prices["date"].unique())
    max_h = 20
    usable = all_dates[:-max_h] if len(all_dates) > max_h else all_dates
    eval_dates = usable[-args.days:]
    logger.info(f"평가 기간: {eval_dates[0]} ~ {eval_dates[-1]} ({len(eval_dates)}일)")

    # ── 4. 평가 실행 ──
    evaluator = AlphaEvaluator()

    # 개별 결과 수집 (regime IC, decay 출력용)
    individual_results = []
    for alpha in alphas:
        logger.info(f"  Evaluating {alpha.name}...")
        result = evaluator.evaluate(
            alpha, prices, features,
            eval_dates=eval_dates,
            forward_horizons=[1, 2, 3, 5, 10, 20],
        )
        individual_results.append(result)

    # 비교 테이블
    results_df = pd.DataFrame([r.summary_row() for r in individual_results])

    # ── 5. 결과 출력 ──
    print_results(results_df)
    print_regime_ic(individual_results)
    print_decay_profile(individual_results)

    # ── 6. 상관행렬 ──
    if args.correlation and len(alphas) > 1:
        logger.info("시그널 상관행렬 계산 중...")
        corr = evaluator.compute_correlation_matrix(
            alphas, prices, features, eval_dates=eval_dates[:30]
        )
        print(f"\n{'=' * 70}")
        print("SIGNAL CORRELATION MATRIX (Spearman)")
        print(f"{'=' * 70}")
        print(corr.to_string(float_format="%.3f"))
        print()

    # ── 7. CSV 저장 ──
    if args.output:
        results_df.to_csv(args.output, index=False)
        logger.info(f"결과 저장됨: {args.output}")

    logger.info("평가 완료!")


if __name__ == "__main__":
    main()
