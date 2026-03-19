#!/usr/bin/env python3
"""
전략 백테스트

저장된 시그널 로그 + 바이낸스 가격 데이터로 수익률 시뮬레이션.
실제 DecisionEngine 규칙(비대칭 임계값, SL/TP, 트레일링, 쿨다운)을 그대로 적용.

사용법:
    # 기본 (저장된 로그로 백테스트)
    python scripts/backtest.py

    # 기간 지정
    python scripts/backtest.py --start 2026-03-16 --end 2026-03-20

    # 로그 없이 과거 데이터로 처음부터 시뮬레이션 (알파 재계산)
    python scripts/backtest.py --simulate --days 30

출력:
    - 거래별 상세 (진입/청산 가격, PnL, 보유시간)
    - 누적 수익률 곡선
    - 승률, 평균 손익, 최대 낙폭
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    COOLDOWN_MINUTES,
    LEVERAGE,
    LOGS_DIR,
    LONG_ENTRY_THRESHOLD,
    LONG_SL_PCT,
    LONG_TP_PCT,
    MIN_HOLD_MINUTES,
    SHORT_ENTRY_THRESHOLD,
    SHORT_SL_PCT,
    SHORT_TP_PCT,
    TRAILING_ACTIVATION_PCT,
    TRAILING_DISTANCE_PCT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backtest")


# ======================================================================
# 데이터 구조
# ======================================================================

@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime | None = None
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0  # 레버리지 미포함
    pnl_leveraged: float = 0.0  # 레버리지 포함


@dataclass
class BacktestResult:
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[tuple[datetime, float]] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.trades:
            return {"n_trades": 0}

        closed = [t for t in self.trades if t.exit_time]
        if not closed:
            return {"n_trades": 0}

        pnls = [t.pnl_leveraged for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        cumulative = np.cumprod([1 + p for p in pnls])
        peak = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / peak - 1
        max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0

        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 60
            for t in closed if t.exit_time
        ]

        return {
            "n_trades": len(closed),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_return": float(cumulative[-1] - 1) if len(cumulative) > 0 else 0,
            "avg_pnl": float(np.mean(pnls)),
            "avg_win": float(np.mean(wins)) if wins else 0,
            "avg_loss": float(np.mean(losses)) if losses else 0,
            "max_drawdown": max_dd,
            "avg_hold_min": float(np.mean(durations)) if durations else 0,
            "best_trade": float(max(pnls)),
            "worst_trade": float(min(pnls)),
        }


# ======================================================================
# 백테스트 엔진
# ======================================================================

class BacktestEngine:
    """
    시그널 → 규칙 기반 매매 시뮬레이션.

    5분봉 단위로 순회하면서:
    1. 시그널 체크 → 진입/청산 결정
    2. SL/TP/트레일링 체크 (봉 단위)
    3. 포지션 관리
    """

    def __init__(self):
        self.current_trade: BacktestTrade | None = None
        self.trades: List[BacktestTrade] = []
        self.last_exit_time: datetime | None = None
        self.last_exit_symbol: str = ""
        self.peak_pnl: float = 0.0
        self.trailing_active: bool = False

    def run_from_signals(
        self,
        signal_log: List[dict],
        prices_5m: dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """
        저장된 시그널 로그로 백테스트.

        signal_log: [{"timestamp": "...", "scores": {"SYM": 0.5, ...}}, ...]
        prices_5m: {symbol: DataFrame[timestamp, open, high, low, close, volume]}
        """
        result = BacktestResult()
        equity = 1.0

        for entry in signal_log:
            ts = datetime.fromisoformat(entry["timestamp"])
            scores = entry.get("scores", {})

            # 포지션 있으면 SL/TP/시그널 청산 체크
            if self.current_trade:
                sym = self.current_trade.symbol
                if sym in prices_5m:
                    price_now = self._get_price_at(prices_5m[sym], ts)
                    if price_now > 0:
                        exit_reason = self._check_exit(price_now, scores, ts)
                        if exit_reason:
                            self._close_trade(price_now, ts, exit_reason)
                            equity *= (1 + self.trades[-1].pnl_leveraged)
                            result.equity_curve.append((ts, equity))

            # 포지션 없으면 진입 체크
            if not self.current_trade:
                order = self._check_entry(scores, ts)
                if order:
                    sym, direction = order
                    if sym in prices_5m:
                        price = self._get_price_at(prices_5m[sym], ts)
                        if price > 0:
                            self._open_trade(sym, direction, price, ts)

        # 마지막 미청산 포지션 강제 청산
        if self.current_trade:
            sym = self.current_trade.symbol
            if sym in prices_5m and not prices_5m[sym].empty:
                last_price = float(prices_5m[sym]["close"].iloc[-1])
                self._close_trade(last_price, datetime.now(timezone.utc), "END")
                equity *= (1 + self.trades[-1].pnl_leveraged)

        result.trades = self.trades
        result.equity_curve.append((datetime.now(timezone.utc), equity))
        return result

    def run_simulation(
        self,
        alphas: list,
        data_mgr,
        days: int = 30,
    ) -> BacktestResult:
        """
        과거 데이터에서 알파를 직접 재계산하며 백테스트.

        일봉 단위로 순회 (5분봉 시뮬레이션은 비용이 너무 큼).
        """
        result = BacktestResult()
        equity = 1.0
        bundle = data_mgr.to_bundle()

        # 일봉 기준 날짜 목록
        all_dates = set()
        for sym, df in bundle.ohlcv_1d.items():
            if not df.empty:
                for ts in df["timestamp"]:
                    all_dates.add(ts)
        sorted_dates = sorted(all_dates)[-days:]

        for ts in sorted_dates:
            # 알파 계산
            scores = {}
            for sym in bundle.universe:
                total = 0.0
                total_w = 0.0
                for alpha in alphas:
                    try:
                        sig = asyncio.get_event_loop().run_until_complete(
                            alpha.compute(sym, bundle)
                        )
                        w = alpha.weight
                        total += sig.score * sig.confidence * w
                        total_w += w * sig.confidence
                    except Exception:
                        pass
                if total_w > 0:
                    scores[sym] = np.clip(total / total_w, -1, 1)

            # 포지션 체크
            if self.current_trade:
                sym = self.current_trade.symbol
                if sym in bundle.ohlcv_1d:
                    df = bundle.ohlcv_1d[sym]
                    row = df[df["timestamp"] <= ts]
                    if not row.empty:
                        price = float(row["close"].iloc[-1])
                        exit_reason = self._check_exit(price, scores, ts)
                        if exit_reason:
                            self._close_trade(price, ts, exit_reason)
                            equity *= (1 + self.trades[-1].pnl_leveraged)
                            result.equity_curve.append((ts, equity))

            if not self.current_trade:
                order = self._check_entry(scores, ts)
                if order:
                    sym, direction = order
                    if sym in bundle.ohlcv_1d:
                        df = bundle.ohlcv_1d[sym]
                        row = df[df["timestamp"] <= ts]
                        if not row.empty:
                            price = float(row["close"].iloc[-1])
                            self._open_trade(sym, direction, price, ts)

        result.trades = self.trades
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_entry(
        self, scores: dict, ts: datetime
    ) -> Optional[tuple[str, str]]:
        # 쿨다운
        if self.last_exit_time:
            elapsed = (ts - self.last_exit_time).total_seconds() / 60
            if elapsed < COOLDOWN_MINUTES:
                cooldown_sym = self.last_exit_symbol
            else:
                cooldown_sym = ""
        else:
            cooldown_sym = ""

        long_candidates = [
            (s, sc) for s, sc in scores.items()
            if sc > LONG_ENTRY_THRESHOLD and s != cooldown_sym
        ]
        short_candidates = [
            (s, sc) for s, sc in scores.items()
            if sc < SHORT_ENTRY_THRESHOLD and s != cooldown_sym
        ]

        all_c = (
            [(s, sc, "LONG") for s, sc in long_candidates]
            + [(s, sc, "SHORT") for s, sc in short_candidates]
        )
        if not all_c:
            return None

        best = max(all_c, key=lambda x: abs(x[1]))
        return (best[0], best[2])

    def _check_exit(
        self, price: float, scores: dict, ts: datetime
    ) -> Optional[str]:
        trade = self.current_trade
        if not trade:
            return None

        # 최소 보유
        hold_min = (ts - trade.entry_time).total_seconds() / 60
        pnl = self._calc_pnl(trade, price)

        # SL
        if trade.direction == "LONG" and pnl <= LONG_SL_PCT:
            return "SL"
        if trade.direction == "SHORT" and pnl <= SHORT_SL_PCT:
            return "SL"

        # 트레일링
        if pnl >= TRAILING_ACTIVATION_PCT:
            self.trailing_active = True
            self.peak_pnl = max(self.peak_pnl, pnl)
            if self.peak_pnl - pnl >= TRAILING_DISTANCE_PCT:
                return "TRAILING"

        # TP (트레일링 미활성 시)
        if not self.trailing_active:
            tp = LONG_TP_PCT if trade.direction == "LONG" else SHORT_TP_PCT
            if pnl >= tp:
                return "TP"

        # 시그널 반전 (최소 보유 이후)
        if hold_min >= MIN_HOLD_MINUTES:
            score = scores.get(trade.symbol, 0)
            if trade.direction == "LONG" and score < SHORT_ENTRY_THRESHOLD:
                return "SIGNAL"
            if trade.direction == "SHORT" and score > LONG_ENTRY_THRESHOLD:
                return "SIGNAL"

            # 시그널 소멸 (2시간+)
            if hold_min >= 120 and abs(score) < 0.10:
                return "SIGNAL"

        return None

    def _open_trade(self, sym: str, direction: str, price: float, ts: datetime):
        self.current_trade = BacktestTrade(
            symbol=sym, direction=direction,
            entry_time=ts, entry_price=price,
        )
        self.trailing_active = False
        self.peak_pnl = 0.0

    def _close_trade(self, price: float, ts: datetime, reason: str):
        trade = self.current_trade
        if not trade:
            return
        trade.exit_price = price
        trade.exit_time = ts
        trade.exit_reason = reason
        trade.pnl_pct = self._calc_pnl(trade, price)
        trade.pnl_leveraged = trade.pnl_pct * LEVERAGE
        self.trades.append(trade)
        self.last_exit_time = ts
        self.last_exit_symbol = trade.symbol
        self.current_trade = None

    @staticmethod
    def _calc_pnl(trade: BacktestTrade, price: float) -> float:
        if trade.entry_price <= 0:
            return 0.0
        if trade.direction == "LONG":
            return (price - trade.entry_price) / trade.entry_price
        else:
            return (trade.entry_price - price) / trade.entry_price

    @staticmethod
    def _get_price_at(df: pd.DataFrame, ts: datetime) -> float:
        if df.empty:
            return 0.0
        before = df[df["timestamp"] <= ts]
        if before.empty:
            return 0.0
        return float(before["close"].iloc[-1])


# ======================================================================
# 로그 로딩
# ======================================================================

def load_signal_logs(
    start: str | None = None,
    end: str | None = None,
) -> List[dict]:
    """logs/signals/ 에서 JSONL 로드."""
    sig_dir = LOGS_DIR / "signals"
    if not sig_dir.exists():
        logger.warning(f"No signal logs in {sig_dir}")
        return []

    entries = []
    for path in sorted(sig_dir.glob("*.jsonl")):
        date_str = path.stem  # "2026-03-16"
        if start and date_str < start:
            continue
        if end and date_str > end:
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

    logger.info(f"Loaded {len(entries)} signal entries from {sig_dir}")
    return entries


async def fetch_prices(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """바이낸스에서 5분봉 가격 조회."""
    from src.data.data_manager import DataManager
    dm = DataManager()

    prices = {}
    for sym in symbols:
        try:
            raw = await dm._exchange.fetch_ohlcv(sym, timeframe="5m", limit=1440)
            if raw:
                df = pd.DataFrame(
                    raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                prices[sym] = df
        except Exception as e:
            logger.debug(f"Price fetch failed {sym}: {e}")

    await dm.close()
    return prices


# ======================================================================
# 출력
# ======================================================================

def print_results(result: BacktestResult):
    s = result.summary()
    if s["n_trades"] == 0:
        print("\n거래 없음 (시그널이 임계값을 넘지 못했거나 로그가 부족합니다)")
        return

    print("\n" + "=" * 70)
    print("백테스트 결과")
    print("=" * 70)
    print(f"  총 거래 수:     {s['n_trades']}")
    print(f"  승률:           {s['win_rate']:.1%}")
    print(f"  누적 수익률:    {s['total_return']:+.2%} (레버리지 {LEVERAGE}배 포함)")
    print(f"  평균 손익:      {s['avg_pnl']:+.2%}")
    print(f"  평균 수익:      {s['avg_win']:+.2%}")
    print(f"  평균 손실:      {s['avg_loss']:+.2%}")
    print(f"  최고 거래:      {s['best_trade']:+.2%}")
    print(f"  최악 거래:      {s['worst_trade']:+.2%}")
    print(f"  최대 낙폭:      {s['max_drawdown']:+.2%}")
    print(f"  평균 보유시간:  {s['avg_hold_min']:.0f}분")

    print(f"\n{'─' * 70}")
    print(f"{'시간':>20s}  {'코인':>6s}  {'방향':>4s}  {'진입가':>10s}  {'청산가':>10s}  {'PnL':>8s}  {'사유':>8s}")
    print(f"{'─' * 70}")
    for t in result.trades:
        if not t.exit_time:
            continue
        sym = t.symbol.split("/")[0]
        ts = t.entry_time.strftime("%m/%d %H:%M") if t.entry_time else ""
        print(
            f"{ts:>20s}  {sym:>6s}  {t.direction:>4s}  "
            f"${t.entry_price:>9,.2f}  ${t.exit_price:>9,.2f}  "
            f"{t.pnl_leveraged:>+7.2%}  {t.exit_reason:>8s}"
        )
    print()


# ======================================================================
# 메인
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="전략 백테스트")
    parser.add_argument("--start", type=str, default=None, help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="끝 날짜 (YYYY-MM-DD)")
    parser.add_argument("--simulate", action="store_true", help="로그 없이 알파 재계산 모드")
    parser.add_argument("--days", type=int, default=30, help="시뮬레이션 기간 (일)")
    args = parser.parse_args()

    if args.simulate:
        logger.info("시뮬레이션 모드 (알파 재계산)...")
        from src.alphas.v2 import ALL_ALPHAS
        from src.data.data_manager import DataManager

        async def run_sim():
            dm = DataManager()
            await dm.initial_fetch()
            alphas = [cls() for cls in ALL_ALPHAS]
            engine = BacktestEngine()
            result = engine.run_simulation(alphas, dm, days=args.days)
            await dm.close()
            return result

        result = asyncio.run(run_sim())
    else:
        # 시그널 로그 기반
        logs = load_signal_logs(args.start, args.end)
        if not logs:
            print("시그널 로그가 없습니다. --simulate 모드를 사용하세요.")
            return

        # 로그에서 심볼 추출
        all_symbols = set()
        for entry in logs:
            all_symbols.update(entry.get("scores", {}).keys())

        logger.info(f"심볼: {len(all_symbols)}개, 가격 데이터 수집 중...")
        prices = asyncio.run(fetch_prices(list(all_symbols)))

        engine = BacktestEngine()
        result = engine.run_from_signals(logs, prices)

    print_results(result)


if __name__ == "__main__":
    main()
