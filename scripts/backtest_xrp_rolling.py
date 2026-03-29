#!/usr/bin/env python3
"""
XRP Funding Contrarian — 롤링 리프레시 백테스트 (Walk-Forward Optimization).

매월 1회 파라미터 재최적화:
  1. 훈련창 (기본 6개월): 그리드서치로 최적 파라미터 선택
  2. 테스트창 (1개월): 최적 파라미터로 OOS 백테스트
  3. 다음 달로 이동 → 반복

이를 통해 과적합을 방지하면서 시장 변화에 적응하는 전략 구현.

Usage:
    python scripts/backtest_xrp_rolling.py                       # 기본 (6개월 훈련, 60개월 전체)
    python scripts/backtest_xrp_rolling.py --train-months 3      # 3개월 훈련창
    python scripts/backtest_xrp_rolling.py --months 36           # 36개월 백테스트
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sniper_v2.config import XRP_CONFIG, SymbolConfig
from src.sniper_v2.strategy import Direction, Signal, ActiveTrade
from src.sniper_v2.funding_strategy import FundingConfig

from scripts.backtest_sniper_v2 import (
    fetch_ohlcv_bulk, fetch_funding_bulk,
    check_exit, record_trade,
)


# ══════════════════════════════════════════════════════════════
# 파라미터 그리드 정의
# ══════════════════════════════════════════════════════════════

def build_param_grid() -> list[tuple[SymbolConfig, FundingConfig, str]]:
    """(SymbolConfig, FundingConfig, label) 리스트 생성."""
    grid = []
    # 롤링용 최소 그리드 (매월 반복, 24조합 × ~0.5s = 12s/월)
    for sl_atr in [2.0, 3.0, 4.0]:
        for z_thresh in [0.75, 1.125, 2.0]:
            for price_z in [0.8]:
                for f_lb in [75]:
                    for pp_trig, pp_exit in [(0, 0)]:
                        for tp1, tp2, tp3 in [(1.5, 3.0, 5.0), (2.0, 3.5, 6.0)]:
                            cfg = replace(
                                XRP_CONFIG,
                                profit_protect=pp_trig > 0,
                                pp_trigger=pp_trig,
                                pp_exit=pp_exit,
                            )
                            fcfg = FundingConfig(
                                sl_atr_mult=sl_atr,
                                z_threshold=z_thresh,
                                price_confirm_z=price_z,
                                funding_lookback=f_lb,
                                tp1_rr=tp1,
                                tp2_rr=tp2,
                                tp3_rr=tp3,
                            )
                            label = f"SL×{sl_atr} Z={z_thresh} PZ={price_z} FL={f_lb} PP={pp_trig}/{pp_exit} TP={tp1}/{tp2}/{tp3}"
                            grid.append((cfg, fcfg, label))
    return grid


# ══════════════════════════════════════════════════════════════
# 단일 기간 백테스트 엔진
# ══════════════════════════════════════════════════════════════

def run_funding_period(
    close_arr, high_arr, low_arr, open_arr,
    atr_arr, rsi_arr, times,
    f_times_i64, f_rates, times_i64,
    cfg: SymbolConfig, fcfg: FundingConfig,
    start_idx: int, end_idx: int,
    warmup: int,
) -> list[dict]:
    """
    주어진 구간 [start_idx, end_idx)에서 펀딩 전략 백테스트.
    지표는 start_idx 이전부터 계산되어야 하므로 warmup 이전부터 시작하되,
    거래 기록은 start_idx 이후만 포함.
    """
    actual_start = max(warmup, start_idx)

    trades = []
    active_trade: ActiveTrade | None = None
    last_direction = 0
    cooldown = 0

    for i in range(warmup, end_idx):
        current_time = times[i]

        # 봉 내 TP/SL/PP 체크 (SL 우선 비관적 순서)
        if active_trade is not None:
            is_long_pos = active_trade.direction == Direction.LONG
            if is_long_pos:
                check_seq = [open_arr[i], low_arr[i], high_arr[i], close_arr[i]]
            else:
                check_seq = [open_arr[i], high_arr[i], low_arr[i], close_arr[i]]
            for check_price in check_seq:
                if active_trade is None:
                    break
                exit_reason = check_exit(active_trade, float(check_price), cfg)
                if exit_reason:
                    if i >= start_idx:  # OOS 구간만 기록
                        record_trade(trades, active_trade, current_time, float(check_price), exit_reason, cfg)
                    if exit_reason == "SL":
                        cooldown = cfg.cooldown_bars
                    active_trade = None
                    break

        if cooldown > 0:
            cooldown -= 1
            continue

        # 펀딩 z-score (searchsorted 최적화)
        n_funding = int(np.searchsorted(f_times_i64, times_i64[i], side='right'))
        if n_funding < 30:
            continue

        rates_window = f_rates[:n_funding]
        n = min(len(rates_window), fcfg.funding_lookback)
        w = rates_window[-n:]
        fr_mean = float(np.mean(w))
        fr_std = float(np.std(w))
        if fr_std < 1e-10:
            continue
        fr_z = (float(rates_window[-1]) - fr_mean) / fr_std

        # 가격 z-score
        z_scores = []
        for period in fcfg.price_z_periods:
            if i < period + 20:
                continue
            ret = close_arr[i] / close_arr[i - period] - 1
            lookback = min(i, period * 4)
            rets = np.diff(close_arr[i - lookback:i + 1]) / close_arr[i - lookback:i]
            vol = float(np.std(rets)) * np.sqrt(period)
            if vol > 1e-10:
                z_scores.append(ret / vol)

        if not z_scores:
            continue
        price_z = float(np.mean(z_scores))

        # 시그널
        zt = fcfg.z_threshold
        pcz = fcfg.price_confirm_z
        is_long = is_short = False
        strength = 0.0

        if fr_z > zt and price_z > pcz:
            is_short = True
            strength = min(fr_z / (zt * 2), 1.0)
        elif fr_z < -zt and price_z < -pcz:
            is_long = True
            strength = min(-fr_z / (zt * 2), 1.0)
        elif fr_z > zt * 1.5:
            is_short = True
            strength = min(fr_z / (zt * 3), 0.5)
        elif fr_z < -zt * 1.5:
            is_long = True
            strength = min(-fr_z / (zt * 3), 0.5)

        if not is_long and not is_short:
            continue
        if strength < 0.3:
            continue

        if is_long and last_direction == 1:
            continue
        if is_short and last_direction == -1:
            continue

        atr_val = float(atr_arr[i])
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        entry = float(close_arr[i])
        risk = atr_val * fcfg.sl_atr_mult
        rsi_val = float(rsi_arr[i]) if not np.isnan(rsi_arr[i]) else 50.0

        if is_long:
            last_direction = 1
            sl = entry - risk
            tp1 = entry + risk * fcfg.tp1_rr
            tp2 = entry + risk * fcfg.tp2_rr
            tp3 = entry + risk * fcfg.tp3_rr
            direction = Direction.LONG
        else:
            last_direction = -1
            sl = entry + risk
            tp1 = entry - risk * fcfg.tp1_rr
            tp2 = entry - risk * fcfg.tp2_rr
            tp3 = entry - risk * fcfg.tp3_rr
            direction = Direction.SHORT

        # 진입은 start_idx 이후만 허용
        if i < start_idx:
            continue

        if active_trade is not None and direction != active_trade.direction:
            record_trade(trades, active_trade, current_time, entry, "REVERSE", cfg)
            active_trade = None

        if active_trade is None:
            active_trade = ActiveTrade(
                direction=direction, entry_price=entry,
                sl_price=sl, tp1_price=tp1, tp2_price=tp2, tp3_price=tp3,
                trail_price=sl, risk=risk, entry_time=current_time,
            )

    # 미결 포지션 — 기간 끝 청산
    if active_trade is not None and end_idx - 1 >= start_idx:
        last_i = end_idx - 1
        record_trade(trades, active_trade, times[last_i], float(close_arr[last_i]), "PERIOD_END", cfg)

    return trades


def evaluate_params(trades: list, cfg: SymbolConfig) -> float:
    """PF 기준 점수 (거래 없으면 0)."""
    if len(trades) < 3:
        return 0.0
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    gp = wins["pnl_pct"].sum() if len(wins) else 0
    gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.01
    pf = gp / gl
    # 거래수 보너스 (너무 적으면 패널티)
    trade_bonus = min(len(trades) / 10, 1.0)
    return pf * trade_bonus


# ══════════════════════════════════════════════════════════════
# 롤링 리프레시 메인
# ══════════════════════════════════════════════════════════════

def rolling_backtest(
    df: pd.DataFrame,
    funding_df: pd.DataFrame,
    train_months: int = 6,
    total_months: int = 60,
):
    print(f"\n{'='*70}", flush=True)
    print(f"  XRP 롤링 리프레시 Walk-Forward Optimization", flush=True)
    print(f"  훈련창: {train_months}개월 | 테스트창: 1개월 | 재최적화: 매월", flush=True)
    print(f"{'='*70}", flush=True)

    # 사전계산 (전체 데이터)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_arr = tr.ewm(alpha=1/14, min_periods=14).mean().values

    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_arr = (100 - 100 / (1 + rs)).values

    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    open_arr = df["open"].values
    times = df.index

    f_times = funding_df["timestamp"].values if len(funding_df) > 0 else np.array([])
    f_rates = funding_df["fundingRate"].values if len(funding_df) > 0 else np.array([])
    f_times_i64 = f_times.astype("int64") if len(f_times) > 0 else np.array([], dtype="int64")
    times_i64 = df.index.values.astype("int64")

    # 파라미터 그리드 생성
    param_grid = build_param_grid()
    print(f"  파라미터 조합: {len(param_grid)}개", flush=True)

    # 월별 구간 정의
    data_start = df.index[0]
    data_end = df.index[-1]

    # 훈련 시작: 데이터 시작 + train_months
    first_test_start = data_start + pd.DateOffset(months=train_months)
    if first_test_start >= data_end:
        print(f"  데이터 부족! 훈련창({train_months}개월)보다 데이터가 적음")
        return

    # 월별 테스트 구간 생성
    test_periods = []
    current = first_test_start.replace(day=1)
    while current < data_end:
        next_month = current + pd.DateOffset(months=1)
        test_periods.append((current, min(next_month, data_end)))
        current = next_month

    print(f"  테스트 기간: {test_periods[0][0].strftime('%Y-%m')} ~ {test_periods[-1][0].strftime('%Y-%m')} ({len(test_periods)}개월)", flush=True)

    # 롤링 루프
    all_oos_trades = []
    monthly_results = []
    chosen_params_history = []

    t0 = time.time()
    for month_idx, (test_start, test_end) in enumerate(test_periods):
        # 훈련 구간: [test_start - train_months, test_start)
        train_start = test_start - pd.DateOffset(months=train_months)
        if train_start < data_start:
            train_start = data_start

        # 인덱스 찾기
        train_start_idx = df.index.searchsorted(train_start)
        train_end_idx = df.index.searchsorted(test_start)
        test_start_idx = train_end_idx
        test_end_idx = df.index.searchsorted(test_end)

        warmup = XRP_CONFIG.warmup_bars

        if train_end_idx - train_start_idx < warmup + 100:
            continue

        # --- 훈련: 그리드서치 ---
        best_score = -999
        best_cfg = None
        best_fcfg = None
        best_label = ""

        for cfg, fcfg, label in param_grid:
            trades = run_funding_period(
                close_arr, high_arr, low_arr, open_arr,
                atr_arr, rsi_arr, times,
                f_times_i64, f_rates, times_i64,
                cfg, fcfg,
                train_start_idx, train_end_idx,
                max(warmup, train_start_idx),
            )
            score = evaluate_params(trades, cfg)
            if score > best_score:
                best_score = score
                best_cfg = cfg
                best_fcfg = fcfg
                best_label = label

        # --- 테스트: OOS 백테스트 ---
        if best_cfg is None:
            continue

        oos_trades = run_funding_period(
            close_arr, high_arr, low_arr, open_arr,
            atr_arr, rsi_arr, times,
            f_times_i64, f_rates, times_i64,
            best_cfg, best_fcfg,
            test_start_idx, test_end_idx,
            max(warmup, test_start_idx - warmup),
        )

        all_oos_trades.extend(oos_trades)

        # 월별 결과
        month_label = test_start.strftime('%Y-%m')
        n_trades = len(oos_trades)
        month_pnl = sum(t["pnl_pct"] for t in oos_trades) if oos_trades else 0.0

        monthly_results.append({
            "month": month_label,
            "n_trades": n_trades,
            "pnl": round(month_pnl, 2),
            "best_params": best_label,
            "train_score": round(best_score, 3),
        })
        chosen_params_history.append(best_label)

        elapsed = time.time() - t0
        print(f"    {month_label}: {n_trades}건 PnL={month_pnl:+.2f}% | 선택: {best_label[:60]} | {elapsed:.0f}s", flush=True)

    total_elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════════
    # 결과 분석
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print(f"  XRP 롤링 리프레시 결과", flush=True)
    print(f"  훈련창: {train_months}개월 | 실행시간: {total_elapsed:.0f}초", flush=True)
    print(f"{'='*70}", flush=True)

    if not all_oos_trades:
        print(f"\n  OOS 거래 없음!", flush=True)
        return

    tdf = pd.DataFrame(all_oos_trades)
    n = len(tdf)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    n_wins, n_losses = len(wins), len(losses)

    total_pnl = tdf["pnl_pct"].sum()
    gp = wins["pnl_pct"].sum() if n_wins else 0
    gl = abs(losses["pnl_pct"].sum()) if n_losses else 0.01
    pf = gp / gl
    wr = n_wins / n * 100

    cum = tdf["pnl_pct"].cumsum()
    max_dd = float((cum - cum.cummax()).min())

    balance = 1000.0
    peak = 1000.0
    max_dd_bal = 0.0
    for p in tdf["pnl_pct"]:
        balance *= (1 + p / 100 * XRP_CONFIG.balance_ratio)
        peak = max(peak, balance)
        dd = (balance - peak) / peak * 100
        max_dd_bal = min(max_dd_bal, dd)

    sl_trades = tdf[tdf["reason"] == "SL"]
    sl_count = len(sl_trades)
    sl_pnl = sl_trades["pnl_pct"].sum() if sl_count else 0

    print(f"\n  --- OOS 종합 성적 ---", flush=True)
    print(f"  총 거래:      {n}건 ({n_wins}W / {n_losses}L)", flush=True)
    print(f"  승률:         {wr:.1f}%", flush=True)
    print(f"  Profit Factor: {pf:.2f}", flush=True)
    print(f"  총 PnL:       {total_pnl:+.2f}%", flush=True)
    print(f"  Max DD (누적): {max_dd:+.2f}%", flush=True)
    print(f"  SL 횟수/손실: {sl_count}건 / {sl_pnl:+.2f}%", flush=True)
    print(flush=True)
    print(f"  --- 복리 시뮬레이션 ($1000, 잔고 {XRP_CONFIG.balance_ratio:.0%}) ---", flush=True)
    print(f"  최종 잔고:    ${balance:,.2f}", flush=True)
    print(f"  총 수익률:    {(balance/1000-1)*100:+.2f}%", flush=True)
    print(f"  Max DD (잔고): {max_dd_bal:+.2f}%", flush=True)

    # 고정 파라미터 비교
    print(f"\n  --- 고정 파라미터 (현재 설정) vs 롤링 비교 ---", flush=True)
    fixed_cfg = XRP_CONFIG
    fixed_fcfg = FundingConfig()
    fixed_trades = run_funding_period(
        close_arr, high_arr, low_arr, open_arr,
        atr_arr, rsi_arr, times,
        f_times_i64, f_rates, times_i64,
        fixed_cfg, fixed_fcfg,
        df.index.searchsorted(test_periods[0][0]),
        len(df),
        max(warmup, 0),
    )
    fixed_pnl = sum(t["pnl_pct"] for t in fixed_trades) if fixed_trades else 0
    fixed_n = len(fixed_trades)
    fixed_wins = sum(1 for t in fixed_trades if t["pnl_pct"] > 0)

    print(f"  {'':15s} {'거래':>6s} {'승률':>7s} {'PF':>6s} {'PnL':>10s} {'잔고':>10s}", flush=True)
    print(f"  {'-'*55}", flush=True)
    print(f"  {'고정':15s} {fixed_n:6d} {fixed_wins/max(fixed_n,1)*100:6.1f}% {'N/A':>6s} {fixed_pnl:+9.2f}% {'N/A':>10s}", flush=True)
    print(f"  {'롤링':15s} {n:6d} {wr:6.1f}% {pf:5.2f}  {total_pnl:+9.2f}% ${balance:>9,.2f}", flush=True)

    # 청산 사유별
    print(f"\n  --- 청산 사유별 ---", flush=True)
    reason_stats = tdf.groupby("reason").agg(
        count=("pnl_pct", "count"),
        avg_pnl=("pnl_pct", "mean"),
        total_pnl=("pnl_pct", "sum"),
    ).sort_values("count", ascending=False)
    for reason, row in reason_stats.iterrows():
        print(f"    {reason:20s}  {int(row['count']):4d}건  avg {row['avg_pnl']:+.2f}%  total {row['total_pnl']:+.2f}%", flush=True)

    # 월별 결과
    print(f"\n  --- 월별 OOS 결과 ---", flush=True)
    print(f"  {'월':>8s} {'거래':>5s} {'PnL':>9s} {'IS점수':>7s} {'선택 파라미터'}", flush=True)
    print(f"  {'-'*90}", flush=True)
    for mr in monthly_results:
        bar = "+" * int(max(0, mr['pnl'])) + "-" * int(max(0, -mr['pnl']))
        print(f"  {mr['month']:>8s} {mr['n_trades']:5d} {mr['pnl']:+8.2f}% {mr['train_score']:6.2f}  {mr['best_params'][:55]} {bar}", flush=True)

    # 파라미터 안정성 분석
    print(f"\n  --- 파라미터 안정성 ---", flush=True)
    from collections import Counter
    param_counts = Counter(chosen_params_history)
    total_months = len(chosen_params_history)
    print(f"  고유 파라미터 셋: {len(param_counts)}개 / {total_months}개월", flush=True)
    print(f"  상위 5개:", flush=True)
    for params, count in param_counts.most_common(5):
        print(f"    {count:3d}회 ({count/total_months*100:.0f}%): {params}", flush=True)

    # 연도별
    tdf["year"] = pd.to_datetime(tdf["exit_time"]).dt.year
    yearly = tdf.groupby("year")["pnl_pct"].agg(["sum", "count", "mean"])
    print(f"\n  --- 연도별 ---", flush=True)
    for year, row in yearly.iterrows():
        print(f"    {year}  {int(row['count']):4d}건  총 {row['sum']:+.2f}%  평균 {row['mean']:+.2f}%", flush=True)

    print(f"\n{'='*70}", flush=True)


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XRP Rolling Refresh Backtest")
    parser.add_argument("--months", type=int, default=60, help="전체 백테스트 기간")
    parser.add_argument("--train-months", type=int, default=6, help="훈련창 (개월)")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.months * 30)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)

    print(f"\n  Fetching XRP 15m data ...", flush=True)
    xrp_df = fetch_ohlcv_bulk("XRP/USDT:USDT", "15m", since_ms, until_ms)
    print(f"  {len(xrp_df)} bars ({xrp_df.index[0].strftime('%Y-%m-%d')} ~ {xrp_df.index[-1].strftime('%Y-%m-%d')})", flush=True)

    print(f"\n  Fetching XRP funding rates ...", flush=True)
    xrp_funding = fetch_funding_bulk("XRP/USDT:USDT", since_ms, until_ms)
    print(f"  {len(xrp_funding)} rates", flush=True)

    rolling_backtest(
        xrp_df, xrp_funding,
        train_months=args.train_months,
        total_months=args.months,
    )
