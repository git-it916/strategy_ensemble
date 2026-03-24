#!/usr/bin/env python3
"""
Precision Sniper Backtest

6개월 과거 데이터로 백테스트.
Usage:
    python scripts/backtest_sniper.py
    python scripts/backtest_sniper.py --symbol ETH/USDT:USDT
    python scripts/backtest_sniper.py --tf 15m --htf 4h
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sniper.config import LEVERAGE
from src.sniper.strategy import PrecisionSniper, Direction


def fetch_ohlcv_bulk(symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    """ccxt 동기로 대량 OHLCV fetch (페이지네이션)."""
    import ccxt
    import yaml

    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"
    with open(keys_path) as f:
        keys = yaml.safe_load(f)

    binance_cfg = keys.get("binance", {})
    exchange = ccxt.binance({
        "apiKey": binance_cfg.get("api_key", ""),
        "secret": binance_cfg.get("api_secret", ""),
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })

    all_ohlcv = []
    cursor = since_ms

    print(f"  Fetching {symbol} {timeframe} ...")
    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1500)
        if not batch:
            break
        all_ohlcv.extend(batch)
        cursor = batch[-1][0] + 1
        # 중복 방지: until 초과 데이터 제거
        if batch[-1][0] >= until_ms:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp")
    df = df[df.index <= pd.Timestamp(until_ms, unit="ms", tz=None)]
    return df


def align_htf(htf_df: pd.DataFrame, current_time: pd.Timestamp) -> pd.DataFrame:
    """현재 시점 이전의 HTF 데이터만 반환 (미래 데이터 누수 방지)."""
    return htf_df[htf_df.index < current_time].copy()


def tf_to_timedelta(tf: str) -> timedelta:
    unit = tf[-1]
    num = int(tf[:-1])
    if unit == "m":
        return timedelta(minutes=num)
    elif unit == "h":
        return timedelta(hours=num)
    elif unit == "d":
        return timedelta(days=num)
    return timedelta(minutes=5)


def run_backtest(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
    htf: str = "4h",
    months: int = 6,
):
    short_sym = symbol.split("/")[0]
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=months * 30)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)

    print(f"={'=' * 60}")
    print(f"Precision Sniper Backtest")
    print(f"Symbol: {short_sym} | TF: {timeframe} | HTF: {htf}")
    print(f"Period: {since.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')} ({months}개월)")
    print(f"Leverage: {LEVERAGE}x")
    print(f"={'=' * 60}")

    # --- 데이터 수집 ---
    df = fetch_ohlcv_bulk(symbol, timeframe, since_ms, until_ms)
    htf_df = fetch_ohlcv_bulk(symbol, htf, since_ms, until_ms)
    print(f"  {timeframe} bars: {len(df)} | {htf} bars: {len(htf_df)}")
    print()

    # --- 백테스트 실행 ---
    sniper = PrecisionSniper()
    trades = []
    lookback = 200  # 시그널 계산에 필요한 최소 봉 수

    active_entry = None     # (direction, entry_price, sl, tp1, tp2, tp3, risk, entry_time, score)
    tp1_hit = False
    tp2_hit = False
    tp3_hit = False
    trail_price = 0.0

    total_bars = len(df)
    report_interval = total_bars // 10

    for i in range(lookback, total_bars):
        if report_interval > 0 and i % report_interval == 0:
            pct = i / total_bars * 100
            print(f"  Processing... {pct:.0f}% ({i}/{total_bars})")

        window = df.iloc[max(0, i - lookback):i + 1].copy()
        current_bar = df.iloc[i]
        current_time = df.index[i]

        # --- 포지션 있으면 봉 내 TP/SL 체크 (high/low 사용) ---
        if active_entry is not None:
            direction, entry_price, sl, tp1, tp2, tp3, risk, entry_time, score = active_entry
            bar_high = float(current_bar["high"])
            bar_low = float(current_bar["low"])
            bar_close = float(current_bar["close"])

            exit_price = None
            exit_reason = None

            if direction == "LONG":
                # TP 체크 (high 기준)
                if bar_high >= tp3 and not tp3_hit:
                    tp3_hit = True
                    trail_price = tp2
                if bar_high >= tp2 and not tp2_hit:
                    tp2_hit = True
                    trail_price = tp1
                if bar_high >= tp1 and not tp1_hit:
                    tp1_hit = True
                    trail_price = entry_price

                # SL/Trail 체크 (low 기준)
                stop = trail_price if tp1_hit else sl
                if bar_low <= stop:
                    exit_price = stop
                    exit_reason = f"TRAIL_TP{'3' if tp3_hit else '2' if tp2_hit else '1'}_HIT" if tp1_hit else "SL"

            else:  # SHORT
                if bar_low <= tp3 and not tp3_hit:
                    tp3_hit = True
                    trail_price = tp2
                if bar_low <= tp2 and not tp2_hit:
                    tp2_hit = True
                    trail_price = tp1
                if bar_low <= tp1 and not tp1_hit:
                    tp1_hit = True
                    trail_price = entry_price

                stop = trail_price if tp1_hit else sl
                if bar_high >= stop:
                    exit_price = stop
                    exit_reason = f"TRAIL_TP{'3' if tp3_hit else '2' if tp2_hit else '1'}_HIT" if tp1_hit else "SL"

            if exit_price is not None:
                if direction == "LONG":
                    pnl_pct = (exit_price - entry_price) / entry_price * 100 * LEVERAGE
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100 * LEVERAGE

                hold_min = (current_time - entry_time).total_seconds() / 60
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": round(pnl_pct, 4),
                    "reason": exit_reason,
                    "score": score,
                    "tp1_hit": tp1_hit,
                    "tp2_hit": tp2_hit,
                    "tp3_hit": tp3_hit,
                    "hold_min": round(hold_min, 1),
                })
                active_entry = None
                tp1_hit = False
                tp2_hit = False
                tp3_hit = False
                trail_price = 0.0

        # --- 시그널 계산 (캔들 클로즈 기준) ---
        htf_window = align_htf(htf_df, current_time)
        if len(htf_window) < 30:
            continue

        signal = sniper.compute(window, htf_window)

        if signal is None:
            continue

        # 반대 시그널 → 기존 청산
        if active_entry is not None:
            direction, entry_price, sl, tp1, tp2, tp3, risk, entry_time, score = active_entry
            if signal.direction.value != direction:
                close_price = float(current_bar["close"])
                if direction == "LONG":
                    pnl_pct = (close_price - entry_price) / entry_price * 100 * LEVERAGE
                else:
                    pnl_pct = (entry_price - close_price) / entry_price * 100 * LEVERAGE

                hold_min = (current_time - entry_time).total_seconds() / 60
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": close_price,
                    "pnl_pct": round(pnl_pct, 4),
                    "reason": "REVERSE",
                    "score": score,
                    "tp1_hit": tp1_hit,
                    "tp2_hit": tp2_hit,
                    "tp3_hit": tp3_hit,
                    "hold_min": round(hold_min, 1),
                })
                active_entry = None
                tp1_hit = False
                tp2_hit = False
                tp3_hit = False
                trail_price = 0.0
                sniper.close_trade()

        # 신규 진입
        if active_entry is None:
            active_entry = (
                signal.direction.value,
                signal.entry_price,
                signal.sl_price,
                signal.tp1_price,
                signal.tp2_price,
                signal.tp3_price,
                signal.risk,
                current_time,
                signal.score,
            )
            trail_price = signal.sl_price
            sniper.open_trade(signal)

    # --- 미결 포지션 정리 ---
    if active_entry is not None:
        direction, entry_price, sl, tp1, tp2, tp3, risk, entry_time, score = active_entry
        close_price = float(df["close"].iloc[-1])
        if direction == "LONG":
            pnl_pct = (close_price - entry_price) / entry_price * 100 * LEVERAGE
        else:
            pnl_pct = (entry_price - close_price) / entry_price * 100 * LEVERAGE
        trades.append({
            "entry_time": entry_time,
            "exit_time": df.index[-1],
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": close_price,
            "pnl_pct": round(pnl_pct, 4),
            "reason": "OPEN_AT_END",
            "score": score,
            "tp1_hit": tp1_hit,
            "tp2_hit": tp2_hit,
            "tp3_hit": tp3_hit,
            "hold_min": round((df.index[-1] - entry_time).total_seconds() / 60, 1),
        })

    # --- 결과 출력 ---
    print_results(trades, short_sym, timeframe, htf, months)
    return trades


def print_results(trades: list, symbol: str, tf: str, htf: str, months: int):
    if not trades:
        print("\n거래 없음!")
        return

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    n_wins = len(wins)
    n_losses = len(losses)

    total_pnl = tdf["pnl_pct"].sum()
    avg_pnl = tdf["pnl_pct"].mean()
    avg_win = wins["pnl_pct"].mean() if n_wins > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if n_losses > 0 else 0
    max_win = tdf["pnl_pct"].max()
    max_loss = tdf["pnl_pct"].min()
    win_rate = n_wins / n * 100

    # Profit factor
    gross_profit = wins["pnl_pct"].sum() if n_wins > 0 else 0
    gross_loss = abs(losses["pnl_pct"].sum()) if n_losses > 0 else 0.01
    profit_factor = gross_profit / gross_loss

    # Max drawdown (누적 수익 기준)
    cum_pnl = tdf["pnl_pct"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    # 연속 승/패
    streak_win = 0
    streak_loss = 0
    max_streak_win = 0
    max_streak_loss = 0
    for pnl in tdf["pnl_pct"]:
        if pnl > 0:
            streak_win += 1
            streak_loss = 0
            max_streak_win = max(max_streak_win, streak_win)
        else:
            streak_loss += 1
            streak_win = 0
            max_streak_loss = max(max_streak_loss, streak_loss)

    # 방향별 통계
    longs = tdf[tdf["direction"] == "LONG"]
    shorts = tdf[tdf["direction"] == "SHORT"]
    long_wr = len(longs[longs["pnl_pct"] > 0]) / len(longs) * 100 if len(longs) > 0 else 0
    short_wr = len(shorts[shorts["pnl_pct"] > 0]) / len(shorts) * 100 if len(shorts) > 0 else 0

    # 청산 사유별 통계
    reason_stats = tdf.groupby("reason").agg(
        count=("pnl_pct", "count"),
        avg_pnl=("pnl_pct", "mean"),
        total_pnl=("pnl_pct", "sum"),
    ).sort_values("count", ascending=False)

    # TP 달성률
    tp1_rate = tdf["tp1_hit"].sum() / n * 100
    tp2_rate = tdf["tp2_hit"].sum() / n * 100
    tp3_rate = tdf["tp3_hit"].sum() / n * 100

    # 평균 보유 시간
    avg_hold = tdf["hold_min"].mean()

    # 가상 잔고 시뮬레이션 (복리)
    balance = 1000.0
    balances = [balance]
    for pnl in tdf["pnl_pct"]:
        balance *= (1 + pnl / 100)
        balances.append(balance)
    final_balance = balances[-1]

    # 월별 수익
    tdf["month"] = pd.to_datetime(tdf["exit_time"]).dt.to_period("M")
    monthly = tdf.groupby("month")["pnl_pct"].agg(["sum", "count", "mean"])

    print()
    print(f"{'=' * 60}")
    print(f"  BACKTEST RESULTS — {symbol} {tf} (HTF: {htf})")
    print(f"  Period: {months}개월 | Leverage: {LEVERAGE}x")
    print(f"{'=' * 60}")
    print()
    print(f"  총 거래 수:       {n}")
    print(f"  승/패:            {n_wins}W / {n_losses}L")
    print(f"  승률:             {win_rate:.1f}%")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print()
    print(f"  --- 수익 ---")
    print(f"  총 수익:          {total_pnl:+.2f}%")
    print(f"  평균 수익:        {avg_pnl:+.2f}%")
    print(f"  평균 수익 (승):   {avg_win:+.2f}%")
    print(f"  평균 손실 (패):   {avg_loss:+.2f}%")
    print(f"  최대 수익:        {max_win:+.2f}%")
    print(f"  최대 손실:        {max_loss:+.2f}%")
    print(f"  Max Drawdown:     {max_dd:+.2f}%")
    print()
    print(f"  --- 시뮬레이션 ($1000 시작, 복리) ---")
    print(f"  최종 잔고:        ${final_balance:,.2f}")
    print(f"  총 수익률:        {(final_balance/1000-1)*100:+.2f}%")
    print()
    print(f"  --- 방향별 ---")
    print(f"  LONG:   {len(longs)}건 (승률 {long_wr:.1f}%) 평균 {longs['pnl_pct'].mean():+.2f}%" if len(longs) > 0 else "  LONG:   0건")
    print(f"  SHORT:  {len(shorts)}건 (승률 {short_wr:.1f}%) 평균 {shorts['pnl_pct'].mean():+.2f}%" if len(shorts) > 0 else "  SHORT:  0건")
    print()
    print(f"  --- TP 달성률 ---")
    print(f"  TP1:  {tp1_rate:.1f}%  |  TP2:  {tp2_rate:.1f}%  |  TP3:  {tp3_rate:.1f}%")
    print()
    print(f"  --- 기타 ---")
    print(f"  평균 보유시간:    {avg_hold:.0f}분 ({avg_hold/60:.1f}시간)")
    print(f"  최대 연승:        {max_streak_win}")
    print(f"  최대 연패:        {max_streak_loss}")
    print()
    print(f"  --- 청산 사유별 ---")
    for reason, row in reason_stats.iterrows():
        print(f"  {reason:20s}  {int(row['count']):4d}건  avg {row['avg_pnl']:+.2f}%  total {row['total_pnl']:+.2f}%")
    print()
    print(f"  --- 월별 수익 ---")
    for period, row in monthly.iterrows():
        print(f"  {period}  {int(row['count']):3d}건  총 {row['sum']:+.2f}%  평균 {row['mean']:+.2f}%")
    print()
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precision Sniper Backtest")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--tf", default="15m")
    parser.add_argument("--htf", default="4h")
    parser.add_argument("--months", type=int, default=6)
    args = parser.parse_args()

    run_backtest(
        symbol=args.symbol,
        timeframe=args.tf,
        htf=args.htf,
        months=args.months,
    )
