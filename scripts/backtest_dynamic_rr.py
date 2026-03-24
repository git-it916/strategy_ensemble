#!/usr/bin/env python3
"""
Dynamic R:R Strategy Backtest

Pine Script "Dynamic R:R Strategy - BTC 15m Optimized" 백테스트.
진입: EMA(20/50) cross + EMA(200) trend filter + RSI(14) 60/40
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


# === 파라미터 ===
EMA_FAST = 20
EMA_SLOW = 50
EMA_TREND = 200
RSI_LEN = 14
RSI_BULL = 60       # RSI > 60 for long
RSI_BEAR = 40       # RSI < 40 for short
ATR_LEN = 14
ATR_MULT = 1.5
SWING_LOOKBACK = 5
SL_METHOD = "ATR"   # "ATR" or "SWING"
TP1_RR = 1.0
TP2_RR = 2.0
TP3_RR = 3.0
LEVERAGE = 3
WARMUP = 210        # EMA(200) + 여유


def fetch_ohlcv_bulk(symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.DataFrame:
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
        if batch[-1][0] >= until_ms:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # EMAs
    df["ema_fast"] = c.ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = c.ewm(span=EMA_SLOW, adjust=False).mean()
    df["ema_trend"] = c.ewm(span=EMA_TREND, adjust=False).mean()

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / RSI_LEN, min_periods=RSI_LEN).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_LEN, min_periods=RSI_LEN).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_LEN, min_periods=ATR_LEN).mean()

    # EMA cross
    df["cross_up"] = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
    df["cross_down"] = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))

    # Swing high/low (rolling)
    df["swing_low"] = l.rolling(SWING_LOOKBACK, min_periods=SWING_LOOKBACK).min()
    df["swing_high"] = h.rolling(SWING_LOOKBACK, min_periods=SWING_LOOKBACK).max()

    return df


def run_backtest(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
    months: int = 6,
    sl_method: str = "ATR",
    atr_mult: float = ATR_MULT,
    rsi_bull: float = RSI_BULL,
    rsi_bear: float = RSI_BEAR,
):
    short_sym = symbol.split("/")[0]
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=months * 30)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)

    label = f"{short_sym} {timeframe} | SL={sl_method} ATR×{atr_mult} | RSI {rsi_bull}/{rsi_bear}"
    print(f"{'=' * 60}")
    print(f"  Dynamic R:R Backtest — {label}")
    print(f"  {since.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')} ({months}개월) | {LEVERAGE}x")
    print(f"{'=' * 60}")

    df = fetch_ohlcv_bulk(symbol, timeframe, since_ms, until_ms)
    print(f"  {timeframe} bars: {len(df)}")

    df = compute_indicators(df)

    # --- 백테스트 ---
    trades = []
    last_direction = 0  # anti-duplicate
    active = None  # (dir, entry, sl, tp1, tp2, tp3, risk, time)
    tp1_hit = tp2_hit = tp3_hit = False
    trail_price = 0.0

    total = len(df)
    for i in range(WARMUP, total):
        row = df.iloc[i]
        idx_time = df.index[i]

        # --- 포지션 TP/SL 체크 ---
        if active is not None:
            d, entry, sl, tp1, tp2, tp3, risk, etime = active
            bh = float(row["high"])
            bl = float(row["low"])

            exit_price = None
            exit_reason = None

            if d == "LONG":
                if bh >= tp3 and not tp3_hit:
                    tp3_hit = True
                    trail_price = tp2
                if bh >= tp2 and not tp2_hit:
                    tp2_hit = True
                    trail_price = tp1
                if bh >= tp1 and not tp1_hit:
                    tp1_hit = True
                    trail_price = entry

                stop = trail_price if tp1_hit else sl
                if bl <= stop:
                    exit_price = stop
                    exit_reason = f"TRAIL_TP{'3' if tp3_hit else '2' if tp2_hit else '1'}_HIT" if tp1_hit else "SL"
            else:
                if bl <= tp3 and not tp3_hit:
                    tp3_hit = True
                    trail_price = tp2
                if bl <= tp2 and not tp2_hit:
                    tp2_hit = True
                    trail_price = tp1
                if bl <= tp1 and not tp1_hit:
                    tp1_hit = True
                    trail_price = entry

                stop = trail_price if tp1_hit else sl
                if bh >= stop:
                    exit_price = stop
                    exit_reason = f"TRAIL_TP{'3' if tp3_hit else '2' if tp2_hit else '1'}_HIT" if tp1_hit else "SL"

            if exit_price:
                pnl = ((exit_price - entry) / entry * 100 * LEVERAGE) if d == "LONG" else ((entry - exit_price) / entry * 100 * LEVERAGE)
                trades.append({
                    "entry_time": etime, "exit_time": idx_time, "direction": d,
                    "entry_price": entry, "exit_price": exit_price,
                    "pnl_pct": round(pnl, 4), "reason": exit_reason,
                    "tp1_hit": tp1_hit, "tp2_hit": tp2_hit, "tp3_hit": tp3_hit,
                    "hold_min": round((idx_time - etime).total_seconds() / 60, 1),
                })
                active = None
                tp1_hit = tp2_hit = tp3_hit = False
                trail_price = 0.0

        # --- 시그널 체크 ---
        if pd.isna(row["ema_trend"]) or pd.isna(row["rsi"]) or pd.isna(row["atr"]):
            continue

        cross_up = bool(row["cross_up"])
        cross_down = bool(row["cross_down"])

        long_sig = cross_up and row["close"] > row["ema_trend"] and row["rsi"] > rsi_bull
        short_sig = cross_down and row["close"] < row["ema_trend"] and row["rsi"] < rsi_bear

        # anti-duplicate
        if long_sig and last_direction == 1:
            long_sig = False
        if short_sig and last_direction == -1:
            short_sig = False
        if long_sig and short_sig:
            short_sig = False

        if not long_sig and not short_sig:
            continue

        # 반대 시그널 → 청산
        if active is not None:
            d, entry, sl, tp1, tp2, tp3, risk, etime = active
            sig_dir = "LONG" if long_sig else "SHORT"
            if sig_dir != d:
                cp = float(row["close"])
                pnl = ((cp - entry) / entry * 100 * LEVERAGE) if d == "LONG" else ((entry - cp) / entry * 100 * LEVERAGE)
                trades.append({
                    "entry_time": etime, "exit_time": idx_time, "direction": d,
                    "entry_price": entry, "exit_price": cp,
                    "pnl_pct": round(pnl, 4), "reason": "REVERSE",
                    "tp1_hit": tp1_hit, "tp2_hit": tp2_hit, "tp3_hit": tp3_hit,
                    "hold_min": round((idx_time - etime).total_seconds() / 60, 1),
                })
                active = None
                tp1_hit = tp2_hit = tp3_hit = False
                trail_price = 0.0

        # 신규 진입
        if active is None:
            entry = float(row["close"])
            atr_val = float(row["atr"])

            if long_sig:
                last_direction = 1
                if sl_method == "ATR":
                    sl = entry - atr_val * atr_mult
                else:
                    sl = float(row["swing_low"]) if not pd.isna(row["swing_low"]) else entry - atr_val * atr_mult
                risk = abs(entry - sl)
                if risk < atr_val * 0.3:
                    risk = atr_val * 0.5
                    sl = entry - risk
                tp1 = entry + risk * TP1_RR
                tp2 = entry + risk * TP2_RR
                tp3 = entry + risk * TP3_RR
                active = ("LONG", entry, sl, tp1, tp2, tp3, risk, idx_time)
                trail_price = sl

            elif short_sig:
                last_direction = -1
                if sl_method == "ATR":
                    sl = entry + atr_val * atr_mult
                else:
                    sl = float(row["swing_high"]) if not pd.isna(row["swing_high"]) else entry + atr_val * atr_mult
                risk = abs(entry - sl)
                if risk < atr_val * 0.3:
                    risk = atr_val * 0.5
                    sl = entry + risk
                tp1 = entry - risk * TP1_RR
                tp2 = entry - risk * TP2_RR
                tp3 = entry - risk * TP3_RR
                active = ("SHORT", entry, sl, tp1, tp2, tp3, risk, idx_time)
                trail_price = sl

    # 미결 정리
    if active is not None:
        d, entry, sl, tp1, tp2, tp3, risk, etime = active
        cp = float(df["close"].iloc[-1])
        pnl = ((cp - entry) / entry * 100 * LEVERAGE) if d == "LONG" else ((entry - cp) / entry * 100 * LEVERAGE)
        trades.append({
            "entry_time": etime, "exit_time": df.index[-1], "direction": d,
            "entry_price": entry, "exit_price": cp,
            "pnl_pct": round(pnl, 4), "reason": "OPEN_AT_END",
            "tp1_hit": tp1_hit, "tp2_hit": tp2_hit, "tp3_hit": tp3_hit,
            "hold_min": round((df.index[-1] - etime).total_seconds() / 60, 1),
        })

    print_results(trades, label)
    return trades


def print_results(trades: list, label: str):
    if not trades:
        print("\n  거래 없음!\n")
        return

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    nw, nl = len(wins), len(losses)

    total_pnl = tdf["pnl_pct"].sum()
    avg_pnl = tdf["pnl_pct"].mean()
    avg_win = wins["pnl_pct"].mean() if nw else 0
    avg_loss = losses["pnl_pct"].mean() if nl else 0
    max_win = tdf["pnl_pct"].max()
    max_loss = tdf["pnl_pct"].min()
    wr = nw / n * 100

    gp = wins["pnl_pct"].sum() if nw else 0
    gl = abs(losses["pnl_pct"].sum()) if nl else 0.01
    pf = gp / gl

    cum = tdf["pnl_pct"].cumsum()
    dd = (cum - cum.cummax()).min()

    # streaks
    sw = sl_ = msw = msl = 0
    for p in tdf["pnl_pct"]:
        if p > 0:
            sw += 1; sl_ = 0; msw = max(msw, sw)
        else:
            sl_ += 1; sw = 0; msl = max(msl, sl_)

    longs = tdf[tdf["direction"] == "LONG"]
    shorts = tdf[tdf["direction"] == "SHORT"]
    lwr = len(longs[longs["pnl_pct"] > 0]) / len(longs) * 100 if len(longs) else 0
    swr = len(shorts[shorts["pnl_pct"] > 0]) / len(shorts) * 100 if len(shorts) else 0

    tp1r = tdf["tp1_hit"].sum() / n * 100
    tp2r = tdf["tp2_hit"].sum() / n * 100
    tp3r = tdf["tp3_hit"].sum() / n * 100

    ah = tdf["hold_min"].mean()

    bal = 1000.0
    for p in tdf["pnl_pct"]:
        bal *= (1 + p / 100)

    reason_stats = tdf.groupby("reason").agg(
        count=("pnl_pct", "count"), avg=("pnl_pct", "mean"), total=("pnl_pct", "sum"),
    ).sort_values("count", ascending=False)

    tdf["month"] = pd.to_datetime(tdf["exit_time"]).dt.to_period("M")
    monthly = tdf.groupby("month")["pnl_pct"].agg(["sum", "count", "mean"])

    print()
    print(f"  {'=' * 56}")
    print(f"  RESULTS — {label}")
    print(f"  {'=' * 56}")
    print(f"  총 거래: {n}  |  {nw}W / {nl}L  |  승률: {wr:.1f}%  |  PF: {pf:.2f}")
    print(f"  총 수익: {total_pnl:+.2f}%  |  평균: {avg_pnl:+.2f}%")
    print(f"  평균승: {avg_win:+.2f}%  |  평균패: {avg_loss:+.2f}%")
    print(f"  최대승: {max_win:+.2f}%  |  최대패: {max_loss:+.2f}%")
    print(f"  Max DD: {dd:+.2f}%  |  $1000→ ${bal:,.2f}")
    print(f"  LONG: {len(longs)}건 WR {lwr:.1f}%  |  SHORT: {len(shorts)}건 WR {swr:.1f}%")
    print(f"  TP1: {tp1r:.1f}%  TP2: {tp2r:.1f}%  TP3: {tp3r:.1f}%")
    print(f"  보유: {ah:.0f}분 ({ah/60:.1f}h)  |  연승: {msw}  |  연패: {msl}")
    print(f"  --- 청산 사유 ---")
    for r, row in reason_stats.iterrows():
        print(f"  {r:20s} {int(row['count']):4d}건  avg {row['avg']:+.2f}%  tot {row['total']:+.2f}%")
    print(f"  --- 월별 ---")
    for period, row in monthly.iterrows():
        print(f"  {period}  {int(row['count']):3d}건  {row['sum']:+.2f}%  avg {row['mean']:+.2f}%")
    print(f"  {'=' * 56}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--tf", default="15m")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--sl", default="ATR", choices=["ATR", "SWING"])
    parser.add_argument("--atr-mult", type=float, default=1.5)
    parser.add_argument("--rsi-bull", type=float, default=60)
    parser.add_argument("--rsi-bear", type=float, default=40)
    args = parser.parse_args()

    run_backtest(
        symbol=args.symbol, timeframe=args.tf, months=args.months,
        sl_method=args.sl, atr_mult=args.atr_mult,
        rsi_bull=args.rsi_bull, rsi_bear=args.rsi_bear,
    )
