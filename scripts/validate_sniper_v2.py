#!/usr/bin/env python3
"""
Sniper V2 IS/OOS Validation — 라이브 파라미터 고정 상태에서 오버피팅 검증.

전략 3종:
  1. BTC: EMA 20/50/200 + SWING_15 + PP 1.5R→0.3R
  2. SOL: EMA 8/40/150 + SWING_3 + ADX>=30
  3. XRP: FundingContrarian (z_threshold=1.125, price_confirm_z=0.8)
  4. AVAX: EMA 12/50/200 + SWING_5 + ADX>=30

방법:
  - 60개월(5년) 데이터 수집
  - IS: 앞 4년 / OOS: 최근 1년 (embargo 7일)
  - 파라미터는 config.py의 라이브 값 고정 (그리드서치 없음)
  - IS/OOS 각각에서 동일 파라미터로 백테스트 → 성과 비교

Usage:
    python scripts/validate_sniper_v2.py              # 전 심볼
    python scripts/validate_sniper_v2.py --symbol BTC  # BTC만
    python scripts/validate_sniper_v2.py --oos-days 180 # OOS 6개월
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sniper_v2.config import (
    AVAX_CONFIG,
    BTC_CONFIG,
    CONFIGS,
    FUNDING_STRATEGY_SYMBOLS,
    SOL_CONFIG,
    SymbolConfig,
    XRP_CONFIG,
)
from src.sniper_v2.funding_strategy import FundingConfig
from src.sniper_v2.strategy import ActiveTrade, Direction, Signal

RESULTS_DIR = Path(__file__).parent.parent / "results" / "sniper_v2_validation"
TIMEFRAME = "15m"


# ──────────────────────────────────────────────
# 데이터 수집 (기존 backtest_sniper_v2에서 가져옴)
# ──────────────────────────────────────────────


def _make_exchange():
    """기존 backtest_sniper_v2.py와 동일한 Exchange 생성 (API 키 포함)."""
    import ccxt
    import yaml
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"
    with open(keys_path) as f:
        keys = yaml.safe_load(f)
    binance_cfg = keys.get("binance", {})
    return ccxt.binance({
        "apiKey": binance_cfg.get("api_key", ""),
        "secret": binance_cfg.get("api_secret", ""),
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })


def fetch_ohlcv_bulk(symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    exchange = _make_exchange()
    all_rows = []
    cursor = since_ms
    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1500)
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + 1
        if batch[-1][0] >= until_ms:
            break
        time.sleep(exchange.rateLimit / 1000)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp")
    df = df[df.index <= pd.Timestamp(until_ms, unit="ms")]
    return df


def fetch_funding_bulk(symbol: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    exchange = _make_exchange()
    base = symbol.replace("/USDT:USDT", "USDT")
    all_rows = []
    cursor = since_ms
    while cursor < until_ms:
        try:
            batch = exchange.fapiPublicGetFundingRate({
                "symbol": base, "startTime": cursor, "endTime": until_ms, "limit": 1000,
            })
        except Exception:
            break
        if not batch:
            break
        all_rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)
    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "fundingRate"])
    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(int(r["fundingTime"]), unit="ms"),
        "fundingRate": float(r["fundingRate"]),
    } for r in all_rows])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df


# ──────────────────────────────────────────────
# 백테스트 엔진 (backtest_sniper_v2에서 가져옴, next-bar-open 적용)
# ──────────────────────────────────────────────


def check_exit(trade: ActiveTrade, price: float, cfg: SymbolConfig) -> str | None:
    is_long = trade.direction == Direction.LONG
    if trade.risk > 0:
        cur_r = (price - trade.entry_price) / trade.risk if is_long else (trade.entry_price - price) / trade.risk
        trade.peak_r = max(trade.peak_r, cur_r)
        if cfg.profit_protect and trade.peak_r >= cfg.pp_trigger and cur_r <= cfg.pp_exit:
            return f"PP_{trade.peak_r:.1f}R"

    if is_long:
        if price >= trade.tp3_price and not trade.tp3_hit:
            trade.tp3_hit = True
            if cfg.use_trailing: trade.trail_price = trade.tp2_price
        if price >= trade.tp2_price and not trade.tp2_hit:
            trade.tp2_hit = True
            if cfg.use_trailing: trade.trail_price = trade.tp1_price
        if price >= trade.tp1_price and not trade.tp1_hit:
            trade.tp1_hit = True
            if cfg.use_trailing: trade.trail_price = trade.entry_price
        stop = trade.trail_price if cfg.use_trailing else trade.sl_price
        if price <= stop:
            return f"TRAIL_TP{'3' if trade.tp3_hit else '2' if trade.tp2_hit else '1'}_HIT" if trade.tp1_hit else "SL"
    else:
        if price <= trade.tp3_price and not trade.tp3_hit:
            trade.tp3_hit = True
            if cfg.use_trailing: trade.trail_price = trade.tp2_price
        if price <= trade.tp2_price and not trade.tp2_hit:
            trade.tp2_hit = True
            if cfg.use_trailing: trade.trail_price = trade.tp1_price
        if price <= trade.tp1_price and not trade.tp1_hit:
            trade.tp1_hit = True
            if cfg.use_trailing: trade.trail_price = trade.entry_price
        stop = trade.trail_price if cfg.use_trailing else trade.sl_price
        if price >= stop:
            return f"TRAIL_TP{'3' if trade.tp3_hit else '2' if trade.tp2_hit else '1'}_HIT" if trade.tp1_hit else "SL"
    return None


def compute_trade_pnl(trade: dict) -> float:
    entry = trade["entry_price"]
    exit_ = trade["exit_price"]
    if trade["direction"] == "LONG":
        return (exit_ / entry - 1)
    else:
        return (entry / exit_ - 1)


def backtest_ema_isoos(df: pd.DataFrame, cfg: SymbolConfig) -> list[dict]:
    """EMA 전략 백테스트 (next-bar-open 체결)."""
    from src.sniper_v2.indicators import compute_all

    df = compute_all(df.copy(), cfg)

    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    open_arr = df["open"].values
    atr_arr = df["atr"].values
    ema_fast_arr = df["ema_fast"].values
    ema_slow_arr = df["ema_slow"].values
    ema_trend_arr = df["ema_trend"].values
    rsi_arr = df["rsi"].values
    adx_arr = df["adx"].values
    atr_sma_arr = df["atr_sma"].values
    bull_cross_arr = df["ema_bull_cross"].values
    bear_cross_arr = df["ema_bear_cross"].values
    swing_low_arr = df["swing_low"].values if "swing_low" in df.columns else None
    swing_high_arr = df["swing_high"].values if "swing_high" in df.columns else None
    times = df.index

    trades = []
    active_trade = None
    pending_signal = None
    last_direction = 0
    cooldown = 0
    total_bars = len(df)
    warmup = cfg.warmup_bars

    for i in range(warmup, total_bars):
        current_time = times[i]

        # stale pending signal 폐기
        if pending_signal is not None and active_trade is not None:
            pending_signal = None

        # pending_signal 체결 (next-bar open)
        if pending_signal is not None and active_trade is None:
            fill_price = float(open_arr[i])
            _ps = pending_signal
            pending_signal = None

            # 갭 체크
            if _ps.direction == Direction.LONG and fill_price <= _ps.sl_price:
                pass
            elif _ps.direction == Direction.SHORT and fill_price >= _ps.sl_price:
                pass
            else:
                atr_i = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else _ps.atr
                if cfg.sl_method == "SWING":
                    sl_adj = _ps.sl_price
                    risk = abs(fill_price - sl_adj)
                    if risk < atr_i * 0.3:
                        risk = 0
                else:
                    risk = abs(fill_price - _ps.sl_price)
                    if risk < atr_i * 0.3:
                        risk = atr_i * 0.5
                    sl_adj = fill_price - risk if _ps.direction == Direction.LONG else fill_price + risk

                if risk >= atr_i * 0.3:
                    if _ps.direction == Direction.LONG:
                        tp1 = fill_price + risk * cfg.tp1_rr
                        tp2 = fill_price + risk * cfg.tp2_rr
                        tp3 = fill_price + risk * cfg.tp3_rr
                    else:
                        tp1 = fill_price - risk * cfg.tp1_rr
                        tp2 = fill_price - risk * cfg.tp2_rr
                        tp3 = fill_price - risk * cfg.tp3_rr
                    active_trade = ActiveTrade(
                        direction=_ps.direction, entry_price=fill_price,
                        sl_price=sl_adj, tp1_price=tp1, tp2_price=tp2, tp3_price=tp3,
                        trail_price=sl_adj, risk=risk, entry_time=current_time,
                    )

        # SL/TP 체크
        if active_trade is not None:
            is_long_pos = active_trade.direction == Direction.LONG
            seq = [open_arr[i], low_arr[i], high_arr[i], close_arr[i]] if is_long_pos else [open_arr[i], high_arr[i], low_arr[i], close_arr[i]]
            for cp in seq:
                if active_trade is None: break
                reason = check_exit(active_trade, float(cp), cfg)
                if reason:
                    trades.append({
                        "direction": active_trade.direction.value,
                        "entry_price": active_trade.entry_price,
                        "exit_price": float(cp),
                        "entry_time": str(active_trade.entry_time),
                        "exit_time": str(current_time),
                        "reason": reason,
                    })
                    if reason == "SL": cooldown = cfg.cooldown_bars
                    active_trade = None
                    break

        if cooldown > 0:
            cooldown -= 1
            continue

        # NaN 체크
        if any(np.isnan(x) for x in [ema_fast_arr[i], ema_slow_arr[i], ema_trend_arr[i], rsi_arr[i], atr_arr[i], adx_arr[i]]):
            continue

        atr_val = float(atr_arr[i])
        atr_sma_val = float(atr_sma_arr[i]) if not np.isnan(atr_sma_arr[i]) else 0
        if atr_sma_val > 0 and atr_val / atr_sma_val > cfg.vol_filter:
            continue
        if float(adx_arr[i]) < cfg.adx_min:
            continue

        close = float(close_arr[i])
        long_sig = bool(bull_cross_arr[i]) and close > float(ema_trend_arr[i]) and float(rsi_arr[i]) > cfg.rsi_bull
        short_sig = bool(bear_cross_arr[i]) and close < float(ema_trend_arr[i]) and float(rsi_arr[i]) < cfg.rsi_bear

        if long_sig and last_direction == 1: long_sig = False
        if short_sig and last_direction == -1: short_sig = False
        if long_sig and short_sig: short_sig = False
        if not long_sig and not short_sig: continue
        if i + 1 >= total_bars: continue

        entry = close
        is_long = long_sig
        if cfg.sl_method == "SWING" and swing_low_arr is not None:
            if is_long:
                sl = float(swing_low_arr[i]) if not np.isnan(swing_low_arr[i]) else entry - atr_val * 1.5
            else:
                sl = float(swing_high_arr[i]) if not np.isnan(swing_high_arr[i]) else entry + atr_val * 1.5
        else:
            sl = entry - atr_val * cfg.sl_atr_mult if is_long else entry + atr_val * cfg.sl_atr_mult

        risk = abs(entry - sl)
        if risk < atr_val * 0.3:
            risk = atr_val * 0.5
            sl = entry - risk if is_long else entry + risk

        direction = Direction.LONG if is_long else Direction.SHORT
        last_direction = 1 if is_long else -1

        signal = Signal(
            direction=direction, entry_price=entry, sl_price=sl,
            tp1_price=entry + risk * cfg.tp1_rr if is_long else entry - risk * cfg.tp1_rr,
            tp2_price=entry + risk * cfg.tp2_rr if is_long else entry - risk * cfg.tp2_rr,
            tp3_price=entry + risk * cfg.tp3_rr if is_long else entry - risk * cfg.tp3_rr,
            risk=risk, rsi=float(rsi_arr[i]), adx=float(adx_arr[i]), atr=atr_val,
        )

        if active_trade is not None and signal.direction != active_trade.direction:
            trades.append({
                "direction": active_trade.direction.value,
                "entry_price": active_trade.entry_price,
                "exit_price": close,
                "entry_time": str(active_trade.entry_time),
                "exit_time": str(current_time),
                "reason": "REVERSE",
            })
            active_trade = None

        if active_trade is None:
            pending_signal = signal

    if active_trade is not None:
        trades.append({
            "direction": active_trade.direction.value,
            "entry_price": active_trade.entry_price,
            "exit_price": float(close_arr[-1]),
            "entry_time": str(active_trade.entry_time),
            "exit_time": str(times[-1]),
            "reason": "OPEN_AT_END",
        })

    return trades


def backtest_funding_isoos(df: pd.DataFrame, funding_df: pd.DataFrame, cfg: SymbolConfig) -> list[dict]:
    """펀딩 전략 백테스트 (next-bar-open 체결)."""
    fcfg = FundingConfig()

    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr_arr = tr.ewm(alpha=1/14, min_periods=14).mean().values

    close_arr = df["close"].values
    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    times = df.index

    f_times = funding_df["timestamp"].values if len(funding_df) > 0 else np.array([])
    f_rates = funding_df["fundingRate"].values if len(funding_df) > 0 else np.array([])

    trades = []
    active_trade = None
    pending_signal = None
    last_direction = 0
    cooldown = 0
    total_bars = len(df)
    warmup = cfg.warmup_bars

    for i in range(warmup, total_bars):
        current_time = times[i]

        if pending_signal is not None and active_trade is not None:
            pending_signal = None

        if pending_signal is not None and active_trade is None:
            fill_price = float(open_arr[i])
            _ps = pending_signal
            pending_signal = None
            atr_i = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else float(atr_arr[i-1])
            risk = atr_i * fcfg.sl_atr_mult

            if _ps.direction == Direction.LONG:
                sl = fill_price - risk
            else:
                sl = fill_price + risk

            # 갭 체크
            if (_ps.direction == Direction.LONG and fill_price <= sl) or \
               (_ps.direction == Direction.SHORT and fill_price >= sl):
                pass  # skip
            else:
                if _ps.direction == Direction.LONG:
                    tp1, tp2, tp3 = fill_price + risk * fcfg.tp1_rr, fill_price + risk * fcfg.tp2_rr, fill_price + risk * fcfg.tp3_rr
                else:
                    tp1, tp2, tp3 = fill_price - risk * fcfg.tp1_rr, fill_price - risk * fcfg.tp2_rr, fill_price - risk * fcfg.tp3_rr
                active_trade = ActiveTrade(
                    direction=_ps.direction, entry_price=fill_price, sl_price=sl,
                    tp1_price=tp1, tp2_price=tp2, tp3_price=tp3,
                    trail_price=sl, risk=risk, entry_time=current_time,
                )

        if active_trade is not None:
            is_long_pos = active_trade.direction == Direction.LONG
            seq = [open_arr[i], low_arr[i], high_arr[i], close_arr[i]] if is_long_pos else [open_arr[i], high_arr[i], low_arr[i], close_arr[i]]
            for cp in seq:
                if active_trade is None: break
                reason = check_exit(active_trade, float(cp), cfg)
                if reason:
                    trades.append({
                        "direction": active_trade.direction.value,
                        "entry_price": active_trade.entry_price,
                        "exit_price": float(cp),
                        "entry_time": str(active_trade.entry_time),
                        "exit_time": str(current_time),
                        "reason": reason,
                    })
                    if reason == "SL": cooldown = cfg.cooldown_bars
                    active_trade = None
                    break

        if cooldown > 0:
            cooldown -= 1
            continue

        # 펀딩비 z-score
        ct_np = np.datetime64(current_time)
        mask = f_times <= ct_np
        n_funding = int(mask.sum())
        if n_funding < 30: continue

        rates_window = f_rates[mask]
        n = min(len(rates_window), fcfg.funding_lookback)
        w = rates_window[-n:]
        fr_mean, fr_std = float(np.mean(w)), float(np.std(w))
        if fr_std < 1e-10: continue
        fr_z = (float(rates_window[-1]) - fr_mean) / fr_std

        z_scores = []
        for period in fcfg.price_z_periods:
            if i < period + 20: continue
            ret = close_arr[i] / close_arr[i - period] - 1
            lookback = min(i, period * 4)
            rets = np.diff(close_arr[i - lookback:i + 1]) / close_arr[i - lookback:i]
            vol = float(np.std(rets)) * np.sqrt(period)
            if vol > 1e-10: z_scores.append(ret / vol)
        if not z_scores: continue
        price_z = float(np.mean(z_scores))

        zt, pcz = fcfg.z_threshold, fcfg.price_confirm_z
        is_long = is_short = False
        strength = 0.0

        if fr_z > zt and price_z > pcz:
            is_short, strength = True, min(fr_z / (zt * 2), 1.0)
        elif fr_z < -zt and price_z < -pcz:
            is_long, strength = True, min(-fr_z / (zt * 2), 1.0)
        elif fr_z > zt * 1.5:
            is_short, strength = True, min(fr_z / (zt * 3), 0.5)
        elif fr_z < -zt * 1.5:
            is_long, strength = True, min(-fr_z / (zt * 3), 0.5)

        if not is_long and not is_short: continue
        if strength < 0.3: continue
        if is_long and last_direction == 1: continue
        if is_short and last_direction == -1: continue

        atr_val = float(atr_arr[i])
        if np.isnan(atr_val) or atr_val <= 0: continue

        entry = float(close_arr[i])
        risk = atr_val * fcfg.sl_atr_mult
        direction = Direction.LONG if is_long else Direction.SHORT
        last_direction = 1 if is_long else -1

        signal = Signal(
            direction=direction, entry_price=entry,
            sl_price=entry - risk if is_long else entry + risk,
            tp1_price=0, tp2_price=0, tp3_price=0,
            risk=risk, rsi=0, adx=0, atr=atr_val,
        )

        if active_trade is not None and signal.direction != active_trade.direction:
            trades.append({
                "direction": active_trade.direction.value,
                "entry_price": active_trade.entry_price,
                "exit_price": entry,
                "entry_time": str(active_trade.entry_time),
                "exit_time": str(current_time),
                "reason": "REVERSE",
            })
            active_trade = None

        if active_trade is None and i + 1 < total_bars:
            pending_signal = signal

    if active_trade is not None:
        trades.append({
            "direction": active_trade.direction.value,
            "entry_price": active_trade.entry_price,
            "exit_price": float(close_arr[-1]),
            "entry_time": str(active_trade.entry_time),
            "exit_time": str(times[-1]),
            "reason": "OPEN_AT_END",
        })

    return trades


# ──────────────────────────────────────────────
# 메트릭 계산
# ──────────────────────────────────────────────


def compute_metrics(trades: list[dict], total_days: int, leverage: int = 3) -> dict:
    if not trades:
        return {"n_trades": 0, "win_rate": 0, "pf": 0, "total_pnl_pct": 0,
                "sharpe": 0, "max_dd_pct": 0, "calmar": 0, "avg_hold_hours": 0}

    pnls = [compute_trade_pnl(t) * leverage for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_profit / gross_loss

    # 에쿼티 커브
    equity = [1.0]
    for p in pnls:
        equity.append(equity[-1] * (1 + p))
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    total_ret = equity[-1] / equity[0] - 1
    years = max(total_days / 365, 0.01)
    ann_ret = (1 + total_ret) ** (1 / years) - 1
    calmar = ann_ret / abs(max_dd) if max_dd < -0.001 else 0

    # 일별 수익률 → Sharpe
    if len(pnls) > 1:
        daily_rets = np.array(pnls)
        sharpe = float(np.mean(daily_rets) / np.std(daily_rets, ddof=1) * np.sqrt(252 / max(years, 0.01))) if np.std(daily_rets) > 0 else 0
    else:
        sharpe = 0

    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(pnls) if pnls else 0,
        "pf": round(pf, 3),
        "total_pnl_pct": round(total_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "avg_pnl_pct": round(np.mean(pnls) * 100, 3) if pnls else 0,
    }


# ──────────────────────────────────────────────
# 메인 검증
# ──────────────────────────────────────────────


SYMBOL_MAP = {
    "BTC": ("BTC/USDT:USDT", BTC_CONFIG, "ema"),
    "SOL": ("SOL/USDT:USDT", SOL_CONFIG, "ema"),
    "XRP": ("XRP/USDT:USDT", XRP_CONFIG, "funding"),
    "AVAX": ("AVAX/USDT:USDT", AVAX_CONFIG, "ema"),
}


def validate_symbol(
    short_sym: str,
    oos_days: int = 365,
    embargo_days: int = 7,
    total_months: int = 60,
) -> dict:
    symbol, cfg, strategy_type = SYMBOL_MAP[short_sym]

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=total_months * 30)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)

    print(f"\n{'='*65}")
    print(f"  {short_sym} Sniper V2 IS/OOS Validation")
    strategy_desc = "FundingContrarian" if strategy_type == "funding" else f"EMA {cfg.ema_fast}/{cfg.ema_slow}/{cfg.ema_trend}"
    sl_desc = f"{'SWING_'+str(cfg.swing_lookback) if cfg.sl_method=='SWING' else 'ATR×'+str(cfg.sl_atr_mult)}"
    print(f"  Strategy: {strategy_desc} | SL: {sl_desc}")
    print(f"{'='*65}")

    # 데이터 수집
    bt_tf = TIMEFRAME
    print(f"\n  Fetching {short_sym} {bt_tf} data ...", flush=True)
    df = fetch_ohlcv_bulk(symbol, bt_tf, since_ms, until_ms)
    print(f"  {bt_tf} bars: {len(df):,}")

    if len(df) < cfg.warmup_bars + 200:
        print(f"  데이터 부족! skip")
        return {"symbol": short_sym, "error": "insufficient data"}

    funding_df = None
    if strategy_type == "funding":
        print(f"  Fetching funding rates ...", flush=True)
        funding_df = fetch_funding_bulk(symbol, since_ms, until_ms)
        print(f"  Funding rates: {len(funding_df):,}")

    # IS/OOS 분할
    oos_cutoff = df.index[-1] - pd.Timedelta(days=oos_days)
    embargo_cutoff = oos_cutoff - pd.Timedelta(days=embargo_days)

    is_df = df[df.index < embargo_cutoff].copy()
    oos_df = df[df.index >= oos_cutoff].copy()

    is_funding, oos_funding = None, None
    if funding_df is not None and len(funding_df) > 0:
        f_ts = pd.to_datetime(funding_df["timestamp"])
        is_funding = funding_df[f_ts < embargo_cutoff].copy()
        oos_funding = funding_df[f_ts >= oos_cutoff].copy()

    print(f"\n  IS:  {is_df.index[0].date()} ~ {is_df.index[-1].date()} ({len(is_df):,} bars, {(is_df.index[-1]-is_df.index[0]).days}일)")
    print(f"  Embargo: {embargo_days}일")
    print(f"  OOS: {oos_df.index[0].date()} ~ {oos_df.index[-1].date()} ({len(oos_df):,} bars, {(oos_df.index[-1]-oos_df.index[0]).days}일)")

    # IS 백테스트
    print(f"\n  Running IS backtest ...", flush=True)
    t0 = time.time()
    if strategy_type == "funding":
        is_trades = backtest_funding_isoos(is_df, is_funding, cfg)
    else:
        is_trades = backtest_ema_isoos(is_df, cfg)
    is_elapsed = time.time() - t0
    print(f"  IS: {len(is_trades)}건 ({is_elapsed:.1f}초)")

    # OOS 백테스트
    print(f"  Running OOS backtest ...", flush=True)
    t0 = time.time()
    if strategy_type == "funding":
        oos_trades = backtest_funding_isoos(oos_df, oos_funding, cfg)
    else:
        oos_trades = backtest_ema_isoos(oos_df, cfg)
    oos_elapsed = time.time() - t0
    print(f"  OOS: {len(oos_trades)}건 ({oos_elapsed:.1f}초)")

    # 메트릭 비교
    is_days = (is_df.index[-1] - is_df.index[0]).days
    oos_days_actual = (oos_df.index[-1] - oos_df.index[0]).days

    is_metrics = compute_metrics(is_trades, is_days, cfg.leverage)
    oos_metrics = compute_metrics(oos_trades, oos_days_actual, cfg.leverage)

    # 오버피팅 진단
    is_pf = is_metrics["pf"]
    oos_pf = oos_metrics["pf"]
    is_pnl = is_metrics["total_pnl_pct"]
    oos_pnl = oos_metrics["total_pnl_pct"]
    is_dd = is_metrics["max_dd_pct"]
    oos_dd = oos_metrics["max_dd_pct"]

    reasons = []
    verdict = "PASS"

    if oos_pf < 1.0:
        reasons.append(f"OOS PF < 1.0: {oos_pf:.2f} (손실 전략)")
        verdict = "OVERFIT"
    if is_pf > 0 and oos_pf > 0 and is_pf / oos_pf > 2.0:
        reasons.append(f"PF 급감: IS={is_pf:.2f} → OOS={oos_pf:.2f}")
        if verdict != "OVERFIT": verdict = "CAUTION"
    if oos_pnl < 0:
        reasons.append(f"OOS 수익 음수: {oos_pnl:.1f}%")
        verdict = "OVERFIT"
    if is_pnl > 0 and oos_pnl > 0 and oos_pnl / is_pnl < 0.2:
        reasons.append(f"수익 급감: IS={is_pnl:.1f}% → OOS={oos_pnl:.1f}%")
        if verdict != "OVERFIT": verdict = "CAUTION"
    if oos_dd < is_dd * 2 and oos_dd < -30:
        reasons.append(f"OOS MDD 과다: {oos_dd:.1f}%")
        if verdict != "OVERFIT": verdict = "CAUTION"
    if oos_metrics["n_trades"] < 5:
        reasons.append(f"OOS 거래 수 부족: {oos_metrics['n_trades']}건")
        if verdict != "OVERFIT": verdict = "CAUTION"
    if not reasons:
        reasons.append("IS/OOS 성과 일관")

    # 출력
    marker = {"PASS": "✓", "CAUTION": "⚠", "OVERFIT": "✗"}
    print(f"\n  {'─'*60}")
    print(f"  {short_sym} IS vs OOS 비교")
    print(f"  {'─'*60}")
    print(f"  {'지표':<20} {'IS':>12} {'OOS':>12}")
    print(f"  {'─'*44}")
    print(f"  {'거래 수':<20} {is_metrics['n_trades']:>12} {oos_metrics['n_trades']:>12}")
    print(f"  {'승률':<20} {is_metrics['win_rate']:>11.1%} {oos_metrics['win_rate']:>11.1%}")
    print(f"  {'PF':<20} {is_metrics['pf']:>12.2f} {oos_metrics['pf']:>12.2f}")
    print(f"  {'총 수익':<20} {is_metrics['total_pnl_pct']:>11.1f}% {oos_metrics['total_pnl_pct']:>11.1f}%")
    print(f"  {'Sharpe':<20} {is_metrics['sharpe']:>12.2f} {oos_metrics['sharpe']:>12.2f}")
    print(f"  {'MDD':<20} {is_metrics['max_dd_pct']:>11.1f}% {oos_metrics['max_dd_pct']:>11.1f}%")
    print(f"  {'Calmar':<20} {is_metrics['calmar']:>12.2f} {oos_metrics['calmar']:>12.2f}")
    print(f"\n  판정: {marker.get(verdict, '?')} {verdict}")
    for r in reasons:
        print(f"    — {r}")

    result = {
        "symbol": short_sym,
        "strategy": strategy_desc,
        "sl": sl_desc,
        "params": {
            "ema": f"{cfg.ema_fast}/{cfg.ema_slow}/{cfg.ema_trend}" if strategy_type == "ema" else "N/A",
            "rsi": f"{cfg.rsi_bull}/{cfg.rsi_bear}",
            "tp_rr": f"{cfg.tp1_rr}/{cfg.tp2_rr}/{cfg.tp3_rr}",
            "adx_min": cfg.adx_min,
            "profit_protect": cfg.profit_protect,
        },
        "is_period": f"{is_df.index[0].date()} ~ {is_df.index[-1].date()}",
        "oos_period": f"{oos_df.index[0].date()} ~ {oos_df.index[-1].date()}",
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "verdict": verdict,
        "reasons": reasons,
    }

    # 저장
    out_dir = RESULTS_DIR / short_sym
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "validation.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser(description="Sniper V2 IS/OOS Validation")
    parser.add_argument("--symbol", type=str, help="심볼 (BTC, SOL, XRP, AVAX)")
    parser.add_argument("--oos-days", type=int, default=365, help="OOS 기간 (기본 365일)")
    parser.add_argument("--months", type=int, default=60, help="총 데이터 기간 (기본 60개월)")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else list(SYMBOL_MAP.keys())
    all_results = []

    for sym in symbols:
        if sym not in SYMBOL_MAP:
            print(f"⚠ {sym}: 미지원 심볼")
            continue
        result = validate_symbol(sym, oos_days=args.oos_days, total_months=args.months)
        all_results.append(result)

    # 종합 요약
    if len(all_results) > 1:
        print(f"\n\n{'='*65}")
        print(f"  종합 요약")
        print(f"{'='*65}")
        print(f"  {'심볼':<8} {'전략':<25} {'IS PF':>8} {'OOS PF':>8} {'IS PnL':>8} {'OOS PnL':>9} {'판정':>8}")
        print(f"  {'─'*70}")
        for r in all_results:
            if "error" in r:
                print(f"  {r['symbol']:<8} {'ERROR':<25}")
                continue
            m = {"PASS": "✓", "CAUTION": "⚠", "OVERFIT": "✗"}
            print(f"  {r['symbol']:<8} {r['strategy']:<25} "
                  f"{r['is_metrics']['pf']:>8.2f} {r['oos_metrics']['pf']:>8.2f} "
                  f"{r['is_metrics']['total_pnl_pct']:>7.1f}% {r['oos_metrics']['total_pnl_pct']:>8.1f}% "
                  f"{m.get(r['verdict'],'?'):>3} {r['verdict']}")

    # 종합 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  결과 저장: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
