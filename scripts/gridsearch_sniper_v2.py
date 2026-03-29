#!/usr/bin/env python3
"""
Sniper V2 — SL 중심 파라미터 그리드서치.

SL 손실이 전략 손실의 주요 원인 → 더 넓은 SL + PP/TP 조합 최적화.
지표 1회 사전계산 후 파라미터만 변경하며 반복.

Usage:
    python scripts/gridsearch_sniper_v2.py                    # 전체 (BTC+SOL+XRP)
    python scripts/gridsearch_sniper_v2.py --symbol BTC       # BTC만
    python scripts/gridsearch_sniper_v2.py --months 36        # 36개월
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sniper_v2.config import (
    BTC_CONFIG, SOL_CONFIG, XRP_CONFIG, AVAX_CONFIG, CONFIGS,
    FUNDING_STRATEGY_SYMBOLS, SymbolConfig,
)
from src.sniper_v2.strategy import Direction, Signal, ActiveTrade
from src.sniper_v2.indicators import compute_all
from src.sniper_v2.funding_strategy import FundingConfig

# backtest_sniper_v2.py에서 데이터 fetch + 포지션 관리 재사용
from scripts.backtest_sniper_v2 import (
    fetch_ohlcv_bulk, fetch_funding_bulk,
    check_exit, record_trade,
)


# ══════════════════════════════════════════════════════════════
# EMA 그리드서치 (BTC, SOL)
# ══════════════════════════════════════════════════════════════

def run_ema_single(df_with_indicators: dict, cfg: SymbolConfig, total_bars: int, times) -> dict:
    """
    사전계산된 지표 배열로 EMA 백테스트 1회 실행.
    Returns: {n, wins, pf, total_pnl, max_dd, calmar, avg_hold_min}
    """
    d = df_with_indicators  # 사전계산된 numpy 배열 dict
    warmup = cfg.warmup_bars

    trades = []
    active_trade: ActiveTrade | None = None
    last_direction = 0
    cooldown = 0

    for i in range(warmup, total_bars):
        current_time = times[i]

        # 봉 내 TP/SL/PP 체크 (SL 우선 비관적 순서)
        if active_trade is not None:
            is_long_pos = active_trade.direction == Direction.LONG
            if is_long_pos:
                check_seq = [d['open'][i], d['low'][i], d['high'][i], d['close'][i]]
            else:
                check_seq = [d['open'][i], d['high'][i], d['low'][i], d['close'][i]]
            for check_price in check_seq:
                if active_trade is None:
                    break
                exit_reason = check_exit(active_trade, float(check_price), cfg)
                if exit_reason:
                    record_trade(trades, active_trade, current_time, float(check_price), exit_reason, cfg)
                    if exit_reason == "SL":
                        cooldown = cfg.cooldown_bars
                    active_trade = None
                    break

        if cooldown > 0:
            cooldown -= 1
            continue

        # NaN
        if (np.isnan(d['ema_fast'][i]) or np.isnan(d['ema_slow'][i]) or
            np.isnan(d['ema_trend'][i]) or np.isnan(d['rsi'][i]) or
            np.isnan(d['atr'][i]) or np.isnan(d['adx'][i])):
            continue

        # Vol filter
        atr_val = float(d['atr'][i])
        atr_sma_val = float(d['atr_sma'][i]) if not np.isnan(d['atr_sma'][i]) else 0
        if atr_sma_val > 0 and atr_val / atr_sma_val > cfg.vol_filter:
            continue

        # ADX
        adx_val = float(d['adx'][i])
        if adx_val < cfg.adx_min:
            continue

        # Entry
        cross_up = bool(d['bull_cross'][i])
        cross_down = bool(d['bear_cross'][i])
        close = float(d['close'][i])
        ema_trend_val = float(d['ema_trend'][i])
        rsi_val = float(d['rsi'][i])

        long_sig = cross_up and close > ema_trend_val and rsi_val > cfg.rsi_bull
        short_sig = cross_down and close < ema_trend_val and rsi_val < cfg.rsi_bear

        if long_sig and last_direction == 1:
            long_sig = False
        if short_sig and last_direction == -1:
            short_sig = False
        if long_sig and short_sig:
            short_sig = False
        if not long_sig and not short_sig:
            continue

        # SL/TP
        entry = close
        is_long = long_sig

        if cfg.sl_method == "SWING" and d.get('swing_low') is not None:
            if is_long:
                sw = float(d['swing_low'][i]) if not np.isnan(d['swing_low'][i]) else entry - atr_val * 1.5
                sl = sw
            else:
                sw = float(d['swing_high'][i]) if not np.isnan(d['swing_high'][i]) else entry + atr_val * 1.5
                sl = sw
        else:
            sl = entry - atr_val * cfg.sl_atr_mult if is_long else entry + atr_val * cfg.sl_atr_mult

        risk = abs(entry - sl)
        if risk < atr_val * 0.3:
            risk = atr_val * 0.5
            sl = entry - risk if is_long else entry + risk

        direction = Direction.LONG if is_long else Direction.SHORT
        if is_long:
            last_direction = 1
            tp1, tp2, tp3 = entry + risk * cfg.tp1_rr, entry + risk * cfg.tp2_rr, entry + risk * cfg.tp3_rr
        else:
            last_direction = -1
            tp1, tp2, tp3 = entry - risk * cfg.tp1_rr, entry - risk * cfg.tp2_rr, entry - risk * cfg.tp3_rr

        signal = Signal(direction=direction, entry_price=entry, sl_price=sl,
                        tp1_price=tp1, tp2_price=tp2, tp3_price=tp3,
                        risk=risk, rsi=rsi_val, adx=adx_val, atr=atr_val,
                        timestamp=current_time)

        if active_trade is not None and signal.direction != active_trade.direction:
            record_trade(trades, active_trade, current_time, close, "REVERSE", cfg)
            active_trade = None

        if active_trade is None:
            active_trade = ActiveTrade(
                direction=signal.direction, entry_price=signal.entry_price,
                sl_price=signal.sl_price, tp1_price=tp1, tp2_price=tp2, tp3_price=tp3,
                trail_price=signal.sl_price, risk=signal.risk, entry_time=signal.timestamp,
            )

    if active_trade is not None:
        record_trade(trades, active_trade, times[-1], float(d['close'][-1]), "OPEN_AT_END", cfg)

    return _summarize(trades, cfg)


def precompute_ema_indicators(df: pd.DataFrame, cfg: SymbolConfig) -> dict:
    """EMA 지표를 한 번 계산하고 numpy 배열로 반환."""
    df = compute_all(df.copy(), cfg)
    result = {
        'close': df['close'].values,
        'high': df['high'].values,
        'low': df['low'].values,
        'open': df['open'].values,
        'ema_fast': df['ema_fast'].values,
        'ema_slow': df['ema_slow'].values,
        'ema_trend': df['ema_trend'].values,
        'rsi': df['rsi'].values,
        'atr': df['atr'].values,
        'atr_sma': df['atr_sma'].values,
        'adx': df['adx'].values,
        'bull_cross': df['ema_bull_cross'].values,
        'bear_cross': df['ema_bear_cross'].values,
    }
    if 'swing_low' in df.columns:
        result['swing_low'] = df['swing_low'].values
        result['swing_high'] = df['swing_high'].values
    return result


def precompute_ema_variants(df: pd.DataFrame, swing_lookbacks: list[int]) -> dict[int, dict]:
    """SWING lookback 별로 swing_low/high만 다르게 계산."""
    # 기본 지표 (EMA/RSI/ATR/ADX 등)는 동일
    base_cfg = BTC_CONFIG  # 기본 EMA 파라미터로 계산
    df_computed = compute_all(df.copy(), base_cfg)

    base = {
        'close': df_computed['close'].values,
        'high': df_computed['high'].values,
        'low': df_computed['low'].values,
        'open': df_computed['open'].values,
        'ema_fast': df_computed['ema_fast'].values,
        'ema_slow': df_computed['ema_slow'].values,
        'ema_trend': df_computed['ema_trend'].values,
        'rsi': df_computed['rsi'].values,
        'atr': df_computed['atr'].values,
        'atr_sma': df_computed['atr_sma'].values,
        'adx': df_computed['adx'].values,
        'bull_cross': df_computed['ema_bull_cross'].values,
        'bear_cross': df_computed['ema_bear_cross'].values,
    }

    variants = {}
    for lb in swing_lookbacks:
        d = dict(base)
        d['swing_low'] = df['low'].rolling(lb, min_periods=lb).min().values
        d['swing_high'] = df['high'].rolling(lb, min_periods=lb).max().values
        variants[lb] = d

    # ATR 전용 (swing 없음)
    d_atr = dict(base)
    d_atr['swing_low'] = None
    d_atr['swing_high'] = None
    variants['ATR'] = d_atr

    return variants


# ══════════════════════════════════════════════════════════════
# Funding 그리드서치 (XRP)
# ══════════════════════════════════════════════════════════════

def run_funding_single(
    close_arr, high_arr, low_arr, open_arr,
    atr_arr, rsi_arr, times,
    f_times_i64, f_rates,
    cfg: SymbolConfig, fcfg: FundingConfig,
    total_bars: int,
    times_i64=None,
) -> dict:
    """펀딩 전략 백테스트 1회 (사전계산 배열 + searchsorted 최적화)."""
    warmup = cfg.warmup_bars
    trades = []
    active_trade: ActiveTrade | None = None
    last_direction = 0
    cooldown = 0

    for i in range(warmup, total_bars):
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

        signal = Signal(direction=direction, entry_price=entry, sl_price=sl,
                        tp1_price=tp1, tp2_price=tp2, tp3_price=tp3,
                        risk=risk, rsi=rsi_val, adx=0.0, atr=atr_val,
                        timestamp=current_time)

        if active_trade is not None and signal.direction != active_trade.direction:
            record_trade(trades, active_trade, current_time, entry, "REVERSE", cfg)
            active_trade = None

        if active_trade is None:
            active_trade = ActiveTrade(
                direction=signal.direction, entry_price=signal.entry_price,
                sl_price=signal.sl_price, tp1_price=tp1, tp2_price=tp2, tp3_price=tp3,
                trail_price=signal.sl_price, risk=signal.risk, entry_time=signal.timestamp,
            )

    if active_trade is not None:
        record_trade(trades, active_trade, times[-1], float(close_arr[-1]), "OPEN_AT_END", cfg)

    return _summarize(trades, cfg)


# ══════════════════════════════════════════════════════════════
# 결과 요약
# ══════════════════════════════════════════════════════════════

def _summarize(trades: list, cfg: SymbolConfig) -> dict:
    if not trades:
        return {"n": 0, "wins": 0, "wr": 0, "pf": 0, "total_pnl": 0,
                "max_dd": 0, "calmar": 0, "final_bal": 1000, "avg_hold": 0,
                "sl_count": 0, "sl_total": 0}

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    n_wins = len(wins)

    total_pnl = tdf["pnl_pct"].sum()
    gp = wins["pnl_pct"].sum() if n_wins else 0
    gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.01
    pf = gp / gl

    cum = tdf["pnl_pct"].cumsum()
    max_dd = float((cum - cum.cummax()).min())

    # 복리
    balance = 1000.0
    peak = 1000.0
    max_dd_bal = 0.0
    for p in tdf["pnl_pct"]:
        balance *= (1 + p / 100 * cfg.balance_ratio)
        peak = max(peak, balance)
        dd = (balance - peak) / peak * 100
        max_dd_bal = min(max_dd_bal, dd)

    calmar = 0
    if max_dd_bal < -0.01 and balance > 0:
        calmar = ((balance / 1000) - 1) / abs(max_dd_bal) * 100

    sl_trades = tdf[tdf["reason"] == "SL"]
    sl_count = len(sl_trades)
    sl_total = sl_trades["pnl_pct"].sum() if sl_count else 0

    return {
        "n": n, "wins": n_wins, "wr": round(n_wins / n * 100, 1),
        "pf": round(pf, 3), "total_pnl": round(total_pnl, 2),
        "max_dd": round(max_dd, 2), "max_dd_bal": round(max_dd_bal, 2),
        "calmar": round(calmar, 3),
        "final_bal": round(balance, 2),
        "avg_hold": round(tdf["hold_min"].mean(), 1),
        "sl_count": sl_count, "sl_total": round(sl_total, 2),
    }


# ══════════════════════════════════════════════════════════════
# 메인 그리드서치
# ══════════════════════════════════════════════════════════════

def gridsearch_btc(df: pd.DataFrame):
    """BTC 그리드서치: SL 방법 + PP 파라미터."""
    print(f"\n{'='*70}", flush=True)
    print(f"  BTC 그리드서치 (SL + PP + TP)", flush=True)
    print(f"{'='*70}", flush=True)

    # SWING lookback 별 사전계산
    swing_lookbacks = [5, 10, 15, 20]
    print(f"  지표 사전계산 중 (SWING {swing_lookbacks} + ATR) ...", flush=True)
    variants = precompute_ema_variants(df, swing_lookbacks)
    total_bars = len(df)
    times = df.index

    # 그리드 정의
    grid = []

    # SWING SL variants
    for lb in swing_lookbacks:
        for pp_trigger, pp_exit in [(0.5, 0.2), (1.0, 0.3), (1.5, 0.3), (1.5, 0.5), (2.0, 0.5)]:
            for tp1, tp2, tp3 in [(1.0, 2.0, 3.0), (1.5, 2.5, 4.0), (1.0, 2.5, 4.0)]:
                grid.append(("SWING", lb, 1.5, pp_trigger, pp_exit, tp1, tp2, tp3))

    # ATR SL variants
    for atr_mult in [1.5, 2.0, 2.5, 3.0, 3.5]:
        for pp_trigger, pp_exit in [(0.5, 0.2), (1.0, 0.3), (1.5, 0.3), (1.5, 0.5), (2.0, 0.5)]:
            for tp1, tp2, tp3 in [(1.0, 2.0, 3.0), (1.5, 2.5, 4.0), (1.0, 2.5, 4.0)]:
                grid.append(("ATR", 5, atr_mult, pp_trigger, pp_exit, tp1, tp2, tp3))

    # PP OFF
    for lb in swing_lookbacks:
        for tp1, tp2, tp3 in [(1.0, 2.0, 3.0), (1.5, 2.5, 4.0)]:
            grid.append(("SWING", lb, 1.5, 0, 0, tp1, tp2, tp3))
    for atr_mult in [2.0, 2.5, 3.0, 3.5]:
        for tp1, tp2, tp3 in [(1.0, 2.0, 3.0), (1.5, 2.5, 4.0)]:
            grid.append(("ATR", 5, atr_mult, 0, 0, tp1, tp2, tp3))

    print(f"  총 {len(grid)} 조합 테스트", flush=True)

    results = []
    t0 = time.time()
    for idx, (sl_method, sl_lb, sl_atr, pp_trig, pp_exit, tp1, tp2, tp3) in enumerate(grid):
        cfg = replace(
            BTC_CONFIG,
            sl_method=sl_method,
            swing_lookback=sl_lb,
            sl_atr_mult=sl_atr,
            profit_protect=pp_trig > 0,
            pp_trigger=pp_trig,
            pp_exit=pp_exit,
            tp1_rr=tp1,
            tp2_rr=tp2,
            tp3_rr=tp3,
        )

        key = sl_lb if sl_method == "SWING" else "ATR"
        d = variants[key]
        stats = run_ema_single(d, cfg, total_bars, times)
        stats["params"] = f"{sl_method}{'_'+str(sl_lb) if sl_method=='SWING' else '×'+str(sl_atr)} PP={pp_trig}/{pp_exit} TP={tp1}/{tp2}/{tp3}"
        stats["sl_method"] = sl_method
        stats["sl_lb"] = sl_lb
        stats["sl_atr"] = sl_atr
        stats["pp_trigger"] = pp_trig
        stats["pp_exit"] = pp_exit
        stats["tp_rr"] = f"{tp1}/{tp2}/{tp3}"
        results.append(stats)

        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{len(grid)} ...", flush=True)

    elapsed = time.time() - t0
    print(f"  완료! {elapsed:.1f}초", flush=True)

    # PF 기준 정렬
    results.sort(key=lambda x: x["pf"], reverse=True)
    _print_top(results, "BTC", "PF")

    # Calmar 기준도
    results_calmar = sorted(results, key=lambda x: x["calmar"], reverse=True)
    _print_top(results_calmar, "BTC", "Calmar")

    return results


def gridsearch_sol(df: pd.DataFrame):
    """SOL 그리드서치."""
    print(f"\n{'='*70}", flush=True)
    print(f"  SOL 그리드서치 (SL + ADX + Cooldown + TP)", flush=True)
    print(f"{'='*70}", flush=True)

    # SOL은 EMA 8/40/150 고정, SWING lookback + 필터 변경
    swing_lookbacks = [3, 5, 7, 10, 15]

    # 기본 지표 계산 (SOL EMA 파라미터 사용)
    df_computed = compute_all(df.copy(), SOL_CONFIG)
    base = {
        'close': df_computed['close'].values,
        'high': df_computed['high'].values,
        'low': df_computed['low'].values,
        'open': df_computed['open'].values,
        'ema_fast': df_computed['ema_fast'].values,
        'ema_slow': df_computed['ema_slow'].values,
        'ema_trend': df_computed['ema_trend'].values,
        'rsi': df_computed['rsi'].values,
        'atr': df_computed['atr'].values,
        'atr_sma': df_computed['atr_sma'].values,
        'adx': df_computed['adx'].values,
        'bull_cross': df_computed['ema_bull_cross'].values,
        'bear_cross': df_computed['ema_bear_cross'].values,
    }

    # SWING variants + ATR
    variants = {}
    for lb in swing_lookbacks:
        d = dict(base)
        d['swing_low'] = df['low'].rolling(lb, min_periods=lb).min().values
        d['swing_high'] = df['high'].rolling(lb, min_periods=lb).max().values
        variants[lb] = d
    d_atr = dict(base)
    d_atr['swing_low'] = None
    d_atr['swing_high'] = None
    variants['ATR'] = d_atr

    total_bars = len(df)
    times = df.index

    grid = []
    # SWING variants
    for lb in swing_lookbacks:
        for adx_min in [15, 20, 25, 30]:
            for cooldown in [5, 10, 20]:
                for rsi_b, rsi_br in [(60, 35), (65, 35), (55, 40)]:
                    for tp1, tp2, tp3 in [(1.25, 3.38, 5.5), (1.5, 3.0, 5.0), (1.0, 2.5, 4.0)]:
                        grid.append(("SWING", lb, 1.5, adx_min, cooldown, rsi_b, rsi_br, tp1, tp2, tp3))

    # ATR variants
    for atr_mult in [2.0, 2.5, 3.0, 3.5]:
        for adx_min in [15, 20, 25]:
            for cooldown in [5, 10, 20]:
                for tp1, tp2, tp3 in [(1.25, 3.38, 5.5), (1.5, 3.0, 5.0), (1.0, 2.5, 4.0)]:
                    grid.append(("ATR", 3, atr_mult, adx_min, cooldown, 65, 35, tp1, tp2, tp3))

    print(f"  총 {len(grid)} 조합 테스트", flush=True)

    results = []
    t0 = time.time()
    for idx, (sl_method, sl_lb, sl_atr, adx_min, cooldown, rsi_b, rsi_br, tp1, tp2, tp3) in enumerate(grid):
        cfg = replace(
            SOL_CONFIG,
            sl_method=sl_method,
            swing_lookback=sl_lb,
            sl_atr_mult=sl_atr,
            adx_min=adx_min,
            cooldown_bars=cooldown,
            rsi_bull=rsi_b,
            rsi_bear=rsi_br,
            tp1_rr=tp1,
            tp2_rr=tp2,
            tp3_rr=tp3,
        )

        key = sl_lb if sl_method == "SWING" else "ATR"
        d = variants[key]
        stats = run_ema_single(d, cfg, total_bars, times)
        stats["params"] = f"{sl_method}{'_'+str(sl_lb) if sl_method=='SWING' else '×'+str(sl_atr)} ADX>={adx_min} CD={cooldown} RSI={rsi_b}/{rsi_br} TP={tp1}/{tp2}/{tp3}"
        results.append(stats)

        if (idx + 1) % 100 == 0:
            print(f"    {idx+1}/{len(grid)} ...", flush=True)

    elapsed = time.time() - t0
    print(f"  완료! {elapsed:.1f}초", flush=True)

    results.sort(key=lambda x: x["pf"], reverse=True)
    _print_top(results, "SOL", "PF")

    results_calmar = sorted(results, key=lambda x: x["calmar"], reverse=True)
    _print_top(results_calmar, "SOL", "Calmar")

    return results


def gridsearch_xrp(df: pd.DataFrame, funding_df: pd.DataFrame):
    """XRP 그리드서치: Funding 파라미터 + SL."""
    print(f"\n{'='*70}", flush=True)
    print(f"  XRP 그리드서치 (Funding + SL + PP)", flush=True)
    print(f"{'='*70}", flush=True)

    # 사전계산
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

    # searchsorted 최적화: int64로 변환
    f_times_i64 = f_times.astype("int64")
    times_i64 = df.index.values.astype("int64")

    total_bars = len(df)

    grid = []
    # 최소 그리드 (5년 60조합 × 7초 ≈ 7분)
    for sl_atr in [2.0, 3.0, 4.0]:
        for z_thresh in [0.75, 1.125, 2.0]:
            for price_z in [0.3, 0.8]:
                for f_lb in [75]:
                    for pp_trig, pp_exit in [(0, 0), (1.5, 0.5)]:
                        for tp1, tp2, tp3 in [(1.5, 3.0, 5.0), (2.0, 3.5, 6.0)]:
                            grid.append((sl_atr, z_thresh, price_z, f_lb, pp_trig, pp_exit, tp1, tp2, tp3))

    print(f"  총 {len(grid)} 조합 테스트", flush=True)

    results = []
    t0 = time.time()
    for idx, (sl_atr, z_thresh, price_z, f_lb, pp_trig, pp_exit, tp1, tp2, tp3) in enumerate(grid):
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

        stats = run_funding_single(
            close_arr, high_arr, low_arr, open_arr,
            atr_arr, rsi_arr, times,
            f_times_i64, f_rates,
            cfg, fcfg, total_bars,
            times_i64=times_i64,
        )
        stats["params"] = f"SL_ATR×{sl_atr} Z={z_thresh} PZ={price_z} FL={f_lb} PP={pp_trig}/{pp_exit} TP={tp1}/{tp2}/{tp3}"
        results.append(stats)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (idx + 1)
            remaining = rate * (len(grid) - idx - 1)
            print(f"    {idx+1}/{len(grid)} ({elapsed:.0f}s / 남은: {remaining:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  완료! {elapsed:.1f}초", flush=True)

    results.sort(key=lambda x: x["pf"], reverse=True)
    _print_top(results, "XRP", "PF", top_n=15)

    results_calmar = sorted(results, key=lambda x: x["calmar"], reverse=True)
    _print_top(results_calmar, "XRP", "Calmar", top_n=15)

    return results


def gridsearch_avax(df: pd.DataFrame):
    """
    AVAX 그리드서치.

    AVAX 특성:
      - ETH 대비 레버리지 플레이 → 높은 변동성, 추세 시 빠른 움직임
      - 서브넷 뉴스 반응 → 이벤트 후 추세 지속
      - SOL과 유사한 고베타 알트코인

    탐색 범위:
      1. EMA 속도: fast 8-15 / slow 30-50 / trend 100-200 (3조합)
      2. SL: SWING 3-15 + ATR 2.0-3.5
      3. ADX 필터: 0(OFF)-30
      4. 쿨다운: 0-20
      5. RSI: bull 55-65, bear 30-40
      6. TP: 보수적~공격적 3조합
      7. PP: ON/OFF
    """
    print(f"\n{'='*70}", flush=True)
    print(f"  AVAX 그리드서치 (EMA + SL + ADX + PP + TP)", flush=True)
    print(f"{'='*70}", flush=True)

    # EMA 파라미터 조합 (3가지)
    ema_combos = [
        (8, 40, 150),    # SOL형 (빠른 크로스)
        (10, 40, 150),   # 약간 느린 크로스
        (12, 50, 200),   # BTC형 (느린 크로스)
    ]

    swing_lookbacks = [3, 5, 7, 10, 15]

    # EMA 조합별 지표 사전계산
    ema_variants = {}
    for ema_f, ema_s, ema_t in ema_combos:
        label = f"EMA_{ema_f}_{ema_s}_{ema_t}"
        print(f"  지표 사전계산: {label} ...", flush=True)
        cfg_tmp = replace(AVAX_CONFIG, ema_fast=ema_f, ema_slow=ema_s, ema_trend=ema_t)
        df_computed = compute_all(df.copy(), cfg_tmp)
        base = {
            'close': df_computed['close'].values,
            'high': df_computed['high'].values,
            'low': df_computed['low'].values,
            'open': df_computed['open'].values,
            'ema_fast': df_computed['ema_fast'].values,
            'ema_slow': df_computed['ema_slow'].values,
            'ema_trend': df_computed['ema_trend'].values,
            'rsi': df_computed['rsi'].values,
            'atr': df_computed['atr'].values,
            'atr_sma': df_computed['atr_sma'].values,
            'adx': df_computed['adx'].values,
            'bull_cross': df_computed['ema_bull_cross'].values,
            'bear_cross': df_computed['ema_bear_cross'].values,
        }

        # SWING variants
        for lb in swing_lookbacks:
            d = dict(base)
            d['swing_low'] = df['low'].rolling(lb, min_periods=lb).min().values
            d['swing_high'] = df['high'].rolling(lb, min_periods=lb).max().values
            ema_variants[(label, lb)] = d

        # ATR variant
        d_atr = dict(base)
        d_atr['swing_low'] = None
        d_atr['swing_high'] = None
        ema_variants[(label, 'ATR')] = d_atr

    total_bars = len(df)
    times = df.index

    # 그리드 정의
    grid = []

    for ema_f, ema_s, ema_t in ema_combos:
        label = f"EMA_{ema_f}_{ema_s}_{ema_t}"

        # SWING SL variants
        for lb in swing_lookbacks:
            for adx_min in [0, 15, 20, 25, 30]:
                for cooldown in [0, 5, 10, 20]:
                    for rsi_b, rsi_br in [(55, 40), (60, 35), (65, 35)]:
                        for tp1, tp2, tp3 in [(1.25, 3.0, 5.0), (1.5, 2.75, 4.0), (2.0, 3.5, 6.0)]:
                            # PP OFF
                            grid.append((label, ema_f, ema_s, ema_t,
                                         "SWING", lb, 1.5, adx_min, cooldown,
                                         rsi_b, rsi_br, tp1, tp2, tp3, 0, 0))
                            # PP ON (가장 유망한 필터 조합만)
                            if adx_min in [20, 25] and cooldown == 10:
                                grid.append((label, ema_f, ema_s, ema_t,
                                             "SWING", lb, 1.5, adx_min, cooldown,
                                             rsi_b, rsi_br, tp1, tp2, tp3, 1.5, 0.3))

        # ATR SL variants (핵심 조합만)
        for atr_mult in [2.0, 2.5, 3.0, 3.5]:
            for adx_min in [0, 20, 25]:
                for cooldown in [5, 10]:
                    for tp1, tp2, tp3 in [(1.5, 2.75, 4.0), (2.0, 3.5, 6.0)]:
                        grid.append((label, ema_f, ema_s, ema_t,
                                     "ATR", 5, atr_mult, adx_min, cooldown,
                                     60, 35, tp1, tp2, tp3, 0, 0))

    print(f"  총 {len(grid)} 조합 테스트", flush=True)

    results = []
    t0 = time.time()
    for idx, (label, ema_f, ema_s, ema_t,
              sl_method, sl_lb, sl_atr, adx_min, cooldown,
              rsi_b, rsi_br, tp1, tp2, tp3, pp_trig, pp_exit) in enumerate(grid):

        cfg = replace(
            AVAX_CONFIG,
            ema_fast=ema_f, ema_slow=ema_s, ema_trend=ema_t,
            sl_method=sl_method,
            swing_lookback=sl_lb,
            sl_atr_mult=sl_atr,
            adx_min=adx_min,
            cooldown_bars=cooldown,
            rsi_bull=rsi_b,
            rsi_bear=rsi_br,
            tp1_rr=tp1, tp2_rr=tp2, tp3_rr=tp3,
            profit_protect=pp_trig > 0,
            pp_trigger=pp_trig,
            pp_exit=pp_exit,
        )

        key = (label, sl_lb if sl_method == "SWING" else "ATR")
        d = ema_variants[key]
        stats = run_ema_single(d, cfg, total_bars, times)

        sl_desc = f"SWING_{sl_lb}" if sl_method == "SWING" else f"ATR×{sl_atr}"
        pp_desc = f"PP={pp_trig}/{pp_exit}" if pp_trig > 0 else "PP=OFF"
        stats["params"] = (
            f"{label} {sl_desc} ADX>={adx_min} CD={cooldown} "
            f"RSI={rsi_b}/{rsi_br} TP={tp1}/{tp2}/{tp3} {pp_desc}"
        )
        results.append(stats)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (idx + 1)
            remaining = rate * (len(grid) - idx - 1)
            print(f"    {idx+1}/{len(grid)} ({elapsed:.0f}s / 남은: {remaining:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  완료! {elapsed:.1f}초", flush=True)

    # PF 기준
    results.sort(key=lambda x: x["pf"], reverse=True)
    _print_top(results, "AVAX", "PF", top_n=15)

    # Calmar 기준
    results_calmar = sorted(results, key=lambda x: x["calmar"], reverse=True)
    _print_top(results_calmar, "AVAX", "Calmar", top_n=15)

    # 잔고 기준
    results_bal = sorted(results, key=lambda x: x["final_bal"], reverse=True)
    _print_top(results_bal, "AVAX", "Final Balance", top_n=15)

    return results


def _print_top(results: list, symbol: str, metric: str, top_n: int = 10):
    """상위 N개 결과 출력."""
    valid = [r for r in results if r["n"] >= 10]  # 최소 10건 이상
    if not valid:
        print(f"\n  {symbol}: 유효 결과 없음 (10건 미만)")
        return

    print(f"\n  --- {symbol} Top {top_n} by {metric} (거래 >= 10건) ---", flush=True)
    print(f"  {'#':>3} {'PF':>6} {'WR':>6} {'PnL':>9} {'MDD':>8} {'잔고':>10} {'Calmar':>7} {'거래':>5} {'SL':>4} {'SL손실':>9} {'파라미터'}", flush=True)
    print(f"  {'-'*100}", flush=True)

    for rank, r in enumerate(valid[:top_n], 1):
        print(
            f"  {rank:3d} {r['pf']:6.2f} {r['wr']:5.1f}% {r['total_pnl']:+8.2f}% {r.get('max_dd_bal', r['max_dd']):+7.2f}% "
            f"${r['final_bal']:>9,.2f} {r['calmar']:+6.2f} {r['n']:5d} {r['sl_count']:4d} {r['sl_total']:+8.2f}% "
            f"{r['params']}",
            flush=True,
        )
    print(flush=True)

    # 현재 vs 최적
    if valid:
        best = valid[0]
        print(f"  >> {symbol} 최적: {best['params']}", flush=True)
        print(f"     PF={best['pf']:.2f} WR={best['wr']:.1f}% PnL={best['total_pnl']:+.2f}% "
              f"MDD={best.get('max_dd_bal', best['max_dd']):+.2f}% 잔고=${best['final_bal']:,.2f} "
              f"Calmar={best['calmar']:+.2f}",
              flush=True)


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sniper V2 Grid Search")
    parser.add_argument("--symbol", default="ALL", help="BTC, SOL, XRP, ALL")
    parser.add_argument("--months", type=int, default=60)
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.months * 30)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)

    targets = args.symbol.upper().split(",") if args.symbol != "ALL" else ["BTC", "SOL", "XRP", "AVAX"]

    if "BTC" in targets:
        print(f"\n  Fetching BTC 15m data ...", flush=True)
        btc_df = fetch_ohlcv_bulk("BTC/USDT:USDT", "15m", since_ms, until_ms)
        print(f"  {len(btc_df)} bars", flush=True)
        gridsearch_btc(btc_df)

    if "SOL" in targets:
        print(f"\n  Fetching SOL 15m data ...", flush=True)
        sol_df = fetch_ohlcv_bulk("SOL/USDT:USDT", "15m", since_ms, until_ms)
        print(f"  {len(sol_df)} bars", flush=True)
        gridsearch_sol(sol_df)

    if "XRP" in targets:
        print(f"\n  Fetching XRP 15m data ...", flush=True)
        xrp_df = fetch_ohlcv_bulk("XRP/USDT:USDT", "15m", since_ms, until_ms)
        print(f"  {len(xrp_df)} bars", flush=True)
        print(f"\n  Fetching XRP funding rates ...", flush=True)
        xrp_funding = fetch_funding_bulk("XRP/USDT:USDT", since_ms, until_ms)
        print(f"  {len(xrp_funding)} rates", flush=True)
        gridsearch_xrp(xrp_df, xrp_funding)

    if "AVAX" in targets:
        print(f"\n  Fetching AVAX 15m data ...", flush=True)
        avax_df = fetch_ohlcv_bulk("AVAX/USDT:USDT", "15m", since_ms, until_ms)
        print(f"  {len(avax_df)} bars", flush=True)
        gridsearch_avax(avax_df)
