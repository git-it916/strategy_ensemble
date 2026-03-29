#!/usr/bin/env python3
"""
Sniper V2 Backtest — BTC, SOL, XRP 전략별 장기 백테스트.

최적화: 지표를 전체 DataFrame에 한 번만 계산 → 시그널/포지션 루프만 순회.

Usage:
    python scripts/backtest_sniper_v2.py                          # BTC 60개월
    python scripts/backtest_sniper_v2.py --symbol SOL/USDT:USDT   # SOL 60개월
    python scripts/backtest_sniper_v2.py --symbol XRP/USDT:USDT   # XRP 60개월
    python scripts/backtest_sniper_v2.py --all                    # 3개 코인 전부
    python scripts/backtest_sniper_v2.py --months 24              # 기간 변경
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

from src.sniper_v2.config import (
    BTC_CONFIG, SOL_CONFIG, XRP_CONFIG, CONFIGS,
    FUNDING_STRATEGY_SYMBOLS, VWAP_MOMENTUM_SYMBOLS, SymbolConfig,
)
from src.sniper_v2.strategy import Direction, Signal, ActiveTrade
from src.sniper_v2.indicators import compute_all
from src.sniper_v2.funding_strategy import FundingConfig
from src.sniper_v2.vwap_momentum_strategy import VWAPMomentumConfig


# ══════════════════════════════════════════════════════════════
# 데이터 수집
# ══════════════════════════════════════════════════════════════

def _make_exchange():
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
    """ccxt 동기로 대량 OHLCV fetch (페이지네이션)."""
    exchange = _make_exchange()
    all_ohlcv = []
    cursor = since_ms

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
    df = df[df.index <= pd.Timestamp(until_ms, unit="ms", tz=None)]
    return df


def fetch_funding_bulk(symbol: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    """Binance 펀딩비 히스토리 fetch (페이지네이션)."""
    exchange = _make_exchange()
    base = symbol.replace("/USDT:USDT", "USDT")
    all_funding = []
    cursor = since_ms

    print(f"  Fetching {symbol} funding rates ...", flush=True)
    while cursor < until_ms:
        try:
            resp = exchange.fapiPublicGetFundingRate({
                "symbol": base, "startTime": cursor, "limit": 1000,
            })
        except Exception as e:
            print(f"  Funding fetch error: {e}", flush=True)
            break
        if not resp:
            break
        all_funding.extend(resp)
        last_ts = int(resp[-1]["fundingTime"])
        if last_ts >= until_ms or last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(0.2)

    if not all_funding:
        return pd.DataFrame(columns=["timestamp", "fundingRate"])

    fdf = pd.DataFrame(all_funding)
    fdf["fundingRate"] = fdf["fundingRate"].astype(float)
    fdf["timestamp"] = pd.to_datetime(fdf["fundingTime"].astype(int), unit="ms")
    fdf = fdf[["timestamp", "fundingRate"]].sort_values("timestamp").drop_duplicates()
    return fdf.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# 포지션 관리 (SniperV2/FundingContrarian 공통)
# ══════════════════════════════════════════════════════════════

def check_exit(trade: ActiveTrade, current_price: float, cfg: SymbolConfig) -> str | None:
    """TP/SL/Trailing/PP 체크. 포지션 상태를 변경(tp_hit, trail_price)."""
    is_long = trade.direction == Direction.LONG

    # 수익보호 (R 단위)
    if trade.risk > 0:
        cur_r = ((current_price - trade.entry_price) / trade.risk if is_long
                 else (trade.entry_price - current_price) / trade.risk)
        trade.peak_r = max(trade.peak_r, cur_r)
        if cfg.profit_protect and trade.peak_r >= cfg.pp_trigger:
            if cur_r <= cfg.pp_exit:
                return f"PP_{trade.peak_r:.1f}R"

    # TP/SL/Trailing
    if is_long:
        if current_price >= trade.tp3_price and not trade.tp3_hit:
            trade.tp3_hit = True
            if cfg.use_trailing:
                trade.trail_price = trade.tp2_price
        if current_price >= trade.tp2_price and not trade.tp2_hit:
            trade.tp2_hit = True
            if cfg.use_trailing:
                trade.trail_price = trade.tp1_price
        if current_price >= trade.tp1_price and not trade.tp1_hit:
            trade.tp1_hit = True
            if cfg.use_trailing:
                trade.trail_price = trade.entry_price
        stop = trade.trail_price if cfg.use_trailing else trade.sl_price
        if current_price <= stop:
            tp_label = "3" if trade.tp3_hit else "2" if trade.tp2_hit else "1"
            return f"TRAIL_TP{tp_label}_HIT" if trade.tp1_hit else "SL"
    else:
        if current_price <= trade.tp3_price and not trade.tp3_hit:
            trade.tp3_hit = True
            if cfg.use_trailing:
                trade.trail_price = trade.tp2_price
        if current_price <= trade.tp2_price and not trade.tp2_hit:
            trade.tp2_hit = True
            if cfg.use_trailing:
                trade.trail_price = trade.tp1_price
        if current_price <= trade.tp1_price and not trade.tp1_hit:
            trade.tp1_hit = True
            if cfg.use_trailing:
                trade.trail_price = trade.entry_price
        stop = trade.trail_price if cfg.use_trailing else trade.sl_price
        if current_price >= stop:
            tp_label = "3" if trade.tp3_hit else "2" if trade.tp2_hit else "1"
            return f"TRAIL_TP{tp_label}_HIT" if trade.tp1_hit else "SL"

    return None


def record_trade(trades: list, trade: ActiveTrade, exit_time, exit_price: float,
                 reason: str, cfg: SymbolConfig):
    """거래 결과 기록."""
    is_long = trade.direction == Direction.LONG
    if is_long:
        pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100 * cfg.leverage
    else:
        pnl_pct = (trade.entry_price - exit_price) / trade.entry_price * 100 * cfg.leverage
    hold_min = (exit_time - trade.entry_time).total_seconds() / 60
    trades.append({
        "entry_time": trade.entry_time,
        "exit_time": exit_time,
        "direction": trade.direction.value,
        "entry_price": trade.entry_price,
        "exit_price": exit_price,
        "pnl_pct": round(pnl_pct, 4),
        "reason": reason,
        "tp1_hit": trade.tp1_hit,
        "tp2_hit": trade.tp2_hit,
        "tp3_hit": trade.tp3_hit,
        "hold_min": round(hold_min, 1),
    })


# ══════════════════════════════════════════════════════════════
# EMA 전략 백테스트 (BTC, SOL)
# ══════════════════════════════════════════════════════════════

def backtest_ema(df: pd.DataFrame, cfg: SymbolConfig) -> list:
    """
    지표를 한 번만 계산하고, 시그널/포지션만 순회.
    기존 SniperV2.compute와 동일한 로직.
    """
    # 지표 전체 계산 (1회)
    df = compute_all(df.copy(), cfg)

    trades = []
    active_trade: ActiveTrade | None = None
    last_direction = 0
    cooldown = 0

    total_bars = len(df)
    warmup = cfg.warmup_bars
    report_interval = max(total_bars // 10, 1)

    # numpy 배열로 추출 (pandas 접근 오버헤드 제거)
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    open_arr = df["open"].values
    ema_fast_arr = df["ema_fast"].values
    ema_slow_arr = df["ema_slow"].values
    ema_trend_arr = df["ema_trend"].values
    rsi_arr = df["rsi"].values
    atr_arr = df["atr"].values
    atr_sma_arr = df["atr_sma"].values
    adx_arr = df["adx"].values
    bull_cross_arr = df["ema_bull_cross"].values
    bear_cross_arr = df["ema_bear_cross"].values
    swing_low_arr = df["swing_low"].values if "swing_low" in df.columns else None
    swing_high_arr = df["swing_high"].values if "swing_high" in df.columns else None
    times = df.index

    # 대기 중인 시그널 (다음 봉 시가에 체결)
    pending_signal: Signal | None = None

    for i in range(warmup, total_bars):
        if i % report_interval == 0:
            print(f"    {i/total_bars*100:.0f}% ({i}/{total_bars})", flush=True)

        current_time = times[i]

        # --- 대기 시그널을 현재 봉 시가(open)에 체결 ---
        # 시그널은 직전 봉의 close 기반으로 생성되었고,
        # 실제 체결은 다음 봉 open에서 발생 (현실 반영)
        # active_trade가 존재하면 대기 시그널은 폐기 (stale signal 방지)
        if pending_signal is not None and active_trade is not None:
            pending_signal = None
        if pending_signal is not None and active_trade is None:
            fill_price = float(open_arr[i])
            _ps = pending_signal
            pending_signal = None

            # 갭 체크: fill_price가 이미 SL을 넘어섰으면 진입 거부
            # (예: 롱인데 갭 다운으로 SL 아래에서 시작)
            if _ps.direction == Direction.LONG and fill_price <= _ps.sl_price:
                pass  # 진입 거부 — 셋업 무효
            elif _ps.direction == Direction.SHORT and fill_price >= _ps.sl_price:
                pass  # 진입 거부 — 셋업 무효
            else:
                # SL/TP를 시가 기준으로 재계산
                atr_i = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else _ps.atr

                if cfg.sl_method == "SWING":
                    # SWING: 구조적 레벨(swing low/high)을 절대 좌표로 유지
                    sl_adj = _ps.sl_price
                    risk = abs(fill_price - sl_adj)
                    # 구조적 레벨과 fill_price가 너무 가까우면 진입 거부
                    if risk < atr_i * 0.3:
                        risk = 0  # 아래에서 걸러짐
                else:
                    # ATR: fill_price 기준으로 SL 재계산
                    risk = abs(fill_price - _ps.sl_price)
                    if risk < atr_i * 0.3:
                        risk = atr_i * 0.5
                    sl_adj = fill_price - risk if _ps.direction == Direction.LONG else fill_price + risk

                # 최소 risk 미달 시 진입 거부
                if risk >= atr_i * 0.3:
                    if _ps.direction == Direction.LONG:
                        tp1_adj = fill_price + risk * cfg.tp1_rr
                        tp2_adj = fill_price + risk * cfg.tp2_rr
                        tp3_adj = fill_price + risk * cfg.tp3_rr
                    else:
                        tp1_adj = fill_price - risk * cfg.tp1_rr
                        tp2_adj = fill_price - risk * cfg.tp2_rr
                        tp3_adj = fill_price - risk * cfg.tp3_rr

                    active_trade = ActiveTrade(
                        direction=_ps.direction,
                        entry_price=fill_price,
                        sl_price=sl_adj,
                        tp1_price=tp1_adj,
                        tp2_price=tp2_adj,
                        tp3_price=tp3_adj,
                        trail_price=sl_adj,
                        risk=risk,
                        entry_time=current_time,
                    )

        # --- 봉 내 TP/SL/PP 체크 ---
        # 롱: open → low(SL 먼저) → high(TP) → close  (비관적 가정)
        # 숏: open → high(SL 먼저) → low(TP) → close
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

        # --- Cooldown ---
        if cooldown > 0:
            cooldown -= 1
            continue

        # --- NaN 체크 ---
        if (np.isnan(ema_fast_arr[i]) or np.isnan(ema_slow_arr[i]) or
            np.isnan(ema_trend_arr[i]) or np.isnan(rsi_arr[i]) or
            np.isnan(atr_arr[i]) or np.isnan(adx_arr[i])):
            continue

        # --- Volatility filter ---
        atr_val = float(atr_arr[i])
        atr_sma_val = float(atr_sma_arr[i]) if not np.isnan(atr_sma_arr[i]) else 0
        if atr_sma_val > 0 and atr_val / atr_sma_val > cfg.vol_filter:
            continue

        # --- ADX filter ---
        adx_val = float(adx_arr[i])
        if adx_val < cfg.adx_min:
            continue

        # --- Entry conditions ---
        # 시그널 판단은 현재 봉의 종가(close) 기반
        # 실제 체결은 다음 봉 시가(open)에서 발생
        cross_up = bool(bull_cross_arr[i])
        cross_down = bool(bear_cross_arr[i])
        close = float(close_arr[i])
        ema_trend_val = float(ema_trend_arr[i])
        rsi_val = float(rsi_arr[i])

        long_sig = cross_up and close > ema_trend_val and rsi_val > cfg.rsi_bull
        short_sig = cross_down and close < ema_trend_val and rsi_val < cfg.rsi_bear

        # Anti-duplicate
        if long_sig and last_direction == 1:
            long_sig = False
        if short_sig and last_direction == -1:
            short_sig = False
        if long_sig and short_sig:
            short_sig = False

        if not long_sig and not short_sig:
            continue

        # 마지막 봉이면 다음 봉이 없으므로 진입 불가
        if i + 1 >= total_bars:
            continue

        # --- SL/TP 계산 (시그널 기준, 체결 시 open 기준으로 재조정됨) ---
        entry = close
        is_long = long_sig

        if cfg.sl_method == "SWING" and swing_low_arr is not None:
            if is_long:
                sw = float(swing_low_arr[i]) if not np.isnan(swing_low_arr[i]) else entry - atr_val * 1.5
                sl = sw
            else:
                sw = float(swing_high_arr[i]) if not np.isnan(swing_high_arr[i]) else entry + atr_val * 1.5
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
            tp1 = entry + risk * cfg.tp1_rr
            tp2 = entry + risk * cfg.tp2_rr
            tp3 = entry + risk * cfg.tp3_rr
        else:
            last_direction = -1
            tp1 = entry - risk * cfg.tp1_rr
            tp2 = entry - risk * cfg.tp2_rr
            tp3 = entry - risk * cfg.tp3_rr

        signal = Signal(
            direction=direction, entry_price=entry, sl_price=sl,
            tp1_price=tp1, tp2_price=tp2, tp3_price=tp3,
            risk=risk, rsi=rsi_val, adx=adx_val, atr=atr_val,
            timestamp=current_time,
        )

        # 반대 시그널 → 기존 포지션 청산 (현재 봉 close에서 청산)
        if active_trade is not None and signal.direction != active_trade.direction:
            record_trade(trades, active_trade, current_time, close, "REVERSE", cfg)
            active_trade = None

        # 신규 진입: 다음 봉 시가(open)에 체결되도록 대기
        if active_trade is None:
            pending_signal = signal

    # 미결 포지션 정리
    if active_trade is not None:
        record_trade(trades, active_trade, times[-1], float(close_arr[-1]), "OPEN_AT_END", cfg)

    return trades


# ══════════════════════════════════════════════════════════════
# 펀딩 전략 백테스트 (XRP)
# ══════════════════════════════════════════════════════════════

def backtest_funding(df: pd.DataFrame, funding_df: pd.DataFrame, cfg: SymbolConfig) -> list:
    """
    펀딩비 z-score 역추세 전략 백테스트.
    ATR/RSI는 전체 사전계산, 펀딩 z-score는 봉별 계산 (경량).
    """
    fcfg = FundingConfig()

    # ATR 전체 사전계산
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1/14, min_periods=14).mean()

    # RSI 전체 사전계산
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - 100 / (1 + rs)

    # numpy 배열
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    open_arr = df["open"].values
    atr_arr = atr_series.values
    rsi_arr = rsi_series.values
    times = df.index

    # 펀딩 데이터 numpy
    f_times = funding_df["timestamp"].values if len(funding_df) > 0 else np.array([])
    f_rates = funding_df["fundingRate"].values if len(funding_df) > 0 else np.array([])

    trades = []
    active_trade: ActiveTrade | None = None
    pending_signal: Signal | None = None  # 다음 봉 open에서 진입할 시그널
    last_direction = 0
    cooldown = 0

    total_bars = len(df)
    warmup = cfg.warmup_bars
    report_interval = max(total_bars // 10, 1)

    for i in range(warmup, total_bars):
        if i % report_interval == 0:
            print(f"    {i/total_bars*100:.0f}% ({i}/{total_bars})", flush=True)

        current_time = times[i]

        # --- pending_signal 처리: 다음 봉 open에서 진입 ---
        if pending_signal is not None and active_trade is None:
            fill_price = float(open_arr[i])
            _ps = pending_signal
            atr_i = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else float(atr_arr[i-1])
            risk = atr_i * fcfg.sl_atr_mult

            if _ps.direction == Direction.LONG:
                sl = fill_price - risk
                tp1 = fill_price + risk * fcfg.tp1_rr
                tp2 = fill_price + risk * fcfg.tp2_rr
                tp3 = fill_price + risk * fcfg.tp3_rr
            else:
                sl = fill_price + risk
                tp1 = fill_price - risk * fcfg.tp1_rr
                tp2 = fill_price - risk * fcfg.tp2_rr
                tp3 = fill_price - risk * fcfg.tp3_rr

            active_trade = ActiveTrade(
                direction=_ps.direction,
                entry_price=fill_price, sl_price=sl,
                tp1_price=tp1, tp2_price=tp2, tp3_price=tp3,
                trail_price=sl, risk=risk,
                entry_time=current_time,
            )
            pending_signal = None

        # --- 봉 내 TP/SL/PP 체크 ---
        # 롱: open → low(SL 먼저) → high(TP) → close  (비관적 가정)
        # 숏: open → high(SL 먼저) → low(TP) → close
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

        # Cooldown
        if cooldown > 0:
            cooldown -= 1
            continue

        # --- 펀딩비 z-score ---
        # 현재 시점 이전 펀딩만
        ct_np = np.datetime64(current_time)
        mask = f_times <= ct_np
        n_funding = int(mask.sum())
        if n_funding < 30:
            continue

        rates_window = f_rates[mask]
        n = min(len(rates_window), fcfg.funding_lookback)
        w = rates_window[-n:]
        fr_mean = float(np.mean(w))
        fr_std = float(np.std(w))
        if fr_std < 1e-10:
            continue
        fr_z = (float(rates_window[-1]) - fr_mean) / fr_std

        # --- 가격 z-score (다중 타임프레임) ---
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

        # --- 시그널 판단 ---
        zt = fcfg.z_threshold
        pcz = fcfg.price_confirm_z
        is_long = False
        is_short = False
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

        # Anti-duplicate
        if is_long and last_direction == 1:
            continue
        if is_short and last_direction == -1:
            continue

        # --- SL/TP (시그널 생성용, 실제 진입은 다음 봉 open에서 재계산) ---
        atr_val = float(atr_arr[i])
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        entry = float(close_arr[i])
        risk = atr_val * fcfg.sl_atr_mult
        rsi_val = float(rsi_arr[i]) if not np.isnan(rsi_arr[i]) else 50.0
        direction = Direction.LONG if is_long else Direction.SHORT

        if is_long:
            last_direction = 1
        else:
            last_direction = -1

        signal = Signal(
            direction=direction, entry_price=entry, sl_price=entry - risk if is_long else entry + risk,
            tp1_price=0, tp2_price=0, tp3_price=0,  # open에서 재계산
            risk=risk, rsi=rsi_val, adx=0.0, atr=atr_val,
            timestamp=current_time,
        )

        # 반대 시그널 → 청산 (close에서)
        if active_trade is not None and signal.direction != active_trade.direction:
            record_trade(trades, active_trade, current_time, entry, "REVERSE", cfg)
            active_trade = None

        # 다음 봉 open에서 진입 예약
        if active_trade is None and i + 1 < total_bars:
            pending_signal = signal

    # 미결 포지션 정리
    if active_trade is not None:
        record_trade(trades, active_trade, times[-1], float(close_arr[-1]), "OPEN_AT_END", cfg)

    return trades


# ══════════════════════════════════════════════════════════════
# VWAP 모멘텀 전략 백테스트 (SOL)
# ══════════════════════════════════════════════════════════════

def backtest_vwap_momentum(df: pd.DataFrame, cfg: SymbolConfig) -> list:
    """
    VWAP 모멘텀 전략 백테스트 (alpha_factory WF-CV 검증).
    5분봉 → 1시간봉 리샘플 → VWAP z-score → 진입/청산.
    """
    vcfg = VWAPMomentumConfig()

    # ATR 전체 사전계산 (5분봉)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1/14, min_periods=14).mean()

    # 1시간봉 리샘플
    df_1h = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    close_1h = df_1h["close"].values
    vol_1h = df_1h["volume"].values
    times_1h = df_1h.index
    nh = len(df_1h)

    # VWAP z-score 전체 사전계산
    # 주의: vwap_z[i]는 봉 i가 **완성된 후** 계산 가능한 값.
    # 따라서 실제 사용 시점은 봉 i+1부터 (다음 봉에서 참조).
    # VWAP = 직전 w개 완성 봉 기준, z = 마지막 완성 봉 종가 기준.
    vwap_z = np.full(nh, np.nan)
    w = vcfg.vwap_hours
    for i in range(w + 1, nh):
        # 직전 w개 완성 봉 (i-w ~ i-1)으로 VWAP 계산
        c = close_1h[i - w - 1 : i - 1]
        v = vol_1h[i - w - 1 : i - 1]
        total_vol = v.sum()
        if total_vol <= 0:
            continue
        vwap = (c * v).sum() / total_vol
        std = c.std(ddof=1)
        if std > 1e-8:
            # 마지막 완성 봉(i-1)의 종가와 비교
            vwap_z[i] = (close_1h[i - 1] - vwap) / std

    # 15분봉 배열
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    open_arr = df["open"].values
    atr_arr = atr_series.values
    times = df.index
    total_bars = len(df)

    trades = []
    active_trade: ActiveTrade | None = None
    last_direction = 0
    cooldown = 0

    warmup = max(cfg.warmup_bars, w * 12 + 20)  # 5분봉: 6h = 72bars
    report_interval = max(total_bars // 10, 1)

    for i in range(warmup, total_bars):
        if i % report_interval == 0:
            print(f"    {i/total_bars*100:.0f}% ({i}/{total_bars})", flush=True)

        current_time = times[i]

        # --- 봉 내 TP/SL/PP 체크 (보수적 순서: SL 먼저) ---
        if active_trade is not None:
            is_long = active_trade.direction == Direction.LONG
            # 롱: Open→Low(SL먼저)→High→Close / 숏: Open→High(SL먼저)→Low→Close
            if is_long:
                check_prices = [open_arr[i], low_arr[i], high_arr[i], close_arr[i]]
            else:
                check_prices = [open_arr[i], high_arr[i], low_arr[i], close_arr[i]]
            for check_price in check_prices:
                if active_trade is None:
                    break
                exit_reason = check_exit(active_trade, float(check_price), cfg)
                if exit_reason:
                    record_trade(trades, active_trade, current_time, float(check_price), exit_reason, cfg)
                    if exit_reason == "SL":
                        cooldown = cfg.cooldown_bars
                    active_trade = None
                    break

        # Cooldown
        if cooldown > 0:
            cooldown -= 1
            continue

        # --- 1시간봉 VWAP z-score 조회 ---
        # 직전 완성된 1시간봉의 z-score만 사용 (look-ahead 방지)
        # searchsorted(side="left")로 현재 시각 이전의 마지막 완성 봉 찾기
        h_idx = np.searchsorted(times_1h, current_time, side="left")
        # times_1h[h_idx]가 current_time 이후이면 h_idx-1이 마지막 완성 봉
        if h_idx > 0 and (h_idx >= nh or times_1h[h_idx] > current_time):
            h_idx -= 1
        if h_idx < w + 1 or h_idx >= nh:
            continue

        z = vwap_z[h_idx]
        if np.isnan(z) or abs(z) < vcfg.entry_z:
            continue

        # 거래량 확인 (직전 완성 봉 기준)
        if h_idx >= 2 and vol_1h[h_idx - 1] < np.mean(vol_1h[max(0, h_idx - w - 1):h_idx - 1]) * vcfg.vol_confirm_ratio:
            continue

        is_long = z > 0
        direction_int = 1 if is_long else -1

        # Anti-duplicate
        if direction_int == last_direction:
            continue

        # ATR 기반 SL/TP
        atr_val = float(atr_arr[i])
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # 진입가 = 다음 봉 시가 (실전과 동일: 시그널 후 다음 봉에서 집행)
        if i + 1 >= total_bars:
            continue
        entry = float(open_arr[i + 1])
        risk = atr_val * vcfg.sl_atr_mult

        if is_long:
            last_direction = 1
            sl = entry - risk
            tp1 = entry + risk * vcfg.tp1_rr
            tp2 = entry + risk * vcfg.tp2_rr
            tp3 = entry + risk * vcfg.tp3_rr
            direction = Direction.LONG
        else:
            last_direction = -1
            sl = entry + risk
            tp1 = entry - risk * vcfg.tp1_rr
            tp2 = entry - risk * vcfg.tp2_rr
            tp3 = entry - risk * vcfg.tp3_rr
            direction = Direction.SHORT

        signal = Signal(
            direction=direction, entry_price=entry, sl_price=sl,
            tp1_price=tp1, tp2_price=tp2, tp3_price=tp3,
            risk=risk, rsi=50.0, adx=0.0, atr=atr_val,
            timestamp=current_time,
        )

        # 반대 시그널 → 청산
        if active_trade is not None and signal.direction != active_trade.direction:
            record_trade(trades, active_trade, current_time, entry, "REVERSE", cfg)
            active_trade = None

        # 신규 진입
        if active_trade is None:
            active_trade = ActiveTrade(
                direction=signal.direction,
                entry_price=signal.entry_price,
                sl_price=signal.sl_price,
                tp1_price=signal.tp1_price,
                tp2_price=signal.tp2_price,
                tp3_price=signal.tp3_price,
                trail_price=signal.sl_price,
                risk=signal.risk,
                entry_time=signal.timestamp,
            )

    # 미결 포지션 정리
    if active_trade is not None:
        record_trade(trades, active_trade, times[-1], float(close_arr[-1]), "OPEN_AT_END", cfg)

    return trades


# ══════════════════════════════════════════════════════════════
# 메인 백테스트
# ══════════════════════════════════════════════════════════════

def run_backtest(
    symbol: str,
    cfg: SymbolConfig,
    months: int = 60,
    timeframe: str = "15m",
):
    short_sym = symbol.split("/")[0]
    is_funding = symbol in FUNDING_STRATEGY_SYMBOLS
    is_vwap = symbol in VWAP_MOMENTUM_SYMBOLS

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=months * 30)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)

    if is_vwap:
        strategy_name = "VWAP Momentum"
    elif is_funding:
        strategy_name = "FundingContrarian"
    else:
        strategy_name = f"EMA {cfg.ema_fast}/{cfg.ema_slow}/{cfg.ema_trend}"
    sl_desc = f"{'SWING_'+str(cfg.swing_lookback) if cfg.sl_method=='SWING' else 'ATR×'+str(cfg.sl_atr_mult)}"

    print(f"\n{'=' * 65}", flush=True)
    print(f"  Sniper V2 Backtest — {short_sym}", flush=True)
    print(f"  Strategy: {strategy_name} | SL: {sl_desc}", flush=True)
    print(f"  Period: {since.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')} ({months}개월)", flush=True)
    print(f"  Leverage: {cfg.leverage}x | Balance Ratio: {cfg.balance_ratio:.0%}", flush=True)
    print(f"{'=' * 65}", flush=True)

    # --- 데이터 수집 ---
    bt_tf = "5m" if is_vwap else timeframe
    print(f"\n  Fetching {short_sym} {bt_tf} data ...", flush=True)
    df = fetch_ohlcv_bulk(symbol, bt_tf, since_ms, until_ms)
    print(f"  {bt_tf} bars: {len(df)}", flush=True)

    if len(df) < cfg.warmup_bars + 100:
        print(f"  데이터 부족! ({len(df)} bars) — 스킵", flush=True)
        return []

    actual_start = df.index[0]
    actual_end = df.index[-1]
    actual_days = (actual_end - actual_start).days
    print(f"  실제 기간: {actual_start.strftime('%Y-%m-%d')} ~ {actual_end.strftime('%Y-%m-%d')} ({actual_days}일)", flush=True)

    funding_df = None
    if is_funding:
        funding_df = fetch_funding_bulk(symbol, since_ms, until_ms)
        print(f"  Funding rates: {len(funding_df)}", flush=True)

    # --- 백테스트 실행 ---
    t0 = time.time()
    print(f"\n  Running backtest (optimized)...", flush=True)

    if is_vwap:
        trades = backtest_vwap_momentum(df, cfg)
    elif is_funding:
        trades = backtest_funding(df, funding_df, cfg)
    else:
        trades = backtest_ema(df, cfg)

    elapsed = time.time() - t0
    print(f"  완료! {elapsed:.1f}초 소요, {len(trades)}건 거래", flush=True)

    # --- 결과 ---
    print_results(trades, short_sym, strategy_name, sl_desc, actual_days, cfg)
    return trades


# ══════════════════════════════════════════════════════════════
# 결과 출력
# ══════════════════════════════════════════════════════════════

def print_results(trades: list, symbol: str, strategy: str, sl_desc: str, days: int, cfg: SymbolConfig):
    if not trades:
        print(f"\n  {symbol}: 거래 없음!", flush=True)
        return

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    n_wins, n_losses = len(wins), len(losses)

    total_pnl = tdf["pnl_pct"].sum()
    avg_pnl = tdf["pnl_pct"].mean()
    avg_win = wins["pnl_pct"].mean() if n_wins else 0
    avg_loss = losses["pnl_pct"].mean() if n_losses else 0
    max_win = tdf["pnl_pct"].max()
    max_loss = tdf["pnl_pct"].min()
    win_rate = n_wins / n * 100

    gross_profit = wins["pnl_pct"].sum() if n_wins else 0
    gross_loss = abs(losses["pnl_pct"].sum()) if n_losses else 0.01
    profit_factor = gross_profit / gross_loss

    # Max drawdown
    cum_pnl = tdf["pnl_pct"].cumsum()
    max_dd = (cum_pnl - cum_pnl.cummax()).min()

    # 연속 승패
    max_streak_win = max_streak_loss = streak_win = streak_loss = 0
    for pnl in tdf["pnl_pct"]:
        if pnl > 0:
            streak_win += 1; streak_loss = 0
            max_streak_win = max(max_streak_win, streak_win)
        else:
            streak_loss += 1; streak_win = 0
            max_streak_loss = max(max_streak_loss, streak_loss)

    # 방향별
    longs = tdf[tdf["direction"] == "LONG"]
    shorts = tdf[tdf["direction"] == "SHORT"]
    long_wr = len(longs[longs["pnl_pct"] > 0]) / len(longs) * 100 if len(longs) else 0
    short_wr = len(shorts[shorts["pnl_pct"] > 0]) / len(shorts) * 100 if len(shorts) else 0

    # 청산 사유
    reason_stats = tdf.groupby("reason").agg(
        count=("pnl_pct", "count"),
        avg_pnl=("pnl_pct", "mean"),
        total_pnl=("pnl_pct", "sum"),
    ).sort_values("count", ascending=False)

    # TP 달성률
    tp1_rate = tdf["tp1_hit"].sum() / n * 100
    tp2_rate = tdf["tp2_hit"].sum() / n * 100
    tp3_rate = tdf["tp3_hit"].sum() / n * 100

    avg_hold = tdf["hold_min"].mean()

    # 복리 잔고 시뮬레이션
    balance = 1000.0
    peak_balance = balance
    max_dd_balance = 0.0
    for pnl in tdf["pnl_pct"]:
        balance *= (1 + pnl / 100 * cfg.balance_ratio)
        peak_balance = max(peak_balance, balance)
        dd = (balance - peak_balance) / peak_balance * 100
        max_dd_balance = min(max_dd_balance, dd)
    final_balance = balance

    # 연환산 수익률
    years = days / 365
    if years > 0 and final_balance > 0:
        cagr = (final_balance / 1000) ** (1 / years) - 1
    else:
        cagr = 0

    # Calmar ratio
    calmar = (cagr * 100) / abs(max_dd_balance) if max_dd_balance < -0.01 else float("inf")

    # 월별 수익
    tdf["month"] = pd.to_datetime(tdf["exit_time"]).dt.to_period("M")
    monthly = tdf.groupby("month")["pnl_pct"].agg(["sum", "count", "mean"])

    # 연도별 수익
    tdf["year"] = pd.to_datetime(tdf["exit_time"]).dt.year
    yearly = tdf.groupby("year")["pnl_pct"].agg(["sum", "count", "mean"])

    print(flush=True)
    print(f"{'=' * 65}", flush=True)
    print(f"  {symbol} — {strategy} | SL: {sl_desc} | {cfg.leverage}x", flush=True)
    print(f"  기간: {days}일 ({days/365:.1f}년)", flush=True)
    print(f"{'=' * 65}", flush=True)
    print(flush=True)
    print(f"  총 거래:         {n}건 ({n_wins}W / {n_losses}L)", flush=True)
    print(f"  승률:            {win_rate:.1f}%", flush=True)
    print(f"  Profit Factor:   {profit_factor:.2f}", flush=True)
    print(flush=True)
    print(f"  --- 수익 (레버리지 {cfg.leverage}x 반영) ---", flush=True)
    print(f"  총 누적 수익:    {total_pnl:+.2f}%", flush=True)
    print(f"  평균 수익/건:    {avg_pnl:+.2f}%", flush=True)
    print(f"  평균 수익 (승):  {avg_win:+.2f}%", flush=True)
    print(f"  평균 손실 (패):  {avg_loss:+.2f}%", flush=True)
    print(f"  최대 단일 수익:  {max_win:+.2f}%", flush=True)
    print(f"  최대 단일 손실:  {max_loss:+.2f}%", flush=True)
    print(f"  Max DD (누적):   {max_dd:+.2f}%", flush=True)
    print(flush=True)
    print(f"  --- 복리 시뮬레이션 ($1000 시작, 잔고의 {cfg.balance_ratio:.0%} 사용) ---", flush=True)
    print(f"  최종 잔고:       ${final_balance:,.2f}", flush=True)
    print(f"  총 수익률:       {(final_balance/1000-1)*100:+.2f}%", flush=True)
    print(f"  CAGR:            {cagr*100:+.2f}%", flush=True)
    print(f"  Max DD (잔고):   {max_dd_balance:+.2f}%", flush=True)
    print(f"  Calmar Ratio:    {calmar:.2f}", flush=True)
    print(flush=True)
    print(f"  --- 방향별 ---", flush=True)
    if len(longs):
        print(f"  LONG:   {len(longs)}건 (승률 {long_wr:.1f}%) 평균 {longs['pnl_pct'].mean():+.2f}%", flush=True)
    else:
        print(f"  LONG:   0건", flush=True)
    if len(shorts):
        print(f"  SHORT:  {len(shorts)}건 (승률 {short_wr:.1f}%) 평균 {shorts['pnl_pct'].mean():+.2f}%", flush=True)
    else:
        print(f"  SHORT:  0건", flush=True)
    print(flush=True)
    print(f"  --- TP 달성률 ---", flush=True)
    print(f"  TP1: {tp1_rate:.1f}%  |  TP2: {tp2_rate:.1f}%  |  TP3: {tp3_rate:.1f}%", flush=True)
    print(flush=True)
    print(f"  --- 기타 ---", flush=True)
    print(f"  평균 보유시간:   {avg_hold:.0f}분 ({avg_hold/60:.1f}시간)", flush=True)
    print(f"  최대 연승:       {max_streak_win}", flush=True)
    print(f"  최대 연패:       {max_streak_loss}", flush=True)
    print(f"  월 평균 거래:    {n / max(days/30, 1):.1f}건", flush=True)
    print(flush=True)
    print(f"  --- 청산 사유별 ---", flush=True)
    for reason, row in reason_stats.iterrows():
        print(f"    {reason:20s}  {int(row['count']):4d}건  avg {row['avg_pnl']:+.2f}%  total {row['total_pnl']:+.2f}%", flush=True)
    print(flush=True)
    print(f"  --- 연도별 수익 ---", flush=True)
    for year, row in yearly.iterrows():
        print(f"    {year}  {int(row['count']):4d}건  총 {row['sum']:+.2f}%  평균 {row['mean']:+.2f}%", flush=True)
    print(flush=True)
    print(f"  --- 월별 수익 (최근 12개월) ---", flush=True)
    for period, row in monthly.tail(12).iterrows():
        bar = "+" * int(max(0, row['sum']) / 2) + "-" * int(max(0, -row['sum']) / 2)
        print(f"    {period}  {int(row['count']):3d}건  {row['sum']:+7.2f}%  {bar}", flush=True)
    print(f"{'=' * 65}", flush=True)


def _summarize_trades(trades: list, cfg: SymbolConfig) -> dict:
    """거래 리스트 → 요약 통계."""
    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "pnl": 0, "bal": 1000, "mdd_bal": 0}
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    wr = len(wins) / n * 100
    gp = wins["pnl_pct"].sum() if len(wins) else 0
    gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.01
    pf = gp / gl
    total = tdf["pnl_pct"].sum()
    bal = 1000.0
    peak = 1000.0
    mdd_bal = 0.0
    for p in tdf["pnl_pct"]:
        bal *= (1 + p / 100 * cfg.balance_ratio)
        peak = max(peak, bal)
        dd = (bal - peak) / peak * 100
        mdd_bal = min(mdd_bal, dd)
    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 2), "pnl": round(total, 2),
            "bal": round(bal, 2), "mdd_bal": round(mdd_bal, 2)}


def run_oos_backtest(symbols: list[str], total_months: int = 60, oos_months: int = 12):
    """
    In-Sample / Out-of-Sample 분할 백테스트.
    데이터 한 번 fetch → IS(앞 N-oos) / OOS(뒤 oos) 분할.
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=total_months * 30)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)
    oos_start = now - timedelta(days=oos_months * 30)
    is_months = total_months - oos_months

    print(f"\n{'='*70}", flush=True)
    print(f"  In-Sample / Out-of-Sample 백테스트", flush=True)
    print(f"  IS: {since.strftime('%Y-%m-%d')} ~ {oos_start.strftime('%Y-%m-%d')} ({is_months}개월)", flush=True)
    print(f"  OOS: {oos_start.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')} ({oos_months}개월)", flush=True)
    print(f"{'='*70}", flush=True)

    results = {}
    for sym in symbols:
        cfg = CONFIGS.get(sym)
        if cfg is None:
            continue
        short = sym.split("/")[0]
        is_funding = sym in FUNDING_STRATEGY_SYMBOLS
        is_vwap = sym in VWAP_MOMENTUM_SYMBOLS
        bt_tf = "5m" if is_vwap else "15m"

        print(f"\n  --- {short} ---", flush=True)
        print(f"  Fetching {bt_tf} data ...", flush=True)
        df = fetch_ohlcv_bulk(sym, bt_tf, since_ms, until_ms)
        print(f"  {len(df)} bars", flush=True)
        if len(df) < cfg.warmup_bars + 100:
            continue

        funding_df = None
        if is_funding:
            funding_df = fetch_funding_bulk(sym, since_ms, until_ms)
            print(f"  Funding: {len(funding_df)} rates", flush=True)

        oos_ts = pd.Timestamp(oos_start.replace(tzinfo=None))

        # IS 백테스트
        df_is = df[df.index < oos_ts]
        print(f"  IS: {len(df_is)} bars ...", flush=True)
        if is_vwap:
            is_trades = backtest_vwap_momentum(df_is, cfg)
        elif is_funding:
            f_is = funding_df[funding_df["timestamp"] < oos_ts] if funding_df is not None else funding_df
            is_trades = backtest_funding(df_is, f_is, cfg)
        else:
            is_trades = backtest_ema(df_is, cfg)

        # OOS 백테스트 (전체 데이터로 지표 계산, OOS 기간 거래만 필터)
        print(f"  OOS: full run + filter {oos_ts.strftime('%Y-%m-%d')}~ ...", flush=True)
        if is_vwap:
            all_trades = backtest_vwap_momentum(df, cfg)
        elif is_funding:
            all_trades = backtest_funding(df, funding_df, cfg)
        else:
            all_trades = backtest_ema(df, cfg)
        oos_trades = [t for t in all_trades if t["entry_time"] >= oos_ts]

        is_s = _summarize_trades(is_trades, cfg)
        oos_s = _summarize_trades(oos_trades, cfg)
        results[sym] = {"is": is_s, "oos": oos_s}

        print(f"  {short} IS:  {is_s['n']:4d}건 WR={is_s['wr']:5.1f}% PF={is_s['pf']:5.2f} "
              f"PnL={is_s['pnl']:+8.2f}% 잔고=${is_s['bal']:>9,.2f} MDD={is_s['mdd_bal']:+.2f}%", flush=True)
        print(f"  {short} OOS: {oos_s['n']:4d}건 WR={oos_s['wr']:5.1f}% PF={oos_s['pf']:5.2f} "
              f"PnL={oos_s['pnl']:+8.2f}% 잔고=${oos_s['bal']:>9,.2f} MDD={oos_s['mdd_bal']:+.2f}%", flush=True)

        if is_s['pf'] > 0 and oos_s['pf'] > 0:
            pf_decay = (oos_s['pf'] - is_s['pf']) / is_s['pf'] * 100 if is_s['pf'] else 0
            verdict = "PASS" if oos_s['pf'] >= 1.0 else "WARN" if oos_s['pf'] >= 0.9 else "FAIL"
            print(f"  PF 변화: IS {is_s['pf']:.2f} → OOS {oos_s['pf']:.2f} ({pf_decay:+.1f}%) → {verdict}", flush=True)

    # 종합
    print(f"\n{'='*70}", flush=True)
    print(f"  IS / OOS 종합 비교", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'코인':6s} | {'IS PF':>6s} {'IS PnL':>9s} {'IS 잔고':>10s} | {'OOS PF':>7s} {'OOS PnL':>9s} {'OOS 잔고':>10s} | {'판정':>4s}", flush=True)
    print(f"  {'─'*78}", flush=True)
    for sym, r in results.items():
        short = sym.split("/")[0]
        is_s, oos_s = r["is"], r["oos"]
        verdict = "PASS" if oos_s['pf'] >= 1.0 else "WARN" if oos_s['pf'] >= 0.9 else "FAIL"
        print(f"  {short:6s} | {is_s['pf']:6.2f} {is_s['pnl']:+8.2f}% ${is_s['bal']:>9,.2f} | "
              f"{oos_s['pf']:7.2f} {oos_s['pnl']:+8.2f}% ${oos_s['bal']:>9,.2f} | {verdict:>4s}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sniper V2 Backtest")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--all", action="store_true", help="전체 코인")
    parser.add_argument("--months", type=int, default=60, help="백테스트 기간 (개월)")
    parser.add_argument("--oos", action="store_true", help="IS/OOS 분할 백테스트")
    parser.add_argument("--oos-months", type=int, default=12, help="OOS 기간 (개월)")
    args = parser.parse_args()

    if args.all:
        symbols = ["BTC/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "AVAX/USDT:USDT"]
    else:
        symbols = [args.symbol]

    if args.oos:
        run_oos_backtest(symbols, total_months=args.months, oos_months=args.oos_months)
    else:
        all_trades = {}
        for sym in symbols:
            cfg = CONFIGS.get(sym)
            if cfg is None:
                print(f"No config for {sym}, skipping", flush=True)
                continue
            trades = run_backtest(sym, cfg, months=args.months)
            all_trades[sym] = trades

        if len(all_trades) > 1:
            print(f"\n{'=' * 65}", flush=True)
            print(f"  종합 비교", flush=True)
            print(f"{'=' * 65}", flush=True)
            print(f"  {'코인':8s} {'거래수':>6s} {'승률':>7s} {'PF':>6s} {'총PnL':>10s} {'MaxDD':>8s} {'최종잔고':>12s}", flush=True)
            print(f"  {'-'*57}", flush=True)
            for sym, trades in all_trades.items():
                if not trades:
                    continue
                tdf = pd.DataFrame(trades)
                n = len(tdf)
                wr = len(tdf[tdf["pnl_pct"] > 0]) / n * 100
                gp = tdf[tdf["pnl_pct"] > 0]["pnl_pct"].sum()
                gl = abs(tdf[tdf["pnl_pct"] <= 0]["pnl_pct"].sum()) or 0.01
                pf = gp / gl
                total = tdf["pnl_pct"].sum()
                dd = (tdf["pnl_pct"].cumsum() - tdf["pnl_pct"].cumsum().cummax()).min()
                cfg = CONFIGS[sym]
                bal = 1000.0
                for p in tdf["pnl_pct"]:
                    bal *= (1 + p / 100 * cfg.balance_ratio)
                short = sym.split("/")[0]
                print(f"  {short:8s} {n:6d} {wr:6.1f}% {pf:6.2f} {total:+9.2f}% {dd:+7.2f}% ${bal:>10,.2f}", flush=True)
            print(f"{'=' * 65}", flush=True)
