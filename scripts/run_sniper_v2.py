#!/usr/bin/env python3
"""
Sniper V2 Daemon (Multi-Symbol)

BTC + SOL + XRP 동시 실행 가능. 심볼별 최적 파라미터 자동 적용.
XRP는 EMA가 아닌 펀딩비 역추세 전략 (alpha_factory WF-CV 검증).

Usage:
    python scripts/run_sniper_v2.py                           # BTC 단독
    python scripts/run_sniper_v2.py --symbol SOL/USDT:USDT    # SOL 단독
    python scripts/run_sniper_v2.py --symbol XRP/USDT:USDT    # XRP 단독 (펀딩 전략)
    python scripts/run_sniper_v2.py --multi                    # BTC + SOL + XRP 동시
    python scripts/run_sniper_v2.py --dry-run                  # BTC 모의
    python scripts/run_sniper_v2.py --multi --dry-run          # 전체 모의
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sniper_v2.config import (
    BTC_CONFIG,
    CHECK_INTERVAL_SEC,
    CONFIGS,
    FUNDING_STRATEGY_SYMBOLS,
    VWAP_MOMENTUM_SYMBOLS,
    SOL_CONFIG,
    TIMEFRAME,
    SymbolConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def load_keys() -> dict:
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"
    with open(keys_path) as f:
        return yaml.safe_load(f)


def _log_trade(data: dict, log_dir: str):
    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    log_file = d / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(data, default=str) + "\n")


def _tf_to_seconds(tf: str) -> int:
    unit = tf[-1]
    num = int(tf[:-1])
    if unit == "m": return num * 60
    if unit == "h": return num * 3600
    if unit == "d": return num * 86400
    return 300


def _ohlcv_to_df(ohlcv: list):
    import pandas as pd
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("timestamp")


async def _fetch_funding(exchange, symbol: str, logger) -> "pd.DataFrame | None":
    """Binance에서 펀딩비 히스토리 조회 (최근 100개, 8h 간격 ≈ 33일)."""
    import pandas as pd
    try:
        base = symbol.replace("/USDT:USDT", "USDT")
        resp = await exchange.fapiPublicGetFundingRate({"symbol": base, "limit": 100})
        if not resp:
            return None
        df = pd.DataFrame(resp)
        df["fundingRate"] = df["fundingRate"].astype(float)
        df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms")
        return df[["timestamp", "fundingRate"]].sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Funding fetch: {e}")
        return None


async def run_symbol(exchange, symbol: str, cfg: SymbolConfig, telegram, dry_run: bool):
    """단일 심볼 데몬 루프."""
    from src.sniper_v2.strategy import SniperV2, Direction

    logger = logging.getLogger(f"v2.{symbol.split('/')[0]}")
    short_sym = symbol.split("/")[0]

    # 전략 판별: 펀딩 > VWAP모멘텀 > EMA 순서
    is_funding_strategy = symbol in FUNDING_STRATEGY_SYMBOLS
    is_vwap_strategy = symbol in VWAP_MOMENTUM_SYMBOLS

    if is_funding_strategy:
        from src.sniper_v2.funding_strategy import FundingContrarianSniper
        sniper = FundingContrarianSniper(cfg)
        strategy_label = "FundingContrarian"
    elif is_vwap_strategy:
        from src.sniper_v2.vwap_momentum_strategy import VWAPMomentumSniper
        sniper = VWAPMomentumSniper(cfg)
        strategy_label = "VWAP Momentum"
    else:
        sniper = SniperV2(cfg)
        strategy_label = f"EMA {cfg.ema_fast}/{cfg.ema_slow}/{cfg.ema_trend}"

    # RL 상태 로거
    from src.sniper_v2.rl_logger import RLStateLogger
    rl_strategy = "funding_contrarian" if is_funding_strategy else ("vwap_momentum" if is_vwap_strategy else "ema_cross")
    rl_logger = RLStateLogger(symbol, rl_strategy, cfg)

    # 레버리지 설정
    if not dry_run:
        try:
            base = symbol.replace("/USDT:USDT", "USDT")
            await exchange.fapiPrivatePostLeverage({"symbol": base, "leverage": cfg.leverage})
            logger.info(f"Leverage {cfg.leverage}x set for {short_sym}")
        except Exception as e:
            logger.warning(f"Leverage set failed: {e}")

    # 잔고
    balance = 1000.0
    if not dry_run:
        try:
            bal = await exchange.fetch_balance()
            balance = float(bal.get("USDT", {}).get("total", 0))
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
            return

    logger.info(f"{'='*50}")
    logger.info(f"{short_sym} {'[DRY]' if dry_run else '[LIVE]'} | "
                f"{strategy_label} | "
                f"SL {'SWING_'+str(cfg.swing_lookback) if cfg.sl_method=='SWING' else 'ATR×'+str(cfg.sl_atr_mult)} | "
                f"Balance ${balance:.2f} × {cfg.balance_ratio:.0%}")
    logger.info(f"{'='*50}")

    # VWAP 전략은 5분봉 사용 (리서치와 동일)
    symbol_tf = "5m" if is_vwap_strategy else TIMEFRAME
    symbol_tf_limit = 500 if is_vwap_strategy else 300
    tf_seconds = _tf_to_seconds(symbol_tf)
    last_candle_time = 0
    position_qty = 0.0
    funding_df = None
    last_funding_fetch = 0
    df = None           # 최근 OHLCV (RL 로그 + SL/TP 체크용)
    _rl_funding = None  # 최근 펀딩 데이터 (RL 로그용)

    while True:
        try:
            now = time.time()
            current_candle_start = int(now // tf_seconds) * tf_seconds

            if current_candle_start <= last_candle_time:
                # TP/SL 체크
                if sniper.active_trade:
                    try:
                        ticker = await exchange.fetch_ticker(symbol)
                        price = float(ticker["last"])
                        exit_reason = sniper.check_exit(price)
                        if exit_reason:
                            await _close_position(
                                exchange, symbol, sniper, cfg, position_qty,
                                price, exit_reason, telegram, dry_run, logger,
                                rl_logger=rl_logger, df=df,
                                rl_account={"balance_usdt": balance, "leverage": cfg.leverage},
                                rl_funding=_rl_funding if is_funding_strategy else None,
                            )
                            position_qty = 0.0
                    except Exception as e:
                        logger.error(f"Price check: {e}")

                await asyncio.sleep(CHECK_INTERVAL_SEC)
                continue

            last_candle_time = current_candle_start

            # OHLCV
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, symbol_tf, limit=symbol_tf_limit)
            except Exception as e:
                logger.error(f"OHLCV: {e}")
                await asyncio.sleep(10)
                continue

            df = _ohlcv_to_df(ohlcv)
            if len(df) < cfg.warmup_bars:
                await asyncio.sleep(10)
                continue

            # 펀딩 전략: 펀딩비 조회 (30분마다 갱신, 8h 주기이므로 충분)
            if is_funding_strategy and now - last_funding_fetch > 1800:
                funding_df = await _fetch_funding(exchange, symbol, logger)
                last_funding_fetch = now
                if funding_df is not None:
                    logger.info(f"[{short_sym}] Funding refreshed: {len(funding_df)} rates, "
                                f"latest={funding_df['fundingRate'].iloc[-1]:.6f}")

            # 시그널
            if is_funding_strategy:
                signal = sniper.compute(df, funding_df)
            else:
                signal = sniper.compute(df)

            last_close = float(df["close"].iloc[-1])
            pos = sniper.active_trade.direction.value if sniper.active_trade else "FLAT"
            sig = f" sig={signal.direction.value} RSI={signal.rsi:.0f}" if signal else ""
            logger.info(f"[{short_sym}] {last_close:.4f} pos={pos}{sig}")

            # RL 캔들 로그: 매 15분봉 상태 기록
            try:
                _rl_account = {"balance_usdt": balance, "leverage": cfg.leverage}
                _rl_funding = None
                if is_funding_strategy and funding_df is not None and len(funding_df) > 0:
                    _rl_funding = {"latest_rate": float(funding_df["fundingRate"].iloc[-1])}
                _hold_reason = ""
                if signal is None:
                    _hold_reason = "no_signal"
                rl_logger.log_candle(
                    df, sniper.active_trade, _rl_account,
                    signal_generated=signal is not None,
                    hold_reason=_hold_reason,
                    funding_data=_rl_funding,
                )
            except Exception as e:
                logger.debug(f"RL candle log: {e}")

            if signal is None:
                await asyncio.sleep(1)
                continue

            # 반대 시그널 → 청산
            if sniper.active_trade and sniper.check_reverse_signal(signal):
                logger.info(f"REVERSE → {signal.direction.value}")
                try:
                    ticker = await exchange.fetch_ticker(symbol)
                    price = float(ticker["last"])
                except Exception:
                    price = last_close
                await _close_position(
                    exchange, symbol, sniper, cfg, position_qty,
                    price, "REVERSE", telegram, dry_run, logger,
                    rl_logger=rl_logger, df=df,
                    rl_account=_rl_account, rl_funding=_rl_funding,
                )
                position_qty = 0.0

            # 신규 진입
            if sniper.active_trade is None:
                trade = sniper.open_trade(signal)
                emoji = "🟢" if signal.direction == Direction.LONG else "🔴"
                logger.info(
                    f"{emoji} {signal.direction.value} {short_sym} @ {signal.entry_price:.4f} "
                    f"| SL={signal.sl_price:.4f} TP1={signal.tp1_price:.4f} "
                    f"TP2={signal.tp2_price:.4f} TP3={signal.tp3_price:.4f}"
                )

                if not dry_run:
                    # 잔고 재조회 (다른 심볼 포지션 반영)
                    try:
                        bal = await exchange.fetch_balance()
                        balance = float(bal.get("USDT", {}).get("free", 0))
                    except Exception:
                        pass
                    position_qty = await _open_position(
                        exchange, symbol, signal, cfg, balance, logger,
                    )
                    # 주문 실패 시 내부 포지션 롤백 (고스트 포지션 방지)
                    if position_qty == 0.0:
                        logger.warning(f"Order failed — rolling back active_trade")
                        sniper.close_trade("ORDER_FAILED")
                        continue

                _log_trade({
                    "event": "ENTRY", "symbol": symbol,
                    "strategy": "funding_contrarian" if is_funding_strategy else ("vwap_momentum" if is_vwap_strategy else "ema_cross"),
                    "direction": signal.direction.value,
                    "price": signal.entry_price,
                    "sl": signal.sl_price,
                    "tp1": signal.tp1_price, "tp2": signal.tp2_price, "tp3": signal.tp3_price,
                    "risk": signal.risk, "rsi": signal.rsi, "adx": signal.adx,
                    "time": datetime.now(timezone.utc).isoformat(),
                }, cfg.log_dir)

                # RL 진입 로그
                try:
                    rl_logger.log_entry(df, signal, trade, _rl_account, _rl_funding)
                except Exception as e:
                    logger.debug(f"RL entry log: {e}")

                if telegram:
                    strat = "💰 FUNDING" if is_funding_strategy else ("📈 VWAP" if is_vwap_strategy else "📊 EMA")
                    sl_pct = abs(signal.entry_price - signal.sl_price) / signal.entry_price * 100
                    tp1_pct = abs(signal.tp1_price - signal.entry_price) / signal.entry_price * 100
                    msg = (
                        f"<b>{emoji} {signal.direction.value} {short_sym} {strat}</b>\n"
                        f"\n<b>진입 사유:</b>\n{signal.reason_text}\n"
                        f"\n<b>포지션:</b>\n"
                        f"Entry: {signal.entry_price:.4f}\n"
                        f"SL: {signal.sl_price:.4f} (-{sl_pct:.2f}%)\n"
                        f"TP1: {signal.tp1_price:.4f} (+{tp1_pct:.2f}%)\n"
                        f"TP2: {signal.tp2_price:.4f}\n"
                        f"TP3: {signal.tp3_price:.4f}\n"
                        f"Risk: {signal.risk:.4f} | {cfg.leverage}x | 잔고 {cfg.balance_ratio:.0%}"
                    )
                    telegram._send(msg)

            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(10)


async def _open_position(exchange, symbol, signal, cfg, balance, logger) -> float:
    side = "buy" if signal.direction.value == "LONG" else "sell"
    usdt_amount = balance * cfg.balance_ratio * cfg.leverage
    qty = usdt_amount / signal.entry_price
    qty = float(exchange.amount_to_precision(symbol, qty))
    logger.info(f"Opening {side} {qty} {symbol.split('/')[0]}")

    # 매 주문 전 시간 동기화 + 최대 2회 재시도 (WSL 시계 드리프트 대응)
    for attempt in range(3):
        try:
            # 항상 시간차 재계산 (WSL 드리프트 누적 방지)
            await exchange.load_time_difference()
            if attempt > 0:
                logger.info(f"Retry {attempt}/2 after time sync (offset={exchange.options.get('timeDifference', '?')}ms)")
            order = await exchange.create_market_order(symbol, side, qty)
            logger.info(f"Filled: {order.get('id', '?')}")
            return qty
        except Exception as e:
            if attempt < 2 and "1021" in str(e):
                logger.warning(f"Timestamp error, retrying... ({e})")
                await asyncio.sleep(1)
                continue
            logger.error(f"Order failed: {e}")
            return 0.0
    return 0.0


async def _close_position(exchange, symbol, sniper, cfg, qty, price, reason, telegram, dry_run, logger,
                          rl_logger=None, df=None, rl_account=None, rl_funding=None):
    trade = sniper.close_trade(reason)
    if trade is None:
        logger.error(f"close_trade returned None! reason={reason} — Binance position may be orphaned")
        return

    short_sym = symbol.split("/")[0]
    if trade.direction.value == "LONG":
        pnl_pct = (price - trade.entry_price) / trade.entry_price * 100 * cfg.leverage
    else:
        pnl_pct = (trade.entry_price - price) / trade.entry_price * 100 * cfg.leverage

    hold_sec = (datetime.now(timezone.utc) - trade.entry_time).total_seconds()
    emoji = "✅" if pnl_pct > 0 else "❌"

    logger.info(
        f"{emoji} CLOSE {short_sym} | {reason} | PnL: {pnl_pct:+.2f}% "
        f"| {hold_sec/60:.0f}min | {trade.entry_price:.2f}→{price:.2f}"
    )

    if not dry_run and qty > 0:
        close_side = "sell" if trade.direction.value == "LONG" else "buy"
        for attempt in range(3):
            try:
                await exchange.load_time_difference()
                await exchange.create_market_order(symbol, close_side, qty)
                break
            except Exception as e:
                if attempt < 2 and "1021" in str(e):
                    logger.warning(f"Close timestamp error, retrying... ({e})")
                    await asyncio.sleep(1)
                    continue
                logger.error(f"Close failed: {e}")
                break

    _log_trade({
        "event": "EXIT", "symbol": symbol,
        "direction": trade.direction.value,
        "entry_price": trade.entry_price, "exit_price": price,
        "reason": reason, "pnl_pct": round(pnl_pct, 2),
        "hold_min": round(hold_sec / 60, 1),
        "tp1_hit": trade.tp1_hit, "tp2_hit": trade.tp2_hit, "tp3_hit": trade.tp3_hit,
        "time": datetime.now(timezone.utc).isoformat(),
    }, cfg.log_dir)

    # RL 청산 로그
    if rl_logger and df is not None:
        try:
            rl_logger.log_exit(df, trade, price, reason, pnl_pct,
                               rl_account or {}, rl_funding)
        except Exception as e:
            logger.debug(f"RL exit log: {e}")

    if telegram:
        # 청산 사유 상세화
        if reason == "SL":
            reason_detail = "손절 (SL 도달)"
        elif reason == "REVERSE":
            reason_detail = "반대 시그널 → 반전 청산"
        elif reason.startswith("PP_"):
            reason_detail = f"수익보호 ({reason}: 최고점 대비 하락)"
        elif reason.startswith("TRAIL_TP"):
            tp_num = reason.replace("TRAIL_TP", "").replace("_HIT", "")
            reason_detail = f"트레일링 (TP{tp_num} 이후 되돌림)"
        else:
            reason_detail = reason

        # TP 달성 표시
        tp_status = ""
        if trade.tp1_hit or trade.tp2_hit or trade.tp3_hit:
            hit_list = []
            if trade.tp1_hit: hit_list.append("TP1")
            if trade.tp2_hit: hit_list.append("TP2")
            if trade.tp3_hit: hit_list.append("TP3")
            tp_status = f"\n달성: {', '.join(hit_list)}"

        # peak R 표시
        peak_info = ""
        if trade.peak_r > 0:
            peak_info = f"\n최고 미실현: {trade.peak_r:.1f}R ({trade.peak_r * abs(trade.entry_price - trade.sl_price) / trade.entry_price * 100 * cfg.leverage:+.2f}%)"

        # 보유시간 포맷
        if hold_sec < 3600:
            hold_str = f"{hold_sec/60:.0f}분"
        else:
            hold_str = f"{hold_sec/3600:.1f}시간"

        msg = (
            f"<b>{emoji} CLOSE {short_sym} — {reason_detail}</b>\n"
            f"\nPnL: <b>{pnl_pct:+.2f}%</b>"
            f"{peak_info}"
            f"{tp_status}"
            f"\n{trade.entry_price:.4f} → {price:.4f}"
            f"\n보유: {hold_str} | {trade.direction.value}"
        )
        telegram._send(msg)


async def main(symbols: list[str], dry_run: bool):
    import ccxt.async_support as ccxt_async

    keys = load_keys()
    binance_cfg = keys.get("binance", {})
    tg_cfg = keys.get("telegram", {})

    exchange = ccxt_async.binance({
        "apiKey": binance_cfg.get("api_key", ""),
        "secret": binance_cfg.get("api_secret", ""),
        "options": {"defaultType": "future", "adjustForTimeDifference": True, "recvWindow": 60000},
        "enableRateLimit": True,
    })

    # Telegram
    telegram = None
    try:
        from src.notifications.telegram_notifier import TelegramNotifier
        chat_id = str(tg_cfg.get("chat_id", ""))
        if not chat_id:
            bcast = tg_cfg.get("broadcast_chat_ids", [])
            chat_id = str(bcast[0]) if bcast else ""
        telegram = TelegramNotifier(
            bot_token=tg_cfg.get("bot_token", ""),
            chat_id=chat_id,
            enabled=bool(tg_cfg.get("bot_token")) and not dry_run,
        )
    except Exception as e:
        logging.warning(f"Telegram init failed: {e}")

    # WSL 시계 동기화
    if not dry_run:
        try:
            from src.execution.time_sync import sync_time_offset
            sync_time_offset(exchange)
        except Exception:
            pass

    # 마켓 로드 (1회)
    if not dry_run:
        try:
            await exchange.load_markets()
        except Exception:
            pass

    logging.info(f"Sniper V2 — {', '.join(s.split('/')[0] for s in symbols)} {'[DRY]' if dry_run else '[LIVE]'}")

    try:
        # 각 심볼을 별도 태스크로 동시 실행
        tasks = []
        for symbol in symbols:
            cfg = CONFIGS.get(symbol)
            if cfg is None:
                logging.error(f"No config for {symbol}, skipping")
                continue
            tasks.append(run_symbol(exchange, symbol, cfg, telegram, dry_run))

        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        await exchange.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sniper V2 Daemon")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--multi", action="store_true", help="BTC + SOL + XRP 동시 실행")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.multi:
        symbols = ["BTC/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT"]
    else:
        symbols = [args.symbol]

    asyncio.run(main(symbols=symbols, dry_run=args.dry_run))
