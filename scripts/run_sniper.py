#!/usr/bin/env python3
"""
Precision Sniper Daemon

단일 코인 전용 트레이딩 봇.
EMA cross + Confluence scoring + ATR-based TP/SL/Trailing.

Usage:
    python scripts/run_sniper.py                         # BTC 기본
    python scripts/run_sniper.py --symbol ETH/USDT:USDT  # ETH
    python scripts/run_sniper.py --dry-run               # 모의 실행
    python scripts/run_sniper.py --tf 15m --htf 4h       # 15분봉 + 4h HTF
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

from src.sniper.config import (
    CHECK_INTERVAL_SEC,
    LEVERAGE,
    BALANCE_USAGE_RATIO,
    LOG_DIR,
    SNIPER_HTF,
    SNIPER_SYMBOL,
    SNIPER_TIMEFRAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sniper")


def load_keys() -> dict:
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"
    with open(keys_path) as f:
        return yaml.safe_load(f)


def _log_trade(data: dict):
    """거래 로그 기록."""
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_dir / f"{date_str}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(data, default=str) + "\n")


async def main_loop(
    symbol: str = SNIPER_SYMBOL,
    timeframe: str = SNIPER_TIMEFRAME,
    htf: str = SNIPER_HTF,
    dry_run: bool = False,
):
    import ccxt.async_support as ccxt_async
    import pandas as pd

    from src.sniper.strategy import PrecisionSniper, Direction

    keys = load_keys()
    binance_cfg = keys.get("binance", {})
    tg_cfg = keys.get("telegram", {})

    # --- Exchange ---
    exchange = ccxt_async.binance({
        "apiKey": binance_cfg.get("api_key", ""),
        "secret": binance_cfg.get("api_secret", ""),
        "options": {
            "defaultType": "future",
            "adjustForTimeDifference": True,
            "recvWindow": 60000,
        },
        "enableRateLimit": True,
    })

    # --- Telegram ---
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
        logger.warning(f"Telegram init failed: {e}")

    # --- Strategy ---
    sniper = PrecisionSniper()
    short_sym = symbol.split("/")[0]

    # --- WSL 시계 동기화 ---
    if not dry_run:
        try:
            from src.execution.time_sync import sync_time_offset
            sync_time_offset(exchange)
        except Exception:
            pass

    # --- 레버리지 설정 ---
    if not dry_run:
        try:
            base = symbol.replace("/USDT:USDT", "USDT")
            await exchange.fapiPrivatePostLeverage({
                "symbol": base,
                "leverage": LEVERAGE,
            })
            logger.info(f"Leverage set to {LEVERAGE}x for {short_sym}")
        except Exception as e:
            logger.warning(f"Leverage set failed: {e}")

    # --- 잔고 ---
    balance = 1000.0
    if not dry_run:
        try:
            bal = await exchange.fetch_balance()
            balance = float(bal.get("USDT", {}).get("total", 0))
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
            return

    # --- 실제 포지션 동기화 ---
    has_position = False
    if not dry_run:
        try:
            positions = await exchange.fetch_positions([symbol])
            for p in positions:
                amt = float(p.get("contracts", 0))
                if amt > 0:
                    has_position = True
                    logger.info(f"Existing position found: {p['side']} {amt}")
        except Exception:
            pass

    logger.info("=" * 60)
    logger.info(f"Precision Sniper {'[DRY RUN]' if dry_run else '[LIVE]'}")
    logger.info(f"Symbol: {short_sym} | TF: {timeframe} | HTF: {htf}")
    logger.info(f"Balance: ${balance:.2f} | Leverage: {LEVERAGE}x")
    logger.info("=" * 60)

    # --- 타임프레임 → 초 변환 ---
    tf_seconds = _tf_to_seconds(timeframe)

    # --- 메인 루프 ---
    last_candle_time = 0
    position_qty = 0.0

    try:
        while True:
            now = time.time()

            # --- 캔들 완성 대기 ---
            # 매 캔들 클로즈 직후에 시그널 계산
            current_candle_start = int(now // tf_seconds) * tf_seconds
            if current_candle_start <= last_candle_time:
                # 포지션이 있으면 5초마다 TP/SL 체크
                if sniper.active_trade and not dry_run:
                    try:
                        ticker = await exchange.fetch_ticker(symbol)
                        price = float(ticker["last"])
                        exit_reason = sniper.check_exit(price)
                        if exit_reason:
                            await _close_position(
                                exchange, symbol, sniper, position_qty,
                                price, exit_reason, balance, telegram,
                                dry_run,
                            )
                            position_qty = 0.0
                    except Exception as e:
                        logger.error(f"Price check error: {e}")

                await asyncio.sleep(CHECK_INTERVAL_SEC)
                continue

            last_candle_time = current_candle_start

            # --- OHLCV 데이터 가져오기 ---
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=200)
                htf_ohlcv = await exchange.fetch_ohlcv(symbol, htf, limit=100)
            except Exception as e:
                logger.error(f"OHLCV fetch error: {e}")
                await asyncio.sleep(10)
                continue

            df = _ohlcv_to_df(ohlcv)
            htf_df = _ohlcv_to_df(htf_ohlcv)

            if len(df) < 60:
                logger.warning(f"Insufficient data: {len(df)} bars")
                await asyncio.sleep(10)
                continue

            # --- 시그널 계산 ---
            signal = sniper.compute(df, htf_df)

            last_close = float(df["close"].iloc[-1])
            logger.info(
                f"[{short_sym}] close={last_close:.2f} "
                f"pos={'LONG' if sniper.active_trade and sniper.active_trade.direction.value == 'LONG' else 'SHORT' if sniper.active_trade else 'FLAT'}"
                + (f" signal={signal.direction.value} score={signal.score:.1f}" if signal else "")
            )

            if signal is None:
                await asyncio.sleep(1)
                continue

            # --- 반대 시그널 → 청산 ---
            if sniper.active_trade and sniper.check_reverse_signal(signal):
                logger.info(f"REVERSE signal: {signal.direction.value} — closing current position")
                try:
                    ticker = await exchange.fetch_ticker(symbol)
                    price = float(ticker["last"])
                except Exception:
                    price = last_close

                await _close_position(
                    exchange, symbol, sniper, position_qty,
                    price, "REVERSE", balance, telegram, dry_run,
                )
                position_qty = 0.0
                # 바로 새 포지션 진입

            # --- 신규 진입 ---
            if sniper.active_trade is None:
                trade = sniper.open_trade(signal)
                logger.info(
                    f"{'🟢' if signal.direction == Direction.LONG else '🔴'} "
                    f"{signal.direction.value} {short_sym} @ {signal.entry_price:.2f} "
                    f"| score={signal.score:.1f} "
                    f"| SL={signal.sl_price:.2f} TP1={signal.tp1_price:.2f} "
                    f"TP2={signal.tp2_price:.2f} TP3={signal.tp3_price:.2f}"
                )

                if not dry_run:
                    position_qty = await _open_position(
                        exchange, symbol, signal, balance,
                    )

                # 로그 기록
                _log_trade({
                    "event": "ENTRY",
                    "symbol": symbol,
                    "direction": signal.direction.value,
                    "price": signal.entry_price,
                    "score": signal.score,
                    "sl": signal.sl_price,
                    "tp1": signal.tp1_price,
                    "tp2": signal.tp2_price,
                    "tp3": signal.tp3_price,
                    "risk": signal.risk,
                    "details": signal.details,
                    "time": datetime.now(timezone.utc).isoformat(),
                })

                # 텔레그램 알림
                if telegram:
                    emoji = "🟢" if signal.direction == Direction.LONG else "🔴"
                    msg = (
                        f"<b>{emoji} {signal.direction.value} {short_sym}</b>\n"
                        f"Entry: {signal.entry_price:.2f}\n"
                        f"Score: {signal.score:.1f}/10\n"
                        f"SL: {signal.sl_price:.2f} | TP1: {signal.tp1_price:.2f}\n"
                        f"TP2: {signal.tp2_price:.2f} | TP3: {signal.tp3_price:.2f}\n"
                        f"Risk: {signal.risk:.2f} ({signal.risk/signal.entry_price*100:.2f}%)"
                    )
                    telegram._send(msg)

            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await exchange.close()


async def _open_position(exchange, symbol: str, signal, balance: float) -> float:
    """시장가 진입."""
    try:
        side = "buy" if signal.direction.value == "LONG" else "sell"
        usdt_amount = balance * BALANCE_USAGE_RATIO * LEVERAGE
        qty = usdt_amount / signal.entry_price

        # LOT_SIZE 정밀도
        try:
            await exchange.load_markets()
            qty = float(exchange.amount_to_precision(symbol, qty))
        except Exception:
            pass

        logger.info(f"Opening {side} {qty} {symbol.split('/')[0]}")
        order = await exchange.create_market_order(symbol, side, qty)
        logger.info(f"Order filled: {order.get('id', '?')}")
        return qty
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return 0.0


async def _close_position(
    exchange, symbol: str, sniper, qty: float,
    price: float, reason: str, balance: float,
    telegram, dry_run: bool,
):
    """포지션 청산."""
    trade = sniper.close_trade()
    if trade is None:
        return

    short_sym = symbol.split("/")[0]

    # PnL 계산
    if trade.direction.value == "LONG":
        pnl_pct = (price - trade.entry_price) / trade.entry_price * 100 * LEVERAGE
    else:
        pnl_pct = (trade.entry_price - price) / trade.entry_price * 100 * LEVERAGE

    hold_sec = (datetime.now(timezone.utc) - trade.entry_time).total_seconds()

    logger.info(
        f"{'✅' if pnl_pct > 0 else '❌'} CLOSE {short_sym} | {reason} "
        f"| PnL: {pnl_pct:+.2f}% | Hold: {hold_sec/60:.0f}min "
        f"| Entry: {trade.entry_price:.2f} → Exit: {price:.2f}"
    )

    if not dry_run and qty > 0:
        try:
            close_side = "sell" if trade.direction.value == "LONG" else "buy"
            order = await exchange.create_market_order(symbol, close_side, qty)
            logger.info(f"Close order: {order.get('id', '?')}")
        except Exception as e:
            logger.error(f"Close order failed: {e}")

    # 로그 기록
    _log_trade({
        "event": "EXIT",
        "symbol": symbol,
        "direction": trade.direction.value,
        "entry_price": trade.entry_price,
        "exit_price": price,
        "reason": reason,
        "pnl_pct": round(pnl_pct, 2),
        "hold_min": round(hold_sec / 60, 1),
        "entry_score": trade.entry_score,
        "status": trade.status.value,
        "peak_price": trade.peak_price,
        "tp1_hit": trade.tp1_hit,
        "tp2_hit": trade.tp2_hit,
        "tp3_hit": trade.tp3_hit,
        "time": datetime.now(timezone.utc).isoformat(),
    })

    # 텔레그램
    if telegram:
        emoji = "✅" if pnl_pct > 0 else "❌"
        msg = (
            f"<b>{emoji} CLOSE {short_sym} — {reason}</b>\n"
            f"PnL: {pnl_pct:+.2f}%\n"
            f"Entry: {trade.entry_price:.2f} → Exit: {price:.2f}\n"
            f"Hold: {hold_sec/60:.0f}min\n"
            f"TP hits: {'TP1 ' if trade.tp1_hit else ''}{'TP2 ' if trade.tp2_hit else ''}{'TP3' if trade.tp3_hit else ''}"
        )
        telegram._send(msg)


def _ohlcv_to_df(ohlcv: list) -> pd.DataFrame:
    """ccxt OHLCV → pandas DataFrame."""
    import pandas as pd
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    return df


def _tf_to_seconds(tf: str) -> int:
    """타임프레임 문자열 → 초."""
    unit = tf[-1]
    num = int(tf[:-1])
    if unit == "m":
        return num * 60
    elif unit == "h":
        return num * 3600
    elif unit == "d":
        return num * 86400
    return 300  # 기본 5분


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precision Sniper Daemon")
    parser.add_argument("--symbol", default=SNIPER_SYMBOL, help="Trading symbol (e.g., BTC/USDT:USDT)")
    parser.add_argument("--tf", default=SNIPER_TIMEFRAME, help="Candle timeframe (e.g., 5m, 15m)")
    parser.add_argument("--htf", default=SNIPER_HTF, help="Higher timeframe (e.g., 1h, 4h)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    args = parser.parse_args()

    asyncio.run(main_loop(
        symbol=args.symbol,
        timeframe=args.tf,
        htf=args.htf,
        dry_run=args.dry_run,
    ))
