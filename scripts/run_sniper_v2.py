#!/usr/bin/env python3
"""
Sniper V2 Daemon

EMA(18/40) cross + EMA(200) trend + RSI(14) 55/40
+ VolFilter + ADX + Cooldown 방어 필터

Usage:
    python scripts/run_sniper_v2.py                          # BTC 기본
    python scripts/run_sniper_v2.py --symbol ETH/USDT:USDT   # ETH
    python scripts/run_sniper_v2.py --dry-run                 # 모의 실행
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
    BALANCE_USAGE_RATIO,
    CHECK_INTERVAL_SEC,
    LEVERAGE,
    LOG_DIR,
    SYMBOL,
    TIMEFRAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sniper_v2")


def load_keys() -> dict:
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"
    with open(keys_path) as f:
        return yaml.safe_load(f)


def _log_trade(data: dict):
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_dir / f"{date_str}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(data, default=str) + "\n")


def _tf_to_seconds(tf: str) -> int:
    unit = tf[-1]
    num = int(tf[:-1])
    if unit == "m": return num * 60
    if unit == "h": return num * 3600
    if unit == "d": return num * 86400
    return 300


async def main_loop(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    dry_run: bool = False,
):
    import ccxt.async_support as ccxt_async
    import pandas as pd
    from src.sniper_v2.strategy import SniperV2, Direction

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
    sniper = SniperV2()
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
            await exchange.fapiPrivatePostLeverage({"symbol": base, "leverage": LEVERAGE})
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

    # --- 마켓 로드 (1회) ---
    if not dry_run:
        try:
            await exchange.load_markets()
        except Exception:
            pass

    logger.info("=" * 60)
    logger.info(f"Sniper V2 {'[DRY RUN]' if dry_run else '[LIVE]'}")
    logger.info(f"Symbol: {short_sym} | TF: {timeframe}")
    logger.info(f"Balance: ${balance:.2f} | Leverage: {LEVERAGE}x")
    logger.info(f"Filters: VolFilter=2.0, ADX>=20, Cooldown=20bars")
    logger.info("=" * 60)

    tf_seconds = _tf_to_seconds(timeframe)
    last_candle_time = 0
    position_qty = 0.0

    try:
        while True:
            now = time.time()
            current_candle_start = int(now // tf_seconds) * tf_seconds

            if current_candle_start <= last_candle_time:
                # 포지션 있으면 TP/SL 체크
                if sniper.active_trade:
                    try:
                        ticker = await exchange.fetch_ticker(symbol)
                        price = float(ticker["last"])
                        exit_reason = sniper.check_exit(price)
                        if exit_reason:
                            await _close_position(
                                exchange, symbol, sniper, position_qty,
                                price, exit_reason, telegram, dry_run,
                            )
                            position_qty = 0.0
                    except Exception as e:
                        logger.error(f"Price check error: {e}")

                await asyncio.sleep(CHECK_INTERVAL_SEC)
                continue

            last_candle_time = current_candle_start

            # --- OHLCV ---
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=300)
            except Exception as e:
                logger.error(f"OHLCV fetch error: {e}")
                await asyncio.sleep(10)
                continue

            df = _ohlcv_to_df(ohlcv)
            if len(df) < 220:
                logger.warning(f"Insufficient data: {len(df)} bars")
                await asyncio.sleep(10)
                continue

            # --- 시그널 ---
            signal = sniper.compute(df)

            last_close = float(df["close"].iloc[-1])
            pos_str = "FLAT"
            if sniper.active_trade:
                pos_str = sniper.active_trade.direction.value
            sig_str = f" sig={signal.direction.value} RSI={signal.rsi:.0f} ADX={signal.adx:.0f}" if signal else ""
            logger.info(f"[{short_sym}] close={last_close:.2f} pos={pos_str}{sig_str}")

            if signal is None:
                await asyncio.sleep(1)
                continue

            # --- 반대 시그널 → 청산 ---
            if sniper.active_trade and sniper.check_reverse_signal(signal):
                logger.info(f"REVERSE: {signal.direction.value}")
                try:
                    ticker = await exchange.fetch_ticker(symbol)
                    price = float(ticker["last"])
                except Exception:
                    price = last_close
                await _close_position(
                    exchange, symbol, sniper, position_qty,
                    price, "REVERSE", telegram, dry_run,
                )
                position_qty = 0.0

            # --- 신규 진입 ---
            if sniper.active_trade is None:
                trade = sniper.open_trade(signal)
                d_emoji = "🟢" if signal.direction == Direction.LONG else "🔴"
                logger.info(
                    f"{d_emoji} {signal.direction.value} {short_sym} @ {signal.entry_price:.2f} "
                    f"| SL={signal.sl_price:.2f} TP1={signal.tp1_price:.2f} "
                    f"TP2={signal.tp2_price:.2f} TP3={signal.tp3_price:.2f} "
                    f"| RSI={signal.rsi:.0f} ADX={signal.adx:.0f}"
                )

                if not dry_run:
                    position_qty = await _open_position(exchange, symbol, signal, balance)

                _log_trade({
                    "event": "ENTRY",
                    "symbol": symbol,
                    "direction": signal.direction.value,
                    "price": signal.entry_price,
                    "sl": signal.sl_price,
                    "tp1": signal.tp1_price,
                    "tp2": signal.tp2_price,
                    "tp3": signal.tp3_price,
                    "risk": signal.risk,
                    "rsi": signal.rsi,
                    "adx": signal.adx,
                    "atr": signal.atr,
                    "time": datetime.now(timezone.utc).isoformat(),
                })

                if telegram:
                    msg = (
                        f"<b>{d_emoji} {signal.direction.value} {short_sym}</b>\n"
                        f"Entry: {signal.entry_price:.2f}\n"
                        f"SL: {signal.sl_price:.2f} | TP1: {signal.tp1_price:.2f}\n"
                        f"TP2: {signal.tp2_price:.2f} | TP3: {signal.tp3_price:.2f}\n"
                        f"RSI: {signal.rsi:.0f} | ADX: {signal.adx:.0f}\n"
                        f"Risk: {signal.risk:.2f} ({signal.risk/signal.entry_price*100:.2f}%)"
                    )
                    telegram._send(msg)

            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await exchange.close()


async def _open_position(exchange, symbol: str, signal, balance: float) -> float:
    try:
        side = "buy" if signal.direction.value == "LONG" else "sell"
        usdt_amount = balance * BALANCE_USAGE_RATIO * LEVERAGE
        qty = usdt_amount / signal.entry_price
        qty = float(exchange.amount_to_precision(symbol, qty))
        logger.info(f"Opening {side} {qty} {symbol.split('/')[0]}")
        order = await exchange.create_market_order(symbol, side, qty)
        logger.info(f"Order filled: {order.get('id', '?')}")
        return qty
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return 0.0


async def _close_position(
    exchange, symbol: str, sniper, qty: float,
    price: float, reason: str, telegram, dry_run: bool,
):
    trade = sniper.close_trade()
    if trade is None:
        return

    short_sym = symbol.split("/")[0]
    if trade.direction.value == "LONG":
        pnl_pct = (price - trade.entry_price) / trade.entry_price * 100 * LEVERAGE
    else:
        pnl_pct = (trade.entry_price - price) / trade.entry_price * 100 * LEVERAGE

    hold_sec = (datetime.now(timezone.utc) - trade.entry_time).total_seconds()
    emoji = "✅" if pnl_pct > 0 else "❌"

    logger.info(
        f"{emoji} CLOSE {short_sym} | {reason} | PnL: {pnl_pct:+.2f}% "
        f"| Hold: {hold_sec/60:.0f}min | {trade.entry_price:.2f} → {price:.2f}"
    )

    if not dry_run and qty > 0:
        try:
            close_side = "sell" if trade.direction.value == "LONG" else "buy"
            order = await exchange.create_market_order(symbol, close_side, qty)
            logger.info(f"Close order: {order.get('id', '?')}")
        except Exception as e:
            logger.error(f"Close order failed: {e}")

    _log_trade({
        "event": "EXIT",
        "symbol": symbol,
        "direction": trade.direction.value,
        "entry_price": trade.entry_price,
        "exit_price": price,
        "reason": reason,
        "pnl_pct": round(pnl_pct, 2),
        "hold_min": round(hold_sec / 60, 1),
        "tp1_hit": trade.tp1_hit,
        "tp2_hit": trade.tp2_hit,
        "tp3_hit": trade.tp3_hit,
        "time": datetime.now(timezone.utc).isoformat(),
    })

    if telegram:
        msg = (
            f"<b>{emoji} CLOSE {short_sym} — {reason}</b>\n"
            f"PnL: {pnl_pct:+.2f}%\n"
            f"Entry: {trade.entry_price:.2f} → Exit: {price:.2f}\n"
            f"Hold: {hold_sec/60:.0f}min\n"
            f"TP: {'TP1 ' if trade.tp1_hit else ''}{'TP2 ' if trade.tp2_hit else ''}{'TP3' if trade.tp3_hit else ''}"
        )
        telegram._send(msg)


def _ohlcv_to_df(ohlcv: list):
    import pandas as pd
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sniper V2 Daemon")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--tf", default=TIMEFRAME)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(main_loop(symbol=args.symbol, timeframe=args.tf, dry_run=args.dry_run))
