#!/usr/bin/env python
"""
3. Run Trading Script

실전 트레이딩 실행 스크립트.
Crontab 또는 Windows Task Scheduler로 스케줄링.

Usage:
    python scripts/3_run_trading.py
    python scripts/3_run_trading.py --paper  # 모의투자
    python scripts/3_run_trading.py --dry-run  # 신호만 생성, 실제 주문 X
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TRADING,
)
from config.logging_config import setup_logging

logger = setup_logging("trading")


def load_keys():
    """Load API keys."""
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"

    if not keys_path.exists():
        logger.error("keys.yaml not found! Copy from keys.example.yaml")
        return None

    with open(keys_path) as f:
        return yaml.safe_load(f)


def load_ensemble():
    """Load trained ensemble model."""
    ensemble_path = MODELS_DIR / "weights" / "ensemble.pkl"

    if not ensemble_path.exists():
        logger.error("Ensemble not found. Run 2_train_ensemble.py first.")
        return None

    with open(ensemble_path, "rb") as f:
        return pickle.load(f)


def load_latest_data():
    """Load latest market data."""
    prices_path = PROCESSED_DATA_DIR / "prices.parquet"
    features_path = PROCESSED_DATA_DIR / "features.parquet"

    if not prices_path.exists():
        logger.error("No price data. Run 1_update_data.py first.")
        return None, None

    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])

    features = None
    if features_path.exists():
        features = pd.read_parquet(features_path)
        features["date"] = pd.to_datetime(features["date"])

    return prices, features


def initialize_broker(keys: dict, is_paper: bool = True):
    """Initialize KIS broker."""
    from src.execution import KISApi, KISAuth, OrderManager

    kis_keys = keys.get("kis", {})

    auth = KISAuth(
        app_key=kis_keys.get("app_key", ""),
        app_secret=kis_keys.get("app_secret", ""),
        account_number=kis_keys.get("account_number", ""),
        is_paper=is_paper,
    )

    api = KISApi(auth)
    order_manager = OrderManager(api)

    return order_manager


def initialize_notifier(keys: dict):
    """Initialize Telegram notifier."""
    from src.execution import TelegramNotifier

    telegram_keys = keys.get("telegram", {})
    bot_token = telegram_keys.get("bot_token", "")
    chat_id = telegram_keys.get("chat_id", "")

    if not bot_token or not chat_id or "YOUR" in bot_token:
        logger.warning("Telegram not configured")
        return None

    notifier = TelegramNotifier(
        bot_token=bot_token,
        chat_id=chat_id,
    )

    # Send startup notification
    notifier.send_startup()

    return notifier


def run_trading_cycle(
    ensemble,
    order_manager,
    notifier,
    prices: pd.DataFrame,
    features: pd.DataFrame | None,
    dry_run: bool = False,
):
    """Run one trading cycle."""
    from src.ensemble import RiskParityAllocator

    today = datetime.now().date()
    logger.info(f"Running trading cycle for {today}")

    # Generate signals
    signals = ensemble.generate_signals(
        date=datetime.now(),
        prices=prices,
        features=features,
    )

    logger.info(f"Generated signals for {len(signals.signals)} assets")
    logger.info(f"Strategy weights: {signals.strategy_weights}")

    # Top signals
    top_signals = signals.signals.nlargest(10, "score")
    logger.info(f"Top signals:\n{top_signals}")

    # Allocate
    allocator = RiskParityAllocator(
        top_k=TRADING["max_positions"],
        max_weight=TRADING["max_position_weight"],
    )

    target_weights = allocator.allocate(signals.signals, prices)
    logger.info(f"Target weights:\n{target_weights}")

    # Send signal alert
    if notifier:
        stocks = [
            {"stock_code": row["asset_id"], "name": row["asset_id"], "score": row["score"]}
            for _, row in top_signals.iterrows()
        ]
        notifier.send_signal_alert(
            strategy="Ensemble",
            stocks=stocks,
            regime=signals.regime,
        )

    if dry_run:
        logger.info("DRY RUN - No orders placed")
        return {"status": "dry_run", "signals": signals, "weights": target_weights}

    # Execute rebalancing
    try:
        orders = order_manager.execute_rebalance(target_weights)
        logger.info(f"Placed {len(orders)} orders")

        # Send trade alerts
        if notifier:
            for order in orders:
                if order.status.value == "submitted":
                    notifier.send_trade_alert(
                        stock_code=order.stock_code,
                        stock_name=order.stock_code,
                        side=order.side,
                        quantity=order.quantity,
                        price=order.price or 0,
                        strategy="Ensemble",
                    )

        return {"status": "success", "orders": orders}

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        if notifier:
            notifier.send_error(str(e), "Order execution")
        return {"status": "error", "error": str(e)}


def send_daily_report(order_manager, notifier):
    """Send end-of-day report."""
    if not notifier:
        return

    try:
        balance = order_manager.api.get_balance()

        notifier.send_daily_summary(
            date=datetime.now(),
            total_value=balance["total_eval"],
            daily_pnl=balance["total_profit_loss"],
            daily_return=balance["total_profit_loss"] / balance["total_eval"] * 100 if balance["total_eval"] > 0 else 0,
            positions=balance["holdings"],
            trades_count=len(order_manager.get_order_history()),
        )
    except Exception as e:
        logger.error(f"Failed to send daily report: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run live trading")

    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Use paper trading (default)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real trading (caution!)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate signals only, no orders",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Send daily report only",
    )

    args = parser.parse_args()

    is_paper = not args.real

    logger.info("=" * 60)
    logger.info("Trading Script Started")
    logger.info(f"Time: {datetime.now()}")
    logger.info(f"Mode: {'PAPER' if is_paper else '⚠️ REAL'}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info("=" * 60)

    # Load configuration
    keys = load_keys()
    if keys is None:
        sys.exit(1)

    # Initialize broker
    order_manager = initialize_broker(keys, is_paper)

    # Initialize notifier
    notifier = initialize_notifier(keys)

    if args.report_only:
        send_daily_report(order_manager, notifier)
        return

    # Load ensemble
    ensemble = load_ensemble()
    if ensemble is None:
        sys.exit(1)

    logger.info(f"Loaded ensemble with {len(ensemble.strategies)} strategies")

    # Load data
    prices, features = load_latest_data()
    if prices is None:
        sys.exit(1)

    logger.info(f"Loaded {len(prices)} price records")

    # Run trading
    result = run_trading_cycle(
        ensemble=ensemble,
        order_manager=order_manager,
        notifier=notifier,
        prices=prices,
        features=features,
        dry_run=args.dry_run,
    )

    logger.info(f"Trading result: {result['status']}")

    # End of day report
    if not args.dry_run:
        send_daily_report(order_manager, notifier)

    logger.info("=" * 60)
    logger.info("Trading Script Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
