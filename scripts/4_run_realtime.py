#!/usr/bin/env python
"""
4. Run Real-time Trading with LLM

실시간 데이터 + LLM 앙상블 트레이딩.
KIS WebSocket으로 실시간 체결가/호가를 수신하고,
1분봉 수집 + N분 간격 LLM 앙상블 의사결정 실행.

Usage:
    python scripts/4_run_realtime.py              # Paper trading (기본)
    python scripts/4_run_realtime.py --paper       # Paper trading (명시)
    python scripts/4_run_realtime.py --real         # 실거래
    python scripts/4_run_realtime.py --dry-run      # 주문 없이 시그널만
    python scripts/4_run_realtime.py --no-llm       # 기존 앙상블만 (LLM 비활성)
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TRADING,
    SCHEDULE,
    ENSEMBLE,
    LLM_CONFIG,
    WEBSOCKET_CONFIG,
)
from config.logging_config import setup_logging

logger = setup_logging("realtime_trading")


def load_keys() -> dict | None:
    """Load API keys from config/keys.yaml."""
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"
    if not keys_path.exists():
        logger.error("keys.yaml not found! Copy from keys.example.yaml")
        return None
    with open(keys_path) as f:
        return yaml.safe_load(f)


class RealtimeTradingLoop:
    """
    Main trading loop:
    1. KIS WebSocket -> real-time ticks
    2. CandleAggregator -> 1분봉 수집 (고해상도 데이터)
    3. signal_interval마다 -> ensemble signals (LLM 호출)
    4. LLM orchestrator -> portfolio decision
    5. OrderManager -> execute trades
    6. ReasoningLogger -> JSON logs
    """

    def __init__(
        self,
        keys: dict,
        is_paper: bool = True,
        dry_run: bool = False,
        use_llm: bool = True,
    ):
        self.keys = keys
        self.is_paper = is_paper
        self.dry_run = dry_run
        self.use_llm = use_llm

        self._running = False
        self._processing = False  # 시그널 처리 중 락
        self._ws = None
        self._ensemble = None
        self._order_manager = None
        self._notifier = None
        self._allocator = None
        self._prices = None
        self._features = None
        self._universe: list[str] = []

        # 시그널 생성 주기 제어
        self._candle_count = 0
        self._signal_interval = WEBSOCKET_CONFIG.get("signal_interval_minutes", 5)
        self._candle_interval = WEBSOCKET_CONFIG.get("candle_interval_minutes", 1)
        # N분봉마다 시그널 = signal_interval / candle_interval
        self._candles_per_signal = max(1, self._signal_interval // self._candle_interval)
        self._last_signal_time: datetime | None = None

    def initialize(self) -> None:
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("Initializing Real-time Trading System")
        logger.info(f"  Paper: {self.is_paper}")
        logger.info(f"  Dry Run: {self.dry_run}")
        logger.info(f"  LLM: {self.use_llm}")
        logger.info(f"  Candle: {self._candle_interval}min (data collection)")
        logger.info(f"  Signal: {self._signal_interval}min (LLM decision)")
        logger.info("=" * 60)

        # 1. Load trained ensemble
        self._load_ensemble()

        # 2. Load latest price/feature data
        self._load_data()

        # 3. Initialize broker (KIS API + OrderManager)
        self._init_broker()

        # 4. Initialize WebSocket
        self._init_websocket()

        # 5. Initialize notifier
        self._init_notifier()

        # 6. Initialize allocator
        from src.ensemble.allocator import RiskParityAllocator
        self._allocator = RiskParityAllocator(
            top_k=TRADING["max_positions"],
            max_weight=TRADING["max_position_weight"],
        )

        logger.info("All components initialized")

    def _load_ensemble(self) -> None:
        """Load trained ensemble model."""
        from src.ensemble_models import ModelManager

        manager = ModelManager(MODELS_DIR)
        self._ensemble = manager.load_ensemble()
        logger.info(
            f"Loaded ensemble with {len(self._ensemble.strategies)} strategies: "
            f"{list(self._ensemble.strategies.keys())}"
        )

    def _load_data(self) -> None:
        """Load latest price and feature data."""
        prices_path = PROCESSED_DATA_DIR / "prices.parquet"
        features_path = PROCESSED_DATA_DIR / "features.parquet"

        if prices_path.exists():
            self._prices = pd.read_parquet(prices_path)
            self._prices["date"] = pd.to_datetime(self._prices["date"])
            logger.info(f"Loaded {len(self._prices)} price records")
        else:
            logger.error("prices.parquet not found!")
            sys.exit(1)

        if features_path.exists():
            self._features = pd.read_parquet(features_path)
            self._features["date"] = pd.to_datetime(self._features["date"])
            logger.info(f"Loaded {len(self._features)} feature records")

        # Get universe from most recent data
        latest_date = self._prices["date"].max()
        recent = self._prices[self._prices["date"] == latest_date]
        self._universe = recent["ticker"].tolist()[:100]
        logger.info(f"Universe: {len(self._universe)} stocks")

    def _init_broker(self) -> None:
        """Initialize KIS API and OrderManager."""
        from src.execution import KISApi, KISAuth, OrderManager

        kis_keys = self.keys.get("kis", {})
        auth = KISAuth(
            app_key=kis_keys.get("app_key", ""),
            app_secret=kis_keys.get("app_secret", ""),
            account_number=kis_keys.get("account_number", ""),
            is_paper=self.is_paper,
        )

        api = KISApi(auth)
        self._order_manager = OrderManager(api)
        logger.info(f"Broker initialized ({'paper' if self.is_paper else 'REAL'})")

    def _init_websocket(self) -> None:
        """Initialize KIS WebSocket."""
        from src.execution import KISWebSocket

        kis_keys = self.keys.get("kis", {})
        self._ws = KISWebSocket(
            app_key=kis_keys.get("app_key", ""),
            app_secret=kis_keys.get("app_secret", ""),
            is_paper=self.is_paper,
            candle_interval=self._candle_interval,
        )

        # Set candle completion callback
        self._ws.set_candle_callback(self._on_candle_complete)
        logger.info(
            f"WebSocket initialized "
            f"(candle: {self._candle_interval}min, "
            f"signal: every {self._signal_interval}min = "
            f"every {self._candles_per_signal} candles)"
        )

    def _init_notifier(self) -> None:
        """Initialize Telegram notifier."""
        from src.execution import TelegramNotifier

        tg_keys = self.keys.get("telegram", {})
        bot_token = tg_keys.get("bot_token")
        chat_id = tg_keys.get("chat_id")

        if bot_token and chat_id:
            self._notifier = TelegramNotifier(bot_token, chat_id)
            logger.info("Telegram notifier initialized")
        else:
            logger.warning("Telegram not configured; notifications disabled")

    def run(self) -> None:
        """Main event loop."""
        self._running = True

        # Start WebSocket
        self._ws.start()

        # Subscribe to universe
        for ticker in self._universe[:40]:  # KIS limit: ~40 concurrent subs
            self._ws.subscribe_price(ticker)

        # Notify startup
        if self._notifier:
            try:
                self._notifier.send_startup()
            except Exception:
                pass

        logger.info(
            f"Real-time trading started. "
            f"Subscribed to {min(len(self._universe), 40)} stocks."
        )
        logger.info(
            f"Market hours: {SCHEDULE['market_open']} - {SCHEDULE['market_close']}"
        )

        # Main loop: check for session boundaries and heartbeat
        try:
            while self._running:
                now = datetime.now()
                current_time = now.strftime("%H:%M")

                # Market close -> send daily report
                if current_time == SCHEDULE["eod_report"]:
                    self._send_daily_report()

                time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def _on_candle_complete(self, candle) -> None:
        """
        Callback when a 1-minute candle completes.

        Data collection happens every candle (1min).
        Signal generation happens every signal_interval (e.g. 5min).
        Re-entry guard prevents LLM inference pileup.
        """
        logger.debug(
            f"Candle ({self._candle_interval}m): {candle.stock_code} "
            f"O={candle.open} H={candle.high} L={candle.low} C={candle.close} "
            f"V={candle.volume} ticks={candle.tick_count}"
        )

        # 항상 가격 데이터는 업데이트 (1분봉 수집)
        all_candles = self._ws.aggregator.get_all_completed()
        self._update_prices_from_candles(all_candles)

        # 시그널 생성 주기 확인
        self._candle_count += 1
        if self._candle_count % self._candles_per_signal != 0:
            return  # 아직 시그널 생성 시점 아님

        # 시간 기반 이중 체크 (동일 시그널 중복 방지)
        now = datetime.now()
        if self._last_signal_time:
            elapsed = (now - self._last_signal_time).total_seconds()
            if elapsed < (self._signal_interval * 60) * 0.8:
                return

        logger.info(
            f"Signal trigger: {self._candle_count} candles accumulated, "
            f"generating signals..."
        )

        # Re-entry guard: skip if previous cycle is still processing
        if self._processing:
            logger.warning(
                "Previous signal still processing, skipping this cycle."
            )
            return
        self._processing = True

        # Minimum candle data check
        if len(all_candles) < 5:
            logger.debug("Not enough candle data yet, skipping signal generation")
            self._processing = False
            return

        try:
            self._last_signal_time = now

            # Generate signals
            signal_result = self._ensemble.generate_signals(
                date=now,
                prices=self._prices,
                features=self._features,
                regime=None,
            )

            if signal_result.signals.empty:
                logger.info("No signals generated")
                return

            logger.info(
                f"Signals generated: {len(signal_result.signals)} stocks, "
                f"weights: {signal_result.strategy_weights}"
            )

            # Allocate portfolio
            target_weights = self._allocator.allocate(
                signal_result.signals, self._prices
            )

            if target_weights.empty:
                logger.info("No allocation produced")
                return

            logger.info(
                f"Allocation: {len(target_weights)} positions\n"
                f"{target_weights.to_string()}"
            )

            # Notify
            if self._notifier:
                try:
                    self._notifier.send_signal_alert(
                        strategy="LLM Ensemble",
                        stocks=target_weights["ticker"].tolist(),
                        regime=signal_result.regime,
                    )
                except Exception as e:
                    logger.error(f"Notification failed: {e}")

            # Execute (unless dry run)
            if not self.dry_run:
                self._execute_rebalance(target_weights)
            else:
                logger.info("DRY RUN - No orders placed")

        except Exception as e:
            logger.error(f"Trading cycle error: {e}", exc_info=True)
            if self._notifier:
                try:
                    self._notifier.send_error(str(e), "candle_trading_cycle")
                except Exception:
                    pass
        finally:
            self._processing = False

    def _update_prices_from_candles(
        self, candles: dict[str, list]
    ) -> None:
        """Update price DataFrame with latest candle data."""
        rows = []
        for stock_code, candle_list in candles.items():
            if candle_list:
                latest = candle_list[-1]
                rows.append({
                    "date": pd.Timestamp(latest.start_time.date()),
                    "ticker": stock_code,
                    "open": latest.open,
                    "high": latest.high,
                    "low": latest.low,
                    "close": latest.close,
                    "volume": latest.volume,
                })

        if rows:
            new_prices = pd.DataFrame(rows)
            if not new_prices.empty:
                self._prices = pd.concat(
                    [self._prices, new_prices], ignore_index=True
                ).drop_duplicates(subset=["date", "ticker"], keep="last")

    def _execute_rebalance(self, target_weights: pd.DataFrame) -> None:
        """Execute portfolio rebalance via OrderManager."""
        try:
            orders = self._order_manager.execute_rebalance(
                target_weights=target_weights,
                order_type=TRADING.get("order_type", "limit"),
                sell_first=True,
                notifier=self._notifier,
            )
            logger.info(f"Rebalance executed: {len(orders)} orders submitted")
        except Exception as e:
            logger.error(f"Rebalance execution failed: {e}")

    def _send_daily_report(self) -> None:
        """Send end-of-day report via Telegram."""
        if not self._notifier:
            return

        try:
            balance = self._order_manager.api.get_balance()
            self._notifier.send_daily_summary(
                date=datetime.now().strftime("%Y-%m-%d"),
                total_value=balance.get("total_eval", 0),
                daily_pnl=balance.get("total_profit_loss", 0),
                daily_return=0,
                positions=balance.get("holdings", []),
                trades_count=0,
            )
        except Exception as e:
            logger.error(f"Daily report failed: {e}")

    def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._running = False
        if self._ws:
            self._ws.stop()
        logger.info("Real-time trading stopped")


def main():
    parser = argparse.ArgumentParser(description="Real-time LLM Trading")
    parser.add_argument("--paper", action="store_true", default=True,
                        help="Paper trading (default)")
    parser.add_argument("--real", action="store_true",
                        help="Real trading")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate signals without placing orders")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use traditional ensemble (no LLM)")

    args = parser.parse_args()

    is_paper = not args.real

    if not is_paper:
        logger.warning("=" * 60)
        logger.warning("  REAL TRADING MODE - USE WITH CAUTION!")
        logger.warning("=" * 60)

    # Load keys
    keys = load_keys()
    if keys is None:
        sys.exit(1)

    # Create and run
    loop = RealtimeTradingLoop(
        keys=keys,
        is_paper=is_paper,
        dry_run=args.dry_run,
        use_llm=not args.no_llm,
    )

    # Graceful shutdown handler
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        loop.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    loop.initialize()
    loop.run()


if __name__ == "__main__":
    main()
