#!/usr/bin/env python
"""
4. Run Real-time Trading — Binance USDT-M Perpetual Futures

앙상블 알파 기반 실시간 트레이딩.
BinanceWebSocket → CandleAggregator → EnsembleAgent → BinanceApi 주문.

Pipeline:
    1. Data Loader    — Binance WebSocket 1분 캔들
    2. Signal Engine  — EnsembleAgent (모든 openclaw_1 알파 + technical)
    3. Risk Manager   — 포지션 사이징
    4. Approval Agent — 텔레그램 승인 (선택)
    5. Executor       — Binance API 주문

Usage:
    python scripts/4_run_realtime.py              # 실계정 + 승인 모드
    python scripts/4_run_realtime.py --dry-run     # 주문 없이 시그널만
    python scripts/4_run_realtime.py --no-approval # 자동 실행 (승인 건너뜀)
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    MODELS_DIR,
    TRADING,
    WEBSOCKET_CONFIG,
    PIPELINE,
)
from config.logging_config import setup_logging

logger = setup_logging("realtime_pipeline")


def load_keys() -> dict | None:
    """Load API keys from config/keys.yaml."""
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"
    if not keys_path.exists():
        logger.error("keys.yaml not found! Copy from keys.example.yaml")
        return None
    with open(keys_path) as f:
        return yaml.safe_load(f)


class BinancePipeline:
    """
    Binance USDT-M 실시간 트레이딩 파이프라인.

    Pipeline:
        1. BinanceWebSocket → 1분 캔들 수신
        2. EnsembleAgent → 알파 시그널 생성 (15분마다)
        3. RiskManager → 포지션 사이징
        4. ApprovalAgent → 텔레그램 승인 (선택)
        5. BinanceApi → 주문 실행
    """

    def __init__(
        self,
        keys: dict,
        dry_run: bool = False,
        require_approval: bool = True,
    ):
        self.keys = keys
        self.dry_run = dry_run
        self.require_approval = require_approval

        self._running = False
        self._processing = False

        self._api = None
        self._ws = None
        self._ensemble = None
        self._approval_agent = None
        self._notifier = None

        self._prices: pd.DataFrame | None = None
        self._prices_lock = threading.Lock()
        self._features: pd.DataFrame | None = None
        self._universe: list[str] = []

        self._candle_count = 0
        self._signal_interval = WEBSOCKET_CONFIG.get("signal_interval_minutes", 15)
        self._candle_interval_str = WEBSOCKET_CONFIG.get("candle_interval", "1m")
        self._candles_per_signal = max(1, self._signal_interval)
        self._last_signal_time: datetime | None = None

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        logger.info("=" * 60)
        logger.info("Binance Pipeline — Initializing")
        logger.info(f"  Dry Run: {self.dry_run}")
        logger.info(f"  Approval: {self.require_approval}")
        logger.info(f"  Signal Interval: {self._signal_interval}min")
        logger.info("=" * 60)

        self._init_api()
        self._load_data()
        self._init_ensemble()
        self._init_approval_agent()
        self._init_websocket()
        self._init_notifier()

        logger.info("All pipeline components initialized")

    def _load_data(self) -> None:
        """
        Fetch historical data directly from Binance via ccxt.

        1. Universe: top N symbols by 24h volume
        2. OHLCV: 300 days of daily candles
        3. Funding rates: 90 days, averaged to daily
        4. Features: funding_rate merged onto prices (for FundingRateCarry)
        """
        from config.settings import UNIVERSE

        n_symbols = UNIVERSE.get("top_n_by_volume", 50)
        exclude = UNIVERSE.get("exclude_symbols", [])
        lookback_days = 300

        # 1. Universe
        logger.info(f"Fetching top {n_symbols} symbols by volume from Binance...")
        self._universe = self._api.get_top_symbols(n=n_symbols, exclude=exclude)
        logger.info(f"Universe: {len(self._universe)} symbols")

        # 2. OHLCV
        logger.info(f"Fetching {lookback_days}d OHLCV for {len(self._universe)} symbols...")
        self._prices = self._api.get_ohlcv_batch(
            self._universe, timeframe="1d", days=lookback_days,
        )
        logger.info(
            f"OHLCV loaded: {len(self._prices)} rows, "
            f"{self._prices['ticker'].nunique()} symbols, "
            f"{self._prices['date'].min().date()} ~ {self._prices['date'].max().date()}"
        )

        # 3. Funding rates
        logger.info("Fetching 90d funding rate history...")
        funding = self._api.get_funding_history_batch(self._universe, days=90)

        # 4. Build features (merge funding_rate for FundingRateCarry alpha)
        if not funding.empty:
            self._features = self._prices[["date", "ticker"]].copy()
            self._features = self._features.merge(
                funding[["date", "ticker", "funding_rate"]],
                on=["date", "ticker"],
                how="left",
            )
            logger.info(f"Features built: {len(self._features)} rows (with funding_rate)")
        else:
            self._features = None
            logger.warning("No funding rate data — features will be None")

    def _init_api(self) -> None:
        """Initialize Binance REST API client."""
        from src.execution.binance_api import BinanceApi

        binance_keys = self.keys.get("binance", {})
        api_key = binance_keys.get("api_key", "")
        api_secret = binance_keys.get("api_secret", "")

        self._api = BinanceApi(api_key=api_key, api_secret=api_secret)
        logger.info("BinanceApi initialized")

    def _init_ensemble(self) -> None:
        """Load trained ensemble or build from openclaw_1 alphas."""
        from src.ensemble_models import ModelManager

        manager = ModelManager(MODELS_DIR)
        try:
            self._ensemble = manager.load_ensemble()
            logger.info(f"Loaded ensemble: {list(self._ensemble.strategies.keys())}")
        except FileNotFoundError:
            logger.warning("No trained ensemble found. Using raw alphas (equal weight).")
            self._ensemble = None

    def _init_approval_agent(self) -> None:
        """Initialize Telegram approval agent."""
        if not self.require_approval:
            return

        tg_keys = self.keys.get("telegram", {})
        bot_token = tg_keys.get("bot_token", "")
        chat_id = tg_keys.get("chat_id", "")

        if not bot_token or not chat_id or "YOUR" in bot_token:
            logger.warning("Telegram not configured — approval agent disabled")
            self.require_approval = False
            return

        from src.pipeline import ApprovalAgent

        self._approval_agent = ApprovalAgent(
            bot_token=bot_token,
            chat_id=chat_id,
            timeout_seconds=PIPELINE.get("approval_timeout_seconds", 300),
        )
        logger.info("ApprovalAgent initialized (Telegram)")

    def _init_websocket(self) -> None:
        """Initialize Binance WebSocket."""
        from src.execution.binance_websocket import BinanceWebSocket

        self._ws = BinanceWebSocket(candle_interval=self._candle_interval_str)
        self._ws.set_candle_callback(self._on_candle_complete)
        logger.info(f"BinanceWebSocket: {self._candle_interval_str} candles")

    def _init_notifier(self) -> None:
        """Initialize Telegram notifier."""
        from src.execution import TelegramNotifier

        tg_keys = self.keys.get("telegram", {})
        bot_token = tg_keys.get("bot_token", "")
        chat_id = tg_keys.get("chat_id", "")

        if bot_token and chat_id and "YOUR" not in bot_token:
            self._notifier = TelegramNotifier(bot_token, chat_id)
            logger.info("Telegram notifier initialized")

    # ===================================================================
    # Main Loop
    # ===================================================================

    def run(self) -> None:
        """Main event loop."""
        self._running = True
        self._ws.start()

        # Subscribe to all universe symbols
        for symbol in self._universe:
            self._ws.subscribe(symbol)

        logger.info(f"Subscribed to {len(self._universe)} symbols. Pipeline running.")

        if self._notifier:
            try:
                self._notifier.send_message(
                    "🤖 <b>Binance Pipeline Started</b>\n"
                    f"Universe: {len(self._universe)} symbols\n"
                    f"Approval: {'ON' if self.require_approval else 'OFF'}\n"
                    f"Interval: {self._signal_interval}min"
                )
            except Exception:
                pass

        try:
            while self._running:
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def _on_candle_complete(self, _candle) -> None:
        """Candle completion callback — update prices, check signal timing."""
        # Update realtime prices
        all_candles = self._ws.aggregator.get_all_completed()
        self._update_prices_from_candles(all_candles)

        # Signal interval check
        self._candle_count += 1
        if self._candle_count % self._candles_per_signal != 0:
            return

        now = datetime.now()
        if self._last_signal_time:
            elapsed = (now - self._last_signal_time).total_seconds()
            if elapsed < (self._signal_interval * 60) * 0.8:
                return

        if self._processing:
            logger.warning("Previous pipeline still running, skipping")
            return
        self._processing = True

        try:
            self._last_signal_time = now
            logger.info(f"{'='*40} PIPELINE START {'='*40}")
            self._run_pipeline(now)
            logger.info(f"{'='*40} PIPELINE END {'='*40}")
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            if self._notifier:
                try:
                    self._notifier.send_error(str(e), "pipeline_cycle")
                except Exception:
                    pass
        finally:
            self._processing = False

    # ===================================================================
    # Pipeline Steps
    # ===================================================================

    def _run_pipeline(self, now: datetime) -> None:
        """Execute ensemble → risk → approval → execute pipeline."""

        if self._ensemble is None:
            logger.warning("No ensemble loaded; skipping signal generation")
            return

        # -- Signal Generation --
        logger.info("[Step 1] Generating ensemble signals")
        try:
            with self._prices_lock:
                prices_snapshot = self._prices.copy()
            signals_result = self._ensemble.generate_signals(
                date=now,
                prices=prices_snapshot,
                features=self._features,
            )
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return

        signals = signals_result.signals
        if signals.empty:
            logger.info("No signals generated")
            return

        top = signals.nlargest(10, "score")
        logger.info(f"Top signals:\n{top.to_string(index=False)}")

        # -- Risk / Position Sizing --
        logger.info("[Step 2] Risk: allocating positions")
        from src.ensemble import RiskParityAllocator

        allocator = RiskParityAllocator(
            top_k=PIPELINE.get("max_positions", 20),
            max_weight=PIPELINE.get("max_position_weight", 0.1),
        )
        target_weights = allocator.allocate(signals, self._prices)
        logger.info(f"Target weights: {len(target_weights)} positions")

        # -- Approval --
        if self.dry_run:
            logger.info("[Step 3] DRY RUN — no orders placed")
            if self._notifier:
                summary = "\n".join(
                    f"{row['ticker']}: {row['score']:.3f}"
                    for _, row in top.iterrows()
                )
                self._notifier.send_message(f"🏃 <b>DRY RUN signals</b>\n{summary}")
            return

        if self.require_approval and self._approval_agent:
            summary_text = "\n".join(
                f"{row['ticker']}: w={target_weights.get(row['ticker'], 0):.1%}"
                for _, row in top.head(5).iterrows()
            )
            approved = self._approval_agent.request_approval(
                proposal_text=summary_text,
                regime=getattr(signals_result, "regime", "unknown"),
                confidence=None,
            )
            if not approved:
                logger.info("User REJECTED or timeout. Skipping execution.")
                return
            logger.info("User APPROVED.")
        else:
            logger.info("[Step 3] Auto-execute (no approval)")

        # -- Execution --
        logger.info("[Step 4] Executing orders via Binance API")
        account = self._api.get_account()
        total_usdt = account.get("total_wallet_balance", 0)
        if total_usdt <= 0:
            logger.error("Account balance is 0 — aborting execution")
            return

        for ticker, weight in target_weights.items():
            notional = total_usdt * abs(weight)
            if notional < TRADING.get("min_notional_usdt", 10):
                continue
            try:
                # Fetch latest price
                ticker_prices = prices_snapshot[prices_snapshot["ticker"] == ticker]
                if ticker_prices.empty:
                    continue
                latest_price = float(ticker_prices.sort_values("date")["close"].iloc[-1])
                qty = notional / latest_price

                side = "buy" if weight > 0 else "sell"
                self._api.place_order(symbol=ticker, side=side, quantity=qty)

            except Exception as e:
                logger.error(f"Order failed for {ticker}: {e}")

    # ===================================================================
    # Helpers
    # ===================================================================

    def _update_prices_from_candles(self, candles: dict[str, list]) -> None:
        """Append latest candle data to self._prices."""
        from src.pipeline.universe import normalize_symbol

        rows = []
        for _symbol, candle_list in candles.items():
            if candle_list:
                latest = candle_list[-1]
                # Normalize WebSocket symbol (e.g. 'BTCUSDT') to ccxt format ('BTC/USDT:USDT')
                ticker = normalize_symbol(latest.stock_code)
                if not ticker:
                    continue
                rows.append({
                    "date": pd.Timestamp(latest.start_time.date()),
                    "ticker": ticker,
                    "open": latest.open,
                    "high": latest.high,
                    "low": latest.low,
                    "close": latest.close,
                    "volume": latest.volume,
                })

        if rows and self._prices is not None:
            new_df = pd.DataFrame(rows)
            with self._prices_lock:
                self._prices = pd.concat(
                    [self._prices, new_df], ignore_index=True
                ).drop_duplicates(subset=["date", "ticker"], keep="last")

    def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down Binance pipeline...")
        self._running = False
        if self._ws:
            self._ws.stop()
        logger.info("Pipeline stopped")


def main():
    parser = argparse.ArgumentParser(description="Binance Real-time Trading Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Generate signals only, no orders")
    parser.add_argument("--no-approval", action="store_true", help="Skip human approval")

    args = parser.parse_args()
    require_approval = not args.no_approval

    keys = load_keys()
    if keys is None:
        sys.exit(1)

    binance_keys = keys.get("binance", {})
    if not binance_keys.get("api_key") or "YOUR" in str(binance_keys.get("api_key", "")):
        logger.error("Binance API keys not configured in config/keys.yaml")
        sys.exit(1)

    pipeline = BinancePipeline(
        keys=keys,
        dry_run=args.dry_run,
        require_approval=require_approval,
    )

    def signal_handler(_sig, _frame):
        logger.info("Shutdown signal received")
        pipeline.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    pipeline.initialize()
    pipeline.run()


if __name__ == "__main__":
    main()
