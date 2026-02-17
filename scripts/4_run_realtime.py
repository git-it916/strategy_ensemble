#!/usr/bin/env python
"""
실행코드
python scripts/4_run_realtime.py --real

4. Run Real-time Trading — Sequential Pipeline

순차 파이프라인 기반 실시간 트레이딩.
병렬 투표(EnsembleAgent) 대신, LLM이 최종 "펀드매니저" 역할.

7-Step Pipeline:
    1. Universe Selector  — 시총 1000억+ 필터
    2. Feature Extractor  — ML 모델 → 확률/스코어 (데이터 소스)
    3. Data Loader        — 실시간 가격(KIS) + 뉴스(Telegram)
    4. Reasoning Engine   — qwen2.5-kospi-ft-s3 → JSON 결정
    5. Risk Manager       — 포지션 사이징 + SHORT→인버스 ETF 변환
    6. Approval Agent     — 텔레그램으로 사용자 승인 대기
    7. Executor           — KIS API 주문 실행

Usage:
    python scripts/4_run_realtime.py              # Paper + 승인 모드
    python scripts/4_run_realtime.py --real        # 실거래
    python scripts/4_run_realtime.py --dry-run     # 주문 없이 시그널만
    python scripts/4_run_realtime.py --no-approval # 자동 실행 (승인 건너뜀)
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TRADING,
    SCHEDULE,
    LLM_CONFIG,
    WEBSOCKET_CONFIG,
    PIPELINE,
    INVERSE_MAPPING,
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


class SequentialPipeline:
    """
    순차 파이프라인 메인 루프.

    As-Is (병렬): ML/RSI/LLM 독립 시그널 → 가중 투표
    To-Be (순차): ML스코어+기술지표+뉴스 → LLM(Judge) → 승인 → 실행

    7-Step Pipeline:
        1. Universe Selector
        2. Feature Extractor (ML models as data sources)
        3. Data Loader (real-time prices + news)
        4. Reasoning Engine (LLM Fund Manager)
        5. Risk Manager (position sizing + inverse ETF)
        6. Approval Agent (human-in-the-loop)
        7. Executor (KIS API orders)
    """

    def __init__(
        self,
        keys: dict,
        is_paper: bool = True,
        dry_run: bool = False,
        require_approval: bool = True,
    ):
        self.keys = keys
        self.is_paper = is_paper
        self.dry_run = dry_run
        self.require_approval = require_approval

        self._running = False
        self._processing = False

        # Components (lazy init)
        self._context_builder = None
        self._llm_alpha = None
        self._risk_manager = None
        self._approval_agent = None
        self._order_manager = None
        self._notifier = None
        self._ws = None
        self._news_collector = None

        # Data
        self._prices: pd.DataFrame | None = None
        self._features: pd.DataFrame | None = None
        self._ml_strategies: dict = {}
        self._technical_strategies: dict = {}
        self._universe: list[str] = []

        # Timing
        self._candle_count = 0
        self._signal_interval = WEBSOCKET_CONFIG.get("signal_interval_minutes", 15)
        self._candle_interval = WEBSOCKET_CONFIG.get("candle_interval_minutes", 1)
        self._candles_per_signal = max(
            1, self._signal_interval // self._candle_interval
        )
        self._last_signal_time: datetime | None = None

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        logger.info("=" * 60)
        logger.info("Sequential Pipeline — Initializing")
        logger.info(f"  Mode: {'PAPER' if self.is_paper else 'REAL'}")
        logger.info(f"  Dry Run: {self.dry_run}")
        logger.info(f"  Approval: {self.require_approval}")
        logger.info(f"  Signal Interval: {self._signal_interval}min")
        logger.info(f"  LLM Model: {PIPELINE['llm_model']}")
        logger.info("=" * 60)

        # 1. Load data
        self._load_data()

        # 2. Load ML + Technical strategies (as data sources, not decision makers)
        self._load_strategies()

        # 3. Initialize ContextBuilder
        self._init_context_builder()

        # 4. Initialize LLM (Fund Manager)
        self._init_llm()

        # 5. Initialize Risk Manager
        self._init_risk_manager()

        # 6. Initialize Approval Agent
        self._init_approval_agent()

        # 7. Initialize Broker (KIS API)
        self._init_broker()

        # 8. Initialize WebSocket
        self._init_websocket()

        # 9. Initialize Telegram Notifier
        self._init_notifier()

        logger.info("All pipeline components initialized")

    def _load_data(self) -> None:
        """Load price and feature data."""
        prices_path = PROCESSED_DATA_DIR / "prices.parquet"
        features_path = PROCESSED_DATA_DIR / "features.parquet"

        if prices_path.exists():
            self._prices = pd.read_parquet(prices_path)
            self._prices["date"] = pd.to_datetime(self._prices["date"])
            logger.info(f"Loaded {len(self._prices)} price records")
        else:
            logger.error("prices.parquet not found! Run 0_prepare_prices.py first.")
            sys.exit(1)

        if features_path.exists():
            self._features = pd.read_parquet(features_path)
            self._features["date"] = pd.to_datetime(self._features["date"])
            logger.info(f"Loaded {len(self._features)} feature records")

        # Universe
        latest_date = self._prices["date"].max()
        recent = self._prices[self._prices["date"] == latest_date]
        self._universe = recent["ticker"].tolist()[
            : PIPELINE.get("max_positions", 20) * 5
        ]
        logger.info(f"Universe: {len(self._universe)} stocks")

    def _load_strategies(self) -> None:
        """Load trained ML and technical strategies as DATA SOURCES."""
        from src.ensemble_models import ModelManager

        manager = ModelManager(MODELS_DIR)

        try:
            ensemble = manager.load_ensemble()
            all_strategies = ensemble.strategies

            # Separate ML vs Technical strategies
            ml_names = set(PIPELINE.get("ml_strategies", []))
            tech_names = set(PIPELINE.get("technical_strategies", []))

            for name, strategy in all_strategies.items():
                if name in ml_names:
                    self._ml_strategies[name] = strategy
                    logger.info(f"  ML data source: {name}")
                elif name in tech_names:
                    self._technical_strategies[name] = strategy
                    logger.info(f"  Technical data source: {name}")

            logger.info(
                f"Loaded {len(self._ml_strategies)} ML + "
                f"{len(self._technical_strategies)} technical strategies"
            )

        except FileNotFoundError as e:
            logger.warning(f"Could not load trained models: {e}")
            logger.warning("Pipeline will run with price data only")

    def _init_context_builder(self) -> None:
        """Initialize ContextBuilder."""
        from src.pipeline import ContextBuilder

        self._context_builder = ContextBuilder(
            ml_strategies=self._ml_strategies,
            technical_strategies=self._technical_strategies,
            news_collector=self._news_collector,
        )
        logger.info("ContextBuilder initialized")

    def _init_llm(self) -> None:
        """Initialize LLM Alpha (Fund Manager role)."""
        from src.alphas.llm.llm_alpha import LLMAlpha

        self._llm_alpha = LLMAlpha(
            name="fund_manager",
            config={
                "model": PIPELINE["llm_model"],
                "temperature": LLM_CONFIG.get("ollama_temperature", 0.3),
                "timeout": LLM_CONFIG.get("ollama_timeout", 120.0),
            },
        )
        self._llm_alpha.is_fitted = True
        logger.info(f"LLM Fund Manager: {PIPELINE['llm_model']}")

    def _init_risk_manager(self) -> None:
        """Initialize Risk Manager."""
        from src.pipeline import RiskManager

        self._risk_manager = RiskManager(
            inverse_mapping=INVERSE_MAPPING,
            max_position_weight=PIPELINE.get("max_position_weight", 0.1),
            max_positions=PIPELINE.get("max_positions", 20),
            max_leverage=TRADING.get("max_leverage", 1.0),
            min_trade_value=TRADING.get("min_trade_value", 100_000),
        )
        logger.info("RiskManager initialized (inverse ETF mapping active)")

    def _init_approval_agent(self) -> None:
        """Initialize Approval Agent."""
        if not self.require_approval:
            logger.info("Approval agent DISABLED (auto-execute mode)")
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
        logger.info("ApprovalAgent initialized (Telegram human-in-the-loop)")

    def _init_broker(self) -> None:
        """Initialize KIS API + OrderManager."""
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
        self._ws.set_candle_callback(self._on_candle_complete)
        logger.info(
            f"WebSocket: {self._candle_interval}min candles, "
            f"signal every {self._signal_interval}min"
        )

    def _init_notifier(self) -> None:
        """Initialize Telegram notifier (separate from approval agent)."""
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

        # Start WebSocket
        self._ws.start()

        # Subscribe to universe
        for ticker in self._universe[:40]:
            self._ws.subscribe_price(ticker)

        if self._notifier:
            try:
                self._notifier.send_message(
                    "\U0001f916 <b>Sequential Pipeline Started</b>\n"
                    f"Model: {PIPELINE['llm_model']}\n"
                    f"Approval: {'ON' if self.require_approval else 'OFF'}\n"
                    f"Interval: {self._signal_interval}min"
                )
            except Exception:
                pass

        logger.info(
            f"Pipeline running. Subscribed to {min(len(self._universe), 40)} stocks."
        )

        try:
            while self._running:
                now = datetime.now()

                # EOD report
                if now.strftime("%H:%M") == SCHEDULE["eod_report"]:
                    self._send_daily_report()

                time.sleep(30)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def _on_candle_complete(self, candle) -> None:
        """1-min candle callback — check signal generation timing."""
        # Always update prices
        all_candles = self._ws.aggregator.get_all_completed()
        self._update_prices_from_candles(all_candles)

        # Check signal interval
        self._candle_count += 1
        if self._candle_count % self._candles_per_signal != 0:
            return

        # Duplicate prevention
        now = datetime.now()
        if self._last_signal_time:
            elapsed = (now - self._last_signal_time).total_seconds()
            if elapsed < (self._signal_interval * 60) * 0.8:
                return

        # Re-entry guard
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
    # 7-Step Pipeline
    # ===================================================================

    def _run_pipeline(self, now: datetime) -> None:
        """
        Execute the 7-step sequential pipeline.

        1. Universe Selector
        2. Feature Extractor (ML scores)
        3. Data Loader (prices + news)
        4. Reasoning Engine (LLM)
        5. Risk Manager
        6. Approval Agent
        7. Executor
        """

        # -- Step 1: Universe Selector --
        logger.info("[Step 1] Universe: filtering candidates")
        # Universe is already filtered at init time

        # -- Step 2+3: ContextBuilder (ML scores + prices + news) --
        logger.info("[Step 2-3] Building context (ML + technical + news)")
        news_messages = None
        if self._news_collector is not None:
            try:
                news_messages = self._news_collector.get_recent_summary(20)
            except Exception as e:
                logger.warning(f"News fetch failed: {e}")

        ctx = self._context_builder.build(
            date=now,
            prices=self._prices,
            features=self._features,
            news_messages=news_messages,
            top_k=PIPELINE.get("max_positions", 20) * 3,
        )
        logger.info(
            f"  Context: {ctx.n_stocks} stocks, regime={ctx.regime}, "
            f"{len(ctx.news_headlines)} news"
        )

        # -- Step 4: Reasoning Engine (LLM Fund Manager) --
        logger.info("[Step 4] LLM Fund Manager reasoning")
        llm_result = self._llm_alpha.generate_from_context(ctx)

        signals = llm_result.get("signals", [])
        confidence = llm_result.get("confidence")
        reasoning = llm_result.get("reasoning", "")

        if not signals:
            logger.info("  LLM returned no signals. Pipeline stops.")
            return

        logger.info(
            f"  LLM decision: {len(signals)} signals, " f"confidence={confidence}"
        )
        if reasoning:
            logger.info(f"  Reasoning: {reasoning[:200]}")

        # -- Step 5: Risk Manager --
        logger.info("[Step 5] Risk Manager: sizing + inverse ETF conversion")
        current_positions = None
        try:
            current_positions = self._order_manager.get_positions()
        except Exception as e:
            logger.warning(f"Could not fetch positions: {e}")

        orders = self._risk_manager.process_llm_decisions(
            llm_signals=signals,
            current_positions=current_positions,
        )

        if not orders:
            logger.info("  No actionable orders after risk check.")
            return

        buy_count = sum(1 for o in orders if o.side == "BUY")
        sell_count = sum(1 for o in orders if o.side == "SELL")
        inv_count = sum(1 for o in orders if o.is_inverse)
        logger.info(
            f"  Orders: {buy_count} BUY, {sell_count} SELL, " f"{inv_count} inverse ETF"
        )

        # -- Step 6: Approval Agent (Human-in-the-Loop) --
        if self.dry_run:
            proposal = self._risk_manager.format_proposal(orders)
            logger.info(f"[Step 6] DRY RUN — proposal:\n{proposal}")
            if self._notifier:
                self._notifier.send_message(
                    f"\U0001f3c3 <b>DRY RUN</b> (no orders)\n\n{proposal}"
                )
            return

        if self.require_approval and self._approval_agent:
            logger.info("[Step 6] Requesting user approval via Telegram")

            total_value = 0
            try:
                total_value = self._order_manager.get_cash()
                positions = self._order_manager.get_positions()
                if not positions.empty:
                    total_value += positions["eval_amount"].sum()
            except Exception:
                pass

            proposal = self._risk_manager.format_proposal(orders, total_value)
            approved = self._approval_agent.request_approval(
                proposal_text=proposal,
                regime=ctx.regime,
                confidence=confidence,
            )

            if not approved:
                logger.info("  User REJECTED or timeout. Skipping execution.")
                return

            logger.info("  User APPROVED. Proceeding to execution.")
        else:
            logger.info("[Step 6] Auto-execute (no approval required)")

        # -- Step 7: Executor --
        logger.info("[Step 7] Executing orders via KIS API")
        target_weights = self._risk_manager.to_target_weights(orders)

        if target_weights.empty:
            logger.info("  No target weights to execute.")
            return

        try:
            executed_orders = self._order_manager.execute_rebalance(
                target_weights=target_weights,
                order_type=TRADING.get("order_type", "limit"),
                sell_first=True,
                notifier=self._notifier,
            )
            logger.info(f"  Executed {len(executed_orders)} orders")

            # Report back
            if self._approval_agent and self.require_approval:
                self._approval_agent.send_execution_result(
                    [
                        {
                            "stock_code": o.stock_code,
                            "side": o.side,
                            "quantity": o.quantity,
                        }
                        for o in executed_orders
                    ]
                )

        except Exception as e:
            logger.error(f"  Execution failed: {e}")
            if self._notifier:
                self._notifier.send_error(str(e), "order_execution")

    # ===================================================================
    # Helpers
    # ===================================================================

    def _update_prices_from_candles(self, candles: dict[str, list]) -> None:
        """Update price DataFrame with latest candle data."""
        rows = []
        for stock_code, candle_list in candles.items():
            if candle_list:
                latest = candle_list[-1]
                rows.append(
                    {
                        "date": pd.Timestamp(latest.start_time.date()),
                        "ticker": stock_code,
                        "open": latest.open,
                        "high": latest.high,
                        "low": latest.low,
                        "close": latest.close,
                        "volume": latest.volume,
                    }
                )

        if rows:
            new_prices = pd.DataFrame(rows)
            if not new_prices.empty:
                self._prices = pd.concat(
                    [self._prices, new_prices], ignore_index=True
                ).drop_duplicates(subset=["date", "ticker"], keep="last")

    def _send_daily_report(self) -> None:
        """Send end-of-day report."""
        if not self._notifier:
            return
        try:
            balance = self._order_manager.api.get_balance()
            self._notifier.send_daily_summary(
                date=datetime.now(),
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
        logger.info("Shutting down pipeline...")
        self._running = False
        if self._ws:
            self._ws.stop()
        logger.info("Pipeline stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Sequential Pipeline Real-time Trading"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Paper trading (default)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Real trading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate signals without placing orders",
    )
    parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Skip human approval (auto-execute)",
    )

    args = parser.parse_args()
    is_paper = not args.real
    require_approval = not args.no_approval

    if not is_paper:
        logger.warning("=" * 60)
        logger.warning("  REAL TRADING MODE — USE WITH CAUTION!")
        logger.warning("=" * 60)

    keys = load_keys()
    if keys is None:
        sys.exit(1)

    pipeline = SequentialPipeline(
        keys=keys,
        is_paper=is_paper,
        dry_run=args.dry_run,
        require_approval=require_approval,
    )

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        pipeline.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    pipeline.initialize()
    pipeline.run()


if __name__ == "__main__":
    main()
