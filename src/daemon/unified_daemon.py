"""
Unified Daemon

Single-process daemon merging OpenClaw (research) and Trading Pipeline (execution).

Main loop:
  - Telegram command polling: every 5 seconds
  - Signal generation + rebalance: every 15 minutes
  - Lifecycle check: once per day
  - Auto research: every 24 hours (optional)
  - WebSocket: continuous 1m candle collection (separate thread)
"""

from __future__ import annotations

import logging
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
import yaml

from config.settings import (
    ENSEMBLE,
    MODELS_DIR,
    SONNET_DECISION,
    STRATEGIES,
    TRADING,
    UNIVERSE,
    WEBSOCKET_CONFIG,
)

from src.daemon.signal_aggregator import SignalAggregator, AggregatedSignal
from src.daemon.sonnet_decision_maker import SonnetDecisionMaker, SonnetDecision, DECISION_LOG_DIR
from src.daemon.position_store import PositionStore
from src.daemon.sltp_monitor import SLTPMonitor
from src.daemon.trade_proposal import TradeProposalBuilder

logger = logging.getLogger(__name__)


class UnifiedDaemon:
    """
    Unified daemon combining OpenClaw research with live trading.

    Pipeline per rebalance cycle:
        1. Collect raw signals from 7 alphas (10 coins)
        2. Gather market context (BTC trend, regime)
        3. Call Sonnet with signals + context → structured decision
        4. Execute via Rebalancer
        5. Store SL/TP levels for continuous monitoring

    Between cycles:
        - SL/TP monitor checks prices every 5 seconds
        - Auto-closes positions when stop-loss or take-profit is hit
    """

    def __init__(
        self,
        dry_run: bool = False,
        require_approval: bool = False,
        enable_research: bool = False,
    ):
        self.dry_run = dry_run
        self.require_approval = require_approval
        self.enable_research = enable_research

        self._running = False
        self._paused = False
        self._processing = False

        self._last_rebalance: datetime | None = None
        self._last_lifecycle: datetime | None = None
        self._last_research: datetime | None = None

        # Components (initialized in initialize())
        self.keys: dict = {}
        self.binance_api = None
        self.ws = None
        self.telegram = None
        self.registry = None
        self.tracker = None
        self.lifecycle = None
        self.claude_orchestrator = None
        self.leverage_calc = None
        self.rebalancer = None
        self.openclaw_daemon = None

        self.signal_aggregator = SignalAggregator()
        self.proposal_builder = TradeProposalBuilder()
        self.sonnet_decision_maker = None
        self.position_store = None
        self.sltp_monitor = None

        # Data
        self._prices: pd.DataFrame | None = None
        self._features: pd.DataFrame | None = None
        self._universe: list[str] = []

        # Ensemble (trained model, optional)
        self._ensemble = None

        # openclaw_1 alpha instances
        self._base_alphas: dict[str, Any] = {}

        # Config
        self._rebalance_interval = WEBSOCKET_CONFIG.get("signal_interval_minutes", 15)

    # ==================================================================
    # Initialization
    # ==================================================================

    def initialize(self) -> None:
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("Unified Daemon — Initializing")
        logger.info(f"  Dry Run: {self.dry_run}")
        logger.info(f"  Approval: {self.require_approval}")
        logger.info(f"  Research: {self.enable_research}")
        logger.info(f"  Rebalance Interval: {self._rebalance_interval}min")
        logger.info("=" * 60)

        self._load_keys()
        self._init_binance_api()
        self._load_data()
        self._init_base_alphas()
        self._init_ensemble()
        self._init_openclaw()
        self._init_rebalancer()
        self._init_sonnet()
        self._init_websocket()
        self._register_extra_commands()

        logger.info("All components initialized")

        self.telegram.send_message(
            "<b>Unified Daemon Started</b>\n\n"
            f"Universe: {len(self._universe)} symbols\n"
            f"Base alphas: {len(self._base_alphas)}\n"
            f"Decision: Sonnet ({SONNET_DECISION['model'].split('-')[1]})\n"
            f"SL/TP: Active (5s check)\n"
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}\n"
            f"Max positions: {TRADING.get('max_positions', 3)}"
        )

    def _load_keys(self) -> None:
        keys_path = Path(__file__).parent.parent.parent / "config" / "keys.yaml"
        if not keys_path.exists():
            raise RuntimeError("config/keys.yaml not found!")
        with open(keys_path) as f:
            self.keys = yaml.safe_load(f)

    def _init_binance_api(self) -> None:
        from src.execution.binance_api import BinanceApi

        binance_cfg = self.keys.get("binance", {})
        self.binance_api = BinanceApi(
            api_key=binance_cfg.get("api_key", ""),
            api_secret=binance_cfg.get("api_secret", ""),
        )
        logger.info("BinanceApi initialized")

    def _load_data(self) -> None:
        """Fetch historical data from Binance via ccxt."""
        n_symbols = UNIVERSE.get("top_n_by_volume", 50)
        exclude = UNIVERSE.get("exclude_symbols", [])
        lookback_days = 300

        logger.info(f"Fetching top {n_symbols} symbols by volume...")
        self._universe = self.binance_api.get_top_symbols(n=n_symbols, exclude=exclude)
        logger.info(f"Universe: {len(self._universe)} symbols")

        logger.info(f"Fetching {lookback_days}d OHLCV...")
        self._prices = self.binance_api.get_ohlcv_batch(
            self._universe, timeframe="1d", days=lookback_days,
        )
        logger.info(
            f"OHLCV: {len(self._prices)} rows, "
            f"{self._prices['ticker'].nunique()} symbols"
        )

        logger.info("Fetching 90d funding rates...")
        funding = self.binance_api.get_funding_history_batch(self._universe, days=90)

        if not funding.empty:
            self._features = self._prices[["date", "ticker"]].copy()
            self._features = self._features.merge(
                funding[["date", "ticker", "funding_rate"]],
                on=["date", "ticker"],
                how="left",
            )
            logger.info(f"Features: {len(self._features)} rows (with funding_rate)")
        else:
            self._features = None
            logger.warning("No funding rate data")

    def _init_base_alphas(self) -> None:
        """Load and fit the 7 openclaw_1 rule-based alphas."""
        from src.alphas.openclaw_1 import (
            CSMomentum,
            TimeSeriesMomentum,
            TimeSeriesMeanReversion,
            PriceVolumeDivergence,
            VolumeMomentum,
            LowVolatilityAnomaly,
            FundingRateCarry,
        )

        alpha_classes = [
            CSMomentum,
            TimeSeriesMomentum,
            TimeSeriesMeanReversion,
            PriceVolumeDivergence,
            VolumeMomentum,
            LowVolatilityAnomaly,
            FundingRateCarry,
        ]

        for cls in alpha_classes:
            try:
                alpha = cls()
                alpha.fit(self._prices, self._features)
                self._base_alphas[alpha.name] = alpha
                logger.info(f"  Base alpha loaded: {alpha.name}")
            except Exception as e:
                logger.error(f"Failed to load alpha {cls.__name__}: {e}")

        logger.info(f"Base alphas: {len(self._base_alphas)} loaded")

    def _init_ensemble(self) -> None:
        """Load trained ensemble model (optional)."""
        from src.ensemble_models import ModelManager

        manager = ModelManager(MODELS_DIR)
        try:
            self._ensemble = manager.load_ensemble()
            logger.info(f"Ensemble loaded: {list(self._ensemble.strategies.keys())}")
        except FileNotFoundError:
            logger.info("No trained ensemble found (optional)")
            self._ensemble = None

    def _init_openclaw(self) -> None:
        """Initialize OpenClaw subsystems directly (no duplicate data loading)."""
        import anthropic as anthropic_mod

        from src.openclaw.config import OPENCLAW_DATA_DIR, OPENCLAW_LOGS_DIR, GENERATED_ALPHAS_DIR
        from src.openclaw.registry.alpha_registry import AlphaRegistry
        from src.openclaw.registry.performance_tracker import PerformanceTracker
        from src.openclaw.registry.lifecycle_manager import LifecycleManager
        from src.openclaw.orchestrator.claude_orchestrator import ClaudeEnsembleOrchestrator
        from src.openclaw.orchestrator.weight_optimizer import WeightOptimizer
        from src.openclaw.execution.leverage_calculator import LeverageCalculator
        from src.openclaw.telegram.command_handler import OpenClawTelegramHandler
        from src.openclaw.telegram.formatters import TelegramFormatter

        # Ensure directories
        OPENCLAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        OPENCLAW_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        GENERATED_ALPHAS_DIR.mkdir(parents=True, exist_ok=True)

        # Anthropic client
        api_key = self.keys.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in config/keys.yaml")
        self.anthropic_client = anthropic_mod.Anthropic(api_key=api_key)

        # Core components
        self.registry = AlphaRegistry()
        self.tracker = PerformanceTracker()

        telegram_cfg = self.keys.get("telegram", {})
        self.telegram = OpenClawTelegramHandler(
            bot_token=telegram_cfg.get("bot_token"),
            chat_id=telegram_cfg.get("chat_id"),
        )
        self.formatter = TelegramFormatter()

        self.lifecycle = LifecycleManager(
            registry=self.registry,
            performance_tracker=self.tracker,
            notifier=self.telegram,
        )

        self.weight_optimizer = WeightOptimizer()
        self.claude_orchestrator = ClaudeEnsembleOrchestrator(
            anthropic_client=self.anthropic_client,
        )
        self.leverage_calc = LeverageCalculator()

        # Build OpenClawDaemon for research (lazy — shares our components)
        from src.openclaw.main import OpenClawDaemon
        self.openclaw_daemon = OpenClawDaemon(dry_run=self.dry_run)
        self.openclaw_daemon._running = True
        self.openclaw_daemon.anthropic_client = self.anthropic_client
        self.openclaw_daemon.binance_api = self.binance_api
        self.openclaw_daemon.registry = self.registry
        self.openclaw_daemon.tracker = self.tracker
        self.openclaw_daemon.telegram = self.telegram
        self.openclaw_daemon.formatter = self.formatter
        self.openclaw_daemon.lifecycle = self.lifecycle
        self.openclaw_daemon.weight_optimizer = self.weight_optimizer
        self.openclaw_daemon.claude_orchestrator = self.claude_orchestrator
        self.openclaw_daemon.leverage_calc = self.leverage_calc

        # Research components (lazy init — only loaded when /research is called)
        try:
            from src.openclaw.researcher.brave_search import BraveSearchClient
            from src.openclaw.researcher.idea_parser import IdeaParser
            from src.openclaw.researcher.code_generator import AlphaCodeGenerator
            from src.openclaw.researcher.code_validator import CodeValidator
            from src.openclaw.researcher.experiment_tracker import ExperimentTracker
            from src.openclaw.validator.quality_gates import QualityGateChecker
            from src.openclaw.validator.correlation_checker import CorrelationChecker
            from src.openclaw.validator.summary_builder import SummaryBuilder

            self.openclaw_daemon.brave_search = BraveSearchClient(
                api_key=self.keys.get("BRAVE_API_KEY")
            )
            self.openclaw_daemon.idea_parser = IdeaParser(self.anthropic_client)
            self.openclaw_daemon.code_generator = AlphaCodeGenerator(self.anthropic_client)
            self.openclaw_daemon.code_validator = CodeValidator()
            self.openclaw_daemon.experiment_tracker = ExperimentTracker()
            self.openclaw_daemon.quality_gates = QualityGateChecker()
            self.openclaw_daemon.correlation_checker = CorrelationChecker()
            self.openclaw_daemon.summary_builder = SummaryBuilder()
        except Exception as e:
            logger.warning(f"Research components init failed (non-critical): {e}")

        # Register OpenClaw Telegram commands
        self.telegram.register_command("research", self.openclaw_daemon._cmd_research)
        self.telegram.register_command("status", self.openclaw_daemon._cmd_status)
        self.telegram.register_command("kill", self.openclaw_daemon._cmd_kill)
        self.telegram.register_command("approve", self.openclaw_daemon._cmd_approve)
        self.telegram.register_command("reject", self.openclaw_daemon._cmd_reject)
        self.telegram.register_command("mutate", self.openclaw_daemon._cmd_mutate)

        # Chat handler
        try:
            from src.openclaw.telegram.chat_handler import ChatHandler
            chat_handler = ChatHandler(
                anthropic_client=self.anthropic_client,
                registry=self.registry,
                tracker=self.tracker,
            )
            self.telegram.set_chat_handler(chat_handler)
        except Exception as e:
            logger.warning(f"Chat handler init failed: {e}")

        logger.info(
            f"OpenClaw initialized: {self.registry.active_count} active, "
            f"{len(self.registry.get_paper())} paper, "
            f"{self.registry.total_count} total"
        )

    def _init_rebalancer(self) -> None:
        from src.openclaw.execution.rebalancer import Rebalancer

        self.rebalancer = Rebalancer(
            binance_api=self.binance_api,
            notifier=self.telegram,
            dry_run=self.dry_run,
        )
        logger.info("Rebalancer initialized")

    def _init_sonnet(self) -> None:
        """Initialize Sonnet decision maker, position store, and SL/TP monitor."""
        from src.openclaw.config import OPENCLAW_DATA_DIR

        self.sonnet_decision_maker = SonnetDecisionMaker(
            anthropic_client=self.anthropic_client,
            model=SONNET_DECISION.get("model"),
        )
        logger.info(f"Sonnet decision maker initialized ({SONNET_DECISION['model']})")

        self.position_store = PositionStore(
            path=OPENCLAW_DATA_DIR / "managed_positions.json",
        )
        logger.info(
            f"Position store: {len(self.position_store.get_active())} active positions"
        )

        self.sltp_monitor = SLTPMonitor(
            position_store=self.position_store,
            binance_api=self.binance_api,
            rebalancer=self.rebalancer,
            notifier=self.telegram,
            dry_run=self.dry_run,
        )
        logger.info("SL/TP monitor initialized (5s interval)")

    def _init_websocket(self) -> None:
        from src.execution.binance_websocket import BinanceWebSocket

        candle_interval = WEBSOCKET_CONFIG.get("candle_interval", "1m")
        self.ws = BinanceWebSocket(candle_interval=candle_interval)
        logger.info(f"WebSocket: {candle_interval} candles")

    def _register_extra_commands(self) -> None:
        """Register additional Telegram commands on top of OpenClaw's."""
        self.telegram.register_command("positions", self._cmd_positions)
        self.telegram.register_command("balance", self._cmd_balance)
        self.telegram.register_command("pause", self._cmd_pause)
        self.telegram.register_command("resume", self._cmd_resume)
        self.telegram.register_command("closeall", self._cmd_closeall)

    # ==================================================================
    # Main Loop
    # ==================================================================

    def run(self) -> None:
        """Main event loop."""
        self._running = True

        # Start WebSocket
        self.ws.start()
        for symbol in self._universe:
            self.ws.subscribe(symbol)
        logger.info(f"WebSocket subscribed: {len(self._universe)} symbols")

        try:
            while self._running:
                now = datetime.now()

                # 1. Poll Telegram commands
                events = self.telegram.poll_commands()
                self._handle_trade_callbacks(events)

                # 1.5. Check SL/TP triggers (every 5 seconds)
                if not self._paused and self.sltp_monitor:
                    try:
                        triggered = self.sltp_monitor.check_all()
                        for t in triggered:
                            logger.info(f"SL/TP triggered: {t}")
                    except Exception as e:
                        logger.error(f"SL/TP monitor error: {e}")

                # 2. Rebalance cycle
                if not self._paused and self._should_rebalance(now):
                    if not self._processing:
                        self._processing = True
                        try:
                            self._run_rebalance_cycle(now)
                            self._last_rebalance = now
                        except Exception as e:
                            logger.error(f"Rebalance error: {e}\n{traceback.format_exc()}")
                            self.telegram.send_message(
                                f"<b>Rebalance Error</b>\n{str(e)[:200]}"
                            )
                        finally:
                            self._processing = False

                # 3. Daily lifecycle check
                if self._should_lifecycle(now):
                    try:
                        summary = self.lifecycle.daily_lifecycle_check()
                        self._last_lifecycle = now
                        logger.info(f"Lifecycle check: {summary}")
                    except Exception as e:
                        logger.error(f"Lifecycle error: {e}")

                # 4. Auto research (24h)
                if self.enable_research and self._should_research(now):
                    try:
                        self.telegram.send_message(
                            "🔬 <b>Auto research session starting...</b>"
                        )
                        self.openclaw_daemon.run_research_session()
                        self._last_research = now
                    except Exception as e:
                        logger.error(f"Research error: {e}")

                time.sleep(5)

        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.stop()

    def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down unified daemon...")
        self._running = False
        if self.ws:
            self.ws.stop()
        if self.openclaw_daemon:
            self.openclaw_daemon.stop()
        logger.info("Daemon stopped")

    # ==================================================================
    # Rebalance Cycle (Core Pipeline)
    # ==================================================================

    def _run_rebalance_cycle(self, now: datetime) -> None:
        """
        Rebalance cycle with Sonnet as decision maker.

        1. Collect raw alpha signals
        2. Gather market context
        3. Get current positions + P&L
        4. Call Sonnet → structured decision
        5. Send decision to Telegram
        6. Execute via Rebalancer
        7. Update PositionStore with SL/TP levels
        """
        logger.info(f"{'='*40} REBALANCE CYCLE {'='*40}")

        # -- Step 1: Collect signals --
        alpha_signals, alpha_entries = self._collect_all_signals(now)
        if not alpha_signals:
            logger.info("No signals generated from any source")
            return
        logger.info(f"Signals from {len(alpha_signals)} alphas")

        # -- Step 1.5: Aggregate signals with config weights + regime --
        alpha_weights = self._config_weights_with_regime(alpha_signals)
        aggregated = self.signal_aggregator.aggregate(
            alpha_signals=alpha_signals,
            alpha_weights=alpha_weights,
        )
        logger.info(
            f"Aggregated: {len(aggregated.scores)} tickers, "
            f"top: {sorted(aggregated.scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]}"
        )

        # -- Step 2: Market context --
        market_context = self._build_market_context()
        market_context["aggregated_scores"] = aggregated.scores
        market_context["alpha_weights"] = aggregated.weights_used

        # -- Step 3: Current state --
        current_positions = self.binance_api.get_positions()
        account = self.binance_api.get_account()
        balance = float(account.get("total_wallet_balance", 0))

        # -- Step 4: Sonnet decision --
        managed = self.position_store.get_active() if self.position_store else None
        decision = self.sonnet_decision_maker.make_decision(
            alpha_signals=alpha_signals,
            current_positions=current_positions,
            account_info=account,
            market_context=market_context,
            managed_positions=managed,
        )

        if not decision.positions:
            logger.info("Sonnet: no positions (staying in cash)")
            self.telegram.send_message(
                f"<b>Sonnet Decision</b>: Stay in cash\n"
                f"{decision.market_assessment}\n"
                f"{decision.risk_note}"
            )
            return

        # -- Step 5: Send to Telegram --
        proposal_text = self._format_sonnet_proposal(decision, balance)

        if self.dry_run:
            self.telegram.send_message(
                f"<b>[DRY RUN]</b>\n\n{proposal_text}"
            )
            logger.info("DRY RUN — no orders placed")
            return

        self.telegram.send_message(proposal_text)

        # -- Step 5.5: Approval gate --
        if self.require_approval:
            from config.settings import DAEMON
            timeout = DAEMON.get("trade_approval_timeout_seconds", 300)
            logger.info(f"Waiting for trade approval ({timeout}s timeout)...")
            self.telegram.send_message(
                "승인 대기 중... /approve 또는 /reject 를 입력하세요."
            )
            approved = self._wait_for_trade_approval(timeout=timeout)
            if not approved:
                logger.info("Trade REJECTED or timed out — skipping execution")
                self.telegram.send_message("거래가 거부되었거나 시간 초과되었습니다.")
                return
            logger.info("Trade APPROVED by user")
            self.telegram.send_message("승인됨 — 주문 실행 중...")

        # -- Step 6: Convert to target weights + execute --
        target_weights = {}
        leverage_per_symbol = {}
        for pos in decision.positions:
            if pos.action == "LONG":
                target_weights[pos.ticker] = pos.weight
            elif pos.action == "SHORT":
                target_weights[pos.ticker] = -pos.weight
            elif pos.action == "CLOSE":
                target_weights[pos.ticker] = 0.0
            # HOLD → don't change

            leverage_per_symbol[pos.ticker] = TRADING.get("max_leverage", 2.0)

        if target_weights:
            logger.info(f"Executing: {len(target_weights)} target positions")
            executed = self.rebalancer.rebalance(
                target_weights=target_weights,
                leverage_per_symbol=leverage_per_symbol,
                total_capital=balance,
            )
            logger.info(f"Rebalance complete: {len(executed)} orders")

            # Log each transaction
            for order in executed:
                SonnetDecisionMaker.log_transaction(
                    ticker=order.get("symbol", ""),
                    action=order.get("side", ""),
                    side=order.get("side", ""),
                    quantity=float(order.get("quantity", 0)),
                    price=float(order.get("price", 0)),
                    notional=float(order.get("notional", 0)),
                    status=order.get("status", ""),
                    reason=f"target_weight={order.get('target_weight', 0):.2f}, diff={order.get('diff', 0):+.2f}",
                )

        # -- Step 7: Update position store with SL/TP --
        # Build a map of fill prices from executed orders
        fill_prices = {}
        if target_weights:
            for order in executed:
                symbol = order.get("symbol", "")
                fill_price = float(order.get("price", 0))
                if symbol and fill_price > 0:
                    fill_prices[symbol] = fill_price

        for pos in decision.positions:
            if pos.action in ("LONG", "SHORT"):
                try:
                    # Prefer actual fill price from executed order, fallback to market price
                    price = fill_prices.get(pos.ticker, 0)
                    if price <= 0:
                        price = self.binance_api.get_price(pos.ticker)
                    if price > 0:
                        self.position_store.upsert(
                            ticker=pos.ticker,
                            side=pos.action,
                            entry_price=price,
                            weight=pos.weight,
                            stop_loss_pct=pos.stop_loss_pct,
                            take_profit_pct=pos.take_profit_pct,
                            reasoning=pos.reasoning,
                        )
                except Exception as e:
                    logger.error(f"Failed to store position {pos.ticker}: {e}")
            elif pos.action == "CLOSE":
                self.position_store.remove(pos.ticker, reason="sonnet_close")

        logger.info(f"{'='*40} CYCLE END {'='*40}")

    def _build_market_context(self) -> dict:
        """Build market context dict for Sonnet prompt."""
        regime = self._detect_regime()

        try:
            btc_data = self._prices[
                self._prices["ticker"].str.contains("BTC", case=False)
            ].sort_values("date")
            close = btc_data["close"].values

            ctx = {
                "regime": regime,
                "btc_price": float(close[-1]) if len(close) > 0 else 0,
                "btc_24h_change": float(close[-1] / close[-2] - 1) if len(close) > 1 else 0,
                "btc_7d_change": float(close[-1] / close[-7] - 1) if len(close) > 7 else 0,
                "btc_30d_change": float(close[-1] / close[-30] - 1) if len(close) > 30 else 0,
            }
        except Exception as e:
            logger.warning(f"Market context build failed: {e}")
            ctx = {"regime": regime, "btc_price": 0, "btc_24h_change": 0, "btc_7d_change": 0, "btc_30d_change": 0}

        # Add current funding rates for held/candidate tickers
        try:
            funding_rates = {}
            if self._features is not None and "funding_rate" in self._features.columns:
                latest = self._features.sort_values("date").groupby("ticker").last()
                for ticker, row in latest.iterrows():
                    fr = row.get("funding_rate")
                    if fr is not None and not pd.isna(fr):
                        funding_rates[str(ticker)] = float(fr)
            ctx["funding_rates"] = funding_rates
        except Exception as e:
            logger.warning(f"Funding rates context failed: {e}")
            ctx["funding_rates"] = {}

        return ctx

    def _format_sonnet_proposal(self, decision: SonnetDecision, balance: float) -> str:
        """Format Sonnet's decision as a Telegram message (Korean)."""
        action_kr = {
            "LONG": "롱 진입",
            "SHORT": "숏 진입",
            "CLOSE": "청산",
            "HOLD": "유지",
        }

        lines = [
            "<b>AI 매매 결정</b>",
            f"시장: {decision.market_assessment}",
            f"잔고: ${balance:.2f}",
            "",
        ]

        for p in decision.positions:
            short_ticker = p.ticker.replace("/USDT:USDT", "")
            action_text = action_kr.get(p.action, p.action)

            lines.append(f"  {action_text} {short_ticker} {p.weight:.0%}")
            if p.action in ("LONG", "SHORT"):
                lines.append(f"    손절: {p.stop_loss_pct:+.1%} | 익절: {p.take_profit_pct:+.1%}")
            lines.append(f"    {p.reasoning}")

        if decision.risk_note:
            lines.append(f"\n위험: {decision.risk_note}")

        return "\n".join(lines)

    # ==================================================================
    # Signal Collection
    # ==================================================================

    def _collect_all_signals(
        self, now: datetime
    ) -> tuple[dict[str, pd.DataFrame], list]:
        """
        Collect signals from 3 sources:
          1. openclaw_1 base alphas (7)
          2. Registry live alphas (OpenClaw-generated)
          3. Trained ensemble model (optional)
        """
        alpha_signals: dict[str, pd.DataFrame] = {}
        alpha_entries = []

        # Source 1: Base alphas
        for name, alpha in self._base_alphas.items():
            try:
                result = alpha.generate_signals(now, self._prices, self._features)
                if result and not result.signals.empty:
                    alpha_signals[name] = result.signals
            except Exception as e:
                logger.error(f"Base alpha {name} signal error: {e}")

        # Source 2: Registry live alphas
        for entry in self.registry.get_active():
            if entry.name in alpha_signals:
                continue  # Don't duplicate base alphas
            try:
                instance = self.openclaw_daemon._load_alpha_instance(entry)
                if instance:
                    result = instance.generate_signals(now, self._prices, self._features)
                    if result and not result.signals.empty:
                        alpha_signals[entry.name] = result.signals
                        alpha_entries.append(entry)
            except Exception as e:
                logger.error(f"Registry alpha {entry.name} signal error: {e}")

        # Source 3: Trained ensemble (as a single "ensemble" alpha)
        if self._ensemble is not None:
            try:
                ensemble_result = self._ensemble.generate_signals(
                    date=now,
                    prices=self._prices,
                    features=self._features,
                )
                if not ensemble_result.signals.empty:
                    alpha_signals["trained_ensemble"] = ensemble_result.signals
            except Exception as e:
                logger.error(f"Ensemble signal error: {e}")

        return alpha_signals, alpha_entries

    def _get_alpha_weights(
        self,
        alpha_signals: dict[str, pd.DataFrame],
        alpha_entries: list,
    ) -> dict[str, float]:
        """
        Get alpha weights: risk-parity → config weights → regime adjustment.

        Priority:
          1. Risk-parity (if performance history exists)
          2. Config-defined weights from STRATEGIES (with regime adjustment)

        Regime adjustment prevents contradictory signals from canceling:
          - Bull: boost momentum, suppress mean reversion
          - Bear: boost mean reversion + carry, suppress momentum
          - Sideways: boost mean reversion + carry, moderate momentum
        """
        from src.openclaw.orchestrator.weight_optimizer import WeightOptimizer

        optimizer = WeightOptimizer()

        # Try risk-parity first (needs performance history)
        alpha_returns = {}
        for name in alpha_signals:
            try:
                returns = self.tracker.get_daily_returns(name)
                if returns is not None and len(returns) > 0:
                    alpha_returns[name] = returns
            except Exception:
                pass

        if len(alpha_returns) >= 2:
            try:
                weights = optimizer.risk_parity(alpha_returns)
                logger.info(f"Risk-parity weights: {weights}")
                return weights
            except Exception as e:
                logger.warning(f"Risk-parity failed: {e}")

        # Fallback: config-defined weights + regime adjustment
        weights = self._config_weights_with_regime(alpha_signals)
        return weights

    def _detect_regime(self) -> str:
        """
        Simple regime detection from BTC price data.

        Returns: 'bull', 'bear', or 'sideways'
        """
        try:
            if self._prices is None or self._prices.empty:
                return "sideways"

            # Use BTC as market proxy
            btc_data = self._prices[
                self._prices["ticker"].str.contains("BTC", case=False)
            ].sort_values("date")

            if len(btc_data) < 30:
                return "sideways"

            close = btc_data["close"].values

            # 20-day return
            ret_20d = (close[-1] / close[-20] - 1) if len(close) >= 20 else 0
            # 60-day return
            ret_60d = (close[-1] / close[-60] - 1) if len(close) >= 60 else 0

            # Regime thresholds
            if ret_20d > 0.05 and ret_60d > 0.10:
                regime = "bull"
            elif ret_20d < -0.05 and ret_60d < -0.10:
                regime = "bear"
            else:
                regime = "sideways"

            logger.info(
                f"Regime: {regime} (BTC 20d: {ret_20d:+.1%}, 60d: {ret_60d:+.1%})"
            )
            return regime

        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return "sideways"

    # Map alpha.name (class name) → config key (snake_case)
    _ALPHA_NAME_TO_CONFIG = {
        "CSMomentum": "cs_momentum",
        "TimeSeriesMomentum": "ts_momentum",
        "TimeSeriesMeanReversion": "ts_mean_reversion",
        "PriceVolumeDivergence": "pv_divergence",
        "VolumeMomentum": "volume_momentum",
        "LowVolatilityAnomaly": "low_volatility_anomaly",
        "FundingRateCarry": "funding_rate_carry",
    }

    def _config_weights_with_regime(
        self, alpha_signals: dict[str, pd.DataFrame]
    ) -> dict[str, float]:
        """
        Use STRATEGIES config weights adjusted by current regime.

        Instead of equal 14% for all 7 alphas, uses the config-defined
        weights (e.g. funding_rate_carry=0.25, cs_momentum=0.20) and
        adjusts them based on the market regime to prevent contradictory
        signals from canceling each other out.
        """
        regime = self._detect_regime()
        regime_multipliers = ENSEMBLE.get("regime_preferences", {}).get(regime, {})

        raw_weights = {}
        for name in alpha_signals:
            # Map class name → config key
            config_key = self._ALPHA_NAME_TO_CONFIG.get(name, name)

            # Get base weight from config
            config_w = STRATEGIES.get(config_key, {}).get("weight", 0.10)

            # Apply regime multiplier (also uses config_key)
            multiplier = regime_multipliers.get(config_key, 1.0)
            raw_weights[name] = config_w * multiplier

        # Normalize to sum=1
        total = sum(raw_weights.values())
        if total <= 0:
            n = len(alpha_signals)
            return {name: 1.0 / n for name in alpha_signals}

        weights = {name: w / total for name, w in raw_weights.items()}

        logger.info(
            f"Config+regime weights ({regime}): "
            + ", ".join(f"{k}={v:.1%}" for k, v in sorted(weights.items(), key=lambda x: -x[1]))
        )

        return weights

    def _compute_leverage(
        self,
        alpha_entries: list,
        alpha_weights: dict[str, float],
    ) -> dict[str, float]:
        """Compute per-symbol leverage."""
        try:
            leverages = self.leverage_calc.calculate_per_alpha(
                alpha_entries, self.tracker
            )
            avg = self.leverage_calc.calculate_portfolio_leverage(
                leverages, alpha_weights
            )
        except Exception:
            avg = 1.0

        # Apply same leverage to all symbols
        return {t: avg for t in self._universe}

    def _get_current_weights(self) -> dict[str, float]:
        """Get current position weights from Binance."""
        try:
            positions = self.binance_api.get_positions()
            account = self.binance_api.get_account()
            balance = account.get("total_wallet_balance", 0)
            if balance <= 0 or positions.empty:
                return {}

            weights = {}
            for _, row in positions.iterrows():
                size = float(row.get("size", 0))
                entry_price = float(row.get("entry_price", 0))
                notional = size * entry_price
                side = row.get("side", "long")
                w = notional / balance
                if side == "short":
                    w = -w
                weights[row["ticker"]] = w
            return weights
        except Exception as e:
            logger.error(f"Failed to get current weights: {e}")
            return {}

    # ==================================================================
    # Approval Workflow
    # ==================================================================

    def _wait_for_trade_approval(self, timeout: int = 300) -> bool:
        """Wait for trade_approve / trade_reject callback."""
        deadline = time.time() + timeout

        while time.time() < deadline and self._running:
            events = self.telegram.poll_commands()
            for event in events:
                if event.get("type") == "callback":
                    data = event.get("data", "")
                    if data == "trade_approve":
                        return True
                    elif data == "trade_reject":
                        return False
            time.sleep(2)

        return False  # Timeout

    def _handle_trade_callbacks(self, events: list[dict]) -> None:
        """Handle trade approval callbacks from poll_commands results."""
        # Trade callbacks are handled in _wait_for_trade_approval
        # This catches callbacks that arrive outside the approval window
        pass

    # ==================================================================
    # Extra Telegram Commands
    # ==================================================================

    def _cmd_positions(self, args: str) -> None:
        """Handle /positions — show current Binance positions."""
        try:
            positions = self.binance_api.get_positions()
            if positions.empty:
                self.telegram.send_message("포지션 없음")
                return

            lines = ["<b>현재 포지션</b>\n"]
            for _, row in positions.iterrows():
                ticker = row["ticker"].replace("/USDT:USDT", "")
                side = row["side"].upper()
                size = row["size"]
                pnl = row.get("unrealized_pnl", 0)
                lev = row.get("leverage", 1)
                lines.append(
                    f"  {ticker} {side} {size:.4f} "
                    f"PnL: ${float(pnl):+,.2f} ({lev}x)"
                )

            self.telegram.send_message("\n".join(lines))
        except Exception as e:
            self.telegram.send_message(f"Error: {e}")

    def _cmd_balance(self, args: str) -> None:
        """Handle /balance — show account balance."""
        try:
            account = self.binance_api.get_account()
            total = account.get("total_wallet_balance", 0)
            available = account.get("available_balance", 0)
            upnl = account.get("total_unrealized_pnl", 0)
            self.telegram.send_message(
                f"<b>계좌 잔고</b>\n\n"
                f"총 자산: ${total:,.2f}\n"
                f"가용 잔고: ${available:,.2f}\n"
                f"미실현 PnL: ${upnl:+,.2f}"
            )
        except Exception as e:
            self.telegram.send_message(f"Error: {e}")

    def _cmd_pause(self, args: str) -> None:
        """Handle /pause — pause trading."""
        self._paused = True
        self.telegram.send_message("⏸ 매매 일시 중지됨")
        logger.info("Trading paused by user")

    def _cmd_resume(self, args: str) -> None:
        """Handle /resume — resume trading."""
        self._paused = False
        self.telegram.send_message("▶ 매매 재개")
        logger.info("Trading resumed by user")

    def _cmd_closeall(self, args: str) -> None:
        """Handle /closeall — close all positions."""
        self._paused = True
        self.telegram.send_message(
            "🔴 <b>전량 청산 시작</b>\n매매 자동 중지됨",
        )
        logger.info("Close all positions requested")

        try:
            results = self.rebalancer.close_all_positions()
            closed = [r for r in results if r.get("status") in ("closed", "dry_run_close")]
            errors = [r for r in results if r.get("status") == "error"]

            lines = [f"<b>청산 완료</b>\n"]
            for r in closed:
                lines.append(f"  {r['symbol']} — 청산됨")
            for r in errors:
                lines.append(f"  {r['symbol']} — 실패: {r.get('error', '?')}")

            if not results:
                lines.append("열린 포지션 없음")

            lines.append(f"\n매매 재개: /resume")
            self.telegram.send_message("\n".join(lines))
        except Exception as e:
            logger.error(f"Close all failed: {e}")
            self.telegram.send_message(f"청산 실패: {e}")

    # ==================================================================
    # Timing Helpers
    # ==================================================================

    def _should_rebalance(self, now: datetime) -> bool:
        if self._last_rebalance is None:
            return True  # First rebalance runs immediately
        elapsed = (now - self._last_rebalance).total_seconds()
        return elapsed >= self._rebalance_interval * 60

    def _should_lifecycle(self, now: datetime) -> bool:
        if self._last_lifecycle is None:
            # Delay first lifecycle check by 1 hour to avoid startup congestion
            self._last_lifecycle = now - timedelta(hours=23)
            return False
        return (now - self._last_lifecycle) > timedelta(hours=24)

    def _should_research(self, now: datetime) -> bool:
        if self._last_research is None:
            # Delay first research by 2 hours to avoid startup congestion
            self._last_research = now - timedelta(hours=22)
            return False
        return (now - self._last_research) > timedelta(hours=24)
