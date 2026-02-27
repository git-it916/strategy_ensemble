"""
OpenClaw Main Daemon

Entry point for the autonomous alpha research and trading system.

Main loop:
  - Telegram command polling: every 5 seconds
  - Signal generation + rebalance: every 15 minutes
  - Lifecycle check: once per day
  - Research sessions: triggered by /research command
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
import yaml

from src.openclaw.config import (
    ASSETS,
    EXECUTION_POLICY,
    GENERATED_ALPHAS_DIR,
    MAX_LOOKBACK_DAYS,
    OPENCLAW_DATA_DIR,
    OPENCLAW_LOGS_DIR,
    QUALITY_GATES,
    RESEARCH_POLICY,
)
from src.openclaw.execution.leverage_calculator import LeverageCalculator
from src.openclaw.execution.rebalancer import Rebalancer
from src.openclaw.mutation.mutation_orchestrator import MutationOrchestrator
from src.openclaw.orchestrator.claude_orchestrator import ClaudeEnsembleOrchestrator
from src.openclaw.orchestrator.weight_optimizer import WeightOptimizer
from src.openclaw.registry.alpha_registry import AlphaEntry, AlphaRegistry
from src.openclaw.registry.lifecycle_manager import LifecycleManager
from src.openclaw.registry.performance_tracker import PerformanceTracker
from src.openclaw.researcher.brave_search import BraveSearchClient
from src.openclaw.researcher.code_generator import AlphaCodeGenerator
from src.openclaw.researcher.code_validator import CodeValidator
from src.openclaw.researcher.experiment_tracker import ExperimentTracker
from src.openclaw.researcher.idea_parser import IdeaParser
from src.openclaw.telegram.command_handler import OpenClawTelegramHandler
from src.openclaw.telegram.formatters import TelegramFormatter
from src.openclaw.validator.backtest_runner import SingleAlphaBacktestRunner
from src.openclaw.validator.correlation_checker import CorrelationChecker
from src.openclaw.validator.quality_gates import QualityGateChecker
from src.openclaw.validator.summary_builder import SummaryBuilder

logger = logging.getLogger(__name__)


class OpenClawDaemon:
    """
    Main daemon orchestrating all OpenClaw subsystems.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._running = False
        self._last_lifecycle_check: datetime | None = None
        self._last_rebalance: datetime | None = None
        self._llm_calls_this_session = 0

        # Initialize directories
        OPENCLAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        OPENCLAW_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        GENERATED_ALPHAS_DIR.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing OpenClaw daemon...")

        # Anthropic client - load from keys.yaml
        keys_path = Path(__file__).parent.parent.parent / "config" / "keys.yaml"
        if not keys_path.exists():
            raise RuntimeError("config/keys.yaml not found! Copy from keys.example.yaml")
        with open(keys_path) as f:
            keys = yaml.safe_load(f)
        api_key = keys.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in config/keys.yaml")
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)

        # Binance API client
        from src.execution.binance_api import BinanceApi
        binance_cfg = keys.get("binance", {})
        self.binance_api = BinanceApi(
            api_key=binance_cfg.get("api_key", ""),
            api_secret=binance_cfg.get("api_secret", ""),
        )

        # Core components
        self.registry = AlphaRegistry()
        self.tracker = PerformanceTracker()
        telegram_cfg = keys.get("telegram", {})
        self.telegram = OpenClawTelegramHandler(
            bot_token=telegram_cfg.get("bot_token"),
            chat_id=telegram_cfg.get("chat_id"),
        )
        self.formatter = TelegramFormatter()

        # Lifecycle
        self.lifecycle = LifecycleManager(
            registry=self.registry,
            performance_tracker=self.tracker,
            notifier=self.telegram,
        )

        # Orchestrator
        self.weight_optimizer = WeightOptimizer()
        self.claude_orchestrator = ClaudeEnsembleOrchestrator(
            anthropic_client=self.anthropic_client,
        )
        self.leverage_calc = LeverageCalculator()

        # Research components
        self.brave_search = BraveSearchClient(api_key=keys.get("BRAVE_API_KEY"))
        self.idea_parser = IdeaParser(self.anthropic_client)
        self.code_generator = AlphaCodeGenerator(self.anthropic_client)
        self.code_validator = CodeValidator()
        self.experiment_tracker = ExperimentTracker()
        self.quality_gates = QualityGateChecker()
        self.correlation_checker = CorrelationChecker()
        self.summary_builder = SummaryBuilder()

        # Chat handler (natural language)
        from src.openclaw.telegram.chat_handler import ChatHandler
        self.chat_handler = ChatHandler(
            anthropic_client=self.anthropic_client,
            registry=self.registry,
            tracker=self.tracker,
        )
        self.telegram.set_chat_handler(self.chat_handler)

        # Register Telegram commands
        self.telegram.register_command("research", self._cmd_research)
        self.telegram.register_command("status", self._cmd_status)
        self.telegram.register_command("kill", self._cmd_kill)
        self.telegram.register_command("approve", self._cmd_approve)
        self.telegram.register_command("reject", self._cmd_reject)
        self.telegram.register_command("mutate", self._cmd_mutate)

        logger.info(
            f"OpenClaw initialized: "
            f"{self.registry.active_count} active, "
            f"{len(self.registry.get_paper())} paper, "
            f"{self.registry.total_count} total alphas"
        )

        # Send startup notification
        self.telegram.send_message(
            f"<b>OpenClaw Started</b>\n\n"
            f"Active: {self.registry.active_count}\n"
            f"Paper: {len(self.registry.get_paper())}\n"
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}"
        )

    def run(self) -> None:
        """
        Main event loop.

        - Poll Telegram commands every 5 seconds
        - Run lifecycle check once per day
        - Rebalance on signal interval (15 min)
        """
        self._running = True
        logger.info("OpenClaw daemon running...")

        while self._running:
            try:
                now = datetime.now()

                # 1. Poll Telegram commands
                self.telegram.poll_commands()

                # 2. Daily lifecycle check
                if self._should_run_lifecycle(now):
                    self._run_lifecycle_check()
                    self._last_lifecycle_check = now

                # 3. Periodic rebalance
                if self._should_rebalance(now):
                    self._run_rebalance()
                    self._last_rebalance = now

                # Sleep between polls
                time.sleep(5)

            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                self._running = False
            except Exception as e:
                logger.error(f"Main loop error: {e}\n{traceback.format_exc()}")
                try:
                    self.telegram.send_message(
                        self.formatter.error_alert(str(e), "main_loop")
                    )
                except Exception:
                    pass
                time.sleep(30)

        logger.info("OpenClaw daemon stopped")

    def stop(self) -> None:
        """Stop the daemon."""
        self._running = False

    # ── Command Handlers ──────────────────────────────────────────────

    def _cmd_research(self, args: str) -> None:
        """Handle /research [query] command."""
        query = args.strip() if args else None
        self.telegram.send_message(
            self.formatter.research_started(query or "default themes")
        )

        try:
            self.run_research_session(query=query)
        except Exception as e:
            logger.error(f"Research session failed: {e}")
            self.telegram.send_message(
                self.formatter.error_alert(str(e), "research")
            )

    def _cmd_status(self, args: str) -> None:
        """Handle /status command."""
        active = []
        for entry in self.registry.get_active():
            summary = self.tracker.get_summary(entry.name)
            active.append({
                "name": entry.name,
                "sharpe": summary["sharpe"],
                "mdd": summary["mdd"],
                "leverage": entry.current_leverage,
                "weight": entry.current_weight,
            })

        paper = []
        for entry in self.registry.get_paper():
            summary = self.tracker.get_summary(entry.name)
            paper.append({
                "name": entry.name,
                "sharpe": summary["sharpe"],
                "n_days": summary["n_days"],
            })

        total_pnl = sum(
            self.tracker.get_summary(e.name).get("total_return", 0)
            for e in self.registry.get_active()
        )

        msg = self.formatter.status_report(active, paper, total_pnl)
        self.telegram.send_message(msg)

    def _cmd_kill(self, args: str) -> None:
        """Handle /kill <name> command."""
        name = args.strip()
        if not name:
            self.telegram.send_message("Usage: /kill <alpha_name>")
            return

        entry = self.registry.get(name)
        if not entry:
            self.telegram.send_message(f"Alpha not found: {name}")
            return

        self.registry.kill(name, reason="manual_kill")
        summary = self.tracker.get_summary(name)
        msg = self.formatter.alpha_killed(
            name, "Manual kill via Telegram",
            summary.get("total_return", 0),
            summary.get("sharpe", 0),
            summary.get("n_days", 0),
        )
        self.telegram.send_message(msg)

    def _cmd_approve(self, args: str) -> None:
        """Handle /approve <name> command."""
        name = args.strip()
        if not name:
            self.telegram.send_message("Usage: /approve <alpha_name>")
            return

        entry = self.registry.get(name)
        if not entry:
            self.telegram.send_message(f"Alpha not found: {name}")
            return

        if entry.status != "pending":
            self.telegram.send_message(
                f"{name} is not pending (status: {entry.status})"
            )
            return

        self.registry.activate(name, status="paper")
        self.telegram.send_message(
            f"Approved: {name} → paper trading (14 days)"
        )

    def _cmd_reject(self, args: str) -> None:
        """Handle /reject <name> command."""
        name = args.strip()
        if not name:
            self.telegram.send_message("Usage: /reject <alpha_name>")
            return

        entry = self.registry.get(name)
        if entry:
            self.registry.kill(name, reason="rejected_by_user")
            self.telegram.send_message(f"Rejected: {name}")
        else:
            self.telegram.send_message(f"Alpha not found: {name}")

    def _cmd_mutate(self, args: str) -> None:
        """Handle /mutate [name] command."""
        self.telegram.send_message("Starting mutation cycle...")

        target = [args.strip()] if args.strip() else None

        try:
            prices, features = self._load_data()
            runner = SingleAlphaBacktestRunner(prices, features)

            orchestrator = MutationOrchestrator(
                backtest_runner=runner,
                registry=self.registry,
                notifier=self.telegram,
            )

            results = orchestrator.run_mutation_cycle(target_alphas=target)
            self.telegram.send_message(
                f"Mutation complete: {len(results)} results"
            )
        except Exception as e:
            self.telegram.send_message(
                self.formatter.error_alert(str(e), "mutation")
            )

    # ── Research Session ──────────────────────────────────────────────

    def run_research_session(self, query: str | None = None) -> None:
        """
        Full research pipeline:
        1. BraveSearch → AlphaIdeas
        2. IdeaParser → AlphaSpecs
        3. CodeGenerator → GeneratedAlpha
        4. CodeValidator → pass/fail
        5. BacktestRunner → IS/OOS metrics
        6. QualityGates → pass/fail
        7. CorrelationChecker → pass/fail
        8. Telegram approval → pending/rejected
        """
        self._llm_calls_this_session = 0

        # 1. Search
        self.telegram.send_message(
            self.formatter.research_progress("Searching for ideas...")
        )
        ideas = self.brave_search.search_alpha_ideas(query=query)

        if not ideas:
            self.telegram.send_message("No ideas found. Try a different query.")
            return

        # 2. Rank and parse
        self.telegram.send_message(
            self.formatter.research_progress(
                f"Found {len(ideas)} ideas, ranking..."
            )
        )
        ranked_ideas = self.idea_parser.rank_ideas(ideas)
        self._llm_calls_this_session += 1

        # 3. Load data for backtesting
        prices, features = self._load_data()
        runner = SingleAlphaBacktestRunner(prices, features)

        # 4. Process top ideas
        max_ideas = RESEARCH_POLICY.max_ideas_per_session
        processed = 0

        for idea in ranked_ideas[:max_ideas * 2]:  # extra buffer for skips
            if not self._running:
                logger.info("Research interrupted — daemon stopping")
                break

            if processed >= max_ideas:
                break

            if self._llm_calls_this_session >= RESEARCH_POLICY.max_llm_calls_per_session:
                self.telegram.send_message(
                    f"LLM call limit reached ({self._llm_calls_this_session})"
                )
                break

            try:
                self._process_single_idea(idea, runner, prices)
                processed += 1
            except Exception as e:
                logger.error(f"Failed to process idea '{idea.title[:50]}': {e}")

        # Summary
        stats = self.experiment_tracker.get_stats()
        self.telegram.send_message(
            f"<b>Research Session Complete</b>\n\n"
            f"Ideas processed: {processed}\n"
            f"LLM calls used: {self._llm_calls_this_session}\n"
            f"Total experiments: {stats['total']}\n"
            f"Pass rate: {stats['pass_rate']:.1%}"
        )

    def _process_single_idea(
        self,
        idea,
        runner: SingleAlphaBacktestRunner,
        prices: pd.DataFrame,
    ) -> None:
        """Process a single alpha idea through the full pipeline."""
        # Fetch content
        content = self.brave_search.fetch_content(idea.url)

        # Parse to spec
        spec = self.idea_parser.parse_idea(idea, extra_content=content)
        self._llm_calls_this_session += 1

        if spec is None:
            return

        # Check duplicate
        if self.experiment_tracker.is_duplicate(spec):
            logger.info(f"Skipping duplicate: {spec.name}")
            return

        self.telegram.send_message(
            self.formatter.research_progress(
                f"Generating code for: {spec.name}",
                spec.hypothesis[:100],
            )
        )

        # Generate code
        generated = self.code_generator.generate(spec)
        self._llm_calls_this_session += generated.generation_metadata.get(
            "attempts", 1
        )

        # Validate code (AST)
        validation = self.code_validator.validate(generated.code)
        if not validation.is_valid:
            self.experiment_tracker.record_experiment(
                spec, {"errors": validation.errors}, "failed"
            )
            return

        # Backtest
        self.telegram.send_message(
            self.formatter.research_progress(f"Backtesting: {spec.name}")
        )

        bt_result = runner.run(
            alpha_code=generated.code,
            class_name=generated.class_name,
        )

        # Quality gates
        passed, failures, warnings = self.quality_gates.check_all(
            is_metrics=bt_result["is_metrics"],
            oos_metrics=bt_result["oos_metrics"],
            daily_returns=bt_result["daily_returns"],
            signals_history=bt_result.get("oos_result", {})
            and bt_result["oos_result"].signals_history,
            prices=prices,
            turnover=bt_result.get("turnover", 0),
        )

        # Correlation check
        corr_passed = True
        correlations = {}
        active_returns = {
            e.name: self.tracker.get_daily_returns(e.name)
            for e in self.registry.get_active()
        }
        if active_returns:
            corr_passed, correlations = self.correlation_checker.check(
                bt_result["oos_daily_returns"],
                active_returns,
            )

        all_passed = passed and corr_passed

        # Record experiment
        self.experiment_tracker.record_experiment(
            spec,
            {
                "is_metrics": bt_result["is_metrics"],
                "oos_metrics": bt_result["oos_metrics"],
                "turnover": bt_result.get("turnover", 0),
                "correlations": correlations,
                "gate_passed": all_passed,
                "failures": failures,
            },
            status="passed" if all_passed else "failed",
            code_hash=AlphaRegistry.compute_code_hash(generated.code),
        )

        if not all_passed:
            self.telegram.send_message(
                f"<b>{spec.name}</b> failed gates:\n"
                + "\n".join(f"  - {f}" for f in failures)
            )
            return

        # Save generated code
        code_path = GENERATED_ALPHAS_DIR / f"{generated.module_name}.py"
        code_path.write_text(generated.code)

        # Build approval summary
        summary_text, keyboard = self.formatter.alpha_approval_request(
            alpha_name=spec.name,
            hypothesis=spec.hypothesis,
            source=spec.source_title,
            oos_sharpe=bt_result["oos_metrics"].get("sharpe_ratio", 0),
            oos_mdd=bt_result["oos_metrics"].get("max_drawdown", 0),
            oos_return=bt_result["oos_metrics"].get("total_return", 0),
            is_sharpe=bt_result["is_metrics"].get("sharpe_ratio", 0),
            turnover=bt_result.get("turnover", 0),
            correlations=correlations,
            gate_passed=True,
        )

        # Register as pending
        entry = AlphaEntry(
            name=spec.name,
            class_name=generated.class_name,
            module_path=str(code_path),
            status="pending",
            source_url=spec.source_url,
            source_title=spec.source_title,
            hypothesis=spec.hypothesis,
            is_sharpe=bt_result["is_metrics"].get("sharpe_ratio", 0),
            oos_sharpe=bt_result["oos_metrics"].get("sharpe_ratio", 0),
            oos_mdd=bt_result["oos_metrics"].get("max_drawdown", 0),
            code_hash=AlphaRegistry.compute_code_hash(generated.code),
        )
        self.registry.add(entry)

        # Request approval via Telegram
        approved = self.telegram.request_alpha_approval(
            alpha_name=spec.name,
            summary_text=summary_text,
            reply_markup=keyboard,
        )

        if approved:
            self.registry.activate(spec.name, status="paper")
            logger.info(f"Alpha approved for paper trading: {spec.name}")
        else:
            self.registry.kill(spec.name, reason="rejected_by_user")
            logger.info(f"Alpha rejected: {spec.name}")

    # ── Trading Loop ──────────────────────────────────────────────────

    def _run_rebalance(self) -> None:
        """Run signal generation + rebalance cycle."""
        active_alphas = self.registry.get_active()
        if not active_alphas:
            return

        try:
            prices, features = self._load_data()

            # Generate signals from each alpha
            all_signals = {}
            for entry in active_alphas:
                try:
                    alpha = self._load_alpha_instance(entry)
                    if alpha and alpha.is_fitted:
                        result = alpha.generate_signals(
                            datetime.now(), prices, features
                        )
                        all_signals[entry.name] = result
                except Exception as e:
                    logger.error(f"Signal gen failed for {entry.name}: {e}")

            if not all_signals:
                return

            # Get weights from orchestrator
            perf_data = {
                name: self.tracker.get_summary(name)
                for name in all_signals
            }
            corr_matrix = self.tracker.get_correlation_matrix(
                list(all_signals.keys())
            )

            # Try Opus orchestrator, fallback to risk-parity
            try:
                alpha_weights = self.claude_orchestrator.decide_weights(
                    active_alphas, perf_data, corr_matrix
                )
            except Exception:
                alpha_returns = {
                    name: self.tracker.get_daily_returns(name)
                    for name in all_signals
                }
                rp_weights = self.weight_optimizer.risk_parity(alpha_returns)
                alpha_weights = rp_weights

            # Combine signals weighted by alpha weights
            combined_scores: dict[str, float] = {}
            for alpha_name, result in all_signals.items():
                weight = alpha_weights.get(alpha_name, 0)
                for _, row in result.signals.iterrows():
                    ticker = row["ticker"]
                    score = row["score"] * weight
                    combined_scores[ticker] = (
                        combined_scores.get(ticker, 0) + score
                    )

            # Calculate leverage
            leverages = self.leverage_calc.calculate_per_alpha(
                active_alphas, self.tracker
            )
            avg_leverage = self.leverage_calc.calculate_portfolio_leverage(
                leverages, alpha_weights
            )

            # Convert to target weights
            total_abs = sum(abs(v) for v in combined_scores.values())
            if total_abs > 0:
                target_weights = {
                    k: v / total_abs for k, v in combined_scores.items()
                    if abs(v / total_abs) > 0.02  # min 2% weight
                }
            else:
                target_weights = {}

            # Update registry weights
            for name, w in alpha_weights.items():
                self.registry.update_weight(name, w)

            logger.info(
                f"Rebalance: {len(target_weights)} positions, "
                f"avg leverage={avg_leverage:.1f}x"
            )

        except Exception as e:
            logger.error(f"Rebalance failed: {e}\n{traceback.format_exc()}")

    def _run_lifecycle_check(self) -> None:
        """Run daily lifecycle check."""
        summary = self.lifecycle.daily_lifecycle_check()
        logger.info(f"Lifecycle check: {summary}")

    # ── Helpers ────────────────────────────────────────────────────────

    def _should_run_lifecycle(self, now: datetime) -> bool:
        if self._last_lifecycle_check is None:
            return True
        return (now - self._last_lifecycle_check) > timedelta(hours=24)

    def _should_rebalance(self, now: datetime) -> bool:
        if self._last_rebalance is None:
            return True
        interval = timedelta(
            minutes=EXECUTION_POLICY.signal_interval_minutes
        )
        return (now - self._last_rebalance) > interval

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Fetch 1m candles directly from Binance via ccxt with pagination."""
        from datetime import timezone

        total_minutes = MAX_LOOKBACK_DAYS * 24 * 60
        since = datetime.now(timezone.utc) - timedelta(days=MAX_LOOKBACK_DAYS)
        logger.info(f"Fetching {MAX_LOOKBACK_DAYS}d 1m candles for {ASSETS} from Binance...")

        all_records: list[pd.DataFrame] = []
        for symbol in ASSETS:
            try:
                candles: list = []
                fetch_since = int(since.timestamp() * 1000)
                while len(candles) < total_minutes:
                    batch = self.binance_api.get_ohlcv(
                        symbol, timeframe="1m", since=datetime.fromtimestamp(fetch_since / 1000, tz=timezone.utc), limit=1500,
                    )
                    if not batch:
                        break
                    candles.extend(batch)
                    fetch_since = batch[-1][0] + 60_000  # next minute
                    time.sleep(0.2)

                if candles:
                    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
                    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                    df["ticker"] = symbol
                    df = df.drop(columns=["ts"])
                    all_records.append(df)
                    logger.info(f"  {symbol}: {len(df)} candles")
            except Exception as e:
                logger.error(f"OHLCV fetch failed for {symbol}: {e}")

        if not all_records:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"]), None

        prices = pd.concat(all_records, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
        logger.info(f"Loaded {len(prices)} total rows from Binance")
        return prices, None

    def _load_alpha_instance(self, entry: AlphaEntry):
        """Load an alpha instance from its saved module."""
        try:
            from src.openclaw.validator.backtest_runner import SingleAlphaBacktestRunner

            module_path = Path(entry.module_path)
            if not module_path.exists():
                return None

            code = module_path.read_text()

            import types
            import numpy as np
            from src.alphas.base_alpha import AlphaResult, BaseAlpha

            module = types.ModuleType(f"openclaw_{entry.name}")
            module.__dict__["np"] = np
            module.__dict__["numpy"] = np
            module.__dict__["pd"] = pd
            module.__dict__["pandas"] = pd
            module.__dict__["BaseAlpha"] = BaseAlpha
            module.__dict__["AlphaResult"] = AlphaResult
            module.__dict__["datetime"] = datetime
            module.__dict__["Any"] = Any

            exec(code, module.__dict__)

            alpha_cls = module.__dict__.get(entry.class_name)
            if alpha_cls:
                instance = alpha_cls()
                instance.is_fitted = True
                return instance

        except Exception as e:
            logger.error(f"Failed to load alpha {entry.name}: {e}")

        return None
