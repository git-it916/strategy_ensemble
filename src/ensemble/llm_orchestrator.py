"""
LLM Ensemble Orchestrator

Replaces the weighted average signal combination with Gemini 2.0 Flash reasoning.
Maintains full backward compatibility with EnsembleAgent interface.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd

from .agent import EnsembleAgent, EnsembleSignal
from ..alphas.base_alpha import BaseAlpha
from ..llm.gemini_client import GeminiClient, MODEL_GEMINI_FLASH
from ..llm.prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    build_orchestrator_prompt,
)
from ..llm.reasoning_logger import ReasoningLogger

logger = logging.getLogger(__name__)


class LLMEnsembleOrchestrator(EnsembleAgent):
    """
    LLM-enhanced ensemble agent using Gemini 2.0 Flash
    for signal combination instead of weighted average.

    Inherits from EnsembleAgent to maintain full compatibility:
        - Same __init__ signature
        - Same generate_signals() return type (EnsembleSignal)
        - Same fit() behavior
        - Falls back to weighted average if LLM is unavailable

    Overrides only _combine_signals() to use LLM reasoning.
    """

    def __init__(
        self,
        strategies: list[BaseAlpha],
        config: dict[str, Any] | None = None,
        gemini_client: GeminiClient | None = None,
        reasoning_logger: ReasoningLogger | None = None,
    ):
        super().__init__(strategies, config)

        self._llm_client = gemini_client
        self._reasoning_logger = reasoning_logger

        # LLM-specific config
        from config.settings import LLM_CONFIG
        self.llm_model = (config or {}).get("llm_model", LLM_CONFIG.get("gemini_model", MODEL_GEMINI_FLASH))
        self.llm_temperature = (config or {}).get("llm_temperature", LLM_CONFIG.get("gemini_temperature", 0.3))
        self.use_llm = (config or {}).get("use_llm_orchestrator", True)
        self.llm_timeout = (config or {}).get("llm_timeout", LLM_CONFIG.get("gemini_timeout", 60.0))

        # Store LLM-suggested weights for analysis
        self._llm_suggested_weights: dict[str, float] = {}

    @property
    def llm_client(self) -> GeminiClient:
        """Lazy init of Gemini client."""
        if self._llm_client is None:
            from config.settings import LLM_CONFIG
            import yaml
            from pathlib import Path

            # Load API key from keys.yaml
            keys_path = Path(__file__).parent.parent.parent / "config" / "keys.yaml"
            with open(keys_path) as f:
                keys = yaml.safe_load(f)
            api_key = keys.get("gemini", {}).get("api_key", "")

            self._llm_client = GeminiClient(
                api_key=api_key,
                default_model=self.llm_model,
                timeout=self.llm_timeout,
                max_retries=LLM_CONFIG.get("max_retries", 2),
                retry_delay=LLM_CONFIG.get("retry_delay", 5.0),
            )
        return self._llm_client

    @property
    def reasoning_logger(self) -> ReasoningLogger:
        """Lazy init of reasoning logger."""
        if self._reasoning_logger is None:
            self._reasoning_logger = ReasoningLogger()
        return self._reasoning_logger

    def _combine_signals(
        self,
        all_signals: dict[str, pd.DataFrame],
        date: datetime,
    ) -> pd.DataFrame:
        """
        Override: Use Gemini 2.0 Flash for signal combination.
        Falls back to parent's weighted average if LLM is unavailable.
        """
        if not self.use_llm or not all_signals:
            return super()._combine_signals(all_signals, date)

        try:
            # Gather context
            strategy_performance = self.score_board.get_recent_performance(
                lookback=self.performance_lookback
            )
            perf_correlations = self.score_board.get_correlation_matrix()
            signal_correlations = self._build_signal_correlation_matrix(all_signals)
            strategy_correlations = signal_correlations
            if strategy_correlations.empty:
                strategy_correlations = perf_correlations

            # Risk constraints from config
            from config.settings import TRADING
            risk_constraints = {
                "max_position_weight": TRADING["max_position_weight"],
                "max_positions": TRADING["max_positions"],
                "max_drawdown": TRADING["max_drawdown"],
            }

            # Market context from recent performance data
            market_context = self._build_market_context(strategy_performance)

            # Build prompt
            prompt = build_orchestrator_prompt(
                date=date,
                strategy_signals=all_signals,
                strategy_performance=strategy_performance,
                strategy_correlations=strategy_correlations,
                market_context=market_context,
                regime=None,
                risk_constraints=risk_constraints,
            )

            # Query Gemini
            response = self.llm_client.generate(
                prompt=prompt,
                model=self.llm_model,
                json_mode=True,
                system=ORCHESTRATOR_SYSTEM_PROMPT,
                temperature=self.llm_temperature,
            )

            latency_ms = response.get("latency_ms", 0)
            parsed = response.get("parsed_json")

            if parsed is None:
                logger.warning(
                    "LLM orchestrator: JSON parse failed, "
                    "falling back to weighted average"
                )
                return super()._combine_signals(all_signals, date)

            # Extract final signals
            final_signals = parsed.get("final_signals", [])
            combined_df = pd.DataFrame(final_signals)

            if combined_df.empty or "ticker" not in combined_df.columns:
                logger.warning(
                    "LLM returned no valid signals, "
                    "falling back to weighted average"
                )
                return super()._combine_signals(all_signals, date)

            # Ensure score column
            combined_df["score"] = (
                pd.to_numeric(combined_df["score"], errors="coerce")
                .fillna(0)
                .clip(-1, 1)
            )

            # Store LLM-suggested weights
            llm_weights = parsed.get("strategy_weights", {})
            if llm_weights:
                self._llm_suggested_weights = llm_weights

            # Log reasoning
            self.reasoning_logger.log(
                model=self.llm_model,
                task="ensemble_orchestration",
                reasoning=parsed.get("reasoning", {}),
                signals=final_signals,
                confidence=parsed.get("confidence"),
                latency_ms=latency_ms,
                metadata={
                    "n_input_strategies": len(all_signals),
                    "n_output_signals": len(combined_df),
                    "llm_suggested_weights": llm_weights,
                },
            )

            logger.info(
                f"LLM orchestrator: {len(combined_df)} signals, "
                f"latency={latency_ms}ms"
            )

            return combined_df

        except Exception as e:
            logger.error(
                f"LLM orchestration failed: {e}, "
                f"falling back to weighted average"
            )
            return super()._combine_signals(all_signals, date)

    def _build_signal_correlation_matrix(
        self,
        all_signals: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Build cross-strategy score correlation using current signal cross-section."""
        if not all_signals:
            return pd.DataFrame()

        merged: pd.DataFrame | None = None
        for name, df in all_signals.items():
            if df.empty or "ticker" not in df.columns or "score" not in df.columns:
                continue

            frame = df[["ticker", "score"]].copy()
            frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
            frame = frame.dropna(subset=["score"]).drop_duplicates("ticker")
            frame = frame.rename(columns={"score": name})

            if merged is None:
                merged = frame
            else:
                merged = merged.merge(frame, on="ticker", how="outer")

        if merged is None:
            return pd.DataFrame()

        score_frame = merged.drop(columns=["ticker"], errors="ignore")
        if score_frame.shape[1] < 2:
            return pd.DataFrame()

        return score_frame.corr(min_periods=3)

    def _build_market_context(
        self, strategy_performance: dict[str, dict]
    ) -> dict[str, Any]:
        """Build market context from available data."""
        context: dict[str, Any] = {}

        if strategy_performance:
            sharpes = {
                name: perf.get("sharpe", 0)
                for name, perf in strategy_performance.items()
            }
            avg_sharpe = sum(sharpes.values()) / len(sharpes) if sharpes else 0
            context["avg_strategy_sharpe"] = f"{avg_sharpe:.2f}"
            context["best_strategy"] = max(sharpes, key=sharpes.get) if sharpes else "N/A"
            context["worst_strategy"] = min(sharpes, key=sharpes.get) if sharpes else "N/A"

        return context

    def get_state(self) -> dict[str, Any]:
        """Extend parent state with LLM config."""
        state = super().get_state()
        state["llm_config"] = {
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "use_llm": self.use_llm,
            "llm_timeout": self.llm_timeout,
        }
        state["llm_suggested_weights"] = self._llm_suggested_weights
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore LLM config alongside parent state."""
        super().restore_state(state)
        llm_config = state.get("llm_config", {})
        self.llm_model = llm_config.get("llm_model", self.llm_model)
        self.llm_temperature = llm_config.get("llm_temperature", self.llm_temperature)
        self.use_llm = llm_config.get("use_llm", self.use_llm)
        self.llm_timeout = llm_config.get("llm_timeout", self.llm_timeout)
        self._llm_suggested_weights = state.get("llm_suggested_weights", {})
