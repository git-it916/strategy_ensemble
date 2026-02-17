"""
LLM Alpha Strategy

Uses Qwen2.5:32B via Ollama to analyze market data and generate trading signals.
Implements BaseAlpha interface identically to rule-based and ML alphas.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..base_alpha import BaseAlpha, AlphaResult
from ...llm.ollama_client import OllamaClient, MODEL_QWEN
from ...llm.prompts import (
    SIGNAL_SYSTEM_PROMPT,
    FUND_MANAGER_SYSTEM_PROMPT,
    build_signal_prompt,
    build_fund_manager_prompt,
)
from ...llm.reasoning_logger import ReasoningLogger

logger = logging.getLogger(__name__)


class LLMAlpha(BaseAlpha):
    """
    LLM-based alpha strategy using Qwen2.5:32B.

    Instead of mathematical rules or ML models, this strategy:
    1. Constructs a prompt with market context
    2. Sends to Qwen2.5:32B for analysis
    3. Parses JSON response into AlphaResult
    4. Logs reasoning for every decision

    Implements the same interface as RSIReversalAlpha, ReturnPredictionAlpha, etc.
    EnsembleAgent treats it identically to any other strategy.
    """

    def __init__(
        self,
        name: str = "llm_alpha",
        config: dict[str, Any] | None = None,
        ollama_client: OllamaClient | None = None,
        reasoning_logger: ReasoningLogger | None = None,
    ):
        super().__init__(name, config)
        self.config = config or {}

        # LLM client (lazy init if not provided)
        self._client = ollama_client
        self._reasoning_logger = reasoning_logger

        # Configuration (파인튜닝 모델 사용)
        from config.settings import LLM_CONFIG
        self.model = self.config.get("model", LLM_CONFIG.get("ollama_model", MODEL_QWEN))
        self.temperature = self.config.get("temperature", LLM_CONFIG.get("ollama_temperature", 0.3))
        self.max_stocks_in_prompt = self.config.get("max_stocks_in_prompt", LLM_CONFIG.get("max_stocks_in_prompt", 50))
        self.timeout = self.config.get("timeout", LLM_CONFIG.get("ollama_timeout", 120.0))

        # Cache for backtesting
        self._response_cache: dict[str, dict] = {}

    @property
    def client(self) -> OllamaClient:
        """Lazy initialization of Ollama client."""
        if self._client is None:
            from config.settings import LLM_CONFIG
            self._client = OllamaClient(
                base_url=LLM_CONFIG.get("ollama_url", "http://localhost:11434"),
                default_model=self.model,
                timeout=self.timeout,
            )
        return self._client

    @property
    def reasoning_logger(self) -> ReasoningLogger:
        """Lazy initialization of reasoning logger."""
        if self._reasoning_logger is None:
            self._reasoning_logger = ReasoningLogger()
        return self._reasoning_logger

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        LLM alpha has no training phase.
        Validates Ollama connectivity and marks as fitted.
        """
        self.is_fitted = True
        self._fit_date = datetime.now()

        available = self.client.is_available()
        if available:
            models = self.client.list_models()
            logger.info(f"Ollama available. Models: {models}")
        else:
            logger.warning("Ollama not reachable; LLM alpha will fail at signal time")

        return {
            "status": "fitted",
            "model": self.model,
            "ollama_available": available,
        }

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate signals by querying Qwen2.5:32B.

        Steps:
        1. Filter data up to date (no lookahead)
        2. Build prompt with market context
        3. Query Qwen2.5:32B with JSON mode
        4. Parse response into DataFrame with ticker, score
        5. Log reasoning
        """
        # Check cache (for backtesting)
        cache_key = str(date.date()) if hasattr(date, "date") else str(date)
        if cache_key in self._response_cache:
            cached = self._response_cache[cache_key]
            return AlphaResult(
                date=date,
                signals=cached["signals"],
                metadata=cached["metadata"],
            )

        # Filter data (no lookahead)
        prices_filtered = prices[prices["date"] <= pd.Timestamp(date)]
        features_filtered = None
        if features is not None:
            features_filtered = features[features["date"] <= pd.Timestamp(date)]

        # Detect simple regime from price data
        regime = self._detect_regime(prices_filtered)

        # Build prompt
        prompt = build_signal_prompt(
            date=date,
            prices=prices_filtered,
            features=features_filtered,
            regime=regime,
            top_k=self.max_stocks_in_prompt,
        )

        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                json_mode=True,
                system=SIGNAL_SYSTEM_PROMPT,
                temperature=self.temperature,
            )
            latency_ms = response.get("latency_ms", 0)

            # Parse JSON response
            parsed = response.get("parsed_json")
            if parsed is None:
                logger.warning("Failed to parse LLM JSON; returning empty signals")
                return self._empty_result(date, "JSON parse failed")

            # Extract signals
            raw_signals = parsed.get("signals", [])
            signals_df = pd.DataFrame(raw_signals)

            if signals_df.empty or "ticker" not in signals_df.columns:
                signals_df = pd.DataFrame(columns=["ticker", "score", "side", "reason"])
            else:
                signals_df["score"] = (
                    pd.to_numeric(signals_df["score"], errors="coerce")
                    .fillna(0)
                    .clip(-1, 1)
                )
                if "side" not in signals_df.columns:
                    signals_df["side"] = signals_df["score"].apply(
                        lambda x: "long" if x > 0 else "short"
                    )
                if "reason" not in signals_df.columns:
                    signals_df["reason"] = signals_df.get("rationale", "")

            # Build metadata
            metadata = {
                "strategy": self.name,
                "model": self.model,
                "confidence": parsed.get("confidence"),
                "market_context": parsed.get("market_context"),
                "regime_assessment": parsed.get("regime_assessment"),
                "latency_ms": latency_ms,
            }

            # Log reasoning
            self.reasoning_logger.log(
                model=self.model,
                task="signal_generation",
                reasoning={
                    "market_context": parsed.get("market_context", ""),
                    "analysis": parsed.get("analysis", ""),
                    "factors_considered": parsed.get("factors_considered", []),
                    "decision": f"Generated {len(signals_df)} signals",
                },
                signals=raw_signals,
                confidence=parsed.get("confidence"),
                latency_ms=latency_ms,
            )

            # Cache for backtesting
            self._response_cache[cache_key] = {
                "signals": signals_df,
                "metadata": metadata,
            }

            return AlphaResult(date=date, signals=signals_df, metadata=metadata)

        except Exception as e:
            logger.error(f"LLM signal generation failed: {e}")
            return self._empty_result(date, str(e))

    def generate_from_context(
        self,
        ctx: "PipelineContext",
    ) -> dict[str, Any]:
        """
        순차 파이프라인용: PipelineContext → LLM 판단.

        EnsembleAgent를 거치지 않고, ContextBuilder가 조립한 컨텍스트를
        직접 LLM에 전달하여 최종 결정을 받음.

        Returns:
            LLM의 parsed JSON 응답 (signals, reasoning, confidence 포함)
        """
        prompt = build_fund_manager_prompt(ctx)

        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                json_mode=True,
                system=FUND_MANAGER_SYSTEM_PROMPT,
                temperature=self.temperature,
            )
            latency_ms = response.get("latency_ms", 0)

            parsed = response.get("parsed_json")
            if parsed is None:
                logger.warning("Fund Manager LLM: JSON parse failed")
                return {"signals": [], "error": "JSON parse failed"}

            # Log reasoning
            self.reasoning_logger.log(
                model=self.model,
                task="fund_manager_decision",
                reasoning={
                    "analysis": parsed.get("reasoning", ""),
                    "regime": parsed.get("regime_assessment", ""),
                },
                signals=parsed.get("signals", []),
                confidence=parsed.get("confidence"),
                latency_ms=latency_ms,
            )

            logger.info(
                f"Fund Manager: {len(parsed.get('signals', []))} decisions, "
                f"confidence={parsed.get('confidence')}, latency={latency_ms}ms"
            )

            return parsed

        except Exception as e:
            logger.error(f"Fund Manager LLM failed: {e}")
            return {"signals": [], "error": str(e)}

    def _detect_regime(self, prices: pd.DataFrame) -> str | None:
        """Simple regime detection from price data for prompt context."""
        if prices.empty:
            return None

        try:
            latest = prices.sort_values("date").groupby("ticker")["close"].last()
            recent = prices[prices["date"] >= prices["date"].max() - pd.Timedelta(days=20)]
            earliest = recent.sort_values("date").groupby("ticker")["close"].first()

            common = latest.index.intersection(earliest.index)
            if len(common) < 5:
                return None

            returns = (latest[common] / earliest[common] - 1).median()

            if returns > 0.03:
                return "bull"
            elif returns < -0.03:
                return "bear"
            else:
                return "sideways"
        except Exception:
            return None

    def _empty_result(self, date: datetime, error: str) -> AlphaResult:
        """Return empty AlphaResult on failure."""
        return AlphaResult(
            date=date,
            signals=pd.DataFrame(columns=["ticker", "score"]),
            metadata={"strategy": self.name, "error": error},
        )

    # --- State persistence ---

    def _get_extra_state(self) -> dict[str, Any]:
        """Save LLM-specific state."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_stocks_in_prompt": self.max_stocks_in_prompt,
            "response_cache": self._response_cache,
        }

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        """Restore LLM-specific state."""
        self.model = state.get("model", MODEL_QWEN)
        self.temperature = state.get("temperature", 0.3)
        self.max_stocks_in_prompt = state.get("max_stocks_in_prompt", 50)
        self._response_cache = state.get("response_cache", {})

    def save_state(self, path: Path) -> dict[str, Any]:
        """Override to set type='llm' in registry metadata."""
        meta = super().save_state(path)
        meta["type"] = "llm"
        return meta

    def clear_cache(self) -> None:
        """Clear response cache."""
        self._response_cache.clear()
