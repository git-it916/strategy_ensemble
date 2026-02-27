"""
Claude Ensemble Orchestrator

Uses Claude Opus to decide ensemble weights based on:
- Each alpha's recent performance
- Inter-alpha correlations
- Market context
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import pandas as pd

from src.openclaw.config import RESEARCH_POLICY
from src.openclaw.orchestrator.weight_optimizer import WeightOptimizer
from src.openclaw.registry.alpha_registry import AlphaEntry

logger = logging.getLogger(__name__)


ENSEMBLE_SYSTEM_PROMPT = """You are a quantitative portfolio manager deciding ensemble weights
for a set of alpha strategies trading crypto futures (BTC, ETH, SOL).

Given:
1. Performance data for each active alpha (Sharpe, MDD, win rate, recent returns)
2. Correlation matrix between alphas
3. Current market context (recent BTC price action, volatility regime)

Decide the weight allocation across alphas. Return ONLY a JSON object:
{
  "weights": {"alpha_name_1": 0.25, "alpha_name_2": 0.35, ...},
  "reasoning": "Brief explanation of allocation rationale"
}

Rules:
- Weights must sum to 1.0
- Each weight must be between 0.05 and 0.40
- Prefer alphas with higher recent Sharpe and lower correlation with others
- In volatile markets, favor mean-reversion and low-volatility alphas
- In trending markets, favor momentum alphas
- Downweight alphas showing performance degradation
- If an alpha has consecutive losses, reduce its weight significantly"""


class ClaudeEnsembleOrchestrator:
    """
    Claude Opus-based ensemble weight allocation.

    Falls back to risk-parity (WeightOptimizer) if LLM fails.
    """

    def __init__(
        self,
        anthropic_client,
        model: str | None = None,
    ):
        self.client = anthropic_client
        self.model = model or RESEARCH_POLICY.ensemble_model
        self.fallback = WeightOptimizer()

    def decide_weights(
        self,
        active_alphas: list[AlphaEntry],
        performance_data: dict[str, dict],
        correlation_matrix: pd.DataFrame,
        market_context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """
        Use Claude Opus to decide ensemble weights.

        Args:
            active_alphas: List of active alpha entries
            performance_data: {alpha_name: {sharpe, mdd, win_rate, ...}}
            correlation_matrix: Correlation DataFrame between alphas
            market_context: Optional market state info

        Returns:
            {alpha_name: weight} summing to 1.0
        """
        if not active_alphas:
            return {}

        alpha_names = [a.name for a in active_alphas]

        try:
            user_prompt = self._build_prompt(
                active_alphas, performance_data,
                correlation_matrix, market_context
            )

            start = time.time()
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=ENSEMBLE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw = self._extract_text(message)
            data = self._parse_json(raw)
            latency = time.time() - start

            weights = data.get("weights", {})
            reasoning = data.get("reasoning", "")

            # Validate weights
            weights = self._validate_weights(weights, alpha_names)

            logger.info(
                f"Opus ensemble weights ({latency:.1f}s): "
                f"{weights} | Reason: {reasoning[:100]}"
            )

            return weights

        except Exception as e:
            logger.warning(
                f"Claude orchestrator failed: {e}. "
                f"Falling back to risk-parity."
            )

            # Fallback to equal weight (risk-parity needs returns data)
            return self.fallback.equal_weight(alpha_names)

    def _build_prompt(
        self,
        alphas: list[AlphaEntry],
        perf: dict[str, dict],
        corr: pd.DataFrame,
        market: dict | None,
    ) -> str:
        """Build user prompt with all context for Opus."""
        lines = ["# Active Alphas\n"]

        for a in alphas:
            p = perf.get(a.name, {})
            lines.append(
                f"- {a.name}: "
                f"Sharpe={p.get('sharpe', 0):.2f}, "
                f"MDD={p.get('mdd', 0):.2%}, "
                f"WinRate={p.get('win_rate', 0):.1%}, "
                f"ConsecLoss={p.get('consecutive_loss_days', 0)}, "
                f"Days={p.get('n_days', 0)}, "
                f"TotalReturn={p.get('total_return', 0):.2%}"
            )

        lines.append("\n# Correlation Matrix\n")
        if not corr.empty:
            lines.append(corr.to_string())
        else:
            lines.append("(Insufficient data for correlation)")

        if market:
            lines.append("\n# Market Context\n")
            for k, v in market.items():
                lines.append(f"- {k}: {v}")

        return "\n".join(lines)

    def _validate_weights(
        self,
        weights: dict[str, float],
        expected_names: list[str],
    ) -> dict[str, float]:
        """Validate and fix LLM-returned weights."""
        # Only keep known alpha names
        valid = {
            k: max(0.05, min(0.40, v))
            for k, v in weights.items()
            if k in expected_names
        }

        # Add missing alphas with minimum weight
        for name in expected_names:
            if name not in valid:
                valid[name] = 0.05

        # Normalize to sum to 1.0
        total = sum(valid.values())
        if total > 0:
            valid = {k: v / total for k, v in valid.items()}

        return valid

    @staticmethod
    def _extract_text(message) -> str:
        for block in message.content:
            if block.type == "text":
                return block.text.strip()
        raise ValueError("No text in API response")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
        if match:
            return json.loads(match.group(1))

        last_close = raw.rfind("}")
        if last_close != -1:
            depth = 0
            for i in range(last_close, -1, -1):
                if raw[i] == "}":
                    depth += 1
                elif raw[i] == "{":
                    depth -= 1
                if depth == 0:
                    return json.loads(raw[i:last_close + 1])

        raise ValueError(f"Could not parse JSON from: {raw[:200]}")
