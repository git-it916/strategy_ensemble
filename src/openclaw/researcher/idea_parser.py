"""
Idea Parser

Uses Claude Sonnet to parse raw search results into structured AlphaSpec.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from src.openclaw.config import RESEARCH_POLICY
from src.openclaw.researcher.brave_search import AlphaIdea

logger = logging.getLogger(__name__)


@dataclass
class AlphaSpec:
    """Structured specification of an alpha idea."""

    name: str                    # e.g. "volume_weighted_momentum"
    description: str             # 1-paragraph natural language
    hypothesis: str              # why it should work
    signal_logic: str            # pseudocode-level description
    required_data: list[str]     # ["ohlcv", "funding_rate", "volume"]
    lookback_days: int           # estimated lookback needed
    expected_style: str          # "rule_based" | "ml_based"
    source_url: str
    source_title: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "signal_logic": self.signal_logic,
            "required_data": self.required_data,
            "lookback_days": self.lookback_days,
            "expected_style": self.expected_style,
            "source_url": self.source_url,
            "source_title": self.source_title,
        }


PARSE_PROMPT = """You are a quantitative researcher parsing alpha strategy ideas.

Given a search result (title, snippet, and optionally full content), extract
a structured alpha specification. The alpha must be applicable to:
- Crypto USDT-M perpetual futures (BTC, ETH, SOL)
- Maximum 5-day lookback window
- Long/short signals with score in [-1, 1]

If the content does NOT describe an actionable trading signal or alpha strategy,
respond with: {"actionable": false, "reason": "..."}

If actionable, respond with ONLY this JSON (no markdown, no backticks):
{
  "actionable": true,
  "name": "snake_case_name",
  "description": "1-paragraph description of the strategy",
  "hypothesis": "Why this alpha should generate excess returns",
  "signal_logic": "Step-by-step pseudocode of the signal computation",
  "required_data": ["ohlcv", "funding_rate", "volume"],
  "lookback_days": 5,
  "expected_style": "rule_based"
}

Rules:
- name must be snake_case, max 40 characters
- lookback_days must be <= 5
- required_data can include: ohlcv, funding_rate, volume, open_interest
- expected_style: "rule_based" if no ML model needed, "ml_based" if needs training
- signal_logic should be specific enough to implement in Python
- Keep description under 200 words"""


class IdeaParser:
    """
    Parse raw search results into structured AlphaSpec using Claude Sonnet.
    """

    def __init__(self, anthropic_client, model: str | None = None):
        """
        Args:
            anthropic_client: anthropic.Anthropic instance
            model: Override model (default: from RESEARCH_POLICY)
        """
        self.client = anthropic_client
        self.model = model or RESEARCH_POLICY.code_gen_model

    def parse_idea(
        self,
        idea: AlphaIdea,
        extra_content: str = "",
    ) -> AlphaSpec | None:
        """
        Parse a single AlphaIdea into an AlphaSpec.

        Args:
            idea: Raw search result
            extra_content: Additional fetched page content

        Returns:
            AlphaSpec if actionable, None otherwise.
        """
        user_content = self._build_user_message(idea, extra_content)

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=PARSE_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )

            raw = self._extract_text(message)
            data = self._parse_json(raw)

            if not data.get("actionable", False):
                reason = data.get("reason", "not actionable")
                logger.info(f"Skipping '{idea.title[:50]}': {reason}")
                return None

            # Enforce max lookback
            lookback = min(data.get("lookback_days", 5), 5)

            spec = AlphaSpec(
                name=data["name"],
                description=data["description"],
                hypothesis=data["hypothesis"],
                signal_logic=data["signal_logic"],
                required_data=data.get("required_data", ["ohlcv"]),
                lookback_days=lookback,
                expected_style=data.get("expected_style", "rule_based"),
                source_url=idea.url,
                source_title=idea.title,
            )

            logger.info(f"Parsed alpha spec: {spec.name} from '{idea.title[:50]}'")
            return spec

        except Exception as e:
            logger.error(f"Failed to parse idea '{idea.title[:50]}': {e}")
            return None

    def rank_ideas(
        self,
        ideas: list[AlphaIdea],
        max_to_rank: int = 20,
    ) -> list[AlphaIdea]:
        """
        Rank ideas by novelty and expected alpha potential.

        Uses a single LLM call to score all ideas at once.
        """
        if len(ideas) <= 3:
            return ideas

        # Prepare compact summaries
        summaries = []
        for i, idea in enumerate(ideas[:max_to_rank]):
            summaries.append(f"{i}: [{idea.source_type}] {idea.title} — {idea.snippet[:100]}")

        prompt = (
            "Rank these alpha ideas by novelty and expected profitability "
            "for crypto futures (BTC/ETH/SOL). Return a JSON list of indices "
            "ordered from best to worst. Only include ideas that are actionable "
            "trading strategies. Format: {\"ranking\": [3, 0, 7, ...]}\n\n"
            + "\n".join(summaries)
        )

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = self._extract_text(message)
            data = self._parse_json(raw)
            ranking = data.get("ranking", [])

            ranked = []
            for idx in ranking:
                if 0 <= idx < len(ideas):
                    ideas[idx].relevance_score = 1.0 - (len(ranked) / len(ranking))
                    ranked.append(ideas[idx])

            # Add any unranked ideas at the end
            ranked_urls = {i.url for i in ranked}
            for idea in ideas:
                if idea.url not in ranked_urls:
                    ranked.append(idea)

            return ranked

        except Exception as e:
            logger.warning(f"Failed to rank ideas: {e}, returning original order")
            return ideas

    @staticmethod
    def _build_user_message(idea: AlphaIdea, extra_content: str) -> str:
        parts = [
            f"Title: {idea.title}",
            f"URL: {idea.url}",
            f"Type: {idea.source_type}",
            f"Snippet: {idea.snippet}",
        ]
        if extra_content:
            parts.append(f"\nFull content (truncated):\n{extra_content[:5000]}")
        return "\n".join(parts)

    @staticmethod
    def _extract_text(message) -> str:
        for block in message.content:
            if block.type == "text":
                return block.text.strip()
        raise ValueError("No text in API response")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
        if match:
            return json.loads(match.group(1))

        # Try finding last { ... } block
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
