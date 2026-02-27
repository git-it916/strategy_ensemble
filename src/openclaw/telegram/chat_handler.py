"""
Conversational Chat Handler

Processes natural language messages via Claude API,
injecting real-time system state as context.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """\
You are OpenClaw, an autonomous crypto alpha research and trading system.
You manage a portfolio of algorithmic trading strategies (alphas) on Binance USDT-M perpetual futures.

Your capabilities:
- Discover new alpha strategies via web search + LLM code generation
- Backtest and validate alphas with quality gates (Sharpe, MDD, correlation)
- Paper trade for 14 days before going live
- Auto-kill underperforming alphas
- Rebalance portfolio every 15 minutes using intelligent weight allocation

Available commands you can suggest:
- /research [query] : Search for and generate new alpha strategies
- /status : Show current active/paper alphas and performance
- /kill <name> : Deactivate an alpha
- /approve <name> : Approve a pending alpha for paper trading
- /reject <name> : Reject a pending alpha
- /mutate [name] : Attempt to improve existing alphas
- /help : Show all commands

Rules:
- Be concise and quantitative
- Reply in the same language as the user's message
- When asked about system state, use the provided context data
- If the user wants an action, suggest the appropriate command
- Keep responses under 500 characters for readability
"""


class ChatHandler:
    """
    Handles natural language messages by calling Claude with system context.
    """

    MAX_HISTORY = 10

    def __init__(
        self,
        anthropic_client: anthropic.Anthropic,
        registry: Any,
        tracker: Any,
        model: str = "claude-sonnet-4-20250514",
        fallback_model: str = "claude-haiku-4-5-20251001",
    ):
        self.client = anthropic_client
        self.registry = registry
        self.tracker = tracker
        self.model = model
        self.fallback_model = fallback_model
        self._history: list[dict] = []

    def handle(self, text: str) -> str:
        """Process a natural language message and return a response."""
        context = self._build_context()

        self._history.append({
            "role": "user",
            "content": f"[System State]\n{context}\n\n[User Message]\n{text}",
        })

        # Trim history
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY:]

        models = [self.model, self.fallback_model]
        for model in models:
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=CHAT_SYSTEM_PROMPT,
                    messages=self._history,
                )

                reply = ""
                for block in response.content:
                    if block.type == "text":
                        reply = block.text.strip()
                        break

                self._history.append({"role": "assistant", "content": reply})

                logger.info(
                    f"Chat response ({model}): {len(reply)} chars, "
                    f"tokens={response.usage.input_tokens}+{response.usage.output_tokens}"
                )
                return reply

            except Exception as e:
                logger.warning(f"Chat failed with {model}: {e}")
                if model == models[-1]:
                    self._history.pop()  # Remove failed user message
                    return f"일시적으로 응답할 수 없습니다. 잠시 후 다시 시도해주세요."

    def _build_context(self) -> str:
        """Compile current system state for Claude context."""
        active = self.registry.get_active()
        paper = self.registry.get_paper()

        active_info = []
        for entry in active:
            summary = self.tracker.get_summary(entry.name)
            active_info.append(
                f"  - {entry.name}: Sharpe={summary.get('sharpe', 'N/A')}, "
                f"MDD={summary.get('mdd', 'N/A')}, "
                f"weight={entry.current_weight:.1%}, "
                f"leverage={entry.current_leverage:.1f}x"
            )

        paper_info = []
        for entry in paper:
            summary = self.tracker.get_summary(entry.name)
            paper_info.append(
                f"  - {entry.name}: Sharpe={summary.get('sharpe', 'N/A')}, "
                f"days={summary.get('n_days', 0)}"
            )

        total_count = self.registry.total_count

        lines = [
            f"Active alphas: {len(active)}",
            f"Paper alphas: {len(paper)}",
            f"Total registered: {total_count}",
        ]

        if active_info:
            lines.append("Active details:")
            lines.extend(active_info)
        if paper_info:
            lines.append("Paper details:")
            lines.extend(paper_info)

        if not active and not paper:
            lines.append("No alphas currently running. Use /research to discover new strategies.")

        return "\n".join(lines)
