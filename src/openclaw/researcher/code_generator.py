"""
Alpha Code Generator

Uses Claude Sonnet to generate BaseAlpha implementations from AlphaSpec.
Includes retry logic and validation feedback loop.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.openclaw.config import RESEARCH_POLICY
from src.openclaw.researcher.code_validator import CodeValidator, ValidationResult
from src.openclaw.researcher.idea_parser import AlphaSpec
from src.openclaw.researcher.prompts import (
    CODE_FIX_PROMPT,
    CODE_GEN_SYSTEM_PROMPT,
    CODE_GEN_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAlpha:
    """Result of alpha code generation."""

    spec: AlphaSpec
    code: str                        # Full Python source
    class_name: str                  # e.g. "VolumeWeightedMomentum"
    module_name: str                 # e.g. "volume_weighted_momentum"
    generation_metadata: dict = field(default_factory=dict)


class AlphaCodeGenerator:
    """
    Generate BaseAlpha subclass code from AlphaSpec using Claude Sonnet.

    Features:
    - Structured prompt with interface contract + reference implementation
    - Automatic code validation with retry on failure
    - Extracts class name from generated code
    """

    def __init__(
        self,
        anthropic_client,
        model: str | None = None,
        max_retries: int | None = None,
    ):
        self.client = anthropic_client
        self.model = model or RESEARCH_POLICY.code_gen_model
        self.max_retries = max_retries or RESEARCH_POLICY.max_code_gen_retries
        self.validator = CodeValidator()

    def generate(self, spec: AlphaSpec) -> GeneratedAlpha:
        """
        Generate a BaseAlpha subclass from an AlphaSpec.

        Args:
            spec: Structured alpha specification

        Returns:
            GeneratedAlpha with validated code

        Raises:
            RuntimeError: If code generation fails after all retries
        """
        start_time = time.time()
        total_tokens = 0

        # Initial generation
        user_prompt = CODE_GEN_USER_TEMPLATE.format(
            name=spec.name,
            description=spec.description,
            hypothesis=spec.hypothesis,
            signal_logic=spec.signal_logic,
            required_data=", ".join(spec.required_data),
            lookback_days=spec.lookback_days,
            expected_style=spec.expected_style,
        )

        code, tokens = self._call_llm(CODE_GEN_SYSTEM_PROMPT, user_prompt)
        total_tokens += tokens
        code = self._clean_code(code)

        # Validate and retry loop
        for attempt in range(self.max_retries + 1):
            validation = self.validator.validate(code)

            if validation.is_valid:
                class_name = self._extract_class_name(code)
                elapsed = time.time() - start_time

                result = GeneratedAlpha(
                    spec=spec,
                    code=code,
                    class_name=class_name,
                    module_name=spec.name,
                    generation_metadata={
                        "model": self.model,
                        "attempts": attempt + 1,
                        "total_tokens": total_tokens,
                        "elapsed_seconds": round(elapsed, 1),
                        "warnings": validation.warnings,
                    },
                )

                logger.info(
                    f"Generated alpha '{class_name}' in {attempt + 1} attempt(s) "
                    f"({elapsed:.1f}s, {total_tokens} tokens)"
                )
                return result

            if attempt < self.max_retries:
                # Retry with error feedback
                logger.warning(
                    f"Validation failed (attempt {attempt + 1}): "
                    f"{validation.errors}"
                )
                fix_prompt = CODE_FIX_PROMPT.format(
                    errors="\n".join(validation.errors),
                    code=code,
                )
                code, tokens = self._call_llm(CODE_GEN_SYSTEM_PROMPT, fix_prompt)
                total_tokens += tokens
                code = self._clean_code(code)

        # All retries exhausted
        raise RuntimeError(
            f"Code generation failed after {self.max_retries + 1} attempts. "
            f"Last errors: {validation.errors}"
        )

    def _call_llm(self, system: str, user: str) -> tuple[str, int]:
        """
        Call Claude API and return (text, total_tokens).
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        raw = ""
        for block in message.content:
            if block.type == "text":
                raw = block.text.strip()
                break

        tokens = (
            message.usage.input_tokens + message.usage.output_tokens
            if hasattr(message, "usage")
            else 0
        )

        return raw, tokens

    @staticmethod
    def _clean_code(raw: str) -> str:
        """
        Clean LLM output to extract pure Python code.

        Removes markdown code blocks, leading/trailing whitespace,
        and any non-code preamble.
        """
        # Remove markdown code blocks
        match = re.search(r"```(?:python)?\s*([\s\S]+?)\s*```", raw)
        if match:
            return match.group(1).strip()

        # If no code block, check if it starts with imports or docstring
        lines = raw.strip().split("\n")
        code_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("from ", "import ", '"""', "class ", "#")):
                code_start = i
                break

        return "\n".join(lines[code_start:]).strip()

    @staticmethod
    def _extract_class_name(code: str) -> str:
        """Extract the BaseAlpha subclass name from generated code."""
        # Look for class definitions that inherit from BaseAlpha
        pattern = r"class\s+(\w+)\s*\(\s*BaseAlpha\s*\)"
        match = re.search(pattern, code)
        if match:
            return match.group(1)

        # Fallback: any class definition
        pattern = r"class\s+(\w+)\s*\("
        match = re.search(pattern, code)
        if match:
            return match.group(1)

        raise ValueError("No class definition found in generated code")
