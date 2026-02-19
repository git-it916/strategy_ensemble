"""
Gemini LLM Client

Wrapper for Google Gemini API (gemini-3-flash) using google.genai SDK.
Implements the same interface as OllamaClient for drop-in replacement.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from google import genai
from google.genai import types

from .ollama_client import _extract_json

logger = logging.getLogger(__name__)

# Model constants
MODEL_GEMINI_FLASH = "gemini-3-flash"
MODEL_GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"

# Task -> model mapping
TASK_MODEL_MAP = {
    "signal_generation": MODEL_GEMINI_FLASH,
    "ensemble_orchestration": MODEL_GEMINI_FLASH,
    "market_analysis": MODEL_GEMINI_FLASH,
}

# Retryable error messages
_RETRYABLE_ERRORS = ("429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE")


class GeminiClient:
    """
    Google Gemini API client (google.genai SDK).

    Features:
        - generate() and chat() endpoints matching OllamaClient interface
        - JSON mode via response_mime_type
        - System prompt via system_instruction
        - Timeout, retry, and latency tracking
        - Model selection by task type
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = MODEL_GEMINI_FLASH,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_delay: float = 5.0,
    ):
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Create genai client
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = True,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Call Gemini generate endpoint.

        Returns same format as OllamaClient.generate():
            {
                "response": str,
                "model": str,
                "latency_ms": int,
                "parsed_json": dict | None
            }
        """
        model = model or self.default_model

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
        )
        if json_mode:
            config.response_mime_type = "application/json"

        start = time.monotonic()
        response_text = self._request_with_retry(model, prompt, config)
        latency_ms = int((time.monotonic() - start) * 1000)

        parsed = _extract_json(response_text) if json_mode else None

        return {
            "response": response_text,
            "model": model,
            "latency_ms": latency_ms,
            "parsed_json": parsed,
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        json_mode: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Call Gemini chat endpoint with conversation history.

        Returns same format as OllamaClient.chat().
        """
        model = model or self.default_model

        # Extract system message if present
        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
        )
        if json_mode:
            config.response_mime_type = "application/json"

        # Convert messages to Gemini Content format
        contents = []
        for msg in chat_messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )

        start = time.monotonic()
        last_error: Exception | None = None
        response_text = ""

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                response_text = response.text or ""
                break
            except Exception as e:
                last_error = e
                logger.error(
                    f"Gemini chat error (attempt {attempt + 1}): {e}"
                )
                if self._is_retryable(e) and attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                elif not self._is_retryable(e):
                    raise
        else:
            raise ConnectionError(
                f"Gemini chat failed after {self.max_retries + 1} attempts: "
                f"{last_error}"
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        parsed = _extract_json(response_text) if json_mode else None

        return {
            "response": response_text,
            "model": model,
            "latency_ms": latency_ms,
            "parsed_json": parsed,
        }

    def generate_for_task(
        self,
        task: str,
        prompt: str,
        system: str | None = None,
        json_mode: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate using the model mapped to a specific task."""
        model = TASK_MODEL_MAP.get(task, self.default_model)
        return self.generate(
            prompt=prompt,
            model=model,
            system=system,
            json_mode=json_mode,
            **kwargs,
        )

    def _request_with_retry(
        self,
        model: str,
        prompt: str,
        config: types.GenerateContentConfig,
    ) -> str:
        """Make API request with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                last_error = e
                logger.error(
                    f"Gemini API error (attempt {attempt + 1}): {e}"
                )
                if self._is_retryable(e) and attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                elif not self._is_retryable(e):
                    raise

        raise ConnectionError(
            f"Gemini request failed after {self.max_retries + 1} attempts: "
            f"{last_error}"
        )

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Check if the error is retryable."""
        error_str = str(error).upper()
        return any(code in error_str for code in _RETRYABLE_ERRORS)

    def is_available(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            response = self._client.models.generate_content(
                model=self.default_model,
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            return response.text is not None
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available Gemini models."""
        try:
            models = self._client.models.list()
            return [m.name for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def close(self) -> None:
        """No persistent connections to close."""
        pass
