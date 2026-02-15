"""
Ollama LLM Client

Wrapper for Ollama HTTP API (localhost:11434).
Supports Qwen2.5:32B and DeepSeek R1:32B.
All responses include latency tracking and optional JSON parsing.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Model name constants
MODEL_QWEN = "qwen2.5:32b"
MODEL_DEEPSEEK = "deepseek-r1:32b"

# Task -> model mapping (24GB VRAM: 단일 모델로 통일, 스왑 없음)
TASK_MODEL_MAP = {
    "signal_generation": MODEL_QWEN,
    "ensemble_orchestration": MODEL_QWEN,
    "market_analysis": MODEL_QWEN,
}


def _extract_json(text: str) -> dict | None:
    """Try to extract JSON from text that may contain non-JSON content."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


class OllamaClient:
    """
    Ollama HTTP API client.

    Features:
        - Chat and generate endpoints
        - JSON mode for structured output
        - Timeout and retry handling
        - Model selection by task type
        - Latency tracking for reasoning logs
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = MODEL_QWEN,
        timeout: float = 300.0,
        max_retries: int = 2,
        retry_delay: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = httpx.Client(timeout=httpx.Timeout(timeout))

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
        Call Ollama generate endpoint.

        Args:
            prompt: User prompt
            model: Model name (None = default)
            json_mode: If True, set format="json"
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            {
                "response": str,
                "model": str,
                "latency_ms": int,
                "parsed_json": dict | None
            }
        """
        model = model or self.default_model

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if json_mode:
            payload["format"] = "json"
        if system:
            payload["system"] = system

        start = time.monotonic()
        raw = self._request_with_retry("/api/generate", payload)
        latency_ms = int((time.monotonic() - start) * 1000)

        response_text = raw.get("response", "")
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
        Call Ollama chat endpoint.

        Args:
            messages: List of {"role": ..., "content": ...}
            model: Model name
            json_mode: Structured JSON output
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            Same format as generate()
        """
        model = model or self.default_model

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if json_mode:
            payload["format"] = "json"

        start = time.monotonic()
        raw = self._request_with_retry("/api/chat", payload)
        latency_ms = int((time.monotonic() - start) * 1000)

        msg = raw.get("message", {})
        response_text = msg.get("content", "")
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
        endpoint: str,
        payload: dict,
    ) -> dict[str, Any]:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}{endpoint}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
            except httpx.ConnectError as e:
                last_error = e
                logger.error(
                    f"Ollama connection failed (attempt {attempt + 1}): {e}"
                )
            except httpx.TimeoutException as e:
                last_error = e
                logger.error(
                    f"Ollama timeout (attempt {attempt + 1}): {e}"
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.error(
                    f"Ollama HTTP error {e.response.status_code} "
                    f"(attempt {attempt + 1}): {e}"
                )
            except Exception as e:
                last_error = e
                logger.error(
                    f"Ollama request error (attempt {attempt + 1}): {e}"
                )

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        raise ConnectionError(
            f"Ollama request failed after {self.max_retries + 1} attempts: "
            f"{last_error}"
        )

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models on Ollama server."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def ensure_model_loaded(self, model: str | None = None) -> bool:
        """Pre-load a model to avoid cold-start latency."""
        model = model or self.default_model
        try:
            self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
            )
            logger.info(f"Model {model} loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model}: {e}")
            return False

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()
