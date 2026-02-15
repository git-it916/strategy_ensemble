"""
LLM Module

Local LLM integration via Ollama for signal generation and ensemble orchestration.
"""

from .ollama_client import OllamaClient
from .reasoning_logger import ReasoningLogger

__all__ = ["OllamaClient", "ReasoningLogger"]
