"""
BaseAlphaV2 + AlphaSignal — CLAUDE.md 섹션 4-1 기준.

모든 알파는 이 인터페이스를 따른다:
    async def compute(self, symbol: str, data: DataBundle) -> AlphaSignal

score: -1.0 ~ +1.0 (양수=롱, 음수=숏)
confidence: 0.0 ~ 1.0 (시그널 확신도)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AlphaSignal:
    """단일 심볼에 대한 알파 시그널."""

    score: float = 0.0        # -1.0 ~ +1.0
    confidence: float = 0.0   # 0.0 ~ 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # NaN/inf 방어
        if math.isnan(self.score) or math.isinf(self.score):
            self.score = 0.0
        if math.isnan(self.confidence) or math.isinf(self.confidence):
            self.confidence = 0.0
        self.score = max(-1.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))


class BaseAlphaV2(ABC):
    """
    알파 기본 클래스.

    모든 알파는 이 클래스를 상속하고 compute()를 구현한다.
    compute()는 단일 심볼에 대해 호출되며, AlphaSignal을 반환한다.
    데이터 부족 시 score=0, confidence=0을 반환해야 한다 (에러 아님).
    """

    def __init__(
        self,
        name: str,
        weight: float,
        category: str,
        required_data: List[str],
    ):
        self.name = name
        self.weight = weight
        self.category = category  # "momentum"|"carry"|"mean_reversion"|"micro"|"auxiliary"
        self.required_data = required_data

    @abstractmethod
    async def compute(self, symbol: str, data: "DataBundle") -> AlphaSignal:
        """
        단일 심볼에 대한 시그널 계산.

        Args:
            symbol: 거래 심볼 (예: "ETH/USDT:USDT")
            data: DataBundle — 전체 유니버스 데이터

        Returns:
            AlphaSignal(score, confidence, metadata)
            - 데이터 부족: score=0, confidence=0
            - 예외 발생 시: 호출부에서 catch → score=0 처리
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', w={self.weight})"
