"""
Funding Rate Carry Alpha

Binance USDT-M 무기한 선물의 펀딩비율을 이용한 Carry 전략.

로직:
    - 음의 펀딩비율 → 숏 포지션 보유자가 롱에게 지불 → 롱 유리 (score +)
    - 양의 펀딩비율 → 롱 포지션 보유자가 숏에게 지불 → 숏 유리 (score -)
    - 이동평균(rolling mean)으로 노이즈 제거
    - Cross-sectional rank 정규화 → [-1, 1]

데이터:
    features DataFrame의 'funding_rate' 컬럼 (Binance USDT-M 8h 펀딩, 일평균)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.alphas.base_alpha import AlphaResult, BaseAlpha


class FundingRateCarry(BaseAlpha):
    """
    Funding Rate Carry alpha for Binance USDT-M perpetual futures.

    Tickers with persistently negative funding rates are likely to mean
    that shorts are paying longs → long positions benefit from carry.
    We rank inversely on funding rate to favor low (negative) values.
    """

    def __init__(
        self,
        name: str = "FundingRateCarry",
        lookback_days: int = 7,
    ):
        super().__init__(name)
        self.lookback_days = lookback_days

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Rule-based: no training needed."""
        self.is_fitted = True
        self._fit_date = datetime.now()
        return {
            "status": "fitted",
            "lookback_days": self.lookback_days,
        }

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate carry signals from funding rate data.

        Args:
            date: Signal date
            prices: Price data (unused directly, for interface compatibility)
            features: Must contain columns ['date', 'ticker', 'funding_rate']

        Returns:
            AlphaResult with score = -(rank of avg_funding_rate)
            so that negative funding → high score (long favored).
        """
        empty = AlphaResult(
            date=date,
            signals=pd.DataFrame(columns=["ticker", "score"]),
        )

        if features is None or features.empty:
            return empty

        required = {"date", "ticker", "funding_rate"}
        if not required.issubset(features.columns):
            missing = required - set(features.columns)
            return empty

        # Look-ahead bias prevention
        past = features[features["date"] <= date].copy()
        if past.empty:
            return empty

        past = past.sort_values("date")

        # Compute rolling mean of funding rate over lookback window per ticker
        cutoff = past["date"].max() - pd.Timedelta(days=self.lookback_days)
        window = past[past["date"] >= cutoff]

        if window.empty:
            return empty

        avg_funding = (
            window.groupby("ticker")["funding_rate"]
            .mean()
            .reset_index()
            .rename(columns={"funding_rate": "avg_funding"})
        )

        avg_funding = avg_funding.dropna(subset=["avg_funding"])
        if avg_funding.empty:
            return empty

        # Score: rank inversely (low/negative funding → high score → long)
        avg_funding["score"] = (
            (avg_funding["avg_funding"].rank(pct=True) - 0.5) * -2
        )

        signals = avg_funding[["ticker", "score"]].reset_index(drop=True)

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={
                "lookback_days": self.lookback_days,
                "n_tickers": len(signals),
                "mean_funding_rate": float(avg_funding["avg_funding"].mean()),
            },
        )

    def _get_extra_state(self) -> dict[str, Any]:
        return {"lookback_days": self.lookback_days}

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        self.lookback_days = state.get("lookback_days", 7)
