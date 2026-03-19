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
        lookback_days: int = 14,
        abs_threshold: float = 0.0001,
    ):
        super().__init__(name)
        self.lookback_days = lookback_days
        self.abs_threshold = abs_threshold

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

        # Filter out coins where funding rate is too close to zero (no edge)
        avg_funding = avg_funding[
            avg_funding["avg_funding"].abs() >= self.abs_threshold
        ]
        if avg_funding.empty:
            return empty

        # Score: rank inversely (low/negative funding → high score → long)
        avg_funding["score"] = (
            (avg_funding["avg_funding"].rank(pct=True) - 0.5) * -2
        )

        # Dampen signal when price trend is strongly aligned with carry direction.
        # Negative funding → positive score (long), but if price already surging → don't chase
        # Positive funding → negative score (short), but if price already crashing → don't chase
        for ticker in avg_funding["ticker"].unique():
            ticker_prices = prices[prices["ticker"] == ticker].sort_values(
                "date" if "date" in prices.columns else "datetime"
            )
            if len(ticker_prices) >= 5:
                recent = ticker_prices["close"].iloc[-5:]
                price_return = (recent.iloc[-1] / recent.iloc[0]) - 1
                funding_val = avg_funding.loc[
                    avg_funding["ticker"] == ticker, "avg_funding"
                ].iloc[0]
                score_val = avg_funding.loc[
                    avg_funding["ticker"] == ticker, "score"
                ].iloc[0]

                # Caution: negative funding → long signal, but price already rising fast
                if funding_val < 0 and price_return > 0.03:
                    dampen = max(0.2, 1.0 - price_return * 5)  # 3%→0.85, 10%→0.5, 16%→0.2
                    avg_funding.loc[
                        avg_funding["ticker"] == ticker, "score"
                    ] = score_val * dampen
                # Caution: positive funding → short signal, but price already falling fast
                elif funding_val > 0 and price_return < -0.03:
                    dampen = max(0.2, 1.0 + price_return * 5)
                    avg_funding.loc[
                        avg_funding["ticker"] == ticker, "score"
                    ] = score_val * dampen

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
        return {"lookback_days": self.lookback_days, "abs_threshold": self.abs_threshold}

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        self.lookback_days = state.get("lookback_days", 14)
        self.abs_threshold = state.get("abs_threshold", 0.0001)
