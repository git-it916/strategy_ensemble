"""
Sentiment Long Alpha

Momentum strategy based on price and fundamental sentiment.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..base_alpha import BaseAlpha, AlphaResult


class SentimentLongAlpha(BaseAlpha):
    """
    Price momentum + quality sentiment strategy.

    Logic:
        - Combine price momentum with fundamental quality
        - Buy: Strong price momentum + improving fundamentals
        - Avoid: Negative momentum or deteriorating quality

    Components:
        1. Price momentum (3M, 6M returns)
        2. Earnings momentum (if available)
        3. Analyst sentiment (if available)
        4. Quality score (ROE, margins)

    Best in:
        - Bull markets
        - Quality rallies
    """

    def __init__(
        self,
        name: str = "sentiment_long",
        momentum_weight: float = 0.6,
        quality_weight: float = 0.4,
        momentum_lookback: int = 60,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize sentiment strategy.

        Args:
            name: Strategy name
            momentum_weight: Weight for price momentum
            quality_weight: Weight for quality score
            momentum_lookback: Days for momentum calculation
            config: Additional configuration
        """
        super().__init__(name, config)
        self.momentum_weight = momentum_weight
        self.quality_weight = quality_weight
        self.momentum_lookback = momentum_lookback

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Fit strategy."""
        self.is_fitted = True
        self._fit_date = datetime.now()

        return {
            "status": "fitted",
            "momentum_weight": self.momentum_weight,
            "quality_weight": self.quality_weight,
        }

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate sentiment-based signals.

        Args:
            date: Signal date
            prices: Price data
            features: Optional fundamental features

        Returns:
            AlphaResult with signals
        """
        signals_list = []

        # Filter prices up to date
        prices = prices[prices["date"] <= pd.Timestamp(date)]

        for asset_id in prices["asset_id"].unique():
            asset_prices = prices[prices["asset_id"] == asset_id].sort_values("date")

            if len(asset_prices) < self.momentum_lookback:
                continue

            # Calculate price momentum
            current_price = asset_prices.iloc[-1]["close"]
            past_price = asset_prices.iloc[-self.momentum_lookback]["close"]

            if past_price > 0:
                momentum = (current_price - past_price) / past_price
            else:
                momentum = 0

            # Normalize momentum to z-score (rough approximation)
            momentum_score = np.clip(momentum * 5, -1, 1)  # Scale and clip

            # Calculate quality score if features available
            quality_score = 0

            if features is not None:
                asset_features = features[
                    (features["asset_id"] == asset_id) &
                    (features["date"] <= pd.Timestamp(date))
                ]

                if not asset_features.empty:
                    latest = asset_features.iloc[-1]

                    # ROE contribution
                    if "roe" in latest:
                        roe = latest["roe"]
                        if roe > 0.15:
                            quality_score += 0.4
                        elif roe > 0.1:
                            quality_score += 0.2

                    # Low debt contribution
                    if "debt_to_equity" in latest:
                        if latest["debt_to_equity"] < 0.5:
                            quality_score += 0.3
                        elif latest["debt_to_equity"] < 1.0:
                            quality_score += 0.1

                    # Market cap (prefer large caps for quality)
                    if "market_cap" in latest:
                        if latest["market_cap"] > 1e12:  # > 1조
                            quality_score += 0.3

            # Combine scores
            combined_score = (
                self.momentum_weight * momentum_score +
                self.quality_weight * quality_score
            )

            # Apply momentum filter - don't buy falling knives
            if momentum_score < -0.3:
                combined_score *= 0.5  # Reduce signal for negative momentum

            signals_list.append({
                "asset_id": asset_id,
                "score": combined_score,
                "momentum_score": momentum_score,
                "quality_score": quality_score,
            })

        signals = pd.DataFrame(signals_list)

        if signals.empty:
            signals = pd.DataFrame(columns=["asset_id", "score"])

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={
                "strategy": self.name,
                "n_positive_momentum": len(signals[signals.get("momentum_score", 0) > 0]) if "momentum_score" in signals.columns else 0,
            }
        )
