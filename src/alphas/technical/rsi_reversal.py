"""
RSI Reversal Alpha

Mean reversion strategy based on RSI extremes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..base_alpha import BaseAlpha, AlphaResult


class RSIReversalAlpha(BaseAlpha):
    """
    RSI-based mean reversion strategy.

    Logic:
        - Buy oversold stocks (RSI < 30)
        - Sell overbought stocks (RSI > 70)
        - Score = inverse of RSI deviation from 50

    Best in:
        - Range-bound markets
        - Low volatility regimes
    """

    def __init__(
        self,
        name: str = "rsi_reversal",
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize RSI reversal strategy.

        Args:
            name: Strategy name
            rsi_period: RSI calculation period
            oversold: Oversold threshold
            overbought: Overbought threshold
            config: Additional configuration
        """
        super().__init__(name, config)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Fit strategy (calibrate thresholds if needed).

        For RSI reversal, we can optionally calibrate oversold/overbought
        thresholds based on historical performance.
        """
        self.is_fitted = True
        self._fit_date = datetime.now()

        # Could add threshold optimization here
        return {
            "status": "fitted",
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate mean reversion signals based on RSI.

        Args:
            date: Signal date
            prices: Price data up to date
            features: Optional features (may include pre-calculated RSI)

        Returns:
            AlphaResult with signals
        """
        signals_list = []

        # Filter data up to date (no lookahead)
        prices = prices[prices["date"] <= pd.Timestamp(date)]

        for ticker in prices["ticker"].unique():
            asset_prices = prices[prices["ticker"] == ticker].sort_values("date")

            if len(asset_prices) < self.rsi_period + 1:
                continue

            # Check if RSI is in features
            if features is not None and "rsi_14" in features.columns:
                asset_features = features[
                    (features["ticker"] == ticker) &
                    (features["date"] <= pd.Timestamp(date))
                ]
                if not asset_features.empty:
                    rsi = asset_features.iloc[-1]["rsi_14"]
                else:
                    rsi = self._calculate_rsi(asset_prices["close"].values)
            else:
                rsi = self._calculate_rsi(asset_prices["close"].values)

            # Generate score based on RSI
            # Oversold (RSI < 30) -> positive score (buy)
            # Overbought (RSI > 70) -> negative score (sell/avoid)
            if rsi < self.oversold:
                # Strong buy signal - more oversold = higher score
                score = (self.oversold - rsi) / self.oversold
            elif rsi > self.overbought:
                # Negative signal
                score = -(rsi - self.overbought) / (100 - self.overbought)
            else:
                # Neutral zone
                score = 0.0

            signals_list.append({
                "ticker": ticker,
                "score": score,
                "rsi": rsi,
            })

        signals = pd.DataFrame(signals_list)

        if signals.empty:
            signals = pd.DataFrame(columns=["ticker", "score", "rsi"])

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={
                "strategy": self.name,
                "n_oversold": len(signals[signals["rsi"] < self.oversold]) if not signals.empty else 0,
                "n_overbought": len(signals[signals["rsi"] > self.overbought]) if not signals.empty else 0,
            }
        )

    def _get_extra_state(self) -> dict:
        return {
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }

    def _restore_extra_state(self, state: dict) -> None:
        self.rsi_period = state.get("rsi_period", 14)
        self.oversold = state.get("oversold", 30.0)
        self.overbought = state.get("overbought", 70.0)

    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate RSI from price array."""
        if len(prices) < self.rsi_period + 1:
            return 50.0  # Neutral

        # Calculate price changes
        deltas = np.diff(prices[-(self.rsi_period + 1):])

        # Separate gains and losses
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)

        # Average gains and losses
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
