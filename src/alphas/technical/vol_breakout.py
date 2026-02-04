"""
Volatility Breakout Alpha

Momentum strategy based on volatility breakouts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..base_alpha import BaseAlpha, AlphaResult


class VolatilityBreakoutAlpha(BaseAlpha):
    """
    Volatility breakout momentum strategy.

    Logic:
        - Buy when price breaks above previous range
        - Signal strength = breakout magnitude / volatility
        - Filter by volume confirmation

    Best in:
        - Trending markets
        - High volatility regimes
    """

    def __init__(
        self,
        name: str = "vol_breakout",
        lookback: int = 20,
        breakout_threshold: float = 1.5,
        volume_confirm: bool = True,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize volatility breakout strategy.

        Args:
            name: Strategy name
            lookback: Lookback period for range calculation
            breakout_threshold: ATR multiplier for breakout
            volume_confirm: Require volume confirmation
            config: Additional configuration
        """
        super().__init__(name, config)
        self.lookback = lookback
        self.breakout_threshold = breakout_threshold
        self.volume_confirm = volume_confirm

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
            "lookback": self.lookback,
            "breakout_threshold": self.breakout_threshold,
        }

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate breakout signals.

        Args:
            date: Signal date
            prices: Price data up to date
            features: Optional features

        Returns:
            AlphaResult with signals
        """
        signals_list = []

        # Filter data up to date
        prices = prices[prices["date"] <= pd.Timestamp(date)]

        for asset_id in prices["asset_id"].unique():
            asset_data = prices[prices["asset_id"] == asset_id].sort_values("date")

            if len(asset_data) < self.lookback + 1:
                continue

            recent = asset_data.tail(self.lookback + 1)

            # Current and historical prices
            current_close = recent.iloc[-1]["close"]
            hist_closes = recent.iloc[:-1]["close"].values

            # Calculate range metrics
            highest_high = recent.iloc[:-1]["close"].max()
            lowest_low = recent.iloc[:-1]["close"].min()

            # ATR-like volatility
            atr = np.std(hist_closes) * np.sqrt(252 / self.lookback)

            if atr == 0:
                continue

            # Breakout score
            if current_close > highest_high:
                # Upward breakout
                breakout_magnitude = (current_close - highest_high) / atr
                score = min(breakout_magnitude, 3.0)  # Cap at 3

            elif current_close < lowest_low:
                # Downward breakout (avoid or short)
                breakout_magnitude = (lowest_low - current_close) / atr
                score = -min(breakout_magnitude, 3.0)

            else:
                # No breakout
                # Score based on position in range
                range_size = highest_high - lowest_low
                if range_size > 0:
                    position = (current_close - lowest_low) / range_size
                    score = (position - 0.5) * 0.5  # Mild directional bias
                else:
                    score = 0.0

            # Volume confirmation
            if self.volume_confirm and "PX_VOLUME" in recent.columns:
                avg_volume = recent.iloc[:-1]["PX_VOLUME"].mean()
                current_volume = recent.iloc[-1]["PX_VOLUME"]

                if avg_volume > 0 and current_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    # Boost signal if volume confirms
                    if volume_ratio > 1.5 and score != 0:
                        score *= min(volume_ratio, 2.0)

            signals_list.append({
                "asset_id": asset_id,
                "score": score,
                "breakout_up": current_close > highest_high,
                "breakout_down": current_close < lowest_low,
            })

        signals = pd.DataFrame(signals_list)

        if signals.empty:
            signals = pd.DataFrame(columns=["asset_id", "score"])

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={
                "strategy": self.name,
                "n_breakouts_up": signals["breakout_up"].sum() if "breakout_up" in signals.columns else 0,
                "n_breakouts_down": signals["breakout_down"].sum() if "breakout_down" in signals.columns else 0,
            }
        )
