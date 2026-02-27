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

    def _get_extra_state(self) -> dict:
        return {
            "lookback": self.lookback,
            "breakout_threshold": self.breakout_threshold,
            "volume_confirm": self.volume_confirm,
        }

    def _restore_extra_state(self, state: dict) -> None:
        self.lookback = state.get("lookback", 20)
        self.breakout_threshold = state.get("breakout_threshold", 1.5)
        self.volume_confirm = state.get("volume_confirm", True)

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

        for ticker in prices["ticker"].unique():
            asset_data = prices[prices["ticker"] == ticker].sort_values("date")

            if len(asset_data) < self.lookback + 1:
                continue

            recent = asset_data.tail(self.lookback + 1)

            # Current and historical prices
            current_close = recent.iloc[-1]["close"]
            hist = recent.iloc[:-1]

            # Use actual high/low columns for range (not just close)
            if "high" in hist.columns and "low" in hist.columns:
                highest_high = hist["high"].max()
                lowest_low = hist["low"].min()
            else:
                highest_high = hist["close"].max()
                lowest_low = hist["close"].min()

            # True ATR: max(high-low, |high-prev_close|, |low-prev_close|)
            if "high" in hist.columns and "low" in hist.columns:
                highs = hist["high"].values
                lows = hist["low"].values
                closes = hist["close"].values
                tr_list = [highs[0] - lows[0]]  # first bar: just high-low
                for i in range(1, len(highs)):
                    tr = max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]),
                    )
                    tr_list.append(tr)
                atr = float(np.mean(tr_list))
            else:
                # Fallback: use daily return std as volatility proxy
                hist_closes = hist["close"].values
                returns = np.diff(hist_closes) / hist_closes[:-1]
                atr = float(np.std(returns) * hist_closes[-1]) if len(returns) > 0 else 0

            if atr == 0:
                continue

            # Breakout score — only trigger if magnitude exceeds threshold
            if current_close > highest_high:
                # Upward breakout
                breakout_magnitude = (current_close - highest_high) / atr
                if breakout_magnitude >= self.breakout_threshold:
                    score = min(breakout_magnitude / (self.breakout_threshold * 2), 1.0)
                else:
                    score = breakout_magnitude / (self.breakout_threshold * 2) * 0.5

            elif current_close < lowest_low:
                # Downward breakout
                breakout_magnitude = (lowest_low - current_close) / atr
                if breakout_magnitude >= self.breakout_threshold:
                    score = -min(breakout_magnitude / (self.breakout_threshold * 2), 1.0)
                else:
                    score = -breakout_magnitude / (self.breakout_threshold * 2) * 0.5

            else:
                # No breakout — mild directional bias
                range_size = highest_high - lowest_low
                if range_size > 0:
                    position = (current_close - lowest_low) / range_size
                    score = (position - 0.5) * 0.3
                else:
                    score = 0.0

            # Volume confirmation — boost but keep within [-1, 1]
            if self.volume_confirm and "volume" in recent.columns:
                avg_volume = recent.iloc[:-1]["volume"].mean()
                current_volume = recent.iloc[-1]["volume"]

                if avg_volume > 0 and current_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    if volume_ratio > 1.5 and score != 0:
                        boost = min(volume_ratio / 2.0, 1.5)  # Max 1.5x boost
                        score *= boost

            # Clamp final score to [-1, 1]
            score = max(-1.0, min(1.0, score))

            signals_list.append({
                "ticker": ticker,
                "score": score,
                "breakout_up": current_close > highest_high,
                "breakout_down": current_close < lowest_low,
            })

        signals = pd.DataFrame(signals_list)

        if signals.empty:
            signals = pd.DataFrame(columns=["ticker", "score"])

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={
                "strategy": self.name,
                "n_breakouts_up": signals["breakout_up"].sum() if "breakout_up" in signals.columns else 0,
                "n_breakouts_down": signals["breakout_down"].sum() if "breakout_down" in signals.columns else 0,
            }
        )
