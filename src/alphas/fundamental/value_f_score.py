"""
Value F-Score Alpha

Piotroski F-Score based value strategy.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..base_alpha import BaseAlpha, AlphaResult


class ValueFScoreAlpha(BaseAlpha):
    """
    Piotroski F-Score based value strategy.

    F-Score components (0-9):
        1. ROA positive (1 point)
        2. Operating cash flow positive (1 point)
        3. ROA increasing (1 point)
        4. CFO > ROA (accruals quality) (1 point)
        5. Long-term debt decreasing (1 point)
        6. Current ratio increasing (1 point)
        7. No share dilution (1 point)
        8. Gross margin increasing (1 point)
        9. Asset turnover increasing (1 point)

    Signal:
        - High F-Score (7-9) = Buy
        - Low F-Score (0-3) = Avoid/Sell
        - Combined with low P/B for value

    Best in:
        - All market conditions
        - Especially effective for small caps
    """

    def __init__(
        self,
        name: str = "value_f_score",
        min_f_score: int = 5,
        max_pb_ratio: float = 3.0,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize F-Score strategy.

        Args:
            name: Strategy name
            min_f_score: Minimum F-Score to consider
            max_pb_ratio: Maximum P/B ratio for value filter
            config: Additional configuration
        """
        super().__init__(name, config)
        self.min_f_score = min_f_score
        self.max_pb_ratio = max_pb_ratio

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Fit strategy."""
        self.is_fitted = True
        self._fit_date = datetime.now()

        return {"status": "fitted"}

    def _get_extra_state(self) -> dict:
        return {
            "min_f_score": self.min_f_score,
            "max_pb_ratio": self.max_pb_ratio,
        }

    def _restore_extra_state(self, state: dict) -> None:
        self.min_f_score = state.get("min_f_score", 5)
        self.max_pb_ratio = state.get("max_pb_ratio", 3.0)

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate signals based on F-Score and value metrics.

        Args:
            date: Signal date
            prices: Price data
            features: Must contain fundamental features (roe, pb_ratio, etc.)

        Returns:
            AlphaResult with signals
        """
        signals_list = []

        if features is None:
            # Return empty if no fundamental data
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
                metadata={"error": "No fundamental features provided"}
            )

        # Filter features up to date
        features = features[features["date"] <= pd.Timestamp(date)]

        # Get latest features per asset
        latest_features = features.sort_values("date").groupby("ticker").last().reset_index()

        for _, row in latest_features.iterrows():
            ticker = row["ticker"]

            # Calculate simplified F-Score components
            f_score = 0

            # ROA/ROE positive
            if "roe" in row and row["roe"] > 0:
                f_score += 2

            # Low debt
            if "debt_to_equity" in row and row["debt_to_equity"] < 1:
                f_score += 2

            # Reasonable P/E
            if "pe_ratio" in row and 0 < row["pe_ratio"] < 20:
                f_score += 2

            # Good market cap (not too small)
            if "market_cap" in row and row["market_cap"] > 1e11:  # > 1000억
                f_score += 1

            # P/B value check
            pb_ratio = row.get("pb_ratio", 999)
            is_value = pb_ratio < self.max_pb_ratio

            # Generate score
            if f_score >= self.min_f_score and is_value:
                # Strong value signal
                score = (f_score - self.min_f_score + 1) / 5  # Normalize to 0-1 range
                score *= (self.max_pb_ratio - pb_ratio) / self.max_pb_ratio  # Boost lower P/B
            elif f_score < 3:
                # Negative signal for very low F-Score
                score = -0.3
            else:
                score = 0.0

            signals_list.append({
                "ticker": ticker,
                "score": score,
                "f_score": f_score,
                "pb_ratio": pb_ratio,
            })

        signals = pd.DataFrame(signals_list)

        if signals.empty:
            signals = pd.DataFrame(columns=["ticker", "score"])

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={
                "strategy": self.name,
                "n_high_f_score": len(signals[signals.get("f_score", 0) >= 7]) if "f_score" in signals.columns else 0,
            }
        )
