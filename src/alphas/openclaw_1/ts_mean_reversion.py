"""
Time Series Mean Reversion Alpha

로직: 극한 상승/하락 후 평균으로의 회귀를 포착.
지난 N일 누적 수익률이 극단적으로 높으면 매도, 낮으면 매수.
크립토에서는 단기 과매수/과매도 반전에 유효.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.alphas.base_alpha import AlphaResult, BaseAlpha


class TimeSeriesMeanReversion(BaseAlpha):
    """
    Time Series Mean Reversion strategy.

    Uses z-score of recent returns relative to a longer rolling window
    to detect overbought/oversold conditions. Distinct from TS Momentum
    by using a short signal window (5d) vs long baseline (60d).

    High z-score (recent outperformance) → short signal (expect reversion).
    Low z-score (recent underperformance) → long signal (expect bounce).
    """

    def __init__(
        self,
        name: str = "TimeSeriesMeanReversion",
        signal_window: int = 5,
        baseline_window: int = 60,
    ):
        super().__init__(name)
        self.signal_window = signal_window      # Short-term return window
        self.baseline_window = baseline_window  # Long-term baseline for z-score

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Rule-based: no training needed."""
        self.is_fitted = True
        self._fit_date = datetime.now()
        return {"status": "fitted", "signal_window": self.signal_window, "baseline_window": self.baseline_window}

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate mean reversion signals using z-score approach.

        Z-score = (recent_5d_return - mean_of_5d_returns) / std_of_5d_returns
        Score = -tanh(z_score) — high z = overbought → sell, low z = oversold → buy
        """
        hist = prices[prices["date"] <= date].copy()

        if hist.empty:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
            )

        min_required = self.baseline_window + self.signal_window

        records = []
        for ticker in hist["ticker"].unique():
            tkr = hist[hist["ticker"] == ticker].sort_values("date").tail(min_required + 1)
            if len(tkr) < self.signal_window + 10:
                continue

            closes = tkr["close"].values
            # Compute rolling short-term returns
            ret_series = []
            for i in range(self.signal_window, len(closes)):
                if closes[i - self.signal_window] == 0:
                    continue
                r = (closes[i] - closes[i - self.signal_window]) / closes[i - self.signal_window]
                ret_series.append(r)

            if len(ret_series) < 10:
                continue

            recent_ret = ret_series[-1]
            mean_ret = float(np.mean(ret_series))
            std_ret = float(np.std(ret_series, ddof=1))

            if std_ret < 1e-8:
                continue

            z_score = (recent_ret - mean_ret) / std_ret
            # Invert: high z-score (overbought) → negative score (sell)
            score = float(-np.tanh(z_score * 0.5))
            records.append({"ticker": ticker, "score": score})

        signals = pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "score"])

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={"signal_window": self.signal_window, "baseline_window": self.baseline_window, "n_tickers": len(signals)},
        )

    def _get_extra_state(self) -> dict[str, Any]:
        return {"signal_window": self.signal_window, "baseline_window": self.baseline_window}

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        self.signal_window = state.get("signal_window", 5)
        self.baseline_window = state.get("baseline_window", 60)
