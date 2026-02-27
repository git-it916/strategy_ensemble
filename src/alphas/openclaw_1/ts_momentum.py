"""
Time Series Momentum Alpha

로직: 각 종목의 자체 수익률 추세를 포착.
지난 N일간 누적 수익률이 양수면 매수(롱), 음수면 매도(숏).
크립토에서는 단기(20일) 추세 추종이 유효.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.alphas.base_alpha import AlphaResult, BaseAlpha


class TimeSeriesMomentum(BaseAlpha):
    """
    Time Series Momentum strategy.

    Captures each asset's own return trend over a lookback window.
    Positive cumulative return → long signal; negative → short signal.
    Score is tanh-normalized to [-1, 1].
    """

    def __init__(self, name: str = "TimeSeriesMomentum", lookback_days: int = 20):
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
        return {"status": "fitted", "lookback_days": self.lookback_days}

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate TS momentum signals.

        Returns score = tanh(cumulative_return * 10) for each ticker.
        """
        hist = prices[prices["date"] <= date].copy()

        if hist.empty:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
            )

        records = []
        for ticker in hist["ticker"].unique():
            tkr = hist[hist["ticker"] == ticker].sort_values("date").tail(
                self.lookback_days + 1
            )
            if len(tkr) < self.lookback_days + 1:
                continue

            first_close = tkr["close"].iloc[0]
            if first_close == 0 or pd.isna(first_close):
                continue

            ret = (tkr["close"].iloc[-1] - first_close) / first_close
            # Use tanh(ret * 3) — avoids saturation for typical crypto moves
            # tanh(0.1*3)=0.29, tanh(0.2*3)=0.54, tanh(0.5*3)=0.91
            score = float(np.tanh(ret * 3))
            records.append({"ticker": ticker, "score": score})

        signals = pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "score"])

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={"lookback_days": self.lookback_days, "n_tickers": len(signals)},
        )

    def _get_extra_state(self) -> dict[str, Any]:
        return {"lookback_days": self.lookback_days}

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        self.lookback_days = state.get("lookback_days", 20)
