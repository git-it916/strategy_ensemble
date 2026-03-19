"""
IntradayRSIV2 — RSI alpha adapted to v2 interface.

Kept as-is from v1 — already well-implemented.
Only changes: uses BaseAlphaV2 + DataBundle interface.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.alphas.base_alpha_v2 import AlphaResultV2, BaseAlphaV2
from src.data.data_bundle import DataBundle


class IntradayRSIV2(BaseAlphaV2):
    """1h RSI reversal alpha (v2 interface)."""

    def __init__(
        self,
        name: str = "IntradayRSIV2",
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ):
        super().__init__(name, category="mean_reversion")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def data_requirements(self) -> dict[str, Any]:
        return {"ohlcv_1h": self.rsi_period + 10}

    @property
    def default_signal_decay_hours(self) -> float:
        return 4.0

    @property
    def default_regime_affinity(self) -> str:
        return "sideways"

    def fit(self, prices=None, features=None, labels=None) -> dict[str, Any]:
        self.is_fitted = True
        self._fit_date = datetime.now()
        return {"status": "fitted"}

    def generate_signals_v2(
        self, date: datetime, data: DataBundle
    ) -> AlphaResultV2:
        prices = data.ohlcv_1h
        if prices is None or prices.empty:
            return self._empty_result(date)

        time_col = "datetime" if "datetime" in prices.columns else "date"
        compare_date = date
        if (
            hasattr(prices[time_col].dtype, "tz")
            and prices[time_col].dtype.tz is not None
        ):
            from datetime import timezone
            if not getattr(date, "tzinfo", None):
                compare_date = date.replace(tzinfo=timezone.utc)

        hist = prices[prices[time_col] <= compare_date].copy()
        if hist.empty:
            return self._empty_result(date)

        records = []
        for ticker in hist["ticker"].unique():
            tkr = (
                hist[hist["ticker"] == ticker]
                .sort_values(time_col)
                .tail(self.rsi_period + 10)
            )
            if len(tkr) < self.rsi_period + 1:
                continue

            closes = tkr["close"].values
            rsi = self._compute_rsi(closes, self.rsi_period)
            if rsi is None:
                continue

            if rsi <= self.oversold:
                score = (self.oversold - rsi) / self.oversold
            elif rsi >= self.overbought:
                score = -(rsi - self.overbought) / (100 - self.overbought)
            else:
                score = (50 - rsi) / 100

            records.append({"ticker": ticker, "score": round(float(score), 4)})

        signals = (
            pd.DataFrame(records)
            if records
            else pd.DataFrame(columns=["ticker", "score"])
        )

        return AlphaResultV2(
            date=date,
            signals=signals,
            confidence=0.75,
            regime_affinity=self.default_regime_affinity,
            signal_decay_hours=self.default_signal_decay_hours,
            metadata={"rsi_period": self.rsi_period, "n_tickers": len(signals)},
        )

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int) -> float | None:
        """Compute RSI using Wilder's smoothing."""
        if len(closes) < period + 1:
            return None
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _get_extra_state(self) -> dict[str, Any]:
        return {
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }
