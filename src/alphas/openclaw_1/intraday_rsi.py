"""
Intraday RSI Alpha (1-hour timeframe)

1시간봉 RSI 기반 단기 과매수/과매도 시그널.
RSI < 30 → 매수 (과매도 반전), RSI > 70 → 매도 (과매수 반전).
5분 리밸런싱에 적합한 고빈도 시그널.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.alphas.base_alpha import AlphaResult, BaseAlpha


class IntradayRSI(BaseAlpha):
    """
    1h RSI reversal alpha.

    Uses hourly OHLCV data to compute RSI and generate
    mean-reversion signals at intraday frequency.
    """

    def __init__(
        self,
        name: str = "IntradayRSI",
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ):
        super().__init__(name)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        self.is_fitted = True
        self._fit_date = datetime.now()
        return {"status": "fitted", "rsi_period": self.rsi_period}

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate RSI signals from hourly price data.

        Expects prices DataFrame with columns: date/datetime, ticker, close.
        Works with both daily and hourly data — hourly gives better resolution.
        """
        if prices is None or prices.empty:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
            )

        # Use datetime column if available (hourly data), else date
        time_col = "datetime" if "datetime" in prices.columns else "date"
        # Ensure timezone-aware comparison
        compare_date = date
        if hasattr(prices[time_col].dtype, 'tz') and prices[time_col].dtype.tz is not None:
            from datetime import timezone
            if not getattr(date, 'tzinfo', None):
                compare_date = date.replace(tzinfo=timezone.utc) if hasattr(date, 'replace') else pd.Timestamp(date, tz='UTC')
        hist = prices[prices[time_col] <= compare_date].copy()

        if hist.empty:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
            )

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

            # Map RSI to score: oversold → +1 (buy), overbought → -1 (sell)
            if rsi <= self.oversold:
                # Linearly scale: RSI 30→0, RSI 0→+1
                score = (self.oversold - rsi) / self.oversold
            elif rsi >= self.overbought:
                # Linearly scale: RSI 70→0, RSI 100→-1
                score = -(rsi - self.overbought) / (100 - self.overbought)
            else:
                # Neutral zone — mild signal based on distance from 50
                score = (50 - rsi) / 100  # RSI 40→+0.1, RSI 60→-0.1

            records.append({"ticker": ticker, "score": round(float(score), 4)})

        signals = (
            pd.DataFrame(records)
            if records
            else pd.DataFrame(columns=["ticker", "score"])
        )

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={"rsi_period": self.rsi_period, "n_tickers": len(signals)},
        )

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int) -> float | None:
        """Compute RSI using exponential moving average of gains/losses."""
        if len(closes) < period + 1:
            return None

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Use EMA (Wilder's smoothing)
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

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        self.rsi_period = state.get("rsi_period", 14)
        self.oversold = state.get("oversold", 30.0)
        self.overbought = state.get("overbought", 70.0)
