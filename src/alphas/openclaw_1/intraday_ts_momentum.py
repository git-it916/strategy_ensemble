"""
Intraday Time Series Momentum Alpha (5-minute timeframe)

5분봉 기반 단기 추세 모멘텀.
최근 N개 5분봉 누적 수익률이 양수면 롱, 음수면 숏 시그널을 만든다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.alphas.base_alpha import AlphaResult, BaseAlpha


class IntradayTimeSeriesMomentum(BaseAlpha):
    """
    Intraday TS momentum strategy.

    Uses intraday candles (5m) to detect short-term trend direction.
    """

    def __init__(
        self,
        name: str = "IntradayTimeSeriesMomentum",
        lookback_bars: int = 36,  # 36 x 5m = 180 minutes
        scale: float = 5.0,
    ):
        super().__init__(name)
        self.lookback_bars = lookback_bars
        self.scale = scale

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        self.is_fitted = True
        self._fit_date = datetime.now()
        return {"status": "fitted", "lookback_bars": self.lookback_bars}

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        if prices is None or prices.empty:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
            )

        time_col = "datetime" if "datetime" in prices.columns else "date"
        compare_date = date
        if hasattr(prices[time_col].dtype, "tz") and prices[time_col].dtype.tz is not None:
            from datetime import timezone
            if not getattr(date, "tzinfo", None):
                compare_date = date.replace(tzinfo=timezone.utc)

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
                .tail(self.lookback_bars + 1)
            )
            if len(tkr) < self.lookback_bars + 1:
                continue

            first_close = tkr["close"].iloc[0]
            if first_close == 0 or pd.isna(first_close):
                continue

            ret = (tkr["close"].iloc[-1] - first_close) / first_close
            score = float(np.tanh(ret * self.scale))
            records.append({"ticker": ticker, "score": score})

        signals = pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "score"])

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={"lookback_bars": self.lookback_bars, "n_tickers": len(signals)},
        )

    def _get_extra_state(self) -> dict[str, Any]:
        return {"lookback_bars": self.lookback_bars, "scale": self.scale}

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        self.lookback_bars = state.get("lookback_bars", 36)
        self.scale = state.get("scale", 5.0)
