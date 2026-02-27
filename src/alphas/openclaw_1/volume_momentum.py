"""
Volume Momentum Alpha

로직: 거래량 증가 추세를 따라가는 전략.
최근 거래량이 지속적으로 증가하는 종목은 기관/세력 관심도 상승을 의미 → 매수 신호.
데이터: prices의 'volume' 컬럼 (표준 컬럼명, PX_VOLUME 아님)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.alphas.base_alpha import AlphaResult, BaseAlpha


class VolumeMomentum(BaseAlpha):
    """
    Volume Momentum strategy.

    Measures the trend in trading volume over a rolling window.
    Rising volume = positive score (institutional interest).
    Uses 'volume' column (standard Binance/data contract column name).
    """

    def __init__(self, name: str = "VolumeMomentum", lookback_days: int = 20):
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
        Generate volume momentum signals.

        Score = tanh((last_vol - first_vol) / mean_vol * 10).
        """
        hist = prices[prices["date"] <= date].copy()

        if hist.empty:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
            )

        if "volume" not in hist.columns:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
            )

        records = []
        for ticker in hist["ticker"].unique():
            tkr = hist[hist["ticker"] == ticker].sort_values("date").tail(self.lookback_days)
            if len(tkr) < 2:
                continue

            volumes = tkr["volume"].values.astype(float)
            mean_vol = float(np.mean(volumes))
            if mean_vol <= 0:
                continue

            # Linear regression slope (least-squares fit) instead of naive first/last
            n = len(volumes)
            x = np.arange(n, dtype=float)
            x_mean = (n - 1) / 2.0
            slope = float(np.sum((x - x_mean) * (volumes - mean_vol)) / np.sum((x - x_mean) ** 2))
            trend_strength = slope / mean_vol
            score = float(np.tanh(trend_strength * 10))
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
