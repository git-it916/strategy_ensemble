"""
Low Volatility Anomaly Alpha

로직: 지난 N일간 변동성이 낮은 종목들을 매수.
저변동성 종목이 역설적으로 고수익을 기록하는 이상 현상 (Baker et al., 2011).
크립토에서도 안정적인 코인이 샤프비율 면에서 우수한 경향이 있음.
데이터: prices의 'close' 컬럼
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.alphas.base_alpha import AlphaResult, BaseAlpha


class LowVolatilityAnomaly(BaseAlpha):
    """
    Low Volatility Anomaly strategy.

    Computes daily return volatility per ticker over a rolling window
    and ranks cross-sectionally. Lower volatility → higher score (buy signal).
    Score range: [-1, 1] via rank inversion.
    """

    def __init__(self, name: str = "LowVolatilityAnomaly", lookback_days: int = 20):
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
        Generate low-vol anomaly signals.

        Score = (1 - vol_rank) * 2 - 1  →  low vol gets score near +1.
        """
        hist = prices[prices["date"] <= date].copy()

        if hist.empty:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
            )

        vol_records = []
        for ticker in hist["ticker"].unique():
            tkr = hist[hist["ticker"] == ticker].sort_values("date").tail(self.lookback_days)
            if len(tkr) < 2:
                continue

            rets = tkr["close"].pct_change().dropna()
            if rets.empty:
                continue

            vol_records.append({"ticker": ticker, "vol": float(rets.std())})

        if not vol_records:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
            )

        vol_df = pd.DataFrame(vol_records)
        vol_df["vol_rank"] = vol_df["vol"].rank(pct=True)
        vol_df["score"] = (1 - vol_df["vol_rank"]) * 2 - 1

        signals = vol_df[["ticker", "score"]].reset_index(drop=True)

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={"lookback_days": self.lookback_days, "n_tickers": len(signals)},
        )

    def _get_extra_state(self) -> dict[str, Any]:
        return {"lookback_days": self.lookback_days}

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        self.lookback_days = state.get("lookback_days", 20)
