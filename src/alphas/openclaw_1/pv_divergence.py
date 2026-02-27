"""
Price-Volume Divergence Alpha

로직: 가격과 거래량의 상관 관계를 포착.
- 가격 상승 + 거래량 증가 → 강한 추세 (롱)
- 가격 하락 + 거래량 증가 → 강한 하락 (숏)
- 가격 상승 + 거래량 감소 → 약한 추세 (약 매도)
- 가격 하락 + 거래량 감소 → 약한 추세 (약 매수, 반전 기대)

데이터: prices의 'close', 'volume' 컬럼 (표준 컬럼명)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.alphas.base_alpha import AlphaResult, BaseAlpha


class PriceVolumeDivergence(BaseAlpha):
    """
    Price-Volume Divergence strategy.

    Computes correlation between price returns and volume changes
    over a rolling window. High positive correlation = strong trend.
    """

    def __init__(self, name: str = "PriceVolumeDivergence", lookback_days: int = 20):
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
        Generate price-volume divergence signals.

        Uses 'volume' column (standard Binance/KIS data contract).
        Score = correlation(price_returns, volume_changes).
        """
        hist = prices[prices["date"] <= date].copy()

        if hist.empty:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
            )

        # Validate volume column exists
        if "volume" not in hist.columns:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
            )

        records = []
        for ticker in hist["ticker"].unique():
            tkr = hist[hist["ticker"] == ticker].sort_values("date").tail(self.lookback_days)
            if len(tkr) < 3:
                continue

            price_ret = tkr["close"].pct_change().dropna().values
            # Guard against volume=0 producing inf in pct_change
            vol_series = tkr["volume"].replace(0, np.nan)
            vol_change = vol_series.pct_change().dropna().values

            min_len = min(len(price_ret), len(vol_change))
            if min_len < 2:
                continue

            price_ret = price_ret[-min_len:]
            vol_change = vol_change[-min_len:]

            if np.std(price_ret) > 0 and np.std(vol_change) > 0:
                corr = float(np.corrcoef(price_ret, vol_change)[0, 1])
            else:
                corr = 0.0

            if np.isnan(corr):
                corr = 0.0

            # Make score directional:
            # - Positive corr + price up = strong trend (bullish)  → positive score
            # - Positive corr + price down = strong decline (bearish) → negative score
            # - Negative corr + price up = weak rally (bearish divergence) → mild negative
            # - Negative corr + price down = capitulation (bullish divergence) → mild positive
            price_direction = np.sign(np.mean(price_ret)) if len(price_ret) > 0 else 0.0

            # score = correlation * price_direction → positive when trend is confirmed
            score = float(np.tanh(corr * abs(price_direction) * 3)) * np.sign(price_direction)
            # Clamp
            score = max(-1.0, min(1.0, score))

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
