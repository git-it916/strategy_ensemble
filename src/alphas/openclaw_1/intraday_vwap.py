"""
Intraday VWAP Deviation Alpha (1-hour timeframe)

1시간봉 VWAP 이탈도 기반 평균회귀 시그널.
가격이 VWAP 아래 → 매수 (저평가), VWAP 위 → 매도 (고평가).
거래량 가중 공정가치 대비 이탈을 포착하여 단기 반전 기회 탐색.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.alphas.base_alpha import AlphaResult, BaseAlpha


class IntradayVWAP(BaseAlpha):
    """
    VWAP deviation mean-reversion alpha.

    Computes rolling VWAP from hourly data and scores tickers
    by their deviation from VWAP. Price below VWAP → long signal,
    price above VWAP → short signal.
    """

    def __init__(
        self,
        name: str = "IntradayVWAP",
        lookback_bars: int = 24,  # 24 hours of 1h bars
    ):
        super().__init__(name)
        self.lookback_bars = lookback_bars

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
        """
        Generate VWAP deviation signals.

        Expects prices with columns: date/datetime, ticker, high, low, close, volume.
        """
        if prices is None or prices.empty:
            return AlphaResult(
                date=date,
                signals=pd.DataFrame(columns=["ticker", "score"]),
            )

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
                .tail(self.lookback_bars)
            )
            if len(tkr) < max(6, self.lookback_bars // 4):
                continue

            # Compute VWAP = sum(typical_price * volume) / sum(volume)
            typical_price = (tkr["high"] + tkr["low"] + tkr["close"]) / 3.0
            volumes = tkr["volume"].values

            total_vol = volumes.sum()
            if total_vol == 0:
                continue

            vwap = (typical_price.values * volumes).sum() / total_vol
            current_price = tkr["close"].iloc[-1]

            if vwap == 0:
                continue

            # Deviation as percentage
            deviation_pct = (current_price - vwap) / vwap

            # Score: negative deviation (below VWAP) → positive score (buy)
            # Normalize with tanh to keep in [-1, 1]
            score = float(np.tanh(-deviation_pct * 15))

            # Dampen reversion signal when strong trend exists.
            # In a strong uptrend, price naturally stays above VWAP — don't fight it.
            if len(tkr) >= 12:
                recent_close = tkr["close"].iloc[-1]
                past_close = tkr["close"].iloc[-min(12, len(tkr))]
                if past_close > 0:
                    trend_return = (recent_close / past_close) - 1
                    # Short signal (score < -0.3) but strong uptrend (>5%)
                    if score < -0.3 and trend_return > 0.05:
                        dampen = max(0.2, 1.0 - trend_return * 4)
                        score = score * dampen
                    # Long signal (score > 0.3) but strong downtrend (<-5%)
                    elif score > 0.3 and trend_return < -0.05:
                        dampen = max(0.2, 1.0 + trend_return * 4)
                        score = score * dampen

            records.append({"ticker": ticker, "score": round(score, 4)})

        signals = (
            pd.DataFrame(records)
            if records
            else pd.DataFrame(columns=["ticker", "score"])
        )

        return AlphaResult(
            date=date,
            signals=signals,
            metadata={"lookback_bars": self.lookback_bars, "n_tickers": len(signals)},
        )

    def _get_extra_state(self) -> dict[str, Any]:
        return {"lookback_bars": self.lookback_bars}

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        self.lookback_bars = state.get("lookback_bars", 24)
