"""
Feature Engineer

Build ML features from daily OHLCV and minute-bar data.
All features are computed per (date, ticker) with no lookahead.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Compute features for ML alphas from raw price/volume data.

    Three feature sets:
        - Daily technical features (from daily OHLCV)
        - Intraday microstructure features (from minute bars, aggregated to daily)
        - Market-level features (cross-sectional, for regime classification)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # 1. Daily technical features
    # ------------------------------------------------------------------

    def build_daily_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Build per-stock daily features from OHLCV.

        Args:
            prices: DataFrame with date, ticker, open, high, low, close, volume

        Returns:
            DataFrame with date, ticker, and feature columns
        """
        prices = prices.sort_values(["ticker", "date"]).copy()
        groups = prices.groupby("ticker")

        # --- Returns ---
        prices["ret_1d"] = groups["close"].pct_change(1)
        prices["ret_5d"] = groups["close"].pct_change(5)
        prices["ret_20d"] = groups["close"].pct_change(20)
        prices["ret_60d"] = groups["close"].pct_change(60)
        prices["log_ret_1d"] = np.log1p(prices["ret_1d"])

        # --- Moving average ratios ---
        for window in [5, 10, 20, 60]:
            ma = groups["close"].transform(lambda x: x.rolling(window).mean())
            prices[f"ma_ratio_{window}"] = prices["close"] / ma - 1

        # --- RSI ---
        prices["rsi_14"] = groups["close"].transform(
            lambda x: self._compute_rsi(x, 14)
        )

        # --- Bollinger Band %B ---
        ma20 = groups["close"].transform(lambda x: x.rolling(20).mean())
        std20 = groups["close"].transform(lambda x: x.rolling(20).std())
        prices["bb_pct_b"] = (prices["close"] - (ma20 - 2 * std20)) / (4 * std20)
        prices["bb_pct_b"] = prices["bb_pct_b"].clip(0, 1)

        # --- MACD ---
        ema12 = groups["close"].transform(lambda x: x.ewm(span=12).mean())
        ema26 = groups["close"].transform(lambda x: x.ewm(span=26).mean())
        prices["macd"] = ema12 - ema26
        prices["macd_signal"] = prices.groupby("ticker")["macd"].transform(
            lambda x: x.ewm(span=9).mean()
        )

        # --- Volatility ---
        prices["vol_5d"] = groups["log_ret_1d"].transform(
            lambda x: x.rolling(5).std() * np.sqrt(252)
        )
        prices["vol_20d"] = groups["log_ret_1d"].transform(
            lambda x: x.rolling(20).std() * np.sqrt(252)
        )

        # --- Parkinson volatility (high-low based) ---
        if "high" in prices.columns and "low" in prices.columns:
            hl_ratio = np.log(prices["high"] / prices["low"])
            prices["parkinson_vol"] = prices.groupby("ticker")[
                hl_ratio.name if hasattr(hl_ratio, "name") else "hl_ratio"
            ].transform(lambda x: x.rolling(20).apply(
                lambda w: np.sqrt(w.pow(2).sum() / (4 * len(w) * np.log(2)))
            )) if False else None
            # Simpler approach
            prices["_hl_log"] = hl_ratio
            prices["parkinson_vol"] = groups["_hl_log"].transform(
                lambda x: x.rolling(20).apply(
                    lambda w: np.sqrt((w ** 2).mean() / (4 * np.log(2)))
                )
            )
            prices.drop(columns=["_hl_log"], inplace=True)

        # --- Garman-Klass volatility ---
        if all(c in prices.columns for c in ["open", "high", "low", "close"]):
            prices["garman_klass_vol"] = self._garman_klass(prices, groups)

        # --- Volume features ---
        vol_col = self._get_volume_col(prices)
        if vol_col:
            vol_ma20 = groups[vol_col].transform(lambda x: x.rolling(20).mean())
            prices["volume_ratio_20d"] = prices[vol_col] / vol_ma20
            prices["volume_ratio_20d"] = prices["volume_ratio_20d"].replace(
                [np.inf, -np.inf], np.nan
            )

        # --- Absolute return MA (for vol prediction) ---
        prices["ret_abs_ma5"] = groups["ret_1d"].transform(
            lambda x: x.abs().rolling(5).mean()
        )

        # --- Range ratio ---
        if "high" in prices.columns and "low" in prices.columns:
            prices["range_ratio"] = (prices["high"] - prices["low"]) / prices["close"]
            prices["range_ratio_ma20"] = groups["range_ratio"].transform(
                lambda x: x.rolling(20).mean()
            )

        # --- Vol of vol ---
        prices["vol_of_vol"] = groups["vol_20d"].transform(
            lambda x: x.rolling(20).std()
        )

        # --- Realized vol ratios (short/long for mean-reversion in vol) ---
        prices["vol_ratio_5_20"] = prices["vol_5d"] / prices["vol_20d"]
        prices["vol_ratio_5_20"] = prices["vol_ratio_5_20"].replace(
            [np.inf, -np.inf], np.nan
        )

        # Select feature columns
        feature_cols = [
            "ret_1d", "ret_5d", "ret_20d", "ret_60d",
            "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60",
            "rsi_14", "bb_pct_b",
            "macd", "macd_signal",
            "vol_5d", "vol_20d", "vol_of_vol", "vol_ratio_5_20",
            "ret_abs_ma5",
        ]

        # Conditionally add columns
        for col in [
            "parkinson_vol", "garman_klass_vol",
            "volume_ratio_20d",
            "range_ratio", "range_ratio_ma20",
        ]:
            if col in prices.columns:
                feature_cols.append(col)

        result = prices[["date", "ticker"] + feature_cols].copy()
        logger.info(f"Built {len(feature_cols)} daily features, {len(result)} rows")
        return result

    # ------------------------------------------------------------------
    # 2. Intraday features (minute bars -> daily aggregation)
    # ------------------------------------------------------------------

    def build_intraday_features(
        self,
        minute_bars: pd.DataFrame,
        market_open: str = "09:00",
        market_close: str = "15:30",
    ) -> pd.DataFrame:
        """
        Aggregate minute bars to daily intraday microstructure features.

        Args:
            minute_bars: DataFrame with datetime, ticker, open, high, low, close, volume
                         datetime column should be full timestamp (date + time)
            market_open: Market open time (HH:MM)
            market_close: Market close time (HH:MM)

        Returns:
            DataFrame with date, ticker, and intraday feature columns
        """
        df = minute_bars.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date.astype("datetime64[ns]")
        df["time"] = df["datetime"].dt.time

        # Define time windows
        morning_end = pd.Timestamp("1900-01-01 10:00").time()
        lunch_start = pd.Timestamp("1900-01-01 11:30").time()
        lunch_end = pd.Timestamp("1900-01-01 13:00").time()
        afternoon_start = pd.Timestamp("1900-01-01 14:30").time()
        mkt_close = pd.Timestamp(f"1900-01-01 {market_close}").time()
        mkt_open = pd.Timestamp(f"1900-01-01 {market_open}").time()

        # Minute returns
        df = df.sort_values(["ticker", "datetime"])
        df["bar_ret"] = df.groupby(["ticker", "date"])["close"].pct_change()

        results = []

        for (ticker, date), day_df in df.groupby(["ticker", "date"]):
            if len(day_df) < 10:
                continue

            bar_rets = day_df["bar_ret"].dropna()
            row = {"date": date, "ticker": ticker}

            # --- Overall intraday stats ---
            row["intraday_vol"] = bar_rets.std() * np.sqrt(len(bar_rets))
            row["bar_return_skew"] = bar_rets.skew() if len(bar_rets) > 3 else 0
            row["bar_return_kurtosis"] = bar_rets.kurtosis() if len(bar_rets) > 3 else 0

            # Large move bars (|ret| > 2 * std)
            bar_std = bar_rets.std()
            if bar_std > 0:
                row["large_bar_count"] = (bar_rets.abs() > 2 * bar_std).sum()
                row["large_bar_ratio"] = row["large_bar_count"] / len(bar_rets)
            else:
                row["large_bar_count"] = 0
                row["large_bar_ratio"] = 0

            # --- Time-window returns ---
            morning = day_df[day_df["time"] <= morning_end]
            afternoon = day_df[day_df["time"] >= afternoon_start]

            if len(morning) >= 2:
                row["ret_first_30min"] = (
                    morning.iloc[-1]["close"] / morning.iloc[0]["open"] - 1
                    if morning.iloc[0]["open"] > 0 else 0
                )
            else:
                row["ret_first_30min"] = 0

            if len(afternoon) >= 2:
                row["ret_last_30min"] = (
                    afternoon.iloc[-1]["close"] / afternoon.iloc[0]["open"] - 1
                    if afternoon.iloc[0]["open"] > 0 else 0
                )
            else:
                row["ret_last_30min"] = 0

            # --- AM / PM range ---
            am_bars = day_df[day_df["time"] < lunch_start]
            pm_bars = day_df[day_df["time"] >= lunch_end]

            if len(am_bars) > 0 and am_bars["close"].iloc[0] > 0:
                row["price_range_am"] = (
                    (am_bars["high"].max() - am_bars["low"].min())
                    / am_bars["close"].iloc[0]
                )
            else:
                row["price_range_am"] = 0

            if len(pm_bars) > 0 and pm_bars["close"].iloc[0] > 0:
                row["price_range_pm"] = (
                    (pm_bars["high"].max() - pm_bars["low"].min())
                    / pm_bars["close"].iloc[0]
                )
            else:
                row["price_range_pm"] = 0

            # --- VWAP deviation ---
            vol_col = self._get_volume_col(day_df)
            if vol_col and day_df[vol_col].sum() > 0:
                vwap = (day_df["close"] * day_df[vol_col]).sum() / day_df[vol_col].sum()
                row["vwap_deviation"] = day_df["close"].iloc[-1] / vwap - 1
            else:
                row["vwap_deviation"] = 0

            # --- Volume concentration (top 30 min / total) ---
            if vol_col and day_df[vol_col].sum() > 0:
                total_vol = day_df[vol_col].sum()
                top30_vol = day_df[vol_col].nlargest(30).sum()
                row["volume_concentration"] = top30_vol / total_vol
            else:
                row["volume_concentration"] = 0

            # --- Morning volume ratio ---
            if vol_col:
                morning_vol = morning[vol_col].sum() if len(morning) > 0 else 0
                total_vol = day_df[vol_col].sum()
                row["volume_profile_morning"] = (
                    morning_vol / total_vol if total_vol > 0 else 0
                )
            else:
                row["volume_profile_morning"] = 0

            # --- Open-close gap (vs previous close) ---
            row["open_close_gap"] = 0  # Will be filled after groupby

            # --- Intraday realized vol (for vol prediction) ---
            row["intraday_realized_vol"] = (
                bar_rets.std() * np.sqrt(252 * len(bar_rets))
                if len(bar_rets) > 1 else 0
            )

            results.append(row)

        result_df = pd.DataFrame(results)

        if not result_df.empty:
            # Fill open-close gap using daily first/last
            result_df = result_df.sort_values(["ticker", "date"])
            # This needs previous day close - will be joined with daily data later

        logger.info(
            f"Built intraday features: {len(result_df)} rows, "
            f"{len(result_df.columns) - 2} features"
        )
        return result_df

    # ------------------------------------------------------------------
    # 3. Market-level features (for regime classification)
    # ------------------------------------------------------------------

    def build_market_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Build market-level (cross-sectional) features per date.

        Args:
            prices: Daily OHLCV with date, ticker, close, volume

        Returns:
            DataFrame with date and market feature columns (1 row per date)
        """
        prices = prices.sort_values(["ticker", "date"]).copy()

        # Per-stock daily returns
        prices["_ret"] = prices.groupby("ticker")["close"].pct_change()
        prices["_ma20"] = prices.groupby("ticker")["close"].transform(
            lambda x: x.rolling(20).mean()
        )
        prices["_above_ma20"] = (prices["close"] > prices["_ma20"]).astype(int)

        # 52-week high/low
        prices["_high_52w"] = prices.groupby("ticker")["close"].transform(
            lambda x: x.rolling(252).max()
        )
        prices["_low_52w"] = prices.groupby("ticker")["close"].transform(
            lambda x: x.rolling(252).min()
        )
        prices["_at_52w_high"] = (prices["close"] >= prices["_high_52w"]).astype(int)
        prices["_at_52w_low"] = (prices["close"] <= prices["_low_52w"]).astype(int)

        # Aggregate to market level per date
        daily = prices.groupby("date").agg(
            market_ret=("_ret", "mean"),
            advance_decline_ratio=("_ret", lambda x: (x > 0).sum() / max((x < 0).sum(), 1)),
            pct_above_ma20=("_above_ma20", "mean"),
            market_breadth=("_at_52w_high", lambda x: x.sum()),
            market_breadth_low=("_at_52w_low", lambda x: x.sum()),
            cross_sectional_vol=("_ret", "std"),
            n_stocks=("_ret", "count"),
        ).reset_index()

        daily["market_breadth"] = daily["market_breadth"] - daily["market_breadth_low"]
        daily.drop(columns=["market_breadth_low"], inplace=True)

        # Rolling market features
        daily = daily.sort_values("date")
        daily["market_ret_5d"] = daily["market_ret"].rolling(5).sum()
        daily["market_ret_20d"] = daily["market_ret"].rolling(20).sum()
        daily["market_vol_20d"] = daily["market_ret"].rolling(20).std() * np.sqrt(252)

        # Volume trend
        vol_col = self._get_volume_col(prices)
        if vol_col:
            daily_vol = prices.groupby("date")[vol_col].sum().reset_index()
            daily_vol.columns = ["date", "_total_vol"]
            daily = daily.merge(daily_vol, on="date", how="left")
            daily["volume_trend"] = (
                daily["_total_vol"]
                / daily["_total_vol"].rolling(20).mean()
                - 1
            )
            daily.drop(columns=["_total_vol"], inplace=True)
        else:
            daily["volume_trend"] = 0

        feature_cols = [
            "market_ret", "market_ret_5d", "market_ret_20d",
            "market_vol_20d", "cross_sectional_vol",
            "advance_decline_ratio", "pct_above_ma20",
            "market_breadth", "volume_trend",
        ]

        result = daily[["date"] + feature_cols].copy()
        logger.info(f"Built {len(feature_cols)} market features, {len(result)} rows")
        return result

    # ------------------------------------------------------------------
    # Merge all features
    # ------------------------------------------------------------------

    def build_all(
        self,
        prices_daily: pd.DataFrame,
        minute_bars: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Build all feature sets.

        Returns:
            Dict with keys: "daily", "intraday", "market"
        """
        result = {}

        result["daily"] = self.build_daily_features(prices_daily)
        result["market"] = self.build_market_features(prices_daily)

        if minute_bars is not None:
            result["intraday"] = self.build_intraday_features(minute_bars)

        # Merge daily + intraday into combined features
        combined = result["daily"]
        if "intraday" in result and not result["intraday"].empty:
            combined = combined.merge(
                result["intraday"],
                on=["date", "ticker"],
                how="left",
            )

        result["combined"] = combined
        logger.info(f"Combined features: {combined.shape}")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _get_volume_col(df: pd.DataFrame) -> str | None:
        for col in ["volume", "PX_VOLUME", "Volume"]:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _garman_klass(prices: pd.DataFrame, groups) -> pd.Series:
        """Garman-Klass volatility estimator."""
        hl = np.log(prices["high"] / prices["low"])
        co = np.log(prices["close"] / prices["open"])

        gk_daily = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
        prices["_gk_daily"] = gk_daily
        result = groups["_gk_daily"].transform(
            lambda x: x.rolling(20).mean().apply(np.sqrt)
        )
        prices.drop(columns=["_gk_daily"], inplace=True)
        return result
