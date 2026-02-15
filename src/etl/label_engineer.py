"""
Label Engineer

Build training labels (y) for each ML alpha type.
All labels use forward-looking values, shifted to align with feature dates.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LabelEngineer:
    """
    Create label DataFrames for ML alpha training.

    All outputs have columns: date, ticker, y_reg
    The y_reg column meaning differs per label type:
        - return: forward N-day return
        - volatility: forward N-day realized volatility
        - regime: market regime encoded as int (0=bear, 1=sideways, 2=bull)
    """

    # ------------------------------------------------------------------
    # 1. Forward return labels
    # ------------------------------------------------------------------

    @staticmethod
    def build_return_labels(
        prices: pd.DataFrame,
        horizon: int = 5,
    ) -> pd.DataFrame:
        """
        Forward N-day return per stock.

        Args:
            prices: DataFrame with date, ticker, close
            horizon: Forward return horizon in trading days

        Returns:
            DataFrame with date, ticker, y_reg (forward return)
        """
        df = prices[["date", "ticker", "close"]].copy()
        df = df.sort_values(["ticker", "date"])

        df["y_reg"] = df.groupby("ticker")["close"].pct_change(horizon).shift(-horizon)

        result = df[["date", "ticker", "y_reg"]].dropna(subset=["y_reg"])
        logger.info(
            f"Return labels (horizon={horizon}d): {len(result)} rows, "
            f"mean={result['y_reg'].mean():.4f}, std={result['y_reg'].std():.4f}"
        )
        return result

    # ------------------------------------------------------------------
    # 2. Forward volatility labels
    # ------------------------------------------------------------------

    @staticmethod
    def build_volatility_labels(
        prices: pd.DataFrame,
        horizon: int = 5,
    ) -> pd.DataFrame:
        """
        Forward N-day realized volatility per stock.

        Args:
            prices: DataFrame with date, ticker, close
            horizon: Forward volatility window in trading days

        Returns:
            DataFrame with date, ticker, y_reg (forward realized vol, annualized)
        """
        df = prices[["date", "ticker", "close"]].copy()
        df = df.sort_values(["ticker", "date"])

        df["_ret"] = df.groupby("ticker")["close"].pct_change()

        # Forward rolling std of returns, then annualize
        df["y_reg"] = (
            df.groupby("ticker")["_ret"]
            .transform(lambda x: x.shift(-horizon).rolling(horizon).std())
            * np.sqrt(252)
        )

        result = df[["date", "ticker", "y_reg"]].dropna(subset=["y_reg"])
        logger.info(
            f"Volatility labels (horizon={horizon}d): {len(result)} rows, "
            f"mean={result['y_reg'].mean():.4f}"
        )
        return result

    # ------------------------------------------------------------------
    # 3. Regime labels (market-level)
    # ------------------------------------------------------------------

    @staticmethod
    def build_regime_labels(
        prices: pd.DataFrame,
        horizon: int = 20,
        bull_threshold: float = 0.03,
        bear_threshold: float = -0.03,
    ) -> pd.DataFrame:
        """
        Market regime labels based on forward aggregate market return.

        Args:
            prices: DataFrame with date, ticker, close
            horizon: Forward window for regime determination
            bull_threshold: Return above this = bull
            bear_threshold: Return below this = bear

        Returns:
            DataFrame with date, y_reg (0=bear, 1=sideways, 2=bull),
            regime (str label)
        """
        # Compute market-level return (equal-weight average)
        daily_ret = prices.groupby("date")["close"].mean().pct_change()
        market = daily_ret.reset_index()
        market.columns = ["date", "market_ret"]
        market = market.sort_values("date")

        # Forward cumulative return
        market["fwd_ret"] = (
            market["market_ret"]
            .shift(-1)
            .rolling(horizon)
            .apply(lambda x: (1 + x).prod() - 1)
            .shift(-horizon + 1)
        )

        # Classify regime
        def classify(r):
            if pd.isna(r):
                return np.nan
            if r > bull_threshold:
                return 2  # bull
            elif r < bear_threshold:
                return 0  # bear
            else:
                return 1  # sideways

        market["y_reg"] = market["fwd_ret"].apply(classify)
        market["regime"] = market["y_reg"].map(
            {0: "bear", 1: "sideways", 2: "bull"}
        )

        result = market[["date", "y_reg", "regime"]].dropna(subset=["y_reg"])
        result["y_reg"] = result["y_reg"].astype(int)

        # Log distribution
        dist = result["regime"].value_counts()
        logger.info(f"Regime labels (horizon={horizon}d): {dist.to_dict()}")

        return result

    # ------------------------------------------------------------------
    # 4. Intraday labels (next-day gap or same-day close direction)
    # ------------------------------------------------------------------

    @staticmethod
    def build_intraday_labels(
        prices: pd.DataFrame,
        label_type: str = "next_open_gap",
    ) -> pd.DataFrame:
        """
        Labels for intraday pattern alpha.

        Args:
            prices: DataFrame with date, ticker, open, close
            label_type:
                - "next_open_gap": (next_day_open - today_close) / today_close
                - "next_close_ret": next day close-to-close return

        Returns:
            DataFrame with date, ticker, y_reg
        """
        df = prices[["date", "ticker", "open", "close"]].copy()
        df = df.sort_values(["ticker", "date"])

        if label_type == "next_open_gap":
            df["_next_open"] = df.groupby("ticker")["open"].shift(-1)
            df["y_reg"] = (df["_next_open"] - df["close"]) / df["close"]
            df.drop(columns=["_next_open"], inplace=True)

        elif label_type == "next_close_ret":
            df["y_reg"] = df.groupby("ticker")["close"].pct_change().shift(-1)

        else:
            raise ValueError(f"Unknown label_type: {label_type}")

        result = df[["date", "ticker", "y_reg"]].dropna(subset=["y_reg"])
        logger.info(f"Intraday labels ({label_type}): {len(result)} rows")
        return result

    # ------------------------------------------------------------------
    # Build all labels
    # ------------------------------------------------------------------

    def build_all(
        self,
        prices: pd.DataFrame,
        return_horizon: int = 5,
        vol_horizon: int = 5,
        regime_horizon: int = 20,
    ) -> dict[str, pd.DataFrame]:
        """
        Build all label types at once.

        Returns:
            Dict with keys: "return", "volatility", "regime", "intraday"
        """
        return {
            "return": self.build_return_labels(prices, return_horizon),
            "volatility": self.build_volatility_labels(prices, vol_horizon),
            "regime": self.build_regime_labels(prices, regime_horizon),
            "intraday": self.build_intraday_labels(prices, "next_open_gap"),
        }
