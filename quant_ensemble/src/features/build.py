"""
Feature Construction Module

Builds features from raw price and fundamental data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..common import get_logger, load_yaml, validate_features_df

logger = get_logger(__name__)


class FeatureBuilder:
    """
    Builds features from raw market data.

    Supports:
    - Momentum features (returns at various horizons)
    - Volatility features
    - Microstructure features (volume, liquidity)
    - Regime features (market-wide)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize feature builder.

        Args:
            config: Feature configuration dictionary
        """
        self.config = config or {}

    @classmethod
    def from_config_file(cls, config_path: str) -> "FeatureBuilder":
        """Create FeatureBuilder from config file."""
        config = load_yaml(config_path)
        return cls(config.get("features", {}))

    def build_all_features(
        self,
        prices_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame | None = None,
        flows_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build all features from raw data.

        Args:
            prices_df: Price data with columns [date, asset_id, open, high, low, close, volume]
            fundamentals_df: Fundamental data with columns [date, asset_id, pbr, per, ...]
            flows_df: Flow data with columns [date, asset_id, foreign_flow, inst_flow, ...]

        Returns:
            Features DataFrame with columns [date, asset_id, feature_1, feature_2, ...]
        """
        logger.info("Building all features...")

        # Ensure datetime
        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        # Build each feature category
        momentum_features = self.build_momentum_features(prices_df)
        volatility_features = self.build_volatility_features(prices_df)
        microstructure_features = self.build_microstructure_features(prices_df)

        # Merge all features
        features_df = momentum_features

        for df in [volatility_features, microstructure_features]:
            features_df = features_df.merge(
                df,
                on=["date", "asset_id"],
                how="outer",
            )

        # Add fundamental features if available
        if fundamentals_df is not None:
            fundamental_features = self.build_fundamental_features(fundamentals_df)
            features_df = features_df.merge(
                fundamental_features,
                on=["date", "asset_id"] if "date" in fundamental_features.columns else ["asset_id"],
                how="left",
            )

        # Add flow features if available
        if flows_df is not None:
            flow_features = self.build_flow_features(flows_df)
            features_df = features_df.merge(
                flow_features,
                on=["date", "asset_id"],
                how="left",
            )

        # Sort and validate
        features_df = features_df.sort_values(["date", "asset_id"]).reset_index(drop=True)

        logger.info(f"Built features: {features_df.shape[0]} rows, {features_df.shape[1]} columns")

        return features_df

    def build_momentum_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build momentum features.

        Features:
        - ret_1w: 1-week return
        - ret_1m: 1-month return (21 days)
        - ret_3m: 3-month return (63 days)
        - ret_6m: 6-month return (126 days)
        - ret_12m: 12-month return (252 days)
        - ret_12m_1m: 12-1 month momentum (skip recent month)
        - ret_6m_1m: 6-1 month momentum

        Args:
            prices_df: Price DataFrame

        Returns:
            Momentum features DataFrame
        """
        logger.info("Building momentum features...")

        # Pivot to wide format for calculations
        close_wide = prices_df.pivot(
            index="date",
            columns="asset_id",
            values="close"
        ).sort_index()

        features = {}

        # Simple returns at various horizons
        horizons = {
            "ret_1w": 5,
            "ret_1m": 21,
            "ret_3m": 63,
            "ret_6m": 126,
            "ret_12m": 252,
        }

        for name, window in horizons.items():
            features[name] = close_wide.pct_change(window)

        # Skip-month momentum (skip recent 21 days)
        features["ret_12m_1m"] = close_wide.shift(21).pct_change(252 - 21)
        features["ret_6m_1m"] = close_wide.shift(21).pct_change(126 - 21)

        # Convert back to long format
        result_dfs = []
        for name, df in features.items():
            long_df = df.reset_index().melt(
                id_vars=["date"],
                var_name="asset_id",
                value_name=name,
            )
            result_dfs.append(long_df)

        # Merge all features
        result = result_dfs[0]
        for df in result_dfs[1:]:
            result = result.merge(df, on=["date", "asset_id"], how="outer")

        return result

    def build_volatility_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build volatility features.

        Features:
        - realized_vol_5d: 5-day realized volatility
        - realized_vol_20d: 20-day realized volatility
        - realized_vol_60d: 60-day realized volatility
        - vol_of_vol_20d: Volatility of volatility
        - max_drawdown_20d: 20-day max drawdown
        - max_drawdown_60d: 60-day max drawdown
        - downside_vol_20d: Downside volatility

        Args:
            prices_df: Price DataFrame

        Returns:
            Volatility features DataFrame
        """
        logger.info("Building volatility features...")

        # Calculate returns
        prices_sorted = prices_df.sort_values(["asset_id", "date"]).copy()
        prices_sorted["return"] = prices_sorted.groupby("asset_id")["close"].pct_change()

        # Pivot returns to wide format
        returns_wide = prices_sorted.pivot(
            index="date",
            columns="asset_id",
            values="return"
        ).sort_index()

        close_wide = prices_sorted.pivot(
            index="date",
            columns="asset_id",
            values="close"
        ).sort_index()

        features = {}

        # Realized volatility at various windows (annualized)
        for window in [5, 20, 60]:
            features[f"realized_vol_{window}d"] = (
                returns_wide.rolling(window).std() * np.sqrt(252)
            )

        # Volatility of volatility
        vol_20d = returns_wide.rolling(20).std()
        features["vol_of_vol_20d"] = vol_20d.rolling(20).std()

        # Max drawdown
        for window in [20, 60]:
            rolling_max = close_wide.rolling(window).max()
            drawdown = (close_wide - rolling_max) / rolling_max
            features[f"max_drawdown_{window}d"] = drawdown.rolling(window).min()

        # Downside volatility (only negative returns)
        negative_returns = returns_wide.where(returns_wide < 0, 0)
        features["downside_vol_20d"] = (
            negative_returns.rolling(20).std() * np.sqrt(252)
        )

        # Convert to long format
        result_dfs = []
        for name, df in features.items():
            long_df = df.reset_index().melt(
                id_vars=["date"],
                var_name="asset_id",
                value_name=name,
            )
            result_dfs.append(long_df)

        result = result_dfs[0]
        for df in result_dfs[1:]:
            result = result.merge(df, on=["date", "asset_id"], how="outer")

        return result

    def build_microstructure_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build microstructure features.

        Features:
        - volume_zscore_5d: Volume z-score (5d vs 60d)
        - volume_zscore_20d: Volume z-score (20d vs 252d)
        - turnover_zscore_20d: Turnover z-score
        - amihud_illiquidity_20d: Amihud illiquidity measure

        Args:
            prices_df: Price DataFrame

        Returns:
            Microstructure features DataFrame
        """
        logger.info("Building microstructure features...")

        # Pivot to wide format
        volume_wide = prices_df.pivot(
            index="date",
            columns="asset_id",
            values="volume"
        ).sort_index()

        close_wide = prices_df.pivot(
            index="date",
            columns="asset_id",
            values="close"
        ).sort_index()

        features = {}

        # Volume z-scores
        for short_window, long_window in [(5, 60), (20, 252)]:
            vol_short = volume_wide.rolling(short_window).mean()
            vol_long_mean = volume_wide.rolling(long_window).mean()
            vol_long_std = volume_wide.rolling(long_window).std()
            features[f"volume_zscore_{short_window}d"] = (vol_short - vol_long_mean) / vol_long_std

        # Turnover (volume * price)
        if "turnover" in prices_df.columns:
            turnover_wide = prices_df.pivot(
                index="date",
                columns="asset_id",
                values="turnover"
            ).sort_index()
        else:
            turnover_wide = volume_wide * close_wide

        turnover_20d = turnover_wide.rolling(20).mean()
        turnover_252d_mean = turnover_wide.rolling(252).mean()
        turnover_252d_std = turnover_wide.rolling(252).std()
        features["turnover_zscore_20d"] = (turnover_20d - turnover_252d_mean) / turnover_252d_std

        # Amihud illiquidity: |return| / volume
        returns_wide = close_wide.pct_change()
        illiquidity = returns_wide.abs() / (volume_wide * close_wide)
        features["amihud_illiquidity_20d"] = illiquidity.rolling(20).mean()

        # Convert to long format
        result_dfs = []
        for name, df in features.items():
            long_df = df.reset_index().melt(
                id_vars=["date"],
                var_name="asset_id",
                value_name=name,
            )
            result_dfs.append(long_df)

        result = result_dfs[0]
        for df in result_dfs[1:]:
            result = result.merge(df, on=["date", "asset_id"], how="outer")

        return result

    def build_fundamental_features(
        self,
        fundamentals_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build fundamental features.

        Features are typically already in the fundamentals data:
        - pbr: Price to Book Ratio
        - per: Price to Earnings Ratio
        - roe: Return on Equity
        - debt_to_equity: Debt to Equity ratio

        Args:
            fundamentals_df: Fundamental DataFrame

        Returns:
            Fundamental features DataFrame (as-is or with date if available)
        """
        logger.info("Building fundamental features...")

        # Keep relevant columns
        keep_cols = ["date", "asset_id", "pbr", "per", "roe", "debt_to_equity", "dividend_yield"]
        available_cols = [c for c in keep_cols if c in fundamentals_df.columns]

        return fundamentals_df[available_cols].copy()

    def build_flow_features(self, flows_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build investor flow features.

        Features:
        - foreign_flow_5d: 5-day cumulative foreign net buying
        - foreign_flow_20d: 20-day cumulative foreign net buying
        - inst_flow_5d: 5-day institutional net buying
        - inst_flow_20d: 20-day institutional net buying

        Args:
            flows_df: Flow DataFrame with columns [date, asset_id, foreign_flow, inst_flow]

        Returns:
            Flow features DataFrame
        """
        logger.info("Building flow features...")

        features = []

        for flow_col in ["foreign_flow", "inst_flow"]:
            if flow_col not in flows_df.columns:
                continue

            # Pivot to wide
            flow_wide = flows_df.pivot(
                index="date",
                columns="asset_id",
                values=flow_col,
            ).sort_index()

            # Rolling sums
            for window in [5, 20]:
                rolling_sum = flow_wide.rolling(window).sum()
                long_df = rolling_sum.reset_index().melt(
                    id_vars=["date"],
                    var_name="asset_id",
                    value_name=f"{flow_col}_{window}d",
                )
                features.append(long_df)

        if not features:
            return pd.DataFrame(columns=["date", "asset_id"])

        result = features[0]
        for df in features[1:]:
            result = result.merge(df, on=["date", "asset_id"], how="outer")

        return result

    def build_regime_features(
        self,
        index_prices_df: pd.DataFrame,
        vix_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build market-wide regime features.

        Features:
        - market_ret_5d: Market 5-day return
        - market_ret_20d: Market 20-day return
        - market_vol_20d: Market 20-day volatility
        - vkospi: Korean VIX equivalent

        Args:
            index_prices_df: Index price DataFrame
            vix_df: VIX DataFrame (optional)

        Returns:
            Regime features DataFrame (date-level, not asset-level)
        """
        logger.info("Building regime features...")

        index_prices_df = index_prices_df.sort_values("date").copy()

        features = {"date": index_prices_df["date"]}

        # Market returns
        features["market_ret_5d"] = index_prices_df["close"].pct_change(5)
        features["market_ret_20d"] = index_prices_df["close"].pct_change(20)

        # Market volatility
        returns = index_prices_df["close"].pct_change()
        features["market_vol_20d"] = returns.rolling(20).std() * np.sqrt(252)

        # VIX if available
        if vix_df is not None and "close" in vix_df.columns:
            vix_df = vix_df.sort_values("date")
            features["vkospi"] = vix_df.set_index("date").reindex(
                index_prices_df["date"]
            )["close"].values

        return pd.DataFrame(features)


def build_features(
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame | None = None,
    flows_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Convenience function to build all features.

    Args:
        prices_df: Price DataFrame
        fundamentals_df: Fundamental DataFrame (optional)
        flows_df: Flow DataFrame (optional)
        config: Feature configuration

    Returns:
        Features DataFrame
    """
    builder = FeatureBuilder(config)
    return builder.build_all_features(prices_df, fundamentals_df, flows_df)
