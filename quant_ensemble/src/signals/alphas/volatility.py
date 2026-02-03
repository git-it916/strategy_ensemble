"""
Volatility Alpha Factor

Risk-based strategies using volatility and drawdown metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...common import get_logger
from ..base import RuleBasedAlpha

logger = get_logger(__name__)


class VolatilityAlpha(RuleBasedAlpha):
    """
    Low volatility alpha factor.

    Lower volatility = higher score (risk-averse strategy).
    Based on the low-volatility anomaly.

    Config:
        vol_window: Volatility lookback window name (default: "realized_vol_20d")
        invert: Whether to prefer low volatility (default: True)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.vol_window = self.config.get("vol_window", "realized_vol_20d")
        self.invert = self.config.get("invert", True)  # Prefer low vol

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute volatility-based scores.

        Args:
            date: Prediction date
            features_df: Features containing volatility metrics

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Find volatility column
        vol_col = None
        for col in df.columns:
            if "vol" in col.lower() and "20" in col:
                vol_col = col
                break

        if vol_col is None:
            for col in df.columns:
                if "vol" in col.lower():
                    vol_col = col
                    break

        if vol_col is None:
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.5] * len(df)),
                confidences=0.0,
            )

        # Rank volatility
        score = df[vol_col].rank(pct=True, na_option="keep")

        # Invert if we prefer low volatility
        if self.invert:
            score = 1.0 - score

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=score,
            confidences=1.0,
        )


class DrawdownAlpha(RuleBasedAlpha):
    """
    Drawdown-based alpha factor.

    Favors stocks with smaller recent drawdowns.

    Config:
        dd_window: Drawdown window name (default: "max_drawdown_60d")
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.dd_window = self.config.get("dd_window", "max_drawdown_60d")

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute drawdown-based scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Find drawdown column
        dd_col = None
        for col in df.columns:
            if "drawdown" in col.lower():
                dd_col = col
                break

        if dd_col is None:
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.5] * len(df)),
                confidences=0.0,
            )

        # Drawdowns are negative, more negative = worse
        # We want smaller (less negative) drawdowns to score higher
        score = df[dd_col].rank(pct=True, ascending=False, na_option="keep")

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=score,
            confidences=1.0,
        )


class RiskAdjustedMomentum(RuleBasedAlpha):
    """
    Risk-adjusted momentum alpha.

    Combines momentum with volatility adjustment (Sharpe-like).

    Config:
        momentum_col: Momentum feature name
        vol_col: Volatility feature name
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.momentum_col = self.config.get("momentum_col", "ret_12m_1m")
        self.vol_col = self.config.get("vol_col", "realized_vol_20d")

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute risk-adjusted momentum scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Find momentum column
        mom_col = None
        for col in df.columns:
            if "ret_12m" in col or "ret_6m" in col:
                mom_col = col
                break

        # Find volatility column
        vol_col = None
        for col in df.columns:
            if "vol" in col.lower() and "20" in col:
                vol_col = col
                break

        if mom_col is None:
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.5] * len(df)),
                confidences=0.0,
            )

        if vol_col is None:
            # Fallback to raw momentum
            score = df[mom_col].rank(pct=True, na_option="keep")
        else:
            # Risk-adjusted momentum: return / volatility
            vol = df[vol_col].clip(lower=0.01)  # Avoid division by zero
            risk_adj = df[mom_col] / vol
            score = risk_adj.rank(pct=True, na_option="keep")

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=score,
            confidences=1.0,
        )


class BetaAlpha(RuleBasedAlpha):
    """
    Beta-based alpha factor.

    Favors low-beta stocks (defensive strategy).

    Config:
        invert: Whether to prefer low beta (default: True)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.invert = self.config.get("invert", True)

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute beta-based scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Find beta column (may need to be pre-computed)
        beta_col = None
        for col in df.columns:
            if "beta" in col.lower():
                beta_col = col
                break

        if beta_col is None:
            # Fallback: use volatility as proxy for beta
            for col in df.columns:
                if "vol" in col.lower():
                    beta_col = col
                    break

        if beta_col is None:
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.5] * len(df)),
                confidences=0.0,
            )

        score = df[beta_col].rank(pct=True, na_option="keep")

        if self.invert:
            score = 1.0 - score

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=score,
            confidences=1.0,
        )


class VolatilityTiming(RuleBasedAlpha):
    """
    Volatility timing alpha.

    Adjusts exposure based on market volatility regime.
    Higher market vol -> lower scores (reduce exposure).

    Config:
        market_vol_col: Market volatility column name
        threshold: Volatility threshold for regime switch
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.market_vol_col = self.config.get("market_vol_col", "market_vol_20d")
        self.threshold = self.config.get("threshold", 0.20)

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute volatility timing scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame (should include market vol)

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Find market volatility
        market_vol = None
        for col in df.columns:
            if "market_vol" in col.lower():
                market_vol = df[col].iloc[0]
                break

        if market_vol is None:
            # Assume normal regime
            market_vol = 0.15

        # Scale factor based on market volatility
        # Higher vol -> lower scale
        scale = min(1.0, self.threshold / max(market_vol, 0.05))

        # This alpha returns a uniform score scaled by market conditions
        # In high vol, all scores are reduced (reduce exposure)
        base_score = pd.Series([0.5] * len(df), index=df.index)
        scaled_score = base_score * scale + (1 - scale) * 0.5

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=scaled_score,
            confidences=scale,  # Confidence reflects regime
        )
