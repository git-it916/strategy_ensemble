"""
Momentum Alpha Factor

Classic momentum strategies based on historical returns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...common import get_logger
from ..base import RuleBasedAlpha

logger = get_logger(__name__)


class MomentumAlpha(RuleBasedAlpha):
    """
    Classic momentum alpha factor.

    Computes momentum score based on historical returns at various horizons.
    Implements the 12-1 month momentum (skip recent month to avoid reversal).

    Config:
        lookback_days: Primary lookback period (default: 252 for 12 months)
        skip_days: Days to skip (default: 21 for 1 month)
        horizons: List of (lookback, skip) tuples for multi-horizon
        combine_method: How to combine multiple horizons ('average', 'rank_average')
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.lookback_days = self.config.get("lookback_days", 252)
        self.skip_days = self.config.get("skip_days", 21)
        self.horizons = self.config.get("horizons", None)
        self.combine_method = self.config.get("combine_method", "rank_average")

        if self.horizons is None:
            self.horizons = [
                (252, 21),  # 12-1 month momentum
                (126, 21),  # 6-1 month momentum
            ]

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute momentum scores.

        Args:
            date: Prediction date
            features_df: Features containing momentum return columns

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Look for pre-computed momentum features
        momentum_cols = []
        for lookback, skip in self.horizons:
            col_name = f"ret_{lookback - skip}d_skip{skip}d"
            if col_name not in df.columns:
                # Try alternative naming conventions
                if f"ret_12m_1m" in df.columns and lookback == 252 and skip == 21:
                    col_name = "ret_12m_1m"
                elif f"ret_6m_1m" in df.columns and lookback == 126 and skip == 21:
                    col_name = "ret_6m_1m"
                elif f"ret_{lookback // 21}m" in df.columns:
                    col_name = f"ret_{lookback // 21}m"

            if col_name in df.columns:
                momentum_cols.append(col_name)

        # Fallback: use any available return columns
        if not momentum_cols:
            return_cols = [c for c in df.columns if c.startswith("ret_")]
            if return_cols:
                momentum_cols = return_cols[:2]  # Use first 2
            else:
                logger.warning("No momentum features found, returning neutral scores")
                return self._create_score_df(
                    date=date,
                    asset_ids=df["asset_id"],
                    scores=pd.Series([0.0] * len(df)),
                    confidences=0.0,
                )

        # Compute combined score
        if self.combine_method == "rank_average":
            # Rank each horizon, then average ranks
            ranks = []
            for col in momentum_cols:
                rank = df[col].rank(pct=True, na_option="keep")
                ranks.append(rank)
            combined_score = pd.concat(ranks, axis=1).mean(axis=1)

        elif self.combine_method == "average":
            # Simple average of returns
            combined_score = df[momentum_cols].mean(axis=1)
            # Convert to rank
            combined_score = combined_score.rank(pct=True, na_option="keep")

        else:
            # Use first momentum column
            combined_score = df[momentum_cols[0]].rank(pct=True, na_option="keep")

        # Calculate confidence based on data availability
        n_available = df[momentum_cols].notna().sum(axis=1)
        confidence = n_available / len(momentum_cols)

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=combined_score,
            confidences=confidence,
        )


class TimeSeriesMomentum(RuleBasedAlpha):
    """
    Time-series momentum (trend following).

    Unlike cross-sectional momentum, this looks at each asset's own trend.

    Config:
        lookback_days: Lookback period for trend (default: 252)
        vol_target: Target volatility for scaling (default: 0.15)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.lookback_days = self.config.get("lookback_days", 252)
        self.vol_target = self.config.get("vol_target", 0.15)

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute time-series momentum scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Look for return and volatility columns
        ret_col = None
        vol_col = None

        for col in df.columns:
            if "ret_12m" in col or "ret_252" in col:
                ret_col = col
            if "vol_20d" in col.lower() or "realized_vol" in col.lower():
                vol_col = col

        if ret_col is None:
            logger.warning("No return column found for TSMOM")
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.0] * len(df)),
                confidences=0.0,
            )

        # Calculate signal: sign of return
        signal = np.sign(df[ret_col])

        # Scale by volatility if available
        if vol_col is not None:
            vol = df[vol_col].clip(lower=0.01)  # Avoid division by zero
            scale = self.vol_target / vol
            score = signal * scale.clip(0.1, 3.0)  # Clip extreme scaling
        else:
            score = signal

        # Convert to rank for cross-sectional comparability
        score = score.rank(pct=True, na_option="keep")

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=score,
            confidences=1.0,
        )


class MomentumReversal(RuleBasedAlpha):
    """
    Short-term reversal alpha.

    Bets against recent short-term winners (mean reversion).

    Config:
        lookback_days: Short-term lookback (default: 5)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.lookback_days = self.config.get("lookback_days", 5)

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute reversal scores (negative of short-term momentum).

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Find short-term return column
        ret_col = None
        for col in df.columns:
            if "ret_1w" in col or "ret_5d" in col or "ret_1m" in col.lower():
                ret_col = col
                break

        if ret_col is None:
            ret_cols = [c for c in df.columns if c.startswith("ret_")]
            if ret_cols:
                ret_col = ret_cols[0]

        if ret_col is None:
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.0] * len(df)),
                confidences=0.0,
            )

        # Reversal: negative of short-term return
        score = -df[ret_col]
        score = score.rank(pct=True, na_option="keep")

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=score,
            confidences=1.0,
        )
