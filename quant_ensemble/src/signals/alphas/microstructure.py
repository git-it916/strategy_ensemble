"""
Microstructure Alpha Factor

Volume, liquidity, and flow-based strategies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...common import get_logger
from ..base import RuleBasedAlpha

logger = get_logger(__name__)


class MicrostructureAlpha(RuleBasedAlpha):
    """
    Microstructure composite alpha.

    Combines volume, liquidity, and flow signals.

    Config:
        components: List of components to use
        weights: Component weights
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.components = self.config.get(
            "components", ["volume", "liquidity", "flow"]
        )
        self.weights = self.config.get("weights", None)

        if self.weights is None:
            self.weights = {c: 1.0 / len(self.components) for c in self.components}

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute microstructure composite scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        scores = []
        weights_used = []

        for component in self.components:
            score = self._compute_component_score(df, component)
            if score is not None:
                scores.append(score)
                weights_used.append(self.weights.get(component, 1.0))

        if not scores:
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.5] * len(df)),
                confidences=0.0,
            )

        # Weighted average
        total_weight = sum(weights_used)
        combined = sum(s * w for s, w in zip(scores, weights_used)) / total_weight

        confidence = len(scores) / len(self.components)

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=combined,
            confidences=confidence,
        )

    def _compute_component_score(
        self,
        df: pd.DataFrame,
        component: str,
    ) -> pd.Series | None:
        """Compute score for a single component."""
        if component == "volume":
            # Higher volume = more attention
            for col in df.columns:
                if "volume_zscore" in col.lower():
                    return df[col].rank(pct=True, na_option="keep")

        elif component == "liquidity":
            # Higher liquidity = better (lower illiquidity)
            for col in df.columns:
                if "illiquidity" in col.lower() or "amihud" in col.lower():
                    return 1.0 - df[col].rank(pct=True, na_option="keep")

        elif component == "flow":
            # Higher foreign/institutional flow = positive
            for col in df.columns:
                if "foreign_flow" in col.lower():
                    return df[col].rank(pct=True, na_option="keep")
                elif "inst_flow" in col.lower():
                    return df[col].rank(pct=True, na_option="keep")

        return None


class VolumeAlpha(RuleBasedAlpha):
    """
    Volume-based alpha.

    High unusual volume often precedes price moves.

    Config:
        zscore_threshold: Threshold for "unusual" volume
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.zscore_threshold = self.config.get("zscore_threshold", 1.5)

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute volume-based scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Find volume z-score column
        vol_col = None
        for col in df.columns:
            if "volume_zscore" in col.lower():
                vol_col = col
                break

        if vol_col is None:
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.5] * len(df)),
                confidences=0.0,
            )

        # Higher volume z-score = higher score
        score = df[vol_col].rank(pct=True, na_option="keep")

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=score,
            confidences=1.0,
        )


class LiquidityAlpha(RuleBasedAlpha):
    """
    Liquidity-based alpha.

    Prefers more liquid stocks (lower transaction costs, easier execution).

    Config:
        invert: Whether to prefer high liquidity (default: True)
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
        Compute liquidity-based scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Find illiquidity column (Amihud)
        liq_col = None
        for col in df.columns:
            if "illiquidity" in col.lower() or "amihud" in col.lower():
                liq_col = col
                break

        if liq_col is None:
            # Try turnover as proxy
            for col in df.columns:
                if "turnover" in col.lower():
                    liq_col = col
                    break

        if liq_col is None:
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.5] * len(df)),
                confidences=0.0,
            )

        score = df[liq_col].rank(pct=True, na_option="keep")

        # For illiquidity, invert (lower illiquidity = higher score)
        if "illiquidity" in liq_col.lower() or "amihud" in liq_col.lower():
            score = 1.0 - score
        # For turnover, keep as is (higher turnover = higher score)

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=score,
            confidences=1.0,
        )


class FlowAlpha(RuleBasedAlpha):
    """
    Investor flow alpha.

    Follows institutional/foreign investor flows.

    Config:
        flow_type: Type of flow to track ('foreign', 'institutional', 'both')
        lookback: Which lookback to use (5 or 20 days)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.flow_type = self.config.get("flow_type", "both")
        self.lookback = self.config.get("lookback", 20)

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute flow-based scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        scores = []

        # Foreign flow
        if self.flow_type in ["foreign", "both"]:
            for col in df.columns:
                if "foreign_flow" in col.lower() and str(self.lookback) in col:
                    scores.append(df[col].rank(pct=True, na_option="keep"))
                    break

        # Institutional flow
        if self.flow_type in ["institutional", "both"]:
            for col in df.columns:
                if "inst_flow" in col.lower() and str(self.lookback) in col:
                    scores.append(df[col].rank(pct=True, na_option="keep"))
                    break

        if not scores:
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.5] * len(df)),
                confidences=0.0,
            )

        # Average scores
        combined = pd.concat(scores, axis=1).mean(axis=1)

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=combined,
            confidences=1.0,
        )


class SmartMoney(RuleBasedAlpha):
    """
    Smart money alpha.

    Combines flow and volume signals to identify informed trading.

    Config:
        None
    """

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute smart money scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Find flow columns
        flow_score = pd.Series([0.5] * len(df), index=df.index)
        for col in df.columns:
            if "foreign_flow" in col.lower():
                flow_score = df[col].rank(pct=True, na_option="keep")
                break

        # Find volume column
        vol_score = pd.Series([0.5] * len(df), index=df.index)
        for col in df.columns:
            if "volume_zscore" in col.lower():
                vol_score = df[col].rank(pct=True, na_option="keep")
                break

        # Smart money: high flow + high unusual volume
        combined = (flow_score + vol_score) / 2

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=combined,
            confidences=1.0,
        )
