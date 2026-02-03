"""
Value Alpha Factor

Value-based strategies using fundamental data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...common import get_logger
from ..base import RuleBasedAlpha

logger = get_logger(__name__)


class ValueAlpha(RuleBasedAlpha):
    """
    Value composite alpha factor.

    Combines multiple value metrics (PBR, PER, dividend yield) into a single score.
    Lower valuation = higher score (contrarian value investing).

    Config:
        metrics: List of value metrics to use
        weights: Weights for each metric (default: equal)
        invert: List of metrics to invert (higher = worse)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.metrics = self.config.get("metrics", ["pbr", "per"])
        self.weights = self.config.get("weights", None)
        self.invert = self.config.get("invert", ["pbr", "per"])  # Lower is better

        if self.weights is None:
            self.weights = {m: 1.0 / len(self.metrics) for m in self.metrics}

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute value scores.

        Args:
            date: Prediction date
            features_df: Features containing value metrics

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Find available value columns
        value_scores = []
        total_weight = 0.0

        for metric in self.metrics:
            # Try to find the column
            col = None
            for c in df.columns:
                if metric.lower() in c.lower():
                    col = c
                    break

            if col is None:
                continue

            # Compute rank
            rank = df[col].rank(pct=True, na_option="keep")

            # Invert if needed (lower value ratio = higher score)
            if metric in self.invert:
                rank = 1.0 - rank

            weight = self.weights.get(metric, 1.0 / len(self.metrics))
            value_scores.append((rank, weight))
            total_weight += weight

        if not value_scores:
            logger.warning("No value metrics found, returning neutral scores")
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.5] * len(df)),
                confidences=0.0,
            )

        # Combine weighted scores
        combined = sum(score * weight for score, weight in value_scores) / total_weight

        # Confidence based on data availability
        n_available = sum(1 for score, _ in value_scores if score.notna().any())
        confidence = n_available / len(self.metrics)

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=combined,
            confidences=confidence,
        )


class BookToMarket(RuleBasedAlpha):
    """
    Book-to-market (inverse PBR) alpha.

    Simple value strategy: buy cheap stocks (high book-to-market).

    Config:
        None
    """

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute book-to-market scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Find PBR column
        pbr_col = None
        for col in df.columns:
            if "pbr" in col.lower() or "book" in col.lower():
                pbr_col = col
                break

        if pbr_col is None:
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.5] * len(df)),
                confidences=0.0,
            )

        # Book-to-market = 1/PBR (higher = cheaper)
        # But rank negative PBR to invert: lower PBR rank = higher score
        score = df[pbr_col].rank(pct=True, ascending=True, na_option="keep")
        score = 1.0 - score  # Invert: lower PBR = higher score

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=score,
            confidences=1.0,
        )


class EarningsYield(RuleBasedAlpha):
    """
    Earnings yield (inverse PE) alpha.

    Buy stocks with high earnings yield (low PE).

    Config:
        None
    """

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute earnings yield scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Find PER column
        per_col = None
        for col in df.columns:
            if "per" in col.lower() or "pe_" in col.lower() or "earning" in col.lower():
                per_col = col
                break

        if per_col is None:
            return self._create_score_df(
                date=date,
                asset_ids=df["asset_id"],
                scores=pd.Series([0.5] * len(df)),
                confidences=0.0,
            )

        # Earnings yield = 1/PE (higher = cheaper)
        # Filter negative PE (loss-making companies)
        per = df[per_col].copy()
        per = per.where(per > 0, np.nan)  # Exclude negative PE

        # Lower PE rank = higher score
        score = per.rank(pct=True, ascending=True, na_option="keep")
        score = 1.0 - score

        # Assign neutral score to negative PE companies
        score = score.fillna(0.5)

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=score,
            confidences=1.0,
        )


class QualityValue(RuleBasedAlpha):
    """
    Quality-adjusted value alpha.

    Combines value (cheap) with quality (profitable).
    Avoids "value traps" by requiring profitability.

    Config:
        value_weight: Weight for value component (default: 0.5)
        quality_weight: Weight for quality component (default: 0.5)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.value_weight = self.config.get("value_weight", 0.5)
        self.quality_weight = self.config.get("quality_weight", 0.5)

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute quality-value composite scores.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            Scores DataFrame
        """
        df = features_df.copy()

        # Value component (inverse PBR or PER)
        value_score = pd.Series([0.5] * len(df), index=df.index)
        for col in df.columns:
            if "pbr" in col.lower():
                value_score = 1.0 - df[col].rank(pct=True, na_option="keep")
                break
            elif "per" in col.lower():
                per = df[col].where(df[col] > 0, np.nan)
                value_score = 1.0 - per.rank(pct=True, na_option="keep")
                break

        # Quality component (ROE or profit margin)
        quality_score = pd.Series([0.5] * len(df), index=df.index)
        for col in df.columns:
            if "roe" in col.lower():
                quality_score = df[col].rank(pct=True, na_option="keep")
                break

        # Combine
        combined = (
            self.value_weight * value_score.fillna(0.5) +
            self.quality_weight * quality_score.fillna(0.5)
        )

        return self._create_score_df(
            date=date,
            asset_ids=df["asset_id"],
            scores=combined,
            confidences=1.0,
        )
