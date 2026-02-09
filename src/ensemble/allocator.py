"""
Portfolio Allocator

Convert signals to portfolio weights with risk management.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Allocator(ABC):
    """Base class for portfolio allocators."""

    @abstractmethod
    def allocate(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Convert signals to portfolio weights.

        Args:
            signals: DataFrame with ticker, score
            prices: Optional price data for vol calculation
            constraints: Position constraints

        Returns:
            DataFrame with ticker, weight
        """
        pass


class TopKAllocator(Allocator):
    """
    Simple top-K allocation.

    Select top K assets by score, equal weight.
    """

    def __init__(self, k: int = 20, long_only: bool = True):
        """
        Initialize allocator.

        Args:
            k: Number of top assets to select
            long_only: Only long positions
        """
        self.k = k
        self.long_only = long_only

    def allocate(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Allocate to top K assets."""
        if signals.empty:
            return pd.DataFrame(columns=["ticker", "weight"])

        # Sort by score
        sorted_signals = signals.sort_values("score", ascending=False)

        if self.long_only:
            # Only positive scores
            sorted_signals = sorted_signals[sorted_signals["score"] > 0]

        # Select top K
        top_k = sorted_signals.head(self.k)

        if top_k.empty:
            return pd.DataFrame(columns=["ticker", "weight"])

        # Equal weight
        weight = 1.0 / len(top_k)

        result = pd.DataFrame({
            "ticker": top_k["ticker"],
            "weight": weight,
        })

        return result


class ScoreWeightedAllocator(Allocator):
    """
    Score-weighted allocation.

    Weight proportional to signal score.
    """

    def __init__(
        self,
        top_k: int | None = None,
        max_weight: float = 0.1,
        long_only: bool = True,
    ):
        """
        Initialize allocator.

        Args:
            top_k: Limit to top K (None = all)
            max_weight: Maximum weight per asset
            long_only: Only long positions
        """
        self.top_k = top_k
        self.max_weight = max_weight
        self.long_only = long_only

    def allocate(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Allocate based on scores."""
        if signals.empty:
            return pd.DataFrame(columns=["ticker", "weight"])

        df = signals.copy()

        if self.long_only:
            df = df[df["score"] > 0]

        if df.empty:
            return pd.DataFrame(columns=["ticker", "weight"])

        # Sort and limit
        df = df.sort_values("score", ascending=False)
        if self.top_k:
            df = df.head(self.top_k)

        # Calculate weights proportional to score
        if self.long_only:
            # All positive scores
            weights = df["score"] / df["score"].sum()
        else:
            # Handle long/short
            long_mask = df["score"] > 0
            short_mask = df["score"] < 0

            weights = pd.Series(0.0, index=df.index)

            if long_mask.any():
                long_total = df.loc[long_mask, "score"].sum()
                weights.loc[long_mask] = df.loc[long_mask, "score"] / long_total * 0.5

            if short_mask.any():
                short_total = abs(df.loc[short_mask, "score"].sum())
                weights.loc[short_mask] = df.loc[short_mask, "score"] / short_total * 0.5

        # Apply max weight constraint
        weights = weights.clip(upper=self.max_weight)

        # Renormalize
        weights = weights / weights.sum()

        result = pd.DataFrame({
            "ticker": df["ticker"],
            "weight": weights.values,
        })

        return result


class RiskParityAllocator(Allocator):
    """
    Risk parity allocation.

    Weight inversely proportional to volatility.
    """

    def __init__(
        self,
        top_k: int = 20,
        vol_lookback: int = 60,
        max_weight: float = 0.15,
    ):
        """
        Initialize allocator.

        Args:
            top_k: Number of assets to select
            vol_lookback: Lookback for volatility calculation
            max_weight: Maximum weight per asset
        """
        self.top_k = top_k
        self.vol_lookback = vol_lookback
        self.max_weight = max_weight

    def allocate(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Allocate using risk parity."""
        if signals.empty:
            return pd.DataFrame(columns=["ticker", "weight"])

        # Select top K by score
        df = signals[signals["score"] > 0].sort_values("score", ascending=False)
        df = df.head(self.top_k)

        if df.empty:
            return pd.DataFrame(columns=["ticker", "weight"])

        # Calculate volatilities if prices provided
        vols = {}
        if prices is not None:
            for ticker in df["ticker"]:
                asset_prices = prices[prices["ticker"] == ticker].sort_values("date")
                if len(asset_prices) >= self.vol_lookback:
                    returns = asset_prices["close"].pct_change().dropna()
                    vols[ticker] = returns.tail(self.vol_lookback).std() * np.sqrt(252)
                else:
                    vols[ticker] = 0.2  # Default vol

        # Risk parity weights (inverse vol)
        weights = {}
        for ticker in df["ticker"]:
            vol = vols.get(ticker, 0.2)
            weights[ticker] = 1 / max(vol, 0.05)  # Floor vol at 5%

        # Normalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Apply max weight
        weights = {k: min(v, self.max_weight) for k, v in weights.items()}

        # Renormalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        result = pd.DataFrame([
            {"ticker": k, "weight": v}
            for k, v in weights.items()
        ])

        return result


class BlackLittermanAllocator(Allocator):
    """
    Black-Litterman allocation.

    Combine market equilibrium with alpha views.
    """

    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        max_weight: float = 0.15,
    ):
        """
        Initialize allocator.

        Args:
            tau: Uncertainty in prior
            risk_aversion: Risk aversion coefficient
            max_weight: Maximum weight per asset
        """
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight

    def allocate(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Allocate using Black-Litterman.

        Simplified implementation - full BL requires covariance matrix.
        """
        if signals.empty or prices is None:
            return pd.DataFrame(columns=["ticker", "weight"])

        # For simplicity, use score-weighted with vol scaling
        df = signals[signals["score"] > 0].sort_values("score", ascending=False)
        df = df.head(20)

        if df.empty:
            return pd.DataFrame(columns=["ticker", "weight"])

        # Calculate vol-adjusted scores
        adjusted_scores = {}
        for _, row in df.iterrows():
            ticker = row["ticker"]
            score = row["score"]

            asset_prices = prices[prices["ticker"] == ticker].sort_values("date")
            if len(asset_prices) >= 60:
                vol = asset_prices["close"].pct_change().tail(60).std() * np.sqrt(252)
            else:
                vol = 0.2

            # Score / volatility (Sharpe-like)
            adjusted_scores[ticker] = score / max(vol, 0.05)

        # Normalize
        total = sum(max(v, 0) for v in adjusted_scores.values())
        if total > 0:
            weights = {k: max(v, 0) / total for k, v in adjusted_scores.items()}
        else:
            weights = {k: 1 / len(adjusted_scores) for k in adjusted_scores}

        # Apply max weight
        weights = {k: min(v, self.max_weight) for k, v in weights.items()}

        # Renormalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        result = pd.DataFrame([
            {"ticker": k, "weight": v}
            for k, v in weights.items()
        ])

        return result
