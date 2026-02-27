"""
Signal Aggregator

Combines signals from multiple alpha sources with per-alpha weights,
tracking each alpha's contribution to the final score for transparency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSignal:
    """Result of signal aggregation."""

    # ticker → final combined score
    scores: dict[str, float]
    # ticker → {alpha_name: contribution}
    contributions: dict[str, dict[str, float]]
    # alpha_name → weight used
    weights_used: dict[str, float]


class SignalAggregator:
    """
    Merge signals from multiple alpha sources into a single ranked list.

    Each alpha produces a DataFrame with columns [ticker, score].
    The aggregator multiplies each score by the alpha's weight, sums across
    alphas, and records per-ticker contributions for explainability.
    """

    def aggregate(
        self,
        alpha_signals: dict[str, pd.DataFrame],
        alpha_weights: dict[str, float],
        min_abs_score: float = 0.01,
    ) -> AggregatedSignal:
        """
        Combine weighted alpha signals.

        Args:
            alpha_signals: {alpha_name: DataFrame[ticker, score]}
            alpha_weights: {alpha_name: weight}  (should sum to ~1.0)
            min_abs_score: Drop tickers with |score| below this threshold.

        Returns:
            AggregatedSignal with scores, contributions, and weights_used.
        """
        combined: dict[str, float] = {}
        contributions: dict[str, dict[str, float]] = {}

        for alpha_name, signals_df in alpha_signals.items():
            weight = alpha_weights.get(alpha_name, 0.0)
            if weight == 0 or signals_df.empty:
                continue

            for _, row in signals_df.iterrows():
                ticker = row["ticker"]
                raw_score = float(row.get("score", 0))
                contribution = raw_score * weight

                combined[ticker] = combined.get(ticker, 0.0) + contribution

                if ticker not in contributions:
                    contributions[ticker] = {}
                contributions[ticker][alpha_name] = contribution

        # Filter out tiny scores
        if min_abs_score > 0:
            combined = {
                k: v for k, v in combined.items() if abs(v) >= min_abs_score
            }
            contributions = {
                k: v for k, v in contributions.items() if k in combined
            }

        return AggregatedSignal(
            scores=combined,
            contributions=contributions,
            weights_used={
                k: v for k, v in alpha_weights.items()
                if k in alpha_signals and v > 0
            },
        )

    @staticmethod
    def scores_to_target_weights(
        scores: dict[str, float],
        max_positions: int = 20,
        max_weight: float = 0.10,
        min_weight: float = 0.02,
    ) -> dict[str, float]:
        """
        Convert raw scores to portfolio target weights.

        Keeps top-K by absolute score, normalizes, and clips.

        Args:
            scores: {ticker: combined_score}
            max_positions: Maximum number of positions.
            max_weight: Max absolute weight per position.
            min_weight: Drop positions below this weight.

        Returns:
            {ticker: signed_weight} summing to <= 1.0 in absolute terms.
        """
        if not scores:
            return {}

        # Rank by absolute score, keep top K
        ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
        top = ranked[:max_positions]

        # Normalize
        total_abs = sum(abs(s) for _, s in top)
        if total_abs == 0:
            return {}

        weights = {}
        for ticker, score in top:
            w = score / total_abs
            if abs(w) < min_weight:
                continue
            # Clip to max
            if abs(w) > max_weight:
                w = max_weight if w > 0 else -max_weight
            weights[ticker] = w

        return weights

    @staticmethod
    def get_top_contributors(
        contributions: dict[str, dict[str, float]],
        ticker: str,
        top_n: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Get the top contributing alphas for a specific ticker.

        Returns:
            [(alpha_name, contribution), ...] sorted by |contribution| desc.
        """
        if ticker not in contributions:
            return []

        items = contributions[ticker].items()
        return sorted(items, key=lambda x: abs(x[1]), reverse=True)[:top_n]
