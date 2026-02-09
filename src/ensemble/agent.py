"""
Ensemble Agent

The brain that combines multiple alpha strategies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import logging

import numpy as np
import pandas as pd

from ..alphas.base_alpha import BaseAlpha, AlphaResult

logger = logging.getLogger(__name__)


@dataclass
class EnsembleSignal:
    """Combined signal from ensemble."""
    date: datetime
    signals: pd.DataFrame  # ticker, score, weight contributions
    strategy_weights: dict[str, float]
    regime: str | None = None


class EnsembleAgent:
    """
    Ensemble agent that combines multiple alpha strategies.

    Features:
        - Dynamic strategy weighting based on recent performance
        - Regime-aware weight adjustment
        - Risk-based signal scaling
    """

    def __init__(
        self,
        strategies: list[BaseAlpha],
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize ensemble agent.

        Args:
            strategies: List of alpha strategies
            config: Configuration dict
        """
        self.strategies = {s.name: s for s in strategies}
        self.config = config or {}

        # Weighting config
        self.base_weights = self.config.get("base_weights", {})
        self.use_dynamic_weights = self.config.get("use_dynamic_weights", True)
        self.performance_lookback = self.config.get("performance_lookback", 21)

        # Current weights
        self._current_weights: dict[str, float] = {}
        self._initialize_weights()

        # Score board for tracking
        from .score_board import ScoreBoard
        self.score_board = ScoreBoard(list(self.strategies.keys()))

    def _initialize_weights(self) -> None:
        """Initialize strategy weights."""
        n_strategies = len(self.strategies)

        for name in self.strategies:
            if name in self.base_weights:
                self._current_weights[name] = self.base_weights[name]
            else:
                # Equal weight by default
                self._current_weights[name] = 1.0 / n_strategies

        # Normalize
        total = sum(self._current_weights.values())
        self._current_weights = {k: v / total for k, v in self._current_weights.items()}

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Fit all strategies.

        Args:
            prices: Price data
            features: Feature data
            labels: Label data

        Returns:
            Fit results per strategy
        """
        results = {}

        for name, strategy in self.strategies.items():
            logger.info(f"Fitting strategy: {name}")
            try:
                result = strategy.fit(prices, features, labels)
                results[name] = {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Failed to fit {name}: {e}")
                results[name] = {"status": "error", "error": str(e)}

        return results

    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        regime: str | None = None,
    ) -> EnsembleSignal:
        """
        Generate combined signals from all strategies.

        Args:
            date: Signal date
            prices: Price data up to date
            features: Feature data
            regime: Current market regime (optional)

        Returns:
            EnsembleSignal with combined scores
        """
        # Update weights based on recent performance
        if self.use_dynamic_weights:
            self._update_weights(regime)

        # Collect signals from all strategies
        all_signals: dict[str, pd.DataFrame] = {}

        for name, strategy in self.strategies.items():
            if not strategy.is_fitted:
                logger.warning(f"Strategy {name} not fitted, skipping")
                continue

            try:
                result = strategy.generate_signals(date, prices, features)
                all_signals[name] = result.signals
            except Exception as e:
                logger.error(f"Signal generation failed for {name}: {e}")

        # Combine signals
        combined = self._combine_signals(all_signals, date)

        return EnsembleSignal(
            date=date,
            signals=combined,
            strategy_weights=self._current_weights.copy(),
            regime=regime,
        )

    def _combine_signals(
        self,
        all_signals: dict[str, pd.DataFrame],
        date: datetime,
    ) -> pd.DataFrame:
        """
        Combine signals from multiple strategies.

        Args:
            all_signals: Dict of strategy_name -> signals DataFrame
            date: Signal date

        Returns:
            Combined signals DataFrame
        """
        if not all_signals:
            return pd.DataFrame(columns=["ticker", "score"])

        # Get all unique assets
        all_assets = set()
        for df in all_signals.values():
            all_assets.update(df["ticker"].tolist())

        # Initialize combined scores
        combined_data = []

        for ticker in all_assets:
            weighted_score = 0.0
            total_weight = 0.0
            contributions = {}

            for strat_name, signals in all_signals.items():
                weight = self._current_weights.get(strat_name, 0)

                asset_signal = signals[signals["ticker"] == ticker]
                if not asset_signal.empty:
                    score = asset_signal.iloc[0]["score"]
                    weighted_score += weight * score
                    total_weight += weight
                    contributions[strat_name] = score * weight

            # Normalize
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 0.0

            combined_data.append({
                "ticker": ticker,
                "score": final_score,
                **{f"{k}_contrib": v for k, v in contributions.items()},
            })

        return pd.DataFrame(combined_data)

    def _update_weights(self, regime: str | None = None) -> None:
        """
        Update strategy weights based on recent performance.

        Args:
            regime: Current market regime
        """
        # Get recent performance from score board
        performances = self.score_board.get_recent_performance(
            lookback=self.performance_lookback
        )

        if not performances:
            return

        # Calculate performance-based weights
        sharpes = {}
        for name, perf in performances.items():
            sharpes[name] = max(perf.get("sharpe", 0), 0.01)  # Floor at 0.01

        # Softmax-like weighting
        total_sharpe = sum(sharpes.values())
        if total_sharpe > 0:
            perf_weights = {k: v / total_sharpe for k, v in sharpes.items()}
        else:
            perf_weights = {k: 1 / len(sharpes) for k in sharpes}

        # Blend with base weights
        blend_factor = self.config.get("performance_blend", 0.5)

        for name in self._current_weights:
            base_w = self.base_weights.get(name, 1 / len(self.strategies))
            perf_w = perf_weights.get(name, base_w)

            self._current_weights[name] = (
                (1 - blend_factor) * base_w +
                blend_factor * perf_w
            )

        # Apply regime adjustments if available
        if regime:
            self._apply_regime_adjustment(regime)

        # Normalize
        total = sum(self._current_weights.values())
        self._current_weights = {k: v / total for k, v in self._current_weights.items()}

    def _apply_regime_adjustment(self, regime: str) -> None:
        """
        Adjust weights based on market regime.

        Args:
            regime: Current regime (e.g., "bull", "bear", "sideways")
        """
        regime_prefs = self.config.get("regime_preferences", {})

        if regime not in regime_prefs:
            return

        adjustments = regime_prefs[regime]

        for strat_name, multiplier in adjustments.items():
            if strat_name in self._current_weights:
                self._current_weights[strat_name] *= multiplier

    def update_performance(
        self,
        date: datetime,
        returns: dict[str, float],
    ) -> None:
        """
        Update strategy performance tracking.

        Args:
            date: Performance date
            returns: Dict of strategy_name -> return
        """
        for name, ret in returns.items():
            if name in self.strategies:
                self.strategies[name].record_performance(date, ret)
                self.score_board.record(name, date, ret)

    def get_weights(self) -> dict[str, float]:
        """Get current strategy weights."""
        return self._current_weights.copy()

    def set_weights(self, weights: dict[str, float]) -> None:
        """Manually set strategy weights."""
        total = sum(weights.values())
        self._current_weights = {k: v / total for k, v in weights.items()}

    def add_strategy(self, strategy: BaseAlpha) -> None:
        """Add a new strategy."""
        self.strategies[strategy.name] = strategy
        self._initialize_weights()

    def remove_strategy(self, name: str) -> None:
        """Remove a strategy."""
        if name in self.strategies:
            del self.strategies[name]
            if name in self._current_weights:
                del self._current_weights[name]
            self._initialize_weights()

    def get_state(self) -> dict[str, Any]:
        """
        Get full ensemble state for serialization.

        Captures all internal state needed for faithful restoration,
        including score_board history and per-strategy performance.

        Returns:
            Serializable state dict
        """
        return {
            "config": self.config,
            "current_weights": self._current_weights,
            "base_weights": self.base_weights,
            "use_dynamic_weights": self.use_dynamic_weights,
            "performance_lookback": self.performance_lookback,
            # ScoreBoard internal state
            "score_board": {
                "strategy_names": self.score_board.strategy_names,
                "history": self.score_board._history,
                "daily_returns": dict(self.score_board._daily_returns),
            },
            # Per-strategy performance history
            "strategy_performance": {
                name: s._performance_history
                for name, s in self.strategies.items()
            },
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """
        Restore ensemble state from saved dict.

        Args:
            state: State dict from get_state()
        """
        self._current_weights = state["current_weights"]
        self.base_weights = state.get("base_weights", self.base_weights)
        self.use_dynamic_weights = state.get("use_dynamic_weights", self.use_dynamic_weights)
        self.performance_lookback = state.get("performance_lookback", self.performance_lookback)

        # Restore ScoreBoard
        sb_state = state.get("score_board")
        if sb_state:
            self.score_board.strategy_names = sb_state["strategy_names"]
            self.score_board._history = sb_state["history"]
            self.score_board._daily_returns = defaultdict(
                list, sb_state["daily_returns"]
            )

        # Restore per-strategy performance history
        perf_map = state.get("strategy_performance", {})
        for name, history in perf_map.items():
            if name in self.strategies:
                self.strategies[name]._performance_history = history
