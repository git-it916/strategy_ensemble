"""
Score Board

Track and analyze strategy performance over time.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScoreBoard:
    """
    Track strategy performance metrics.

    Features:
        - Rolling performance tracking
        - Win rate calculation
        - Sharpe ratio estimation
        - Strategy ranking
    """

    def __init__(self, strategy_names: list[str]):
        """
        Initialize score board.

        Args:
            strategy_names: List of strategy names to track
        """
        self.strategy_names = strategy_names
        self._history: dict[str, list[dict]] = {name: [] for name in strategy_names}
        self._daily_returns: dict[str, list[float]] = defaultdict(list)

    def record(
        self,
        strategy_name: str,
        date: datetime,
        returns: float,
        hit_rate: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Record strategy performance.

        Args:
            strategy_name: Strategy name
            date: Performance date
            returns: Strategy returns
            hit_rate: Win rate (optional)
            metadata: Additional info
        """
        if strategy_name not in self._history:
            self._history[strategy_name] = []

        record = {
            "date": date,
            "returns": returns,
            "hit_rate": hit_rate,
            "metadata": metadata or {},
        }

        self._history[strategy_name].append(record)
        self._daily_returns[strategy_name].append(returns)

        # Keep limited history
        max_history = 252 * 2  # 2 years
        if len(self._history[strategy_name]) > max_history:
            self._history[strategy_name] = self._history[strategy_name][-max_history:]
            self._daily_returns[strategy_name] = self._daily_returns[strategy_name][-max_history:]

    def get_recent_performance(
        self,
        lookback: int = 21,
    ) -> dict[str, dict[str, float]]:
        """
        Get recent performance metrics for all strategies.

        Args:
            lookback: Number of periods to look back

        Returns:
            Dict of strategy_name -> performance metrics
        """
        results = {}

        for name in self.strategy_names:
            returns = self._daily_returns.get(name, [])[-lookback:]

            if len(returns) < 5:
                results[name] = {
                    "mean_return": 0,
                    "sharpe": 0,
                    "win_rate": 0.5,
                    "n_periods": len(returns),
                }
                continue

            mean_ret = np.mean(returns)
            std_ret = np.std(returns) if len(returns) > 1 else 0.0001
            win_rate = np.mean([1 if r > 0 else 0 for r in returns])
            sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0

            results[name] = {
                "mean_return": mean_ret,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "n_periods": len(returns),
                "cumulative_return": np.prod([1 + r for r in returns]) - 1,
            }

        return results

    def get_ranking(self, metric: str = "sharpe") -> list[tuple[str, float]]:
        """
        Rank strategies by metric.

        Args:
            metric: Metric to rank by (sharpe, mean_return, win_rate)

        Returns:
            List of (strategy_name, metric_value) sorted descending
        """
        performance = self.get_recent_performance()

        rankings = [
            (name, perf.get(metric, 0))
            for name, perf in performance.items()
        ]

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_strategy_history(
        self,
        strategy_name: str,
        as_dataframe: bool = True,
    ) -> pd.DataFrame | list[dict]:
        """
        Get full history for a strategy.

        Args:
            strategy_name: Strategy name
            as_dataframe: Return as DataFrame

        Returns:
            History records
        """
        history = self._history.get(strategy_name, [])

        if as_dataframe:
            if not history:
                return pd.DataFrame()
            return pd.DataFrame(history)

        return history

    def get_correlation_matrix(self, lookback: int = 60) -> pd.DataFrame:
        """
        Calculate correlation matrix between strategies.

        Args:
            lookback: Lookback period

        Returns:
            Correlation matrix DataFrame
        """
        # Build returns DataFrame
        returns_dict = {}
        for name in self.strategy_names:
            returns = self._daily_returns.get(name, [])[-lookback:]
            if len(returns) >= lookback:
                returns_dict[name] = returns

        if not returns_dict:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_dict)
        return returns_df.corr()

    def get_drawdown(self, strategy_name: str) -> dict[str, float]:
        """
        Calculate drawdown metrics for a strategy.

        Args:
            strategy_name: Strategy name

        Returns:
            Drawdown metrics
        """
        returns = self._daily_returns.get(strategy_name, [])

        if len(returns) < 2:
            return {"max_drawdown": 0, "current_drawdown": 0}

        # Calculate cumulative returns
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        return {
            "max_drawdown": abs(min(drawdowns)),
            "current_drawdown": abs(drawdowns[-1]),
        }

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all strategies.

        Returns:
            Summary DataFrame
        """
        performance = self.get_recent_performance()

        rows = []
        for name, perf in performance.items():
            dd = self.get_drawdown(name)
            rows.append({
                "strategy": name,
                "mean_return": perf["mean_return"],
                "sharpe": perf["sharpe"],
                "win_rate": perf["win_rate"],
                "max_drawdown": dd["max_drawdown"],
                "n_periods": perf["n_periods"],
            })

        return pd.DataFrame(rows)

    def reset(self, strategy_name: str | None = None) -> None:
        """
        Reset history.

        Args:
            strategy_name: Specific strategy to reset (None = all)
        """
        if strategy_name:
            self._history[strategy_name] = []
            self._daily_returns[strategy_name] = []
        else:
            self._history = {name: [] for name in self.strategy_names}
            self._daily_returns = defaultdict(list)
