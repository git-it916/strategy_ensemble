"""
Quality Gates

Check whether a new alpha passes minimum quality requirements.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.openclaw.config import QUALITY_GATES, QualityGates

logger = logging.getLogger(__name__)


class QualityGateChecker:
    """
    Check backtest results against quality gates.

    All checks must pass for an alpha to be approved.
    """

    def __init__(self, gates: QualityGates | None = None):
        self.gates = gates or QUALITY_GATES

    def check_all(
        self,
        is_metrics: dict[str, float],
        oos_metrics: dict[str, float],
        daily_returns: pd.Series,
        signals_history: list[dict] | None = None,
        prices: pd.DataFrame | None = None,
        turnover: float = 0.0,
    ) -> tuple[bool, list[str], list[str]]:
        """
        Run all quality gate checks.

        Args:
            is_metrics: In-sample backtest metrics
            oos_metrics: Out-of-sample backtest metrics
            daily_returns: Full daily returns series
            signals_history: Optional signal history for IC calculation
            prices: Optional prices for IC calculation
            turnover: Average daily turnover

        Returns:
            (passed, failures, warnings)
        """
        failures = []
        warnings = []

        # 1. OOS Sharpe >= min_sharpe
        oos_sharpe = oos_metrics.get("sharpe_ratio", 0)
        if oos_sharpe < self.gates.min_sharpe:
            failures.append(
                f"OOS Sharpe {oos_sharpe:.2f} < {self.gates.min_sharpe}"
            )

        # 2. OOS Max Drawdown >= max_mdd (both negative)
        oos_mdd = oos_metrics.get("max_drawdown", 0)
        if oos_mdd < self.gates.max_mdd:
            failures.append(
                f"OOS MDD {oos_mdd:.2%} < {self.gates.max_mdd:.2%}"
            )

        # 3. Backtest period >= min_backtest_years
        n_days = len(daily_returns)
        years = n_days / 252
        if years < self.gates.min_backtest_years:
            failures.append(
                f"Backtest period {years:.1f}y < {self.gates.min_backtest_years}y"
            )

        # 4. Turnover <= max_daily_turnover
        if turnover > self.gates.max_daily_turnover:
            failures.append(
                f"Daily turnover {turnover:.2%} > {self.gates.max_daily_turnover:.2%}"
            )

        # 5. IS/OOS divergence check
        is_sharpe = is_metrics.get("sharpe_ratio", 0)
        if is_sharpe > 0 and oos_sharpe > 0:
            divergence = is_sharpe / oos_sharpe
            if divergence > self.gates.max_is_oos_sharpe_divergence:
                warnings.append(
                    f"IS/OOS Sharpe divergence {divergence:.1f}x "
                    f"(IS={is_sharpe:.2f}, OOS={oos_sharpe:.2f}) - "
                    f"possible overfitting"
                )

        # 6. Information Coefficient (if signal history available)
        if signals_history and prices is not None:
            ic = self.compute_ic(signals_history, prices)
            if ic < self.gates.min_ic:
                failures.append(
                    f"IC {ic:.4f} < {self.gates.min_ic}"
                )
        else:
            warnings.append("IC not computed (no signal history)")

        # 7. OOS positive total return
        oos_return = oos_metrics.get("total_return", 0)
        if oos_return <= 0:
            failures.append(
                f"OOS total return {oos_return:.2%} <= 0"
            )

        # 8. Win rate sanity check
        oos_win_rate = oos_metrics.get("win_rate", 0)
        if oos_win_rate < 0.35:
            warnings.append(
                f"Low OOS win rate: {oos_win_rate:.2%}"
            )

        passed = len(failures) == 0

        log_fn = logger.info if passed else logger.warning
        log_fn(
            f"Quality gates: {'PASSED' if passed else 'FAILED'} "
            f"(failures={len(failures)}, warnings={len(warnings)})"
        )

        return passed, failures, warnings

    @staticmethod
    def compute_ic(
        signals_history: list[dict],
        prices: pd.DataFrame,
        forward_days: int = 5,
    ) -> float:
        """
        Compute rank Information Coefficient.

        IC = Spearman correlation between signal scores and
        forward N-day returns, averaged across dates.

        Args:
            signals_history: List of {date, top_signals: [{ticker, score}]}
            prices: Price data with date, ticker, close columns
            forward_days: Forward return horizon

        Returns:
            Average IC across dates.
        """
        ics = []

        for entry in signals_history:
            signal_date = entry.get("date")
            top_signals = entry.get("top_signals", [])

            if not top_signals or signal_date is None:
                continue

            # Get forward returns
            for sig in top_signals:
                ticker = sig.get("ticker")
                score = sig.get("score")

                if ticker is None or score is None:
                    continue

                ticker_prices = prices[
                    (prices["ticker"] == ticker)
                    & (prices["date"] >= pd.Timestamp(signal_date))
                ].sort_values("date")

                if len(ticker_prices) <= forward_days:
                    continue

                p0 = ticker_prices.iloc[0]["close"]
                p1 = ticker_prices.iloc[forward_days]["close"]

                if p0 > 0:
                    sig["fwd_return"] = (p1 / p0) - 1

            # Calculate rank IC for this date
            valid = [s for s in top_signals if "fwd_return" in s]
            if len(valid) >= 3:
                scores = [s["score"] for s in valid]
                returns = [s["fwd_return"] for s in valid]

                corr, _ = stats.spearmanr(scores, returns)
                if not np.isnan(corr):
                    ics.append(corr)

        return float(np.mean(ics)) if ics else 0.0
