"""
Alpha Evaluator

Signal-level evaluation of individual alphas without portfolio construction.
Computes IC, hit rate, turnover, decay profile, and regime-conditional metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass
class AlphaEvalResult:
    """Result of individual alpha evaluation."""

    alpha_name: str

    # IC (Spearman rank correlation: signal vs forward return)
    ic_by_horizon: dict[int, float] = field(default_factory=dict)
    ic_series: dict[int, list[float]] = field(default_factory=dict)

    # Hit rate (sign agreement between signal and forward return)
    hit_rate_by_horizon: dict[int, float] = field(default_factory=dict)

    # Information Ratio (IC mean / IC std)
    ir_by_horizon: dict[int, float] = field(default_factory=dict)

    # Turnover (average fraction of signal change per period)
    avg_turnover: float = 0.0

    # Signal autocorrelation (how sticky are signals)
    signal_autocorrelation: float = 0.0

    # IC decay profile (IC at increasing horizons)
    ic_decay_profile: list[tuple[int, float]] = field(default_factory=list)

    # Regime-conditional IC (regime → IC)
    regime_conditional_ic: dict[str, float] = field(default_factory=dict)

    # Coverage
    n_eval_dates: int = 0
    avg_n_signals_per_date: float = 0.0

    def summary_row(self) -> dict[str, Any]:
        """Single-row dict for comparison tables."""
        return {
            "alpha": self.alpha_name,
            "ic_1d": self.ic_by_horizon.get(1, 0.0),
            "ic_5d": self.ic_by_horizon.get(5, 0.0),
            "ic_10d": self.ic_by_horizon.get(10, 0.0),
            "ir_1d": self.ir_by_horizon.get(1, 0.0),
            "ir_5d": self.ir_by_horizon.get(5, 0.0),
            "hit_1d": self.hit_rate_by_horizon.get(1, 0.0),
            "hit_5d": self.hit_rate_by_horizon.get(5, 0.0),
            "turnover": self.avg_turnover,
            "autocorr": self.signal_autocorrelation,
            "n_dates": self.n_eval_dates,
        }


class AlphaEvaluator:
    """
    Evaluate individual alpha signal quality on historical data.

    Usage:
        evaluator = AlphaEvaluator()
        result = evaluator.evaluate(alpha, prices, features, eval_dates)
        print(result.summary_row())
    """

    def evaluate(
        self,
        alpha,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        eval_dates: list[datetime] | None = None,
        forward_horizons: list[int] | None = None,
        min_signals: int = 5,
    ) -> AlphaEvalResult:
        """
        Run signal-level evaluation of an alpha.

        Args:
            alpha: BaseAlpha or BaseAlphaV2 instance (must have generate_signals).
            prices: Daily OHLCV with columns [date, ticker, open, high, low, close, volume].
            features: Optional features DataFrame.
            eval_dates: Dates to evaluate on. If None, uses last 60 trading dates.
            forward_horizons: Forward return horizons in days. Default [1, 5, 10, 20].
            min_signals: Minimum signals per date to include in evaluation.

        Returns:
            AlphaEvalResult with all metrics.
        """
        if forward_horizons is None:
            forward_horizons = [1, 5, 10, 20]

        # Determine evaluation dates
        all_dates = sorted(prices["date"].unique())
        max_horizon = max(forward_horizons)

        if eval_dates is None:
            # Use last 60 dates, leaving room for forward returns
            usable = all_dates[:-max_horizon] if len(all_dates) > max_horizon else all_dates
            eval_dates = usable[-60:]

        # Pre-compute forward returns for all dates
        fwd_returns = self._compute_forward_returns(prices, forward_horizons)

        # Generate signals for each date
        signals_by_date: dict[datetime, pd.DataFrame] = {}
        for date in eval_dates:
            try:
                prices_up_to = prices[prices["date"] <= date]
                features_up_to = features
                if features is not None and "date" in features.columns:
                    features_up_to = features[features["date"] <= pd.Timestamp(date)]

                result = alpha.generate_signals(date, prices_up_to, features_up_to)
                sig_df = result.signals if hasattr(result, "signals") else result
                if isinstance(sig_df, pd.DataFrame) and len(sig_df) >= min_signals:
                    signals_by_date[date] = sig_df
            except Exception as e:
                logger.debug(f"Signal generation failed at {date}: {e}")

        if not signals_by_date:
            logger.warning(f"No valid signals for {alpha.name}")
            return AlphaEvalResult(alpha_name=alpha.name)

        # Compute metrics
        result = AlphaEvalResult(
            alpha_name=alpha.name,
            n_eval_dates=len(signals_by_date),
            avg_n_signals_per_date=float(
                np.mean([len(df) for df in signals_by_date.values()])
            ),
        )

        # IC and hit rate per horizon
        for h in forward_horizons:
            ics, hit_rates = self._compute_ic_and_hit(
                signals_by_date, fwd_returns, h, min_signals
            )
            result.ic_by_horizon[h] = float(np.mean(ics)) if ics else 0.0
            result.ic_series[h] = ics
            result.ir_by_horizon[h] = (
                float(np.mean(ics) / np.std(ics))
                if ics and np.std(ics) > 0
                else 0.0
            )
            result.hit_rate_by_horizon[h] = (
                float(np.mean(hit_rates)) if hit_rates else 0.0
            )

        # IC decay profile
        decay_horizons = [1, 2, 3, 5, 10, 20]
        for h in decay_horizons:
            if h in result.ic_by_horizon:
                result.ic_decay_profile.append((h, result.ic_by_horizon[h]))
            else:
                ics, _ = self._compute_ic_and_hit(
                    signals_by_date, fwd_returns, h, min_signals
                )
                result.ic_decay_profile.append(
                    (h, float(np.mean(ics)) if ics else 0.0)
                )

        # Turnover
        result.avg_turnover = self._compute_turnover(signals_by_date)

        # Signal autocorrelation
        result.signal_autocorrelation = self._compute_autocorrelation(
            signals_by_date
        )

        # Regime-conditional IC (using simple price-based regime)
        result.regime_conditional_ic = self._compute_regime_ic(
            signals_by_date, fwd_returns, prices
        )

        return result

    def evaluate_batch(
        self,
        alphas: list,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        eval_dates: list[datetime] | None = None,
        forward_horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Evaluate multiple alphas and return a comparison DataFrame.
        """
        rows = []
        for alpha in alphas:
            logger.info(f"Evaluating {alpha.name}...")
            result = self.evaluate(
                alpha, prices, features, eval_dates, forward_horizons
            )
            rows.append(result.summary_row())
        return pd.DataFrame(rows)

    def compute_correlation_matrix(
        self,
        alphas: list,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        eval_dates: list[datetime] | None = None,
    ) -> pd.DataFrame:
        """
        Compute pairwise correlation matrix of alpha signals.

        Returns:
            DataFrame with alpha names as index/columns, values are
            average cross-sectional Spearman correlations.
        """
        all_dates = sorted(prices["date"].unique())
        if eval_dates is None:
            eval_dates = all_dates[-60:]

        # Collect scores: {alpha_name: {date: {ticker: score}}}
        all_scores: dict[str, dict] = {}
        for alpha in alphas:
            scores_per_date = {}
            for date in eval_dates:
                try:
                    p = prices[prices["date"] <= date]
                    f = features
                    if features is not None and "date" in features.columns:
                        f = features[features["date"] <= pd.Timestamp(date)]
                    result = alpha.generate_signals(date, p, f)
                    sig = result.signals if hasattr(result, "signals") else result
                    if isinstance(sig, pd.DataFrame) and not sig.empty:
                        scores_per_date[date] = dict(
                            zip(sig["ticker"], sig["score"])
                        )
                except Exception:
                    pass
            all_scores[alpha.name] = scores_per_date

        # Compute pairwise correlations
        names = [a.name for a in alphas]
        n = len(names)
        corr_matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                corrs = []
                for date in eval_dates:
                    s1 = all_scores[names[i]].get(date, {})
                    s2 = all_scores[names[j]].get(date, {})
                    common = set(s1) & set(s2)
                    if len(common) < 5:
                        continue
                    v1 = [s1[t] for t in common]
                    v2 = [s2[t] for t in common]
                    c, _ = spearmanr(v1, v2)
                    if not np.isnan(c):
                        corrs.append(c)
                avg_corr = float(np.mean(corrs)) if corrs else 0.0
                corr_matrix[i, j] = avg_corr
                corr_matrix[j, i] = avg_corr

        return pd.DataFrame(corr_matrix, index=names, columns=names)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_forward_returns(
        prices: pd.DataFrame,
        horizons: list[int],
    ) -> pd.DataFrame:
        """Pre-compute forward returns for all (date, ticker) pairs."""
        # Pivot to get close prices per ticker per date
        pivot = prices.pivot_table(
            index="date", columns="ticker", values="close", aggfunc="last"
        )
        pivot = pivot.sort_index()

        fwd_dfs = []
        for h in horizons:
            fwd = pivot.shift(-h) / pivot - 1
            fwd_melted = fwd.reset_index().melt(
                id_vars="date", var_name="ticker", value_name=f"fwd_return_{h}d"
            )
            fwd_dfs.append(fwd_melted)

        # Merge all horizons
        result = fwd_dfs[0]
        for df in fwd_dfs[1:]:
            result = result.merge(df, on=["date", "ticker"], how="outer")

        return result

    @staticmethod
    def _compute_ic_and_hit(
        signals_by_date: dict[datetime, pd.DataFrame],
        fwd_returns: pd.DataFrame,
        horizon: int,
        min_signals: int,
    ) -> tuple[list[float], list[float]]:
        """Compute IC and hit rate for a given horizon."""
        ret_col = f"fwd_return_{horizon}d"
        ics = []
        hit_rates = []

        for date, sig_df in signals_by_date.items():
            fwd = fwd_returns[fwd_returns["date"] == date]
            if fwd.empty or ret_col not in fwd.columns:
                continue

            merged = sig_df[["ticker", "score"]].merge(
                fwd[["ticker", ret_col]], on="ticker"
            )
            merged = merged.dropna(subset=["score", ret_col])

            if len(merged) < min_signals:
                continue

            # IC
            corr, _ = spearmanr(merged["score"], merged[ret_col])
            if not np.isnan(corr):
                ics.append(corr)

            # Hit rate (only for non-trivial signals)
            active = merged[merged["score"].abs() > 0.05]
            if len(active) >= 3:
                hits = (
                    np.sign(active["score"]) == np.sign(active[ret_col])
                ).mean()
                hit_rates.append(hits)

        return ics, hit_rates

    @staticmethod
    def _compute_turnover(
        signals_by_date: dict[datetime, pd.DataFrame],
    ) -> float:
        """Average turnover: how much signals change between consecutive dates."""
        dates = sorted(signals_by_date.keys())
        if len(dates) < 2:
            return 0.0

        turnovers = []
        for i in range(1, len(dates)):
            prev = signals_by_date[dates[i - 1]]
            curr = signals_by_date[dates[i]]

            prev_scores = dict(zip(prev["ticker"], prev["score"]))
            curr_scores = dict(zip(curr["ticker"], curr["score"]))
            all_tickers = set(prev_scores) | set(curr_scores)

            if not all_tickers:
                continue

            changes = sum(
                abs(curr_scores.get(t, 0) - prev_scores.get(t, 0))
                for t in all_tickers
            )
            turnovers.append(changes / len(all_tickers))

        return float(np.mean(turnovers)) if turnovers else 0.0

    @staticmethod
    def _compute_autocorrelation(
        signals_by_date: dict[datetime, pd.DataFrame],
    ) -> float:
        """Average cross-sectional autocorrelation of signals."""
        dates = sorted(signals_by_date.keys())
        if len(dates) < 2:
            return 0.0

        corrs = []
        for i in range(1, len(dates)):
            prev = dict(
                zip(
                    signals_by_date[dates[i - 1]]["ticker"],
                    signals_by_date[dates[i - 1]]["score"],
                )
            )
            curr = dict(
                zip(
                    signals_by_date[dates[i]]["ticker"],
                    signals_by_date[dates[i]]["score"],
                )
            )
            common = set(prev) & set(curr)
            if len(common) < 5:
                continue
            v1 = [prev[t] for t in common]
            v2 = [curr[t] for t in common]
            c, _ = spearmanr(v1, v2)
            if not np.isnan(c):
                corrs.append(c)

        return float(np.mean(corrs)) if corrs else 0.0

    @staticmethod
    def _compute_regime_ic(
        signals_by_date: dict[datetime, pd.DataFrame],
        fwd_returns: pd.DataFrame,
        prices: pd.DataFrame,
        lookback: int = 20,
    ) -> dict[str, float]:
        """
        Compute IC conditioned on market regime.

        Regime determined by BTC 20-day return:
            bull: > +5%, bear: < -5%, sideways: else
        """
        # Find BTC ticker
        btc_tickers = [
            t for t in prices["ticker"].unique()
            if t.upper().startswith("BTC/")
        ]
        if not btc_tickers:
            return {}

        btc_ticker = btc_tickers[0]
        btc_prices = (
            prices[prices["ticker"] == btc_ticker]
            .sort_values("date")
            .set_index("date")["close"]
        )

        regime_ics: dict[str, list[float]] = {
            "bull": [], "bear": [], "sideways": []
        }

        for date, sig_df in signals_by_date.items():
            # Determine regime
            btc_before = btc_prices[btc_prices.index <= date]
            if len(btc_before) < lookback:
                continue
            ret_20d = btc_before.iloc[-1] / btc_before.iloc[-lookback] - 1

            if ret_20d > 0.05:
                regime = "bull"
            elif ret_20d < -0.05:
                regime = "bear"
            else:
                regime = "sideways"

            # Compute IC
            fwd = fwd_returns[fwd_returns["date"] == date]
            ret_col = "fwd_return_5d"
            if fwd.empty or ret_col not in fwd.columns:
                continue
            merged = sig_df[["ticker", "score"]].merge(
                fwd[["ticker", ret_col]], on="ticker"
            ).dropna()
            if len(merged) < 5:
                continue
            corr, _ = spearmanr(merged["score"], merged[ret_col])
            if not np.isnan(corr):
                regime_ics[regime].append(corr)

        return {
            r: float(np.mean(ics)) if ics else 0.0
            for r, ics in regime_ics.items()
        }
