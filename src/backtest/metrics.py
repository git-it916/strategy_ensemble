"""
Performance Metrics

Standard quantitative finance performance metrics for backtesting.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.035,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Strategy daily returns
        benchmark_returns: Benchmark daily returns (optional)
        risk_free_rate: Annual risk-free rate (Korean 국고채 3년)
        periods_per_year: Trading days per year

    Returns:
        Dict of metric_name -> value
    """
    if returns.empty:
        return _empty_metrics()

    returns = returns.dropna()
    if len(returns) < 2:
        return _empty_metrics()

    n = len(returns)
    daily_rf = risk_free_rate / periods_per_year

    # Basic return metrics
    total_return = (1 + returns).prod() - 1
    years = n / periods_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Risk metrics
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else 0

    # Ratios
    excess_return = cagr - risk_free_rate
    sharpe = excess_return / ann_vol if ann_vol > 0 else 0
    sortino = excess_return / downside_vol if downside_vol > 0 else 0

    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    max_dd = drawdown.min()

    # Drawdown duration
    dd_duration = _max_drawdown_duration(drawdown)

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win/loss metrics
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / n if n > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")

    # Distribution
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # VaR / CVaR
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95

    metrics: dict[str, float] = {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": dd_duration,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_day": returns.max(),
        "worst_day": returns.min(),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "n_trades": n,
    }

    # Benchmark-relative metrics
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(returns.index).dropna()
        common_idx = returns.index.intersection(bench.index)
        if len(common_idx) > 10:
            r = returns.loc[common_idx]
            b = bench.loc[common_idx]

            cov_matrix = np.cov(r, b)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0
            alpha = (r.mean() - daily_rf - beta * (b.mean() - daily_rf)) * periods_per_year

            tracking_error = (r - b).std() * np.sqrt(periods_per_year)
            info_ratio = (r.mean() - b.mean()) * periods_per_year / tracking_error if tracking_error > 0 else 0

            # Capture ratios
            up_periods = b > 0
            down_periods = b < 0
            up_capture = r[up_periods].mean() / b[up_periods].mean() if up_periods.sum() > 0 and b[up_periods].mean() != 0 else 0
            down_capture = r[down_periods].mean() / b[down_periods].mean() if down_periods.sum() > 0 and b[down_periods].mean() != 0 else 0

            metrics.update({
                "alpha": alpha,
                "beta": beta,
                "information_ratio": info_ratio,
                "tracking_error": tracking_error,
                "up_capture": up_capture,
                "down_capture": down_capture,
            })

    return metrics


def calculate_drawdown_series(returns: pd.Series) -> pd.DataFrame:
    """Calculate drawdown time series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1

    return pd.DataFrame({
        "cumulative_return": cumulative,
        "running_max": running_max,
        "drawdown": drawdown,
    }, index=returns.index)


def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """Calculate rolling performance metrics."""
    rolling_ret = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_ret / rolling_vol

    # Rolling max drawdown
    cumulative = (1 + returns).cumprod()

    def _rolling_dd(x: pd.Series) -> float:
        cum = (1 + x).cumprod()
        rm = cum.cummax()
        return ((cum / rm) - 1).min()

    rolling_dd = returns.rolling(window).apply(_rolling_dd, raw=False)

    return pd.DataFrame({
        "rolling_return": rolling_ret,
        "rolling_volatility": rolling_vol,
        "rolling_sharpe": rolling_sharpe,
        "rolling_max_drawdown": rolling_dd,
    }, index=returns.index)


def compare_strategies(
    strategy_returns: dict[str, pd.Series],
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Compare multiple strategies side by side."""
    rows = []
    for name, rets in strategy_returns.items():
        metrics = calculate_metrics(rets, benchmark_returns)
        metrics["strategy"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if "strategy" in df.columns:
        df = df.set_index("strategy")
    return df


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """Calculate maximum drawdown duration in trading days."""
    if drawdown.empty:
        return 0

    is_dd = drawdown < 0
    if not is_dd.any():
        return 0

    groups = (~is_dd).cumsum()
    dd_groups = is_dd.groupby(groups)
    max_dur = dd_groups.sum().max()
    return int(max_dur)


def _empty_metrics() -> dict[str, float]:
    """Return empty metrics dict."""
    return {
        "total_return": 0.0,
        "cagr": 0.0,
        "annualized_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_duration_days": 0,
        "calmar_ratio": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "best_day": 0.0,
        "worst_day": 0.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
        "var_95": 0.0,
        "cvar_95": 0.0,
        "n_trades": 0,
    }
