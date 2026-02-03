"""
Performance Metrics

Calculate comprehensive backtest performance metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..common import get_logger

logger = get_logger(__name__)


def calculate_metrics(
    returns: pd.Series,
    initial_capital: float = 100_000_000,
    risk_free_rate: float = 0.03,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Daily returns series
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate
        benchmark_returns: Benchmark returns for relative metrics

    Returns:
        Dictionary of metrics
    """
    if returns.empty or len(returns) < 2:
        return {}

    metrics = {}

    # Basic statistics
    n_days = len(returns)
    n_years = n_days / 252

    # Returns
    cumulative_return = (1 + returns).prod() - 1
    total_return = cumulative_return
    annual_return = (1 + cumulative_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    metrics["total_return"] = total_return
    metrics["annual_return"] = annual_return
    metrics["n_days"] = n_days

    # Volatility
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    metrics["daily_volatility"] = daily_vol
    metrics["annual_volatility"] = annual_vol

    # Sharpe Ratio
    excess_return = annual_return - risk_free_rate
    sharpe = excess_return / annual_vol if annual_vol > 0 else 0
    metrics["sharpe_ratio"] = sharpe

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_vol
    sortino = excess_return / downside_vol if downside_vol > 0 else 0
    metrics["sortino_ratio"] = sortino

    # Calmar Ratio
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min())
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
    metrics["calmar_ratio"] = calmar

    # Drawdown metrics
    metrics["max_drawdown"] = max_drawdown
    metrics["avg_drawdown"] = abs(drawdowns[drawdowns < 0].mean()) if len(drawdowns[drawdowns < 0]) > 0 else 0

    # Drawdown duration
    is_drawdown = drawdowns < 0
    drawdown_periods = []
    current_period = 0
    for dd in is_drawdown:
        if dd:
            current_period += 1
        else:
            if current_period > 0:
                drawdown_periods.append(current_period)
            current_period = 0
    if current_period > 0:
        drawdown_periods.append(current_period)

    metrics["max_drawdown_duration"] = max(drawdown_periods) if drawdown_periods else 0
    metrics["avg_drawdown_duration"] = np.mean(drawdown_periods) if drawdown_periods else 0

    # Win rate
    positive_days = (returns > 0).sum()
    win_rate = positive_days / n_days
    metrics["win_rate"] = win_rate

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    metrics["profit_factor"] = profit_factor

    # Average win/loss
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
    metrics["avg_win"] = avg_win
    metrics["avg_loss"] = avg_loss
    metrics["win_loss_ratio"] = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # Tail metrics
    metrics["var_95"] = np.percentile(returns, 5)
    metrics["cvar_95"] = returns[returns <= np.percentile(returns, 5)].mean() if len(returns) > 20 else 0
    metrics["skewness"] = returns.skew()
    metrics["kurtosis"] = returns.kurtosis()

    # Final value
    final_value = initial_capital * (1 + cumulative_return)
    metrics["final_value"] = final_value
    metrics["profit_loss"] = final_value - initial_capital

    # Benchmark comparison
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) > 20:
            aligned.columns = ["strategy", "benchmark"]

            # Beta
            cov = aligned["strategy"].cov(aligned["benchmark"])
            var_bench = aligned["benchmark"].var()
            beta = cov / var_bench if var_bench > 0 else 1.0
            metrics["beta"] = beta

            # Alpha (annualized)
            bench_return = (1 + aligned["benchmark"]).prod() ** (252 / len(aligned)) - 1
            alpha = annual_return - (risk_free_rate + beta * (bench_return - risk_free_rate))
            metrics["alpha"] = alpha

            # Information Ratio
            excess = aligned["strategy"] - aligned["benchmark"]
            tracking_error = excess.std() * np.sqrt(252)
            info_ratio = excess.mean() * 252 / tracking_error if tracking_error > 0 else 0
            metrics["information_ratio"] = info_ratio
            metrics["tracking_error"] = tracking_error

            # Correlation
            metrics["benchmark_correlation"] = aligned["strategy"].corr(aligned["benchmark"])

    return metrics


def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.

    Args:
        returns: Daily returns
        window: Rolling window size

    Returns:
        DataFrame with rolling metrics
    """
    if len(returns) < window:
        return pd.DataFrame()

    rolling_return = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / rolling_vol

    # Rolling drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(window, min_periods=1).max()
    rolling_dd = (cumulative - rolling_max) / rolling_max

    return pd.DataFrame({
        "rolling_return": rolling_return,
        "rolling_volatility": rolling_vol,
        "rolling_sharpe": rolling_sharpe,
        "rolling_drawdown": rolling_dd,
    })


def calculate_monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns table.

    Args:
        returns: Daily returns series

    Returns:
        DataFrame with monthly returns pivoted by year and month
    """
    if returns.empty:
        return pd.DataFrame()

    # Ensure datetime index
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)

    # Calculate monthly returns
    monthly = returns.groupby([returns.index.year, returns.index.month]).apply(
        lambda x: (1 + x).prod() - 1
    )
    monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=["Year", "Month"])

    # Pivot
    monthly_df = monthly.unstack(level=1)
    monthly_df.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ][:len(monthly_df.columns)]

    # Add yearly total
    yearly = returns.groupby(returns.index.year).apply(lambda x: (1 + x).prod() - 1)
    monthly_df["Year Total"] = yearly

    return monthly_df


def calculate_drawdown_analysis(returns: pd.Series) -> pd.DataFrame:
    """
    Detailed drawdown analysis.

    Args:
        returns: Daily returns series

    Returns:
        DataFrame with top drawdowns
    """
    if len(returns) < 10:
        return pd.DataFrame()

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max

    # Find drawdown periods
    in_drawdown = drawdowns < 0

    drawdown_events = []
    start_date = None
    peak_value = None

    for date, (is_dd, dd_value, cum_value, rm_value) in enumerate(
        zip(in_drawdown, drawdowns, cumulative, rolling_max)
    ):
        idx = drawdowns.index[date]

        if is_dd and start_date is None:
            start_date = idx
            peak_value = rm_value

        elif not is_dd and start_date is not None:
            # End of drawdown
            period_dd = drawdowns[start_date:idx]
            trough_date = period_dd.idxmin()
            trough_value = period_dd.min()

            drawdown_events.append({
                "start_date": start_date,
                "trough_date": trough_date,
                "recovery_date": idx,
                "depth": abs(trough_value),
                "duration_to_trough": (trough_date - start_date).days,
                "duration_to_recovery": (idx - start_date).days,
            })

            start_date = None

    # Handle ongoing drawdown
    if start_date is not None:
        period_dd = drawdowns[start_date:]
        trough_date = period_dd.idxmin()
        trough_value = period_dd.min()

        drawdown_events.append({
            "start_date": start_date,
            "trough_date": trough_date,
            "recovery_date": None,
            "depth": abs(trough_value),
            "duration_to_trough": (trough_date - start_date).days,
            "duration_to_recovery": None,
        })

    if not drawdown_events:
        return pd.DataFrame()

    df = pd.DataFrame(drawdown_events)
    df = df.sort_values("depth", ascending=False).head(10).reset_index(drop=True)

    return df


def calculate_regime_performance(
    returns: pd.Series,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """
    Calculate performance by market regime.

    Args:
        returns: Daily returns
        regime_labels: Regime classification per date

    Returns:
        DataFrame with per-regime metrics
    """
    aligned = pd.concat([returns, regime_labels], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame()

    aligned.columns = ["return", "regime"]

    regime_metrics = []

    for regime in aligned["regime"].unique():
        regime_returns = aligned[aligned["regime"] == regime]["return"]

        if len(regime_returns) < 10:
            continue

        annual_return = (1 + regime_returns.mean()) ** 252 - 1
        annual_vol = regime_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        regime_metrics.append({
            "regime": regime,
            "n_days": len(regime_returns),
            "pct_time": len(regime_returns) / len(aligned),
            "mean_return": regime_returns.mean(),
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "win_rate": (regime_returns > 0).mean(),
            "max_drawdown": calculate_max_drawdown(regime_returns),
        })

    return pd.DataFrame(regime_metrics)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    if len(returns) < 2:
        return 0.0

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max

    return abs(drawdowns.min())


def calculate_ic_analysis(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate Information Coefficient analysis.

    Args:
        predictions: Predictions with date, asset_id, score
        returns: Returns with date, asset_id, return

    Returns:
        IC analysis results
    """
    merged = predictions.merge(
        returns,
        on=["date", "asset_id"],
        how="inner",
    )

    if merged.empty:
        return {}

    # Daily IC
    daily_ic = merged.groupby("date").apply(
        lambda x: x["score"].corr(x["return"])
    )

    # Rank IC
    daily_rank_ic = merged.groupby("date").apply(
        lambda x: x["score"].rank().corr(x["return"].rank())
    )

    return {
        "mean_ic": daily_ic.mean(),
        "std_ic": daily_ic.std(),
        "ic_ir": daily_ic.mean() / daily_ic.std() if daily_ic.std() > 0 else 0,
        "pct_positive_ic": (daily_ic > 0).mean(),
        "mean_rank_ic": daily_rank_ic.mean(),
        "std_rank_ic": daily_rank_ic.std(),
        "rank_ic_ir": daily_rank_ic.mean() / daily_rank_ic.std() if daily_rank_ic.std() > 0 else 0,
    }
