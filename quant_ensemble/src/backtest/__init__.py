"""
Backtest Module

Backtesting engine, walk-forward optimization, and reporting.
"""

from .engine import BacktestConfig, BacktestEngine, BacktestResult, DailyResult, run_backtest
from .metrics import (
    calculate_drawdown_analysis,
    calculate_ic_analysis,
    calculate_max_drawdown,
    calculate_metrics,
    calculate_monthly_returns,
    calculate_regime_performance,
    calculate_rolling_metrics,
)
from .report import BacktestReport, compare_strategies, export_report, print_report
from .walkforward import (
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardOptimizer,
    WalkForwardResult,
    create_time_series_splits,
    run_walk_forward,
)

__all__ = [
    # Engine
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "DailyResult",
    "run_backtest",
    # Metrics
    "calculate_metrics",
    "calculate_rolling_metrics",
    "calculate_monthly_returns",
    "calculate_drawdown_analysis",
    "calculate_regime_performance",
    "calculate_max_drawdown",
    "calculate_ic_analysis",
    # Walk-forward
    "WalkForwardConfig",
    "WalkForwardOptimizer",
    "WalkForwardResult",
    "WalkForwardFold",
    "run_walk_forward",
    "create_time_series_splits",
    # Report
    "BacktestReport",
    "print_report",
    "export_report",
    "compare_strategies",
]
