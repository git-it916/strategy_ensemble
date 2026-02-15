"""
Backtesting Module

Historical simulation and performance analysis.
"""

from .engine import BacktestEngine, BacktestResult
from .metrics import calculate_metrics

__all__ = ["BacktestEngine", "BacktestResult", "calculate_metrics"]
