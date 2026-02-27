"""
Performance Tracker

Per-alpha daily PnL tracking with JSONL persistence.
Supports consecutive loss day calculation and rolling Sharpe.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.openclaw.config import PERFORMANCE_DIR

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Track daily performance of each alpha.

    Stores data in JSONL files (one per alpha) and provides
    computed metrics for lifecycle decisions.
    """

    def __init__(self, tracker_dir: Path | None = None):
        self._dir = tracker_dir or PERFORMANCE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[dict]] = defaultdict(list)
        self._load_all()

    def _alpha_path(self, alpha_name: str) -> Path:
        return self._dir / f"{alpha_name}.jsonl"

    def _load_all(self) -> None:
        """Load all existing performance files into cache."""
        for path in self._dir.glob("*.jsonl"):
            alpha_name = path.stem
            try:
                with open(path) as f:
                    records = [json.loads(line) for line in f if line.strip()]
                self._cache[alpha_name] = records
            except Exception as e:
                logger.error(f"Failed to load performance for {alpha_name}: {e}")

    def record_daily(
        self,
        alpha_name: str,
        date: datetime,
        pnl: float,
        positions: dict[str, Any] | None = None,
    ) -> None:
        """
        Record daily PnL for an alpha.

        Args:
            alpha_name: Alpha identifier
            date: Trading date
            pnl: Daily return (as decimal, e.g. 0.01 = 1%)
            positions: Optional position snapshot
        """
        record = {
            "date": date.isoformat() if isinstance(date, datetime) else str(date),
            "pnl": pnl,
            "positions": positions or {},
            "recorded_at": datetime.now().isoformat(),
        }

        self._cache[alpha_name].append(record)

        path = self._alpha_path(alpha_name)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_daily_returns(
        self,
        alpha_name: str,
        lookback: int | None = None,
    ) -> pd.Series:
        """Get daily returns as a pandas Series."""
        records = self._cache.get(alpha_name, [])
        if not records:
            return pd.Series(dtype=float)

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")["pnl"]

        if lookback:
            df = df.tail(lookback)

        return df

    def get_consecutive_loss_days(self, alpha_name: str) -> int:
        """
        Count consecutive loss days from the most recent date.

        Returns:
            Number of consecutive days with negative PnL.
        """
        records = self._cache.get(alpha_name, [])
        if not records:
            return 0

        # Sort by date descending
        sorted_records = sorted(records, key=lambda r: r["date"], reverse=True)

        count = 0
        for r in sorted_records:
            if r["pnl"] < 0:
                count += 1
            else:
                break

        return count

    def get_rolling_sharpe(
        self,
        alpha_name: str,
        window: int = 63,
    ) -> float:
        """
        Calculate rolling Sharpe ratio over the most recent window.

        Args:
            alpha_name: Alpha identifier
            window: Rolling window (default ~3 months)

        Returns:
            Annualized Sharpe ratio (0.0 if insufficient data).
        """
        returns = self.get_daily_returns(alpha_name, lookback=window)

        if len(returns) < 10:
            return 0.0

        mean_ret = returns.mean()
        std_ret = returns.std()

        if std_ret == 0 or np.isnan(std_ret):
            return 0.0

        return float(mean_ret / std_ret * np.sqrt(252))

    def get_rolling_mdd(self, alpha_name: str, window: int = 252) -> float:
        """
        Calculate maximum drawdown over the most recent window.

        Returns:
            Maximum drawdown as negative float (e.g. -0.15 = -15%).
        """
        returns = self.get_daily_returns(alpha_name, lookback=window)

        if len(returns) < 2:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        return float(drawdown.min())

    def get_summary(self, alpha_name: str) -> dict[str, Any]:
        """Get comprehensive performance summary for an alpha."""
        returns = self.get_daily_returns(alpha_name)

        if returns.empty:
            return {
                "alpha_name": alpha_name,
                "n_days": 0,
                "total_return": 0.0,
                "sharpe": 0.0,
                "mdd": 0.0,
                "win_rate": 0.0,
                "consecutive_loss_days": 0,
            }

        n_days = len(returns)
        total_return = float((1 + returns).prod() - 1)
        sharpe = self.get_rolling_sharpe(alpha_name, window=n_days)
        mdd = self.get_rolling_mdd(alpha_name, window=n_days)
        win_rate = float((returns > 0).mean())
        consec_loss = self.get_consecutive_loss_days(alpha_name)

        return {
            "alpha_name": alpha_name,
            "n_days": n_days,
            "total_return": total_return,
            "sharpe": sharpe,
            "mdd": mdd,
            "win_rate": win_rate,
            "consecutive_loss_days": consec_loss,
            "avg_daily_return": float(returns.mean()),
            "daily_vol": float(returns.std()),
            "best_day": float(returns.max()),
            "worst_day": float(returns.min()),
        }

    def get_all_summaries(self) -> dict[str, dict[str, Any]]:
        """Get summaries for all tracked alphas."""
        return {
            name: self.get_summary(name)
            for name in self._cache
        }

    def get_correlation_matrix(
        self,
        alpha_names: list[str] | None = None,
        lookback: int = 63,
    ) -> pd.DataFrame:
        """Calculate return correlation matrix between alphas."""
        names = alpha_names or list(self._cache.keys())
        returns_dict = {}

        for name in names:
            ret = self.get_daily_returns(name, lookback=lookback)
            if len(ret) >= 10:
                returns_dict[name] = ret

        if not returns_dict:
            return pd.DataFrame()

        return pd.DataFrame(returns_dict).corr()

    def clear(self, alpha_name: str) -> None:
        """Clear performance data for an alpha."""
        self._cache.pop(alpha_name, None)
        path = self._alpha_path(alpha_name)
        if path.exists():
            path.unlink()
            logger.info(f"Cleared performance data for {alpha_name}")
