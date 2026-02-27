"""
Single Alpha Backtest Runner

Wraps BacktestEngine to validate a single generated alpha.
Supports dynamic code loading from string and IS/OOS splitting.
"""

from __future__ import annotations

import logging
import types
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.alphas.base_alpha import BaseAlpha
from src.backtest.engine import BacktestEngine, BacktestResult
from src.ensemble.agent import EnsembleAgent
from src.ensemble.allocator import RiskParityAllocator
from src.openclaw.config import QUALITY_GATES, QualityGates

logger = logging.getLogger(__name__)


class SingleAlphaBacktestRunner:
    """
    Run backtest on a single alpha (loaded dynamically from code string).

    Wraps the existing BacktestEngine with IS/OOS splitting
    and standardized result format.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        backtest_config: dict[str, Any] | None = None,
    ):
        """
        Args:
            prices: Historical prices (date, ticker, open, high, low, close, volume)
            features: Historical features (date, ticker, ...)
            backtest_config: Override backtest params (initial_capital, commission, etc.)
        """
        self.prices = prices
        self.features = features

        config = backtest_config or {}
        self.engine = BacktestEngine(
            initial_capital=config.get("initial_capital", 10_000),
            commission_bps=config.get("commission_bps", 4.0),
            tax_bps=config.get("tax_bps", 0.0),  # Crypto: no tax
            slippage_bps=config.get("slippage_bps", 5.0),
            max_position_weight=config.get("max_position_weight", 0.5),
            rebalance_frequency=config.get("rebalance_frequency", "daily"),
        )

    def run(
        self,
        alpha_code: str,
        class_name: str,
        oos_split_ratio: float = 0.3,
    ) -> dict[str, Any]:
        """
        Run IS/OOS backtest on dynamically loaded alpha.

        Args:
            alpha_code: Python source code of the alpha class
            class_name: Name of the class to instantiate
            oos_split_ratio: Fraction of data for out-of-sample (default 30%)

        Returns:
            Dict with is_result, oos_result, metrics, daily_returns, turnover.
        """
        # 1. Dynamically load the alpha
        alpha = self._dynamic_load(alpha_code, class_name)

        # 2. Determine date split
        dates = sorted(self.prices["date"].unique())
        if len(dates) < 100:
            raise ValueError(
                f"Insufficient data: {len(dates)} dates (need >= 100)"
            )

        split_idx = int(len(dates) * (1 - oos_split_ratio))
        is_end = str(dates[split_idx - 1])[:10]
        oos_start = str(dates[split_idx])[:10]
        full_start = str(dates[0])[:10]
        full_end = str(dates[-1])[:10]

        logger.info(
            f"Backtest split: IS [{full_start} → {is_end}], "
            f"OOS [{oos_start} → {full_end}]"
        )

        # 3. Fit alpha on IS data
        is_prices = self.prices[self.prices["date"] <= pd.Timestamp(is_end)]
        is_features = None
        if self.features is not None and "date" in self.features.columns:
            is_features = self.features[
                self.features["date"] <= pd.Timestamp(is_end)
            ]

        alpha.fit(is_prices, is_features)

        # 4. Create single-alpha ensemble
        ensemble = EnsembleAgent(
            strategies=[alpha],
            config={"use_dynamic_weights": False},
        )

        allocator = RiskParityAllocator(
            top_k=len(self.prices["ticker"].unique()),
            vol_lookback=60,
            max_weight=0.5,
        )

        # 5. Run IS backtest
        is_result = self.engine.run(
            ensemble=ensemble,
            allocator=allocator,
            prices=self.prices,
            features=self.features,
            start_date=full_start,
            end_date=is_end,
        )

        # 6. Run OOS backtest (using IS-fitted model, no refit)
        oos_result = self.engine.run(
            ensemble=ensemble,
            allocator=allocator,
            prices=self.prices,
            features=self.features,
            start_date=oos_start,
            end_date=full_end,
        )

        # 7. Compute turnover
        turnover = self._compute_avg_turnover(is_result.holdings_history)

        return {
            "is_result": is_result,
            "oos_result": oos_result,
            "is_metrics": is_result.metrics,
            "oos_metrics": oos_result.metrics,
            "daily_returns": pd.concat(
                [is_result.returns, oos_result.returns]
            ).sort_index(),
            "oos_daily_returns": oos_result.returns,
            "turnover": turnover,
            "alpha_name": alpha.name,
            "split_date": is_end,
        }

    def run_with_instance(
        self,
        alpha: BaseAlpha,
        oos_split_ratio: float = 0.3,
    ) -> dict[str, Any]:
        """
        Run IS/OOS backtest with an already-instantiated alpha.

        Useful for testing existing alphas or mutated variants.
        """
        dates = sorted(self.prices["date"].unique())
        if len(dates) < 100:
            raise ValueError(
                f"Insufficient data: {len(dates)} dates (need >= 100)"
            )

        split_idx = int(len(dates) * (1 - oos_split_ratio))
        is_end = str(dates[split_idx - 1])[:10]
        oos_start = str(dates[split_idx])[:10]
        full_start = str(dates[0])[:10]
        full_end = str(dates[-1])[:10]

        # Fit on IS data
        is_prices = self.prices[self.prices["date"] <= pd.Timestamp(is_end)]
        is_features = None
        if self.features is not None and "date" in self.features.columns:
            is_features = self.features[
                self.features["date"] <= pd.Timestamp(is_end)
            ]

        alpha.fit(is_prices, is_features)

        ensemble = EnsembleAgent(
            strategies=[alpha],
            config={"use_dynamic_weights": False},
        )

        allocator = RiskParityAllocator(
            top_k=len(self.prices["ticker"].unique()),
            vol_lookback=60,
            max_weight=0.5,
        )

        is_result = self.engine.run(
            ensemble=ensemble, allocator=allocator,
            prices=self.prices, features=self.features,
            start_date=full_start, end_date=is_end,
        )

        oos_result = self.engine.run(
            ensemble=ensemble, allocator=allocator,
            prices=self.prices, features=self.features,
            start_date=oos_start, end_date=full_end,
        )

        turnover = self._compute_avg_turnover(is_result.holdings_history)

        return {
            "is_result": is_result,
            "oos_result": oos_result,
            "is_metrics": is_result.metrics,
            "oos_metrics": oos_result.metrics,
            "daily_returns": pd.concat(
                [is_result.returns, oos_result.returns]
            ).sort_index(),
            "oos_daily_returns": oos_result.returns,
            "turnover": turnover,
            "alpha_name": alpha.name,
            "split_date": is_end,
        }

    def _dynamic_load(self, code: str, class_name: str) -> BaseAlpha:
        """
        Dynamically load an alpha class from source code.

        Creates a temporary module, executes the code in it,
        and returns an instance of the specified class.
        """
        # Create a temporary module
        module = types.ModuleType(f"openclaw_generated_{class_name}")

        # Add required imports to the module namespace
        import numpy as np
        import pandas as pd
        from src.alphas.base_alpha import AlphaResult, BaseAlpha

        module.__dict__["np"] = np
        module.__dict__["numpy"] = np
        module.__dict__["pd"] = pd
        module.__dict__["pandas"] = pd
        module.__dict__["BaseAlpha"] = BaseAlpha
        module.__dict__["AlphaResult"] = AlphaResult
        module.__dict__["datetime"] = datetime
        module.__dict__["Any"] = Any

        try:
            exec(code, module.__dict__)
        except Exception as e:
            raise RuntimeError(f"Failed to execute alpha code: {e}") from e

        if class_name not in module.__dict__:
            available = [
                k for k, v in module.__dict__.items()
                if isinstance(v, type) and issubclass(v, BaseAlpha)
                and v is not BaseAlpha
            ]
            raise ValueError(
                f"Class '{class_name}' not found in generated code. "
                f"Available BaseAlpha subclasses: {available}"
            )

        alpha_cls = module.__dict__[class_name]
        if not issubclass(alpha_cls, BaseAlpha):
            raise TypeError(
                f"'{class_name}' does not subclass BaseAlpha"
            )

        return alpha_cls()

    @staticmethod
    def _compute_avg_turnover(holdings_history: list[dict]) -> float:
        """
        Compute average daily turnover from holdings history.

        Turnover = average of sum(|weight_change|) across rebalance dates.
        """
        if len(holdings_history) < 2:
            return 0.0

        turnovers = []
        for i in range(1, len(holdings_history)):
            prev_weights = holdings_history[i - 1].get("weights", {})
            curr_weights = holdings_history[i].get("weights", {})

            all_tickers = set(prev_weights) | set(curr_weights)
            turnover = sum(
                abs(curr_weights.get(t, 0) - prev_weights.get(t, 0))
                for t in all_tickers
            )
            turnovers.append(turnover)

        return float(np.mean(turnovers)) if turnovers else 0.0
