"""
Portfolio Risk Management

Risk metrics, beta hedging, and drawdown control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..common import get_logger

logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""

    volatility: float
    beta: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float


class RiskManager:
    """
    Portfolio risk management.

    Features:
        - Real-time risk monitoring
        - Beta hedging recommendations
        - Drawdown-based position scaling
        - VaR/CVaR calculations
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize risk manager.

        Args:
            config: Configuration with:
                - lookback_days: Days for risk calculation (default: 252)
                - target_volatility: Target annual volatility (default: 0.15)
                - max_drawdown_threshold: Drawdown for risk reduction (default: 0.1)
                - hedge_ticker: Ticker for beta hedging (default: "KODEX200")
        """
        self.config = config or {}
        self.lookback_days = self.config.get("lookback_days", 252)
        self.target_volatility = self.config.get("target_volatility", 0.15)
        self.max_drawdown_threshold = self.config.get("max_drawdown_threshold", 0.1)
        self.hedge_ticker = self.config.get("hedge_ticker", "KODEX200")

        self._returns_history: list[float] = []
        self._peak_value: float = 1.0
        self._current_value: float = 1.0

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.

        Args:
            returns: Portfolio daily returns
            benchmark_returns: Benchmark returns for beta calculation

        Returns:
            RiskMetrics object
        """
        if len(returns) < 20:
            return RiskMetrics(
                volatility=0.0,
                beta=1.0,
                var_95=0.0,
                cvar_95=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
            )

        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)

        # Beta
        if benchmark_returns is not None and len(benchmark_returns) >= 20:
            aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if len(aligned) >= 20:
                cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
                var_benchmark = aligned.iloc[:, 1].var()
                beta = cov / var_benchmark if var_benchmark > 0 else 1.0
            else:
                beta = 1.0
        else:
            beta = 1.0

        # VaR and CVaR (95%)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        # Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        current_drawdown = drawdowns.iloc[-1] if len(drawdowns) > 0 else 0.0

        # Sharpe ratio (assuming 0 risk-free rate)
        mean_return = returns.mean() * 252
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = mean_return / downside_vol if downside_vol > 0 else 0.0

        return RiskMetrics(
            volatility=volatility,
            beta=beta,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
        )

    def calculate_volatility_scalar(
        self,
        recent_returns: pd.Series,
    ) -> float:
        """
        Calculate position scalar for volatility targeting.

        Args:
            recent_returns: Recent portfolio returns

        Returns:
            Scalar to apply to positions (0 to 2)
        """
        if len(recent_returns) < 20:
            return 1.0

        realized_vol = recent_returns.std() * np.sqrt(252)

        if realized_vol <= 0:
            return 1.0

        # Target vol / realized vol, capped
        scalar = self.target_volatility / realized_vol
        scalar = np.clip(scalar, 0.5, 2.0)

        return scalar

    def calculate_drawdown_scalar(
        self,
        current_value: float,
        peak_value: float,
    ) -> float:
        """
        Calculate position scalar based on drawdown.

        Reduces exposure as drawdown increases.

        Args:
            current_value: Current portfolio value
            peak_value: Peak portfolio value

        Returns:
            Scalar to apply to positions (0 to 1)
        """
        if peak_value <= 0:
            return 1.0

        drawdown = (peak_value - current_value) / peak_value

        if drawdown <= 0:
            return 1.0

        if drawdown >= self.max_drawdown_threshold:
            # Linear reduction: at threshold -> 0.5, at 2x threshold -> 0
            reduction = 0.5 * (drawdown - self.max_drawdown_threshold) / self.max_drawdown_threshold
            scalar = max(0.5 - reduction, 0.0)
        else:
            scalar = 1.0

        return scalar

    def calculate_beta_hedge(
        self,
        weights: pd.Series,
        asset_betas: pd.Series,
        target_beta: float = 0.0,
    ) -> float:
        """
        Calculate hedge ratio for beta neutralization.

        Args:
            weights: Portfolio weights
            asset_betas: Beta of each asset
            target_beta: Target portfolio beta

        Returns:
            Hedge weight (negative = short hedge instrument)
        """
        # Align indices
        common_idx = weights.index.intersection(asset_betas.index)

        if len(common_idx) == 0:
            return 0.0

        # Portfolio beta
        portfolio_beta = (weights.loc[common_idx] * asset_betas.loc[common_idx]).sum()

        # Hedge to achieve target
        # portfolio_beta + hedge_weight * hedge_beta = target_beta
        # For inverse ETF with beta = -1: hedge_weight = (target_beta - portfolio_beta) / (-1)
        hedge_beta = -1.0  # Inverse ETF

        hedge_weight = (target_beta - portfolio_beta) / hedge_beta

        # Limit hedge size
        hedge_weight = np.clip(hedge_weight, 0.0, 0.5)  # Long-only, max 50% hedge

        return hedge_weight

    def get_hedge_recommendation(
        self,
        weights: pd.Series,
        asset_betas: pd.Series,
        market_regime: str | None = None,
    ) -> dict[str, Any]:
        """
        Get hedging recommendation.

        Args:
            weights: Current portfolio weights
            asset_betas: Asset betas
            market_regime: Current market regime

        Returns:
            Hedge recommendation
        """
        portfolio_beta = 0.0
        common_idx = weights.index.intersection(asset_betas.index)

        if len(common_idx) > 0:
            portfolio_beta = (weights.loc[common_idx] * asset_betas.loc[common_idx]).sum()

        # Regime-based target beta
        if market_regime == "high_volatility":
            target_beta = 0.3
        elif market_regime == "bear":
            target_beta = 0.0
        else:
            target_beta = 0.7

        hedge_weight = self.calculate_beta_hedge(weights, asset_betas, target_beta)

        return {
            "hedge_ticker": self.hedge_ticker,
            "hedge_weight": hedge_weight,
            "portfolio_beta": portfolio_beta,
            "target_beta": target_beta,
            "hedged_beta": portfolio_beta - hedge_weight,  # Inverse ETF
            "market_regime": market_regime,
        }

    def update_nav(self, daily_return: float) -> None:
        """
        Update NAV tracking for drawdown calculation.

        Args:
            daily_return: Today's portfolio return
        """
        self._returns_history.append(daily_return)
        self._current_value *= (1 + daily_return)

        if self._current_value > self._peak_value:
            self._peak_value = self._current_value

        # Keep limited history
        if len(self._returns_history) > self.lookback_days * 2:
            self._returns_history = self._returns_history[-self.lookback_days:]

    def get_current_drawdown(self) -> float:
        """Get current drawdown."""
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - self._current_value) / self._peak_value

    def reset_nav(self, initial_value: float = 1.0) -> None:
        """Reset NAV tracking."""
        self._returns_history = []
        self._peak_value = initial_value
        self._current_value = initial_value


class RiskBudgetAllocator:
    """
    Risk budget-based portfolio allocation.

    Allocates risk budget across strategies rather than capital.
    """

    def __init__(
        self,
        total_risk_budget: float = 0.15,
        strategy_risk_limits: dict[str, float] | None = None,
    ):
        """
        Initialize risk budget allocator.

        Args:
            total_risk_budget: Total portfolio volatility target
            strategy_risk_limits: Max vol contribution per strategy
        """
        self.total_risk_budget = total_risk_budget
        self.strategy_risk_limits = strategy_risk_limits or {}

    def allocate_by_risk_parity(
        self,
        strategy_volatilities: dict[str, float],
        correlations: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Allocate capital to achieve equal risk contribution.

        Args:
            strategy_volatilities: Volatility of each strategy
            correlations: Correlation matrix (optional)

        Returns:
            Capital weights per strategy
        """
        strategies = list(strategy_volatilities.keys())
        vols = np.array([strategy_volatilities[s] for s in strategies])

        if len(vols) == 0 or np.all(vols == 0):
            return {s: 1.0 / len(strategies) for s in strategies}

        # Simple inverse volatility for risk parity (ignores correlations)
        inv_vols = 1.0 / np.maximum(vols, 0.01)
        weights = inv_vols / inv_vols.sum()

        # Scale to target volatility
        portfolio_vol = np.sqrt(np.sum((weights * vols) ** 2))
        if portfolio_vol > 0:
            scale = self.total_risk_budget / portfolio_vol
            weights = weights * scale

        # Ensure weights sum to <= 1
        if weights.sum() > 1.0:
            weights = weights / weights.sum()

        return {s: w for s, w in zip(strategies, weights)}

    def allocate_by_risk_budget(
        self,
        strategy_volatilities: dict[str, float],
        risk_budgets: dict[str, float],
    ) -> dict[str, float]:
        """
        Allocate capital based on predefined risk budgets.

        Args:
            strategy_volatilities: Volatility of each strategy
            risk_budgets: Target risk contribution per strategy

        Returns:
            Capital weights per strategy
        """
        weights = {}

        for strategy, vol in strategy_volatilities.items():
            budget = risk_budgets.get(strategy, 1.0 / len(strategy_volatilities))

            if vol > 0:
                # Weight = budget / vol (to achieve target risk contribution)
                target_vol_contribution = budget * self.total_risk_budget
                weight = target_vol_contribution / vol
            else:
                weight = budget

            weights[strategy] = weight

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights
