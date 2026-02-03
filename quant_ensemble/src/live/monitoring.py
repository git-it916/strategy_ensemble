"""
Live Monitoring

Real-time monitoring and alerting for live trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..common import get_logger
from .broker import BrokerInterface, Position

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity level."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Monitoring alert."""

    timestamp: datetime
    level: AlertLevel
    category: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """System health status."""

    is_healthy: bool
    broker_connected: bool
    data_fresh: bool
    model_healthy: bool
    last_trade_time: datetime | None
    alerts: list[Alert]
    timestamp: datetime = field(default_factory=datetime.now)


class LiveMonitor:
    """
    Live trading monitoring system.

    Features:
        - Position monitoring
        - P&L tracking
        - Risk alerts
        - System health checks
    """

    def __init__(
        self,
        broker: BrokerInterface,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize monitor.

        Args:
            broker: Broker interface
            config: Monitoring configuration
        """
        self.broker = broker
        self.config = config or {}

        # Thresholds
        self.max_drawdown_alert = self.config.get("max_drawdown_alert", 0.05)
        self.max_position_size = self.config.get("max_position_size", 0.15)
        self.max_daily_loss = self.config.get("max_daily_loss", 0.03)
        self.stale_data_minutes = self.config.get("stale_data_minutes", 5)

        # State
        self._alerts: list[Alert] = []
        self._pnl_history: list[dict[str, Any]] = []
        self._peak_value: float = 0
        self._day_start_value: float = 0
        self._last_data_time: datetime | None = None
        self._alert_callbacks: list[Callable[[Alert], None]] = []

    def register_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register callback for alerts."""
        self._alert_callbacks.append(callback)

    def _raise_alert(
        self,
        level: AlertLevel,
        category: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Raise an alert."""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            data=data or {},
        )

        self._alerts.append(alert)
        logger.log(
            40 if level in [AlertLevel.ERROR, AlertLevel.CRITICAL] else 30,
            f"[{level.value.upper()}] {category}: {message}",
        )

        # Call registered callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def update(self, current_prices: dict[str, float] | None = None) -> HealthStatus:
        """
        Run monitoring update cycle.

        Args:
            current_prices: Current prices (optional)

        Returns:
            Current health status
        """
        alerts = []

        # Check broker connection
        broker_connected = self.broker.is_connected()
        if not broker_connected:
            self._raise_alert(
                AlertLevel.ERROR,
                "connection",
                "Broker disconnected",
            )

        # Get account info and positions
        try:
            account = self.broker.get_account_info()
            positions = self.broker.get_positions()
        except Exception as e:
            self._raise_alert(
                AlertLevel.ERROR,
                "data",
                f"Failed to get account data: {e}",
            )
            return HealthStatus(
                is_healthy=False,
                broker_connected=broker_connected,
                data_fresh=False,
                model_healthy=True,
                last_trade_time=None,
                alerts=self._alerts[-10:],
            )

        portfolio_value = account.portfolio_value

        # Initialize tracking
        if self._peak_value == 0:
            self._peak_value = portfolio_value
        if self._day_start_value == 0:
            self._day_start_value = portfolio_value

        # Update peak
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        # Check drawdown
        if self._peak_value > 0:
            drawdown = (self._peak_value - portfolio_value) / self._peak_value
            if drawdown > self.max_drawdown_alert:
                self._raise_alert(
                    AlertLevel.WARNING,
                    "risk",
                    f"Drawdown {drawdown:.2%} exceeds threshold {self.max_drawdown_alert:.2%}",
                    {"drawdown": drawdown, "peak_value": self._peak_value},
                )

        # Check daily loss
        if self._day_start_value > 0:
            daily_loss = (self._day_start_value - portfolio_value) / self._day_start_value
            if daily_loss > self.max_daily_loss:
                self._raise_alert(
                    AlertLevel.CRITICAL,
                    "risk",
                    f"Daily loss {daily_loss:.2%} exceeds threshold {self.max_daily_loss:.2%}",
                    {"daily_loss": daily_loss},
                )

        # Check position sizes
        for asset_id, pos in positions.items():
            if portfolio_value > 0:
                position_pct = abs(pos.market_value) / portfolio_value
                if position_pct > self.max_position_size:
                    self._raise_alert(
                        AlertLevel.WARNING,
                        "position",
                        f"{asset_id} position {position_pct:.2%} exceeds limit {self.max_position_size:.2%}",
                        {"asset_id": asset_id, "position_pct": position_pct},
                    )

        # Check data freshness
        data_fresh = True
        if self._last_data_time is not None:
            stale_threshold = datetime.now() - timedelta(minutes=self.stale_data_minutes)
            if self._last_data_time < stale_threshold:
                data_fresh = False
                self._raise_alert(
                    AlertLevel.WARNING,
                    "data",
                    f"Data stale: last update {self._last_data_time}",
                )

        self._last_data_time = datetime.now()

        # Record P&L
        self._pnl_history.append({
            "timestamp": datetime.now(),
            "portfolio_value": portfolio_value,
            "drawdown": (self._peak_value - portfolio_value) / self._peak_value if self._peak_value > 0 else 0,
        })

        # Keep limited history
        if len(self._pnl_history) > 10000:
            self._pnl_history = self._pnl_history[-5000:]

        is_healthy = broker_connected and data_fresh

        return HealthStatus(
            is_healthy=is_healthy,
            broker_connected=broker_connected,
            data_fresh=data_fresh,
            model_healthy=True,
            last_trade_time=None,
            alerts=self._alerts[-10:],
        )

    def get_pnl_summary(self) -> dict[str, Any]:
        """Get P&L summary."""
        if not self._pnl_history:
            return {}

        values = [p["portfolio_value"] for p in self._pnl_history]

        return {
            "current_value": values[-1],
            "peak_value": self._peak_value,
            "day_start_value": self._day_start_value,
            "current_drawdown": (self._peak_value - values[-1]) / self._peak_value if self._peak_value > 0 else 0,
            "daily_pnl": values[-1] - self._day_start_value,
            "daily_return": (values[-1] - self._day_start_value) / self._day_start_value if self._day_start_value > 0 else 0,
            "n_records": len(self._pnl_history),
        }

    def get_position_summary(self) -> pd.DataFrame:
        """Get position summary."""
        positions = self.broker.get_positions()
        account = self.broker.get_account_info()

        rows = []
        for asset_id, pos in positions.items():
            pct = pos.market_value / account.portfolio_value if account.portfolio_value > 0 else 0
            rows.append({
                "asset_id": asset_id,
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "weight": pct,
            })

        return pd.DataFrame(rows)

    def reset_daily(self) -> None:
        """Reset daily tracking (call at market open)."""
        try:
            account = self.broker.get_account_info()
            self._day_start_value = account.portfolio_value
            logger.info(f"Reset daily tracking. Start value: {self._day_start_value:,.0f}")
        except Exception as e:
            logger.error(f"Failed to reset daily tracking: {e}")

    def get_alerts(
        self,
        since: datetime | None = None,
        level: AlertLevel | None = None,
    ) -> list[Alert]:
        """
        Get alerts.

        Args:
            since: Only alerts after this time
            level: Filter by level

        Returns:
            List of alerts
        """
        alerts = self._alerts

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        if level:
            alerts = [a for a in alerts if a.level == level]

        return alerts

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts = []


class RiskMonitor:
    """
    Real-time risk monitoring.

    Monitors:
        - Portfolio VaR
        - Sector exposure
        - Factor exposure
        - Correlation breakdown
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize risk monitor.

        Args:
            config: Risk monitoring configuration
        """
        self.config = config or {}

        self.var_confidence = self.config.get("var_confidence", 0.95)
        self.var_horizon_days = self.config.get("var_horizon_days", 1)
        self.max_sector_exposure = self.config.get("max_sector_exposure", 0.3)

        self._returns_history: list[float] = []

    def update_returns(self, daily_return: float) -> None:
        """Update returns history."""
        self._returns_history.append(daily_return)

        if len(self._returns_history) > 252:
            self._returns_history = self._returns_history[-252:]

    def calculate_var(self) -> dict[str, float]:
        """Calculate Value at Risk."""
        if len(self._returns_history) < 20:
            return {"var": 0, "cvar": 0}

        returns = np.array(self._returns_history)

        # Historical VaR
        var = np.percentile(returns, (1 - self.var_confidence) * 100)

        # CVaR (Expected Shortfall)
        cvar = returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var

        # Scale to horizon
        var_scaled = var * np.sqrt(self.var_horizon_days)
        cvar_scaled = cvar * np.sqrt(self.var_horizon_days)

        return {
            "var": var_scaled,
            "cvar": cvar_scaled,
            "confidence": self.var_confidence,
            "horizon_days": self.var_horizon_days,
        }

    def check_sector_exposure(
        self,
        positions: dict[str, Position],
        portfolio_value: float,
        sector_map: dict[str, str],
    ) -> dict[str, Any]:
        """
        Check sector exposure limits.

        Args:
            positions: Current positions
            portfolio_value: Portfolio value
            sector_map: Asset to sector mapping

        Returns:
            Sector exposure analysis
        """
        sector_values: dict[str, float] = {}

        for asset_id, pos in positions.items():
            sector = sector_map.get(asset_id, "Unknown")
            sector_values[sector] = sector_values.get(sector, 0) + pos.market_value

        sector_weights = {
            k: v / portfolio_value if portfolio_value > 0 else 0
            for k, v in sector_values.items()
        }

        breaches = {
            k: v for k, v in sector_weights.items()
            if v > self.max_sector_exposure
        }

        return {
            "sector_weights": sector_weights,
            "breaches": breaches,
            "max_sector": max(sector_weights.values()) if sector_weights else 0,
            "is_compliant": len(breaches) == 0,
        }
