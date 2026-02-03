"""
Order Execution

Intelligent order execution and trade management.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..common import Order, Trade, get_logger
from .broker import BrokerInterface, OrderStatus, Position

logger = get_logger(__name__)


@dataclass
class ExecutionPlan:
    """Execution plan for rebalancing."""

    target_weights: pd.Series
    current_weights: pd.Series
    orders: list[Order]
    estimated_cost: float
    estimated_turnover: float


@dataclass
class ExecutionResult:
    """Result of execution."""

    trades: list[Trade]
    total_cost: float
    slippage: float
    fill_rate: float
    execution_time: float


class ExecutionEngine:
    """
    Order execution engine.

    Features:
        - Smart order routing
        - TWAP/VWAP execution (simplified)
        - Cost-aware trading
        - Position reconciliation
    """

    def __init__(
        self,
        broker: BrokerInterface,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize execution engine.

        Args:
            broker: Broker interface
            config: Execution configuration
        """
        self.broker = broker
        self.config = config or {}

        # Execution parameters
        self.min_trade_value = self.config.get("min_trade_value", 100_000)  # 10만원
        self.max_order_pct_adv = self.config.get("max_order_pct_adv", 0.05)  # 5% ADV
        self.use_limit_orders = self.config.get("use_limit_orders", False)
        self.limit_offset_bps = self.config.get("limit_offset_bps", 5)

    def create_execution_plan(
        self,
        target_weights: pd.Series,
        portfolio_value: float,
        current_positions: dict[str, Position] | None = None,
        advs: pd.Series | None = None,
    ) -> ExecutionPlan:
        """
        Create execution plan for rebalancing.

        Args:
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            current_positions: Current positions from broker
            advs: Average daily volumes

        Returns:
            ExecutionPlan
        """
        if current_positions is None:
            current_positions = self.broker.get_positions()

        # Calculate current weights
        current_weights = {}
        for asset_id, pos in current_positions.items():
            if portfolio_value > 0:
                current_weights[asset_id] = pos.market_value / portfolio_value

        current_weights = pd.Series(current_weights)

        # Calculate trades needed
        all_assets = set(target_weights.index) | set(current_weights.index)
        orders = []
        total_turnover = 0

        for asset_id in all_assets:
            target_wt = target_weights.get(asset_id, 0)
            current_wt = current_weights.get(asset_id, 0)
            weight_diff = target_wt - current_wt

            trade_value = abs(weight_diff) * portfolio_value
            total_turnover += trade_value

            if trade_value < self.min_trade_value:
                continue

            # Check ADV constraint
            if advs is not None and asset_id in advs.index:
                adv = advs[asset_id]
                if trade_value > adv * self.max_order_pct_adv:
                    logger.warning(
                        f"{asset_id}: Trade {trade_value:,.0f} exceeds "
                        f"{self.max_order_pct_adv:.0%} ADV ({adv:,.0f})"
                    )

            # Get current quote
            quote = self.broker.get_quote(asset_id)
            last_price = quote.get("last", 10000)

            if last_price <= 0:
                continue

            quantity = int(abs(trade_value) / last_price)
            if quantity == 0:
                continue

            side = "BUY" if weight_diff > 0 else "SELL"

            # Determine order price
            if self.use_limit_orders:
                if side == "BUY":
                    price = quote.get("ask", last_price) * (1 + self.limit_offset_bps / 10000)
                else:
                    price = quote.get("bid", last_price) * (1 - self.limit_offset_bps / 10000)
            else:
                price = None  # Market order

            order = Order(
                asset_id=asset_id,
                side=side,
                quantity=quantity,
                price=price,
                order_type="LIMIT" if self.use_limit_orders else "MARKET",
            )
            orders.append(order)

        # Estimate costs (simplified)
        estimated_cost = total_turnover * 0.003  # ~30bps round trip

        return ExecutionPlan(
            target_weights=target_weights,
            current_weights=current_weights,
            orders=orders,
            estimated_cost=estimated_cost,
            estimated_turnover=total_turnover / portfolio_value if portfolio_value > 0 else 0,
        )

    def execute_plan(
        self,
        plan: ExecutionPlan,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """
        Execute trading plan.

        Args:
            plan: Execution plan
            dry_run: If True, don't actually execute

        Returns:
            ExecutionResult
        """
        if not self.broker.is_connected():
            logger.error("Broker not connected")
            return ExecutionResult(
                trades=[],
                total_cost=0,
                slippage=0,
                fill_rate=0,
                execution_time=0,
            )

        start_time = datetime.now()
        trades = []
        total_filled = 0
        total_orders = len(plan.orders)

        # Sort orders: sells first (to free up cash)
        orders_sorted = sorted(plan.orders, key=lambda o: o.side != "SELL")

        for order in orders_sorted:
            if dry_run:
                logger.info(f"[DRY RUN] Would execute: {order.side} {order.quantity} {order.asset_id}")
                continue

            try:
                order_id = self.broker.submit_order(order)

                if order_id:
                    # Wait for fill (simplified - in production would be async)
                    import time
                    time.sleep(0.1)

                    status = self.broker.get_order_status(order_id)

                    if status and status.status == OrderStatus.FILLED:
                        trade = Trade(
                            date=datetime.now(),
                            asset_id=order.asset_id,
                            side=order.side,
                            quantity=status.filled_quantity,
                            price=status.avg_price,
                            cost=0,  # Included in execution price
                        )
                        trades.append(trade)
                        total_filled += 1

            except Exception as e:
                logger.error(f"Order execution failed for {order.asset_id}: {e}")

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Calculate actual costs and slippage
        total_cost = sum(t.cost for t in trades)
        slippage = 0  # Would calculate vs expected price

        return ExecutionResult(
            trades=trades,
            total_cost=total_cost,
            slippage=slippage,
            fill_rate=total_filled / total_orders if total_orders > 0 else 1.0,
            execution_time=execution_time,
        )

    def reconcile_positions(
        self,
        expected_positions: dict[str, float],
    ) -> dict[str, Any]:
        """
        Reconcile positions with broker.

        Args:
            expected_positions: Expected positions (quantity)

        Returns:
            Reconciliation report
        """
        actual_positions = self.broker.get_positions()

        discrepancies = []

        all_assets = set(expected_positions.keys()) | set(actual_positions.keys())

        for asset_id in all_assets:
            expected_qty = expected_positions.get(asset_id, 0)
            actual_qty = actual_positions.get(asset_id, Position(asset_id, 0, 0, 0, 0)).quantity

            diff = actual_qty - expected_qty

            if abs(diff) > 0.5:  # More than 0.5 share difference
                discrepancies.append({
                    "asset_id": asset_id,
                    "expected": expected_qty,
                    "actual": actual_qty,
                    "difference": diff,
                })

        return {
            "is_reconciled": len(discrepancies) == 0,
            "n_discrepancies": len(discrepancies),
            "discrepancies": discrepancies,
            "timestamp": datetime.now(),
        }


class TWAPExecutor:
    """
    TWAP (Time-Weighted Average Price) executor.

    Splits large orders into smaller slices over time.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        duration_minutes: int = 30,
        n_slices: int = 10,
    ):
        """
        Initialize TWAP executor.

        Args:
            broker: Broker interface
            duration_minutes: Total execution duration
            n_slices: Number of order slices
        """
        self.broker = broker
        self.duration_minutes = duration_minutes
        self.n_slices = n_slices

    def execute(
        self,
        asset_id: str,
        total_quantity: int,
        side: str,
    ) -> list[Trade]:
        """
        Execute TWAP order.

        Args:
            asset_id: Asset to trade
            total_quantity: Total quantity
            side: BUY or SELL

        Returns:
            List of executed trades
        """
        slice_qty = total_quantity // self.n_slices
        remainder = total_quantity % self.n_slices

        trades = []
        interval_seconds = (self.duration_minutes * 60) // self.n_slices

        for i in range(self.n_slices):
            qty = slice_qty + (1 if i < remainder else 0)

            if qty <= 0:
                continue

            order = Order(
                asset_id=asset_id,
                side=side,
                quantity=qty,
                price=None,
                order_type="MARKET",
            )

            order_id = self.broker.submit_order(order)

            if order_id:
                # Wait for fill
                import time
                time.sleep(0.1)

                status = self.broker.get_order_status(order_id)

                if status and status.status == OrderStatus.FILLED:
                    trade = Trade(
                        date=datetime.now(),
                        asset_id=asset_id,
                        side=side,
                        quantity=status.filled_quantity,
                        price=status.avg_price,
                        cost=0,
                    )
                    trades.append(trade)

            # Wait for next slice (in production, would be async)
            if i < self.n_slices - 1:
                import time
                time.sleep(interval_seconds)

        return trades


def create_rebalance_orders(
    target_weights: pd.Series,
    current_positions: dict[str, Position],
    portfolio_value: float,
    prices: dict[str, float],
    min_trade_value: float = 100_000,
) -> list[Order]:
    """
    Create orders for rebalancing.

    Args:
        target_weights: Target portfolio weights
        current_positions: Current positions
        portfolio_value: Total portfolio value
        prices: Current prices
        min_trade_value: Minimum trade value

    Returns:
        List of orders
    """
    # Calculate current weights
    current_weights = {}
    for asset_id, pos in current_positions.items():
        current_weights[asset_id] = pos.market_value / portfolio_value

    current_weights = pd.Series(current_weights)

    orders = []
    all_assets = set(target_weights.index) | set(current_weights.index)

    for asset_id in all_assets:
        target_wt = target_weights.get(asset_id, 0)
        current_wt = current_weights.get(asset_id, 0)
        weight_diff = target_wt - current_wt

        trade_value = abs(weight_diff) * portfolio_value

        if trade_value < min_trade_value:
            continue

        price = prices.get(asset_id, 0)
        if price <= 0:
            continue

        quantity = int(trade_value / price)
        if quantity == 0:
            continue

        side = "BUY" if weight_diff > 0 else "SELL"

        order = Order(
            asset_id=asset_id,
            side=side,
            quantity=quantity,
            price=None,
            order_type="MARKET",
        )
        orders.append(order)

    return orders
