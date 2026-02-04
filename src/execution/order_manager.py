"""
Order Manager

Manage order lifecycle and position synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import logging

import pandas as pd

from .kis_api import KISApi

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation."""
    order_id: str
    stock_code: str
    side: str  # "BUY" or "SELL"
    quantity: int
    price: int | None
    order_type: str = "limit"  # "limit" or "market"
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class OrderManager:
    """
    Manage orders and synchronize with broker.

    Features:
        - Order tracking
        - Position reconciliation
        - Rebalancing execution
        - Slippage tracking
    """

    def __init__(self, api: KISApi):
        """
        Initialize order manager.

        Args:
            api: KIS API instance
        """
        self.api = api
        self._pending_orders: dict[str, Order] = {}
        self._order_history: list[Order] = []

    def get_positions(self) -> pd.DataFrame:
        """
        Get current positions from broker.

        Returns:
            DataFrame with positions
        """
        balance = self.api.get_balance()

        holdings = balance.get("holdings", [])

        if not holdings:
            return pd.DataFrame(columns=[
                "stock_code", "name", "quantity", "avg_price",
                "current_price", "eval_amount", "profit_loss"
            ])

        return pd.DataFrame(holdings)

    def get_cash(self) -> int:
        """Get available cash."""
        balance = self.api.get_balance()
        return balance.get("cash", 0)

    def submit_order(
        self,
        stock_code: str,
        side: str,
        quantity: int,
        price: int | None = None,
        order_type: str = "limit",
    ) -> Order:
        """
        Submit an order.

        Args:
            stock_code: Stock code
            side: "BUY" or "SELL"
            quantity: Order quantity
            price: Limit price (None for market)
            order_type: "limit" or "market"

        Returns:
            Order object
        """
        # Create order object
        order = Order(
            order_id=f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{stock_code}",
            stock_code=stock_code,
            side=side.upper(),
            quantity=quantity,
            price=price,
            order_type=order_type,
        )

        try:
            # Submit to broker
            if side.upper() == "BUY":
                result = self.api.buy_stock(
                    stock_code=stock_code,
                    quantity=quantity,
                    price=price,
                    order_type="00" if order_type == "limit" else "01",
                )
            else:
                result = self.api.sell_stock(
                    stock_code=stock_code,
                    quantity=quantity,
                    price=price,
                    order_type="00" if order_type == "limit" else "01",
                )

            order.order_id = result.get("order_no", order.order_id)
            order.status = OrderStatus.SUBMITTED

            self._pending_orders[order.order_id] = order
            logger.info(f"Order submitted: {order}")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Order rejected: {e}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled
        """
        if order_id not in self._pending_orders:
            logger.warning(f"Order not found: {order_id}")
            return False

        order = self._pending_orders[order_id]

        try:
            self.api.cancel_order(
                order_no=order_id,
                stock_code=order.stock_code,
                quantity=order.quantity - order.filled_qty,
            )

            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()

            del self._pending_orders[order_id]
            self._order_history.append(order)

            return True

        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def sync_orders(self) -> None:
        """Synchronize order status with broker."""
        try:
            broker_orders = self.api.get_orders()

            for bo in broker_orders:
                order_id = bo.get("order_no")

                if order_id in self._pending_orders:
                    order = self._pending_orders[order_id]

                    order.filled_qty = bo.get("filled_qty", 0)
                    order.filled_price = bo.get("filled_price", 0)
                    order.updated_at = datetime.now()

                    if order.filled_qty >= order.quantity:
                        order.status = OrderStatus.FILLED
                        del self._pending_orders[order_id]
                        self._order_history.append(order)

                    elif order.filled_qty > 0:
                        order.status = OrderStatus.PARTIAL

        except Exception as e:
            logger.error(f"Order sync failed: {e}")

    def execute_rebalance(
        self,
        target_weights: pd.DataFrame,
        total_value: int | None = None,
    ) -> list[Order]:
        """
        Execute rebalancing to target weights.

        Args:
            target_weights: DataFrame with asset_id, weight
            total_value: Total portfolio value (auto-calculated if None)

        Returns:
            List of orders placed
        """
        # Get current positions
        positions = self.get_positions()
        cash = self.get_cash()

        if total_value is None:
            if positions.empty:
                total_value = cash
            else:
                total_value = positions["eval_amount"].sum() + cash

        # Calculate current weights
        current_weights = {}
        for _, row in positions.iterrows():
            current_weights[row["stock_code"]] = row["eval_amount"] / total_value

        # Calculate trades needed
        orders = []

        for _, row in target_weights.iterrows():
            stock_code = row["asset_id"]
            target_weight = row["weight"]
            current_weight = current_weights.get(stock_code, 0)

            weight_diff = target_weight - current_weight
            trade_value = abs(weight_diff) * total_value

            # Skip small trades (< 10만원)
            if trade_value < 100000:
                continue

            # Get current price
            try:
                price_info = self.api.get_price(stock_code)
                price = price_info["price"]
            except Exception as e:
                logger.error(f"Failed to get price for {stock_code}: {e}")
                continue

            quantity = int(trade_value / price)
            if quantity == 0:
                continue

            side = "BUY" if weight_diff > 0 else "SELL"

            order = self.submit_order(
                stock_code=stock_code,
                side=side,
                quantity=quantity,
                price=price,
                order_type="limit",
            )
            orders.append(order)

        # Handle exits (stocks not in target)
        for stock_code, weight in current_weights.items():
            if stock_code not in target_weights["asset_id"].values:
                # Full exit
                pos = positions[positions["stock_code"] == stock_code]
                if not pos.empty:
                    quantity = pos.iloc[0]["quantity"]
                    price = pos.iloc[0]["current_price"]

                    order = self.submit_order(
                        stock_code=stock_code,
                        side="SELL",
                        quantity=quantity,
                        price=price,
                        order_type="limit",
                    )
                    orders.append(order)

        return orders

    def get_pending_orders(self) -> list[Order]:
        """Get all pending orders."""
        return list(self._pending_orders.values())

    def get_order_history(self, n: int = 100) -> list[Order]:
        """Get recent order history."""
        return self._order_history[-n:]

    def calculate_slippage(self) -> dict[str, float]:
        """
        Calculate execution slippage from filled orders.

        Returns:
            Slippage statistics
        """
        if not self._order_history:
            return {"avg_slippage": 0, "total_slippage": 0}

        slippages = []

        for order in self._order_history:
            if order.status != OrderStatus.FILLED:
                continue
            if order.price is None or order.price == 0:
                continue

            expected_price = order.price
            actual_price = order.filled_price

            if order.side == "BUY":
                slippage = (actual_price - expected_price) / expected_price
            else:
                slippage = (expected_price - actual_price) / expected_price

            slippages.append(slippage)

        if not slippages:
            return {"avg_slippage": 0, "total_slippage": 0}

        return {
            "avg_slippage": sum(slippages) / len(slippages),
            "total_slippage": sum(slippages),
            "n_orders": len(slippages),
        }
