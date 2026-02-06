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
import time

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
    krx_fwdg_ord_orgno: str | None = None
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

    def __init__(self, api: KISApi, notifier: Any | None = None):
        """
        Initialize order manager.

        Args:
            api: KIS API instance
        """
        self.api = api
        self.notifier = notifier
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
        excg_id_dvsn_cd: str | None = None,
        sll_type: str = "",
        cndt_pric: str = "",
        notify_on_submit: bool = False,
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
            order_type_code = "00" if order_type == "limit" else "01"
            if side.upper() == "BUY":
                result = self.api.buy_stock(
                    stock_code=stock_code,
                    quantity=quantity,
                    price=price,
                    order_type=order_type_code,
                    excg_id_dvsn_cd=excg_id_dvsn_cd,
                    sll_type=sll_type,
                    cndt_pric=cndt_pric,
                )
            else:
                result = self.api.sell_stock(
                    stock_code=stock_code,
                    quantity=quantity,
                    price=price,
                    order_type=order_type_code,
                    excg_id_dvsn_cd=excg_id_dvsn_cd,
                    sll_type=sll_type,
                    cndt_pric=cndt_pric,
                )

            order.order_id = result.get("order_no", order.order_id)
            order.krx_fwdg_ord_orgno = result.get("krx_fwdg_ord_orgno")
            order.status = OrderStatus.SUBMITTED

            self._pending_orders[order.order_id] = order
            logger.info(f"Order submitted: {order}")

            if notify_on_submit and self.notifier:
                try:
                    self.notifier.send_order_submitted(
                        stock_code=order.stock_code,
                        stock_name=order.stock_code,
                        side=order.side,
                        quantity=order.quantity,
                        order_type=order.order_type,
                    )
                except Exception as e:
                    logger.warning(f"Failed to send order notification: {e}")

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
                order_type="00" if order.order_type == "limit" else "01",
                krx_fwdg_ord_orgno=order.krx_fwdg_ord_orgno or "",
            )

            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()

            del self._pending_orders[order_id]
            self._order_history.append(order)

            return True

        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def sync_orders(self) -> list[Order]:
        """Synchronize order status with broker."""
        filled_orders: list[Order] = []
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
                        filled_orders.append(order)

                    elif order.filled_qty > 0:
                        order.status = OrderStatus.PARTIAL

        except Exception as e:
            logger.error(f"Order sync failed: {e}")
        return filled_orders

    def _notify_fill(self, order: Order, notifier: Any | None = None) -> None:
        notifier = notifier or self.notifier
        if not notifier:
            return

        fill_price = order.filled_price or order.price or 0
        try:
            if hasattr(notifier, "send_fill_alert"):
                notifier.send_fill_alert(
                    stock_code=order.stock_code,
                    stock_name=order.stock_code,
                    side=order.side,
                    quantity=order.filled_qty or order.quantity,
                    price=fill_price,
                )
            else:
                notifier.send_trade_alert(
                    stock_code=order.stock_code,
                    stock_name=order.stock_code,
                    side=order.side,
                    quantity=order.filled_qty or order.quantity,
                    price=fill_price,
                    strategy="Ensemble",
                )
        except Exception as e:
            logger.warning(f"Failed to send fill notification: {e}")

    def wait_for_fills(
        self,
        orders: list[Order],
        timeout_seconds: int = 120,
        poll_interval: float = 2.0,
        notifier: Any | None = None,
    ) -> list[Order]:
        """Wait for orders to be filled and send notifications."""
        pending_ids = {
            order.order_id
            for order in orders
            if order.status in {OrderStatus.SUBMITTED, OrderStatus.PARTIAL}
        }
        if not pending_ids:
            return []

        end_time = time.time() + timeout_seconds
        while pending_ids and time.time() < end_time:
            filled_orders = self.sync_orders()
            for order in filled_orders:
                if order.order_id in pending_ids:
                    pending_ids.remove(order.order_id)
                    self._notify_fill(order, notifier)

            if pending_ids:
                time.sleep(poll_interval)

        return [order for order in orders if order.order_id in pending_ids]

    def execute_rebalance(
        self,
        target_weights: pd.DataFrame,
        total_value: int | None = None,
        order_type: str = "limit",
        sell_first: bool = True,
        wait_for_fills: bool = False,
        timeout_seconds: int = 120,
        poll_interval: float = 2.0,
        notifier: Any | None = None,
        notify_on_submit: bool = False,
    ) -> list[Order]:
        """
        Execute rebalancing to target weights.

        Args:
            target_weights: DataFrame with asset_id, weight
            total_value: Total portfolio value (auto-calculated if None)
            order_type: "limit" or "market"
            sell_first: Place sell orders before buy orders
            wait_for_fills: Wait for fills and optionally notify
            timeout_seconds: Max wait time per batch
            poll_interval: Poll interval for order status
            notifier: Optional Telegram notifier
            notify_on_submit: Send notifications when orders are submitted

        Returns:
            List of orders placed
        """
        # Normalize order type
        order_type_normalized = str(order_type).lower()
        if order_type_normalized in {"01", "market", "mkt"}:
            order_type_normalized = "market"
        else:
            order_type_normalized = "limit"

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

        sell_specs: list[dict[str, Any]] = []
        buy_specs: list[dict[str, Any]] = []

        def add_spec(stock_code: str, side: str, quantity: int, price: int) -> None:
            spec = {
                "stock_code": stock_code,
                "side": side,
                "quantity": quantity,
                "price": price,
            }
            if side == "SELL":
                sell_specs.append(spec)
            else:
                buy_specs.append(spec)

        # Calculate trades needed
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
            add_spec(stock_code, side, quantity, price)

        # Handle exits (stocks not in target)
        for stock_code, _weight in current_weights.items():
            if stock_code not in target_weights["asset_id"].values:
                # Full exit
                pos = positions[positions["stock_code"] == stock_code]
                if not pos.empty:
                    quantity = pos.iloc[0]["quantity"]
                    price = pos.iloc[0]["current_price"]
                    add_spec(stock_code, "SELL", quantity, price)

        orders: list[Order] = []
        notifier = notifier or self.notifier

        def submit_specs(specs: list[dict[str, Any]]) -> list[Order]:
            batch: list[Order] = []
            for spec in specs:
                order_price = 0 if order_type_normalized == "market" else spec["price"]
                order = self.submit_order(
                    stock_code=spec["stock_code"],
                    side=spec["side"],
                    quantity=spec["quantity"],
                    price=order_price,
                    order_type=order_type_normalized,
                    notify_on_submit=notify_on_submit,
                )
                batch.append(order)
                orders.append(order)
            return batch

        if sell_first:
            sell_orders = submit_specs(sell_specs)
            if wait_for_fills and sell_orders:
                self.wait_for_fills(
                    sell_orders,
                    timeout_seconds=timeout_seconds,
                    poll_interval=poll_interval,
                    notifier=notifier,
                )

            buy_orders = submit_specs(buy_specs)
            if wait_for_fills and buy_orders:
                self.wait_for_fills(
                    buy_orders,
                    timeout_seconds=timeout_seconds,
                    poll_interval=poll_interval,
                    notifier=notifier,
                )
        else:
            all_orders = submit_specs(sell_specs + buy_specs)
            if wait_for_fills and all_orders:
                self.wait_for_fills(
                    all_orders,
                    timeout_seconds=timeout_seconds,
                    poll_interval=poll_interval,
                    notifier=notifier,
                )

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
