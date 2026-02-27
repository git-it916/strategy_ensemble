"""
Rebalancer

Diff current positions against target weights and execute
orders via BinanceApi.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.openclaw.config import ASSETS, EXECUTION_POLICY

logger = logging.getLogger(__name__)


class Rebalancer:
    """
    Execute portfolio rebalancing on Binance futures.

    Computes diffs between current and target positions,
    then places orders to reach targets.
    """

    def __init__(
        self,
        binance_api,          # BinanceApi instance
        notifier=None,        # TelegramNotifier or OpenClawTelegramHandler
        dry_run: bool = False,
    ):
        self.api = binance_api
        self.notifier = notifier
        self.dry_run = dry_run

    def rebalance(
        self,
        target_weights: dict[str, float],
        leverage_per_symbol: dict[str, float],
        total_capital: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rebalance portfolio to match target weights.

        Args:
            target_weights: {symbol: weight} e.g. {"SOLUSDT": 0.3, "ETHUSDT": 0.5}
            leverage_per_symbol: {symbol: leverage}
            total_capital: Override total capital (default: query from API)

        Returns:
            List of executed order dicts.
        """
        # 1. Get current state
        if total_capital is None:
            total_capital = self._get_account_balance()

        current_positions = self._get_current_positions()

        # 2. Compute current weights and build position size lookup
        current_weights = {}
        position_sizes = {}  # {symbol: {"size": float, "side": str}}
        for pos in current_positions:
            symbol = pos["symbol"]
            notional = abs(pos.get("notional", 0))
            if total_capital > 0:
                current_weights[symbol] = notional / total_capital
            position_sizes[symbol] = {
                "size": pos.get("size", 0),
                "side": pos.get("side", "long"),
            }

        # 3. Compute diffs
        all_symbols = set(target_weights) | set(current_weights)
        orders = []

        for symbol in all_symbols:
            target_w = target_weights.get(symbol, 0.0)
            current_w = current_weights.get(symbol, 0.0)
            diff = target_w - current_w

            # Skip small diffs
            if abs(diff) < EXECUTION_POLICY.rebalance_threshold:
                continue

            leverage = leverage_per_symbol.get(symbol, 1.0)
            # notional = margin * leverage (actual position size)
            target_notional = diff * total_capital * leverage

            # Determine if this is reducing/closing an existing position
            is_closing = (
                symbol in position_sizes
                and diff < 0
                and target_w <= EXECUTION_POLICY.rebalance_threshold
            )
            is_reducing = (
                symbol in position_sizes
                and diff < 0
                and not is_closing
            )

            order = {
                "symbol": symbol,
                "side": "BUY" if diff > 0 else "SELL",
                "notional": abs(target_notional),
                "leverage": leverage,
                "target_weight": target_w,
                "current_weight": current_w,
                "diff": diff,
                "reduce_only": is_closing or is_reducing,
            }

            # For closing positions, use the actual position size
            if is_closing and symbol in position_sizes:
                order["close_qty"] = position_sizes[symbol]["size"]

            orders.append(order)

        if not orders:
            logger.info("No rebalancing needed — within threshold")
            return []

        # 4. Execute orders
        executed = []
        for order in orders:
            try:
                result = self._execute_order(order)
                executed.append(result)
            except Exception as e:
                logger.error(
                    f"Failed to execute order for {order['symbol']}: {e}"
                )
                order["error"] = str(e)
                executed.append(order)

        # 5. Notify
        self._send_rebalance_notification(executed, total_capital)

        return executed

    def _get_account_balance(self) -> float:
        """Get total USDT balance from Binance."""
        try:
            account = self.api.get_account()
            for asset in account.get("assets", []):
                if asset.get("asset") == "USDT":
                    return float(asset.get("walletBalance", 0))
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return 0.0

    def _get_current_positions(self) -> list[dict]:
        """Get current open positions from Binance."""
        try:
            positions = self.api.get_positions()
            if positions.empty:
                return []
            rows = []
            for _, row in positions.iterrows():
                size = abs(float(row.get("size", 0)))
                if size > 0:
                    entry_price = float(row.get("entry_price", 0))
                    rows.append({
                        "symbol": row["ticker"],
                        "side": row.get("side", "long"),
                        "notional": size * entry_price,
                        "size": size,
                    })
            return rows
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def _execute_order(self, order: dict) -> dict:
        """Execute a single order."""
        symbol = order["symbol"]
        side = order["side"]
        notional = order["notional"]
        leverage = order["leverage"]
        reduce_only = order.get("reduce_only", False)
        close_qty = order.get("close_qty")  # Pre-computed qty for full close

        # For closing orders, skip min-notional check (must close regardless of size)
        MIN_NOTIONAL = 5.0
        if not reduce_only and notional < MIN_NOTIONAL:
            order["status"] = "skipped_min_notional"
            logger.debug(f"Skipped {symbol}: notional ${notional:.2f} < ${MIN_NOTIONAL}")
            return order

        if self.dry_run:
            logger.info(
                f"[DRY RUN] {side} {symbol} "
                f"notional={notional:.2f} lev={leverage:.1f}x"
                f"{' reduce_only' if reduce_only else ''}"
            )
            order["status"] = "dry_run"
            return order

        # Set leverage (skip for reduce-only since position already has leverage)
        if not reduce_only:
            try:
                self.api.set_leverage(symbol=symbol, leverage=int(leverage))
            except Exception as e:
                logger.warning(f"Failed to set leverage for {symbol}: {e}")

        # Get current price to calculate quantity
        try:
            price = self.api.get_price(symbol)

            if close_qty and close_qty > 0:
                # Full position close — use actual position size
                quantity = close_qty
            else:
                quantity = notional / price if price > 0 else 0

            if quantity <= 0:
                order["status"] = "skipped_zero_qty"
                return order

            # Round quantity to exchange precision
            try:
                quantity = float(
                    self.api._exchange.amount_to_precision(symbol, quantity)
                )
            except Exception:
                pass

            if quantity <= 0:
                order["status"] = "skipped_zero_qty"
                return order

            # Re-check notional after rounding (skip for reduce-only)
            if not reduce_only:
                actual_notional = quantity * price
                if actual_notional < MIN_NOTIONAL:
                    order["status"] = "skipped_min_notional"
                    logger.debug(f"Skipped {symbol}: rounded notional ${actual_notional:.2f} < ${MIN_NOTIONAL}")
                    return order

            # Place market order
            result = self.api.place_order(
                symbol=symbol,
                side=side.lower(),
                quantity=quantity,
                reduce_only=reduce_only,
            )

            order["status"] = "filled"
            order["price"] = price
            order["quantity"] = quantity
            order["api_response"] = result

            logger.info(
                f"Executed: {side} {symbol} qty={quantity:.4f} "
                f"@ {price:.2f} lev={leverage:.1f}x"
                f"{' (reduce_only)' if reduce_only else ''}"
            )

        except Exception as e:
            order["status"] = "error"
            order["error"] = str(e)
            logger.error(f"Order execution failed for {symbol}: {e}")

        return order

    def _send_rebalance_notification(
        self,
        orders: list[dict],
        total_capital: float,
    ) -> None:
        """Send rebalance summary to Telegram."""
        if not self.notifier or not orders:
            return

        status_kr = {
            "filled": "체결",
            "dry_run": "모의",
            "error": "오류",
            "skipped_min_notional": "최소금액미달",
            "skipped_zero_qty": "수량부족",
            "unknown": "알수없음",
        }
        side_kr = {"BUY": "매수", "SELL": "매도"}

        lines = [
            "<b>리밸런스 실행</b>",
            f"자본금: ${total_capital:,.2f}",
            "",
        ]

        for o in orders:
            status = o.get("status", "unknown")
            icon = "" if status in ("filled", "dry_run") else ""
            side = o.get("side", "?")
            symbol = o.get("symbol", "?")
            diff = o.get("diff", 0)
            s_kr = status_kr.get(status, status)
            sd_kr = side_kr.get(side, side)

            lines.append(
                f"{icon} {sd_kr} {symbol} "
                f"({diff:+.1%}) "
                f"[{s_kr}]"
            )

        try:
            self.notifier.send_message("\n".join(lines))
        except Exception as e:
            logger.warning(f"Failed to send rebalance notification: {e}")

    def close_all_positions(self) -> list[dict]:
        """Emergency: close all positions."""
        positions = self._get_current_positions()
        results = []

        for pos in positions:
            symbol = pos["symbol"]
            amt = float(pos.get("positionAmt", 0))

            if amt == 0:
                continue

            side = "SELL" if amt > 0 else "BUY"

            try:
                if not self.dry_run:
                    result = self.api.place_order(
                        symbol=symbol,
                        side=side.lower(),
                        quantity=abs(amt),
                    )
                    results.append({"symbol": symbol, "status": "closed", **result})
                else:
                    results.append({"symbol": symbol, "status": "dry_run_close"})

                logger.info(f"Closed position: {symbol} {side} {abs(amt)}")

            except Exception as e:
                logger.error(f"Failed to close {symbol}: {e}")
                results.append({"symbol": symbol, "status": "error", "error": str(e)})

        return results
