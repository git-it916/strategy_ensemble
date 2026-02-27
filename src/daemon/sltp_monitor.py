"""
Stop-Loss / Take-Profit Monitor

Checks active positions against their SL/TP price levels every tick.
Executes immediate closure when triggered, without waiting for Sonnet.
"""

from __future__ import annotations

import logging
from typing import Any

from src.daemon.position_store import PositionStore, ManagedPosition
from src.daemon.sonnet_decision_maker import SonnetDecisionMaker

logger = logging.getLogger(__name__)


class SLTPMonitor:
    """
    Monitor positions for stop-loss and take-profit triggers.

    Called from the main loop every 5 seconds. When SL/TP is hit,
    immediately closes the position via the rebalancer and notifies
    via Telegram.
    """

    def __init__(
        self,
        position_store: PositionStore,
        binance_api,
        rebalancer,
        notifier=None,
        dry_run: bool = False,
    ):
        self.store = position_store
        self.api = binance_api
        self.rebalancer = rebalancer
        self.notifier = notifier
        self.dry_run = dry_run

    def check_all(self) -> list[dict[str, Any]]:
        """
        Check all active positions against their SL/TP levels.

        Returns list of triggered actions.
        """
        active = self.store.get_active()
        if not active:
            return []

        triggered = []
        for pos in active:
            try:
                current_price = self.api.get_price(pos.ticker)
                if current_price <= 0:
                    continue

                action = self._check_position(pos, current_price)
                if action:
                    self._execute_exit(pos, action, current_price)
                    triggered.append(action)

            except Exception as e:
                logger.error(f"SL/TP check error for {pos.ticker}: {e}")

        return triggered

    def _check_position(
        self, pos: ManagedPosition, price: float
    ) -> dict[str, Any] | None:
        """Check if SL or TP is triggered for a position."""
        if pos.side == "LONG":
            if price <= pos.stop_loss_price:
                pnl_pct = (price / pos.entry_price - 1)
                return {
                    "type": "stop_loss",
                    "ticker": pos.ticker,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "exit_price": price,
                    "sl_price": pos.stop_loss_price,
                    "pnl_pct": pnl_pct,
                }
            if price >= pos.take_profit_price:
                pnl_pct = (price / pos.entry_price - 1)
                return {
                    "type": "take_profit",
                    "ticker": pos.ticker,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "exit_price": price,
                    "tp_price": pos.take_profit_price,
                    "pnl_pct": pnl_pct,
                }
        else:  # SHORT
            if price >= pos.stop_loss_price:
                pnl_pct = -(price / pos.entry_price - 1)
                return {
                    "type": "stop_loss",
                    "ticker": pos.ticker,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "exit_price": price,
                    "sl_price": pos.stop_loss_price,
                    "pnl_pct": pnl_pct,
                }
            if price <= pos.take_profit_price:
                pnl_pct = -(price / pos.entry_price - 1)
                return {
                    "type": "take_profit",
                    "ticker": pos.ticker,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "exit_price": price,
                    "tp_price": pos.take_profit_price,
                    "pnl_pct": pnl_pct,
                }

        return None

    def _execute_exit(
        self, pos: ManagedPosition, action: dict, current_price: float
    ) -> None:
        """Close position and notify via Telegram."""
        ticker = pos.ticker
        action_type = action["type"]
        pnl_pct = action["pnl_pct"]

        logger.info(
            f"SL/TP triggered: {action_type} for {ticker} "
            f"@ {current_price:.2f} (entry: {pos.entry_price:.2f}, "
            f"pnl: {pnl_pct:+.2%})"
        )

        # Close via rebalancer (target weight = 0)
        close_success = False
        if not self.dry_run:
            try:
                side = "sell" if pos.side == "LONG" else "buy"
                # Get current position size from Binance
                positions = self.api.get_positions()
                if not positions.empty:
                    match = positions[positions["ticker"] == ticker]
                    if not match.empty:
                        qty = abs(float(match.iloc[0].get("size", 0)))
                        if qty > 0:
                            self.api.place_order(
                                symbol=ticker,
                                side=side,
                                quantity=qty,
                            )
                            logger.info(f"Closed {ticker}: {side} qty={qty}")
                            close_success = True
                            SonnetDecisionMaker.log_transaction(
                                ticker=ticker,
                                action=action_type,
                                side=side,
                                quantity=qty,
                                price=current_price,
                                notional=qty * current_price,
                                status="closed",
                                reason=f"{action_type} triggered (entry={pos.entry_price:.4f}, exit={current_price:.4f}, pnl={pnl_pct:+.2%})",
                            )
            except Exception as e:
                logger.error(f"Failed to close {ticker}: {e}")
        else:
            # Dry run: log but do NOT remove from store so monitoring continues
            logger.info(f"[DRY RUN] Would close {ticker} ({action_type})")

        # Only remove from position store if close order succeeded (or not dry run)
        if close_success:
            reason = "stopped_out" if action_type == "stop_loss" else "took_profit"
            self.store.remove(ticker, reason=reason)
        elif not self.dry_run:
            logger.warning(
                f"Position {ticker} NOT removed from store — close order failed. "
                f"Will retry on next SL/TP check."
            )

        # Notify via Telegram
        if self.notifier:
            short_ticker = ticker.replace("/USDT:USDT", "")
            if action_type == "stop_loss":
                icon = "STOP LOSS"
                msg = (
                    f"<b>{icon} {short_ticker}</b>\n"
                    f"{pos.side} @ ${pos.entry_price:,.2f} → ${current_price:,.2f}\n"
                    f"P&L: {pnl_pct:+.2%}\n"
                    f"SL level: ${pos.stop_loss_price:,.2f}"
                )
            else:
                icon = "TAKE PROFIT"
                msg = (
                    f"<b>{icon} {short_ticker}</b>\n"
                    f"{pos.side} @ ${pos.entry_price:,.2f} → ${current_price:,.2f}\n"
                    f"P&L: {pnl_pct:+.2%}\n"
                    f"TP level: ${pos.take_profit_price:,.2f}"
                )

            try:
                self.notifier.send_message(msg)
            except Exception as e:
                logger.warning(f"Failed to send SL/TP notification: {e}")
