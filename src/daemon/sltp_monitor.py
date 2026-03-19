"""
Stop-Loss / Take-Profit / Trailing-Stop Monitor

Checks active positions every tick:
  1. Hard stop-loss — immediate exit
  2. Trailing stop — once profit exceeds activation threshold,
     trail the stop at a fixed % below the peak price
  3. Fixed TP fallback — only used if trailing is disabledㅁㄴㄻ
"""

from __future__ import annotations

import logging
from typing import Any

from src.daemon.position_store import PositionStore, ManagedPosition
from src.daemon.sonnet_decision_maker import SonnetDecisionMaker

logger = logging.getLogger(__name__)

# Trailing stop configuration
TRAILING_CONFIG = {
    "enabled": True,
    "activation_pct": 0.025,    # Activate trailing after 2.5% profit (7.5% ROI at 3x)
    "trail_pct": 0.015,         # Trail 1.5% below peak — worst case ROI ~+3%
}


class SLTPMonitor:
    """
    Monitor positions for stop-loss, trailing-stop, and take-profit triggers.

    Called from the main loop every 5 seconds. When triggered,
    immediately closes the position via direct API order.
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

        # Track peak prices for trailing stop {ticker: peak_price}
        self._peak_prices: dict[str, float] = {}
        # Track whether trailing is activated {ticker: True}
        self._trailing_active: dict[str, bool] = {}

    def check_all(self) -> list[dict[str, Any]]:
        """
        Check all active positions against SL/trailing/TP levels.

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

                # Update peak price tracking
                self._update_peak(pos, current_price)

                action = self._check_position(pos, current_price)
                if action:
                    self._execute_exit(pos, action, current_price)
                    triggered.append(action)
                    # Clean up tracking
                    self._peak_prices.pop(pos.ticker, None)
                    self._trailing_active.pop(pos.ticker, None)

            except Exception as e:
                logger.error(f"SL/TP check error for {pos.ticker}: {e}")

        return triggered

    def _update_peak(self, pos: ManagedPosition, price: float) -> None:
        """Track the most favorable price for trailing stop."""
        ticker = pos.ticker
        if pos.side == "LONG":
            # For longs, peak = highest price
            current_peak = self._peak_prices.get(ticker, pos.entry_price)
            self._peak_prices[ticker] = max(current_peak, price)
        else:
            # For shorts, peak = lowest price
            current_peak = self._peak_prices.get(ticker, pos.entry_price)
            self._peak_prices[ticker] = min(current_peak, price)

    def _check_position(
        self, pos: ManagedPosition, price: float
    ) -> dict[str, Any] | None:
        """Check SL, trailing stop, and TP in order of priority."""
        ticker = pos.ticker

        # --- 1. Hard stop-loss (always active) ---
        sl_hit = self._check_stop_loss(pos, price)
        if sl_hit:
            return sl_hit

        # --- 2. Trailing stop (if enabled and activated) ---
        if TRAILING_CONFIG["enabled"]:
            trailing_hit = self._check_trailing_stop(pos, price)
            if trailing_hit:
                return trailing_hit

        # --- 3. Fixed TP (only if trailing is NOT active for this position) ---
        # Once trailing is activated, we let the trailing stop handle the exit
        # instead of the fixed TP, so profits can run beyond the TP level.
        if not self._trailing_active.get(ticker, False):
            tp_hit = self._check_take_profit(pos, price)
            if tp_hit:
                return tp_hit

        return None

    def _check_stop_loss(self, pos: ManagedPosition, price: float) -> dict | None:
        if pos.side == "LONG" and price <= pos.stop_loss_price:
            return {
                "type": "stop_loss",
                "ticker": pos.ticker,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "sl_price": pos.stop_loss_price,
                "pnl_pct": (price / pos.entry_price - 1),
            }
        if pos.side == "SHORT" and price >= pos.stop_loss_price:
            return {
                "type": "stop_loss",
                "ticker": pos.ticker,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "sl_price": pos.stop_loss_price,
                "pnl_pct": -(price / pos.entry_price - 1),
            }
        return None

    def _check_trailing_stop(self, pos: ManagedPosition, price: float) -> dict | None:
        """
        Trailing stop logic:
        - Activate after profit exceeds activation_pct
        - Once active, exit if price retraces trail_pct from peak
        """
        ticker = pos.ticker
        activation_pct = TRAILING_CONFIG["activation_pct"]
        trail_pct = TRAILING_CONFIG["trail_pct"]
        peak = self._peak_prices.get(ticker, pos.entry_price)

        if pos.side == "LONG":
            current_pnl = (price / pos.entry_price) - 1
            # Check activation
            if current_pnl >= activation_pct:
                if not self._trailing_active.get(ticker, False):
                    self._trailing_active[ticker] = True
                    logger.info(
                        f"Trailing stop ACTIVATED for {ticker} LONG "
                        f"(pnl {current_pnl:+.2%}, peak ${peak:.4f})"
                    )

            # Check trailing trigger
            if self._trailing_active.get(ticker, False):
                trail_price = peak * (1 - trail_pct)
                if price <= trail_price:
                    return {
                        "type": "trailing_stop",
                        "ticker": ticker,
                        "side": pos.side,
                        "entry_price": pos.entry_price,
                        "exit_price": price,
                        "peak_price": peak,
                        "trail_price": trail_price,
                        "pnl_pct": current_pnl,
                    }
        else:  # SHORT
            current_pnl = -(price / pos.entry_price - 1)
            if current_pnl >= activation_pct:
                if not self._trailing_active.get(ticker, False):
                    self._trailing_active[ticker] = True
                    logger.info(
                        f"Trailing stop ACTIVATED for {ticker} SHORT "
                        f"(pnl {current_pnl:+.2%}, peak ${peak:.4f})"
                    )

            if self._trailing_active.get(ticker, False):
                trail_price = peak * (1 + trail_pct)
                if price >= trail_price:
                    return {
                        "type": "trailing_stop",
                        "ticker": ticker,
                        "side": pos.side,
                        "entry_price": pos.entry_price,
                        "exit_price": price,
                        "peak_price": peak,
                        "trail_price": trail_price,
                        "pnl_pct": current_pnl,
                    }

        return None

    def _check_take_profit(self, pos: ManagedPosition, price: float) -> dict | None:
        if pos.side == "LONG" and price >= pos.take_profit_price:
            return {
                "type": "take_profit",
                "ticker": pos.ticker,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "tp_price": pos.take_profit_price,
                "pnl_pct": (price / pos.entry_price - 1),
            }
        if pos.side == "SHORT" and price <= pos.take_profit_price:
            return {
                "type": "take_profit",
                "ticker": pos.ticker,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "tp_price": pos.take_profit_price,
                "pnl_pct": -(price / pos.entry_price - 1),
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
            f"@ {current_price:.4f} (entry: {pos.entry_price:.4f}, "
            f"pnl: {pnl_pct:+.2%})"
        )

        # Close via direct API order
        close_success = False
        if not self.dry_run:
            try:
                side = "sell" if pos.side == "LONG" else "buy"
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
            logger.info(f"[DRY RUN] Would close {ticker} ({action_type})")

        # Remove from position store only if close succeeded
        if close_success:
            reason = {
                "stop_loss": "stopped_out",
                "trailing_stop": "trailing_stop",
                "take_profit": "took_profit",
            }.get(action_type, "closed")
            self.store.remove(ticker, reason=reason)
        elif not self.dry_run:
            logger.warning(
                f"Position {ticker} NOT removed from store — close order failed. "
                f"Will retry on next check."
            )

        # Notify via Telegram
        if self.notifier:
            short_ticker = ticker.replace("/USDT:USDT", "")
            roi_pct = pnl_pct * 3  # 3x leverage ROI approximation

            if action_type == "stop_loss":
                msg = (
                    f"<b>STOP LOSS {short_ticker}</b>\n"
                    f"{pos.side} @ ${pos.entry_price:,.4f} -> ${current_price:,.4f}\n"
                    f"P&L: {pnl_pct:+.2%} (ROI ~{roi_pct:+.1%})"
                )
            elif action_type == "trailing_stop":
                peak = action.get("peak_price", current_price)
                msg = (
                    f"<b>TRAILING STOP {short_ticker}</b>\n"
                    f"{pos.side} @ ${pos.entry_price:,.4f} -> ${current_price:,.4f}\n"
                    f"Peak: ${peak:,.4f}\n"
                    f"P&L: {pnl_pct:+.2%} (ROI ~{roi_pct:+.1%})"
                )
            else:
                msg = (
                    f"<b>TAKE PROFIT {short_ticker}</b>\n"
                    f"{pos.side} @ ${pos.entry_price:,.4f} -> ${current_price:,.4f}\n"
                    f"P&L: {pnl_pct:+.2%} (ROI ~{roi_pct:+.1%})"
                )

            try:
                self.notifier.send_message(msg)
            except Exception as e:
                logger.warning(f"Failed to send notification: {e}")
