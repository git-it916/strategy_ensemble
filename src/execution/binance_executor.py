"""
BinanceExecutor — CLAUDE.md 섹션 8 기준.

ccxt async 래퍼. 서버사이드 SL/TP 주문 포함.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import ccxt.async_support as ccxt_async

from config.settings import BALANCE_USAGE_RATIO, LEVERAGE
from src.engine.decision_engine import Order, Position

logger = logging.getLogger(__name__)


class BinanceExecutor:
    """바이낸스 선물 주문 실행기."""

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self._exchange = ccxt_async.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
                "recvWindow": 60000,
            },
            "enableRateLimit": True,
        })
        self.current_position: Position | None = None

    def _sync(self):
        """시간 동기화는 30초 백그라운드 태스크에서 처리. 여기선 no-op."""
        pass

    async def close(self):
        await self._exchange.close()

    async def get_balance(self) -> float:
        """USDT 잔고 조회."""
        try:
            self._sync()
            balance = await self._exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("total", 0))
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
            return 0.0

    async def get_price(self, symbol: str) -> float:
        """현재가 조회."""
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            return float(ticker.get("last", 0))
        except Exception as e:
            logger.error(f"Price fetch failed {symbol}: {e}")
            return 0.0

    async def set_leverage(self, symbol: str) -> None:
        """레버리지 설정."""
        try:
            base = symbol.replace("/USDT:USDT", "USDT")
            await self._exchange.fapiPrivatePostLeverage({
                "symbol": base,
                "leverage": LEVERAGE,
            })
        except Exception as e:
            logger.debug(f"Leverage set failed {symbol}: {e}")

    async def open_position(self, order: Order, balance: float) -> Optional[Position]:
        """
        포지션 진입.

        1. 시장가 주문
        2. 서버사이드 SL/TP 등록
        3. Position 객체 반환
        """
        try:
            self._sync()
            await self.set_leverage(order.symbol)

            # 포지션 크기 계산
            price = await self.get_price(order.symbol)
            if price <= 0:
                return None

            usdt_amount = balance * BALANCE_USAGE_RATIO * LEVERAGE
            qty = usdt_amount / price

            # Binance LOT_SIZE 정밀도 적용
            try:
                await self._exchange.load_markets()
                qty = float(self._exchange.amount_to_precision(order.symbol, qty))
            except Exception:
                pass  # 실패시 원래 qty 사용

            if qty <= 0:
                logger.error(f"Calculated qty is 0 for {order.symbol}")
                return None

            # 시장가 주문
            side = "buy" if order.direction == "LONG" else "sell"
            result = await self._exchange.create_market_order(
                order.symbol, side, qty,
            )
            fill_price = float(result.get("average", price))

            logger.info(f"Order filled: {order.symbol} {side} qty={qty:.4f} @ ${fill_price:,.2f}")

            # SL/TP 가격 계산
            if order.direction == "LONG":
                sl_price = fill_price * (1 + order.sl_pct)
                tp_price = fill_price * (1 + order.tp_pct)
            else:
                sl_price = fill_price * (1 - order.sl_pct)
                tp_price = fill_price * (1 - order.tp_pct)

            # 서버사이드 SL/TP 주문 등록
            await self._place_server_sltp(order.symbol, order.direction, sl_price, tp_price)

            position = Position(
                symbol=order.symbol,
                direction=order.direction,
                entry_price=fill_price,
                entry_time=datetime.now(timezone.utc),
                sl_price=sl_price,
                tp_price=tp_price,
            )
            self.current_position = position
            return position

        except Exception as e:
            logger.error(f"Open position failed: {e}")
            return None

    async def close_position(self, position: Position, reason: str) -> float:
        """
        포지션 청산.

        1. 서버사이드 주문 취소
        2. 시장가 청산
        3. 손익 반환
        """
        try:
            self._sync()
            # 서버사이드 SL/TP 취소
            await self._cancel_server_sltp(position.symbol)

            # 시장가 청산
            side = "sell" if position.direction == "LONG" else "buy"
            # 현재 포지션 크기 조회
            positions = await self._exchange.fetch_positions([position.symbol])
            qty = 0.0
            for p in positions:
                if p["symbol"] == position.symbol and abs(float(p.get("contracts", 0))) > 0:
                    qty = abs(float(p["contracts"]))
                    break

            if qty > 0:
                result = await self._exchange.create_market_order(
                    position.symbol, side, qty, params={"reduceOnly": True},
                )
                exit_price = float(result.get("average", 0))
            else:
                exit_price = await self.get_price(position.symbol)

            # PnL 계산
            if position.direction == "LONG":
                pnl_pct = (exit_price - position.entry_price) / position.entry_price
            else:
                pnl_pct = (position.entry_price - exit_price) / position.entry_price

            logger.info(
                f"Position closed: {position.symbol} {reason} "
                f"PnL={pnl_pct:+.2%}"
            )
            self.current_position = None
            return pnl_pct

        except Exception as e:
            logger.error(f"Close position failed: {e}")
            # 실패 시 current_position 유지 — 바이낸스에 포지션 남아있을 수 있음
            return 0.0

    async def _place_server_sltp(
        self, symbol: str, direction: str, sl_price: float, tp_price: float,
    ) -> None:
        """서버사이드 STOP_MARKET + TAKE_PROFIT_MARKET 등록 (ccxt 표준 API)."""
        close_side = "sell" if direction == "LONG" else "buy"

        # SL
        try:
            await self._exchange.create_order(
                symbol=symbol,
                type="STOP_MARKET",
                side=close_side,
                amount=None,
                price=None,
                params={
                    "stopPrice": sl_price,
                    "closePosition": True,
                    "workingType": "MARK_PRICE",
                },
            )
            logger.info(f"Server SL set: {symbol} @ ${sl_price:,.2f}")
        except Exception as e:
            logger.error(f"Server SL failed {symbol}: {e}")

        # TP
        try:
            await self._exchange.create_order(
                symbol=symbol,
                type="TAKE_PROFIT_MARKET",
                side=close_side,
                amount=None,
                price=None,
                params={
                    "stopPrice": tp_price,
                    "closePosition": True,
                    "workingType": "MARK_PRICE",
                },
            )
            logger.info(f"Server TP set: {symbol} @ ${tp_price:,.2f}")
        except Exception as e:
            logger.error(f"Server TP failed {symbol}: {e}")

    async def _cancel_server_sltp(self, symbol: str) -> None:
        """서버사이드 SL/TP 주문 전체 취소."""
        try:
            orders = await self._exchange.fetch_open_orders(symbol)
            for order in orders:
                otype = order.get("type", "")
                if otype in ("stop_market", "take_profit_market", "STOP_MARKET", "TAKE_PROFIT_MARKET"):
                    await self._exchange.cancel_order(order["id"], symbol)
                    logger.info(f"Server order cancelled: {order['id']}")
        except Exception as e:
            logger.error(f"Cancel SL/TP failed {symbol}: {e}")
