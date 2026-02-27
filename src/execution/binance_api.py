"""
Binance USDT-M Futures API Client

ccxt 기반 Binance 선물 REST 클라이언트.
거래량 상위 50개 USDT 무기한 선물을 대상으로 OHLCV + 펀딩비율 수집 및 주문 실행.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceApi:
    """
    Binance USDT-M Perpetual Futures REST client via ccxt.

    Handles:
        - Symbol discovery (volume-ranked)
        - OHLCV batch fetch
        - Funding rate history
        - Position and account queries
        - Order placement
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        rate_limit_sleep: float = 0.2,
    ):
        self.rate_limit_sleep = rate_limit_sleep

        options: dict = {
            "defaultType": "future",
            "adjustForTimeDifference": True,
            "recvWindow": 10000,  # 10s tolerance (WSL clock drift)
        }
        if testnet:
            options["urls"] = {
                "api": {
                    "public": "https://testnet.binancefuture.com/fapi/v1",
                    "private": "https://testnet.binancefuture.com/fapi/v1",
                }
            }

        self._exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "options": options,
            "enableRateLimit": True,
        })

        if testnet:
            self._exchange.set_sandbox_mode(True)

        # Force time sync to avoid timestamp errors (common in WSL)
        self._sync_time()

    def _sync_time(self) -> None:
        """Sync local clock offset with Binance server time (critical for WSL)."""
        try:
            self._exchange.load_time_difference()
            logger.info(
                f"Time synced with Binance (offset: "
                f"{getattr(self._exchange, 'options', {}).get('timeDifference', 0)}ms)"
            )
        except Exception:
            # Manual fallback: fetch server time and compute offset
            try:
                server_time = self._exchange.fetch_time()
                local_time = int(time.time() * 1000)
                offset = server_time - local_time
                self._exchange.options["timeDifference"] = offset
                logger.info(f"Manual time sync (offset: {offset}ms)")
            except Exception as e:
                logger.warning(f"Time sync failed: {e}. Using recvWindow=10s as fallback.")

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_top_symbols(self, n: int = 50, exclude: list[str] | None = None) -> list[str]:
        """
        Return top-N USDT-M perpetual symbols ranked by 24h quote volume.

        Args:
            n: Number of symbols to return
            exclude: Symbols to exclude (e.g. index composites)

        Returns:
            List of symbol strings, e.g. ['BTC/USDT:USDT', ...]
        """
        exclude = set(exclude or ["BTCDOMUSDT", "DEFIUSDT", "BTCSTUSDT"])
        markets = self._exchange.load_markets()

        # Filter to USDT-M linear perpetuals only
        futures = [
            m for m in markets.values()
            if (
                m.get("linear")
                and m.get("quote") == "USDT"
                and m.get("type") == "swap"
                and m.get("active")
                and m.get("id", "").replace("/", "") not in exclude
            )
        ]

        # Fetch tickers for volume ranking
        tickers = self._exchange.fetch_tickers(
            [m["symbol"] for m in futures]
        )

        ranked = sorted(
            [(sym, t.get("quoteVolume") or 0) for sym, t in tickers.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        return [sym for sym, _ in ranked[:n]]

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: datetime | None = None,
        limit: int = 300,
    ) -> list[list]:
        """
        Fetch OHLCV for a single symbol.

        Returns list of [timestamp_ms, open, high, low, close, volume].
        """
        since_ms = None
        if since:
            since_ms = int(since.replace(tzinfo=timezone.utc).timestamp() * 1000)

        return self._exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=limit
        )

    def get_ohlcv_batch(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        days: int = 300,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for multiple symbols.

        Returns DataFrame with columns:
            date, ticker, open, high, low, close, volume
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)
        records: list[pd.DataFrame] = []

        for symbol in symbols:
            try:
                raw = self.get_ohlcv(symbol, timeframe=timeframe, since=since, limit=days + 10)
                if not raw:
                    logger.warning(f"No OHLCV data for {symbol}")
                    continue

                df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
                df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.date
                df["ticker"] = symbol
                df = df.drop(columns=["ts"])
                records.append(df)

                time.sleep(self.rate_limit_sleep)

            except Exception as e:
                logger.error(f"OHLCV fetch failed for {symbol}: {e}")

        if not records:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

        result = pd.concat(records, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])
        return result.sort_values(["date", "ticker"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Funding Rates
    # ------------------------------------------------------------------

    def get_funding_rates(self, symbols: list[str]) -> pd.DataFrame:
        """
        Get the current funding rate for each symbol.

        Returns DataFrame with columns:
            ticker, funding_rate, next_funding_time
        """
        records = []
        for symbol in symbols:
            try:
                info = self._exchange.fetch_funding_rate(symbol)
                records.append({
                    "ticker": symbol,
                    "funding_rate": info.get("fundingRate"),
                    "next_funding_time": info.get("nextFundingDatetime"),
                })
                time.sleep(self.rate_limit_sleep)
            except Exception as e:
                logger.error(f"Funding rate fetch failed for {symbol}: {e}")

        return pd.DataFrame(records)

    def get_funding_history(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """
        Fetch historical funding rates for a single symbol.

        Returns DataFrame with columns:
            date, ticker, funding_rate
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)
        since_ms = int(since.timestamp() * 1000)

        try:
            raw = self._exchange.fetch_funding_rate_history(
                symbol, since=since_ms, limit=1000
            )
        except Exception as e:
            logger.error(f"Funding history fetch failed for {symbol}: {e}")
            return pd.DataFrame(columns=["date", "ticker", "funding_rate"])

        if not raw:
            return pd.DataFrame(columns=["date", "ticker", "funding_rate"])

        df = pd.DataFrame(raw)
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = symbol
        df = df.rename(columns={"fundingRate": "funding_rate"})
        return df[["date", "ticker", "funding_rate"]].reset_index(drop=True)

    def get_funding_history_batch(
        self, symbols: list[str], days: int = 90
    ) -> pd.DataFrame:
        """
        Fetch funding history for multiple symbols and aggregate to daily.

        Binance pays funding every 8h; we average per day.

        Returns DataFrame with columns:
            date, ticker, funding_rate
        """
        records = []
        for symbol in symbols:
            df = self.get_funding_history(symbol, days=days)
            if not df.empty:
                records.append(df)
            time.sleep(self.rate_limit_sleep)

        if not records:
            return pd.DataFrame(columns=["date", "ticker", "funding_rate"])

        combined = pd.concat(records, ignore_index=True)
        # Average the three 8h payments into one daily rate
        daily = (
            combined.groupby(["date", "ticker"])["funding_rate"]
            .mean()
            .reset_index()
        )
        return daily.sort_values(["date", "ticker"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Account / Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> pd.DataFrame:
        """
        Return all open positions.

        Returns DataFrame with columns:
            ticker, side, size, entry_price, unrealized_pnl, leverage
        """
        try:
            positions = self._exchange.fetch_positions()
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return pd.DataFrame()

        rows = []
        for p in positions:
            size = float(p.get("contracts") or p.get("positionAmt") or 0)
            if size == 0:
                continue
            rows.append({
                "ticker": p["symbol"],
                "side": p.get("side", "long" if size > 0 else "short"),
                "size": abs(size),
                "entry_price": p.get("entryPrice"),
                "unrealized_pnl": p.get("unrealizedPnl"),
                "leverage": p.get("leverage"),
            })

        return pd.DataFrame(rows)

    def get_account(self) -> dict:
        """
        Return account balance summary.

        Returns dict with:
            total_wallet_balance, available_balance, total_unrealized_pnl
        """
        try:
            balance = self._exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            return {
                "total_wallet_balance": float(usdt.get("total") or 0),
                "available_balance": float(usdt.get("free") or 0),
                "total_unrealized_pnl": float(
                    balance.get("info", {}).get("totalUnrealizedProfit") or 0
                ),
            }
        except Exception as e:
            logger.error(f"Failed to fetch account: {e}")
            return {"total_wallet_balance": 0, "available_balance": 0, "total_unrealized_pnl": 0}

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        reduce_only: bool = False,
    ) -> dict:
        """
        Place a futures order.

        Args:
            symbol: e.g. 'BTC/USDT:USDT'
            side: 'buy' or 'sell'
            quantity: Contract quantity (base currency)
            price: Limit price; None for market order
            reduce_only: Only reduce existing position

        Returns:
            Order dict from ccxt
        """
        order_type = "limit" if price else "market"
        params: dict = {}
        if reduce_only:
            params["reduceOnly"] = True

        try:
            if order_type == "limit":
                order = self._exchange.create_limit_order(
                    symbol, side, quantity, price, params=params
                )
            else:
                order = self._exchange.create_market_order(
                    symbol, side, quantity, params=params
                )
            logger.info(
                f"Order placed: {side} {quantity} {symbol} @ "
                f"{'market' if not price else price} → id={order['id']}"
            )
            return order
        except Exception as e:
            logger.error(f"Order failed for {symbol}: {e}")
            raise

    def get_price(self, symbol: str) -> float:
        """Get latest price for a symbol."""
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return float(ticker.get("last") or ticker.get("close") or 0)
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return 0.0

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol."""
        try:
            self._exchange.set_leverage(leverage, symbol)
            logger.info(f"Leverage set: {symbol} x{leverage}")
        except Exception as e:
            logger.warning(f"set_leverage failed for {symbol}: {e}")
