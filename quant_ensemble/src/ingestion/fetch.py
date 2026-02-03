"""
Bloomberg Data Fetching Module

Provides functionality to fetch market data from Bloomberg API.
Includes fallback to local Parquet/CSV files for testing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common import get_data_dir, get_logger, load_yaml, save_parquet

logger = get_logger(__name__)


class BloombergDataFetcher:
    """
    Bloomberg API data fetcher with local file fallback.

    Supports:
    - Historical price data (BDH)
    - Reference data (BDP)
    - Index constituents (BDS)
    - Intraday data
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8194,
        use_cache: bool = True,
        cache_dir: str | Path | None = None,
    ):
        """
        Initialize Bloomberg data fetcher.

        Args:
            host: Bloomberg API host
            port: Bloomberg API port
            use_cache: Whether to use local cache
            cache_dir: Directory for cached data
        """
        self.host = host
        self.port = port
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else get_data_dir() / "raw"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._session = None
        self._connected = False

    def connect(self) -> bool:
        """
        Connect to Bloomberg API.

        Returns:
            True if connected successfully
        """
        try:
            import blpapi

            session_options = blpapi.SessionOptions()
            session_options.setServerHost(self.host)
            session_options.setServerPort(self.port)

            self._session = blpapi.Session(session_options)

            if not self._session.start():
                logger.error("Failed to start Bloomberg session")
                return False

            if not self._session.openService("//blp/refdata"):
                logger.error("Failed to open Bloomberg refdata service")
                return False

            self._connected = True
            logger.info("Connected to Bloomberg API")
            return True

        except ImportError:
            logger.warning("blpapi not installed, using local data fallback")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Bloomberg: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Bloomberg API."""
        if self._session:
            self._session.stop()
            self._connected = False
            logger.info("Disconnected from Bloomberg API")

    def _get_cache_path(self, data_type: str, identifier: str) -> Path:
        """Get cache file path."""
        safe_id = identifier.replace(" ", "_").replace("/", "_")
        return self.cache_dir / f"{data_type}_{safe_id}.parquet"

    def _load_from_cache(
        self,
        data_type: str,
        identifier: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """Load data from cache if available."""
        cache_path = self._get_cache_path(data_type, identifier)

        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)

            if start_date and "date" in df.columns:
                df = df[df["date"] >= pd.Timestamp(start_date)]
            if end_date and "date" in df.columns:
                df = df[df["date"] <= pd.Timestamp(end_date)]

            logger.debug(f"Loaded {len(df)} rows from cache: {cache_path}")
            return df

        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None

    def _save_to_cache(
        self,
        df: pd.DataFrame,
        data_type: str,
        identifier: str,
    ) -> None:
        """Save data to cache."""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(data_type, identifier)

        try:
            # Merge with existing cache if present
            if cache_path.exists():
                existing = pd.read_parquet(cache_path)
                if "date" in df.columns and "date" in existing.columns:
                    df = pd.concat([existing, df]).drop_duplicates(
                        subset=["date", "asset_id"] if "asset_id" in df.columns else ["date"]
                    )
                    df = df.sort_values("date")

            save_parquet(df, cache_path)
            logger.debug(f"Saved {len(df)} rows to cache: {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def fetch_historical_data(
        self,
        tickers: list[str],
        fields: list[str],
        start_date: str,
        end_date: str,
        periodicity: str = "DAILY",
    ) -> pd.DataFrame:
        """
        Fetch historical data from Bloomberg (BDH).

        Args:
            tickers: List of Bloomberg tickers
            fields: List of fields to fetch (e.g., ['PX_LAST', 'PX_VOLUME'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            periodicity: Data frequency (DAILY, WEEKLY, MONTHLY)

        Returns:
            DataFrame with columns [date, asset_id, field1, field2, ...]
        """
        # Try cache first
        if self.use_cache:
            cache_key = f"bdh_{hash(tuple(sorted(tickers)))}_{hash(tuple(sorted(fields)))}"
            cached = self._load_from_cache("bdh", cache_key, start_date, end_date)
            if cached is not None and len(cached) > 0:
                return cached

        # Try Bloomberg API
        if self._connected:
            try:
                df = self._fetch_bdh_api(tickers, fields, start_date, end_date, periodicity)
                if df is not None and len(df) > 0:
                    self._save_to_cache(df, "bdh", cache_key)
                    return df
            except Exception as e:
                logger.error(f"Bloomberg API error: {e}")

        # Fallback to local files
        logger.info("Using local file fallback for historical data")
        return self._fetch_bdh_local(tickers, fields, start_date, end_date)

    def _fetch_bdh_api(
        self,
        tickers: list[str],
        fields: list[str],
        start_date: str,
        end_date: str,
        periodicity: str,
    ) -> pd.DataFrame:
        """Fetch historical data using Bloomberg API."""
        import blpapi

        service = self._session.getService("//blp/refdata")
        request = service.createRequest("HistoricalDataRequest")

        for ticker in tickers:
            request.getElement("securities").appendValue(ticker)
        for field in fields:
            request.getElement("fields").appendValue(field)

        request.set("periodicitySelection", periodicity)
        request.set("startDate", start_date.replace("-", ""))
        request.set("endDate", end_date.replace("-", ""))

        self._session.sendRequest(request)

        all_data = []

        while True:
            event = self._session.nextEvent(500)

            for msg in event:
                if msg.hasElement("securityData"):
                    security_data = msg.getElement("securityData")
                    ticker = security_data.getElementAsString("security")
                    field_data = security_data.getElement("fieldData")

                    for i in range(field_data.numValues()):
                        row = field_data.getValueAsElement(i)
                        row_data = {"asset_id": ticker}

                        for field in fields + ["date"]:
                            if row.hasElement(field):
                                row_data[field.lower()] = row.getElementValue(field)

                        all_data.append(row_data)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        df = pd.DataFrame(all_data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df

    def _fetch_bdh_local(
        self,
        tickers: list[str],
        fields: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical data from local files."""
        all_data = []

        for ticker in tickers:
            cache_path = self._get_cache_path("ticker", ticker)

            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                df["asset_id"] = ticker
                all_data.append(df)
            else:
                logger.warning(f"No local data for ticker: {ticker}")

        if not all_data:
            # Generate synthetic data for testing
            logger.warning("No local data found, generating synthetic data")
            return self._generate_synthetic_data(tickers, fields, start_date, end_date)

        df = pd.concat(all_data, ignore_index=True)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

        return df

    def _generate_synthetic_data(
        self,
        tickers: list[str],
        fields: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Generate synthetic data for testing."""
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        all_data = []

        np.random.seed(42)

        for ticker in tickers:
            # Generate random walk price
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = 10000 * np.exp(np.cumsum(returns))

            for i, date in enumerate(dates):
                row = {
                    "date": date,
                    "asset_id": ticker,
                }

                for field in fields:
                    field_lower = field.lower()
                    if "last" in field_lower or "close" in field_lower:
                        row[field_lower] = prices[i]
                    elif "open" in field_lower:
                        row[field_lower] = prices[i] * (1 + np.random.uniform(-0.01, 0.01))
                    elif "high" in field_lower:
                        row[field_lower] = prices[i] * (1 + np.random.uniform(0, 0.02))
                    elif "low" in field_lower:
                        row[field_lower] = prices[i] * (1 - np.random.uniform(0, 0.02))
                    elif "volume" in field_lower:
                        row[field_lower] = np.random.randint(100000, 10000000)
                    elif "turnover" in field_lower:
                        row[field_lower] = prices[i] * np.random.randint(100000, 10000000)
                    else:
                        row[field_lower] = np.random.uniform(0.5, 2.0)

                all_data.append(row)

        return pd.DataFrame(all_data)

    def fetch_reference_data(
        self,
        tickers: list[str],
        fields: list[str],
    ) -> pd.DataFrame:
        """
        Fetch reference data from Bloomberg (BDP).

        Args:
            tickers: List of Bloomberg tickers
            fields: List of fields to fetch

        Returns:
            DataFrame with columns [asset_id, field1, field2, ...]
        """
        # Try cache first
        if self.use_cache:
            cache_key = f"bdp_{hash(tuple(sorted(tickers)))}_{hash(tuple(sorted(fields)))}"
            cached = self._load_from_cache("bdp", cache_key)
            if cached is not None and len(cached) > 0:
                return cached

        # Try Bloomberg API
        if self._connected:
            try:
                df = self._fetch_bdp_api(tickers, fields)
                if df is not None and len(df) > 0:
                    self._save_to_cache(df, "bdp", cache_key)
                    return df
            except Exception as e:
                logger.error(f"Bloomberg API error: {e}")

        # Generate synthetic data
        logger.warning("Generating synthetic reference data")
        return self._generate_synthetic_reference_data(tickers, fields)

    def _fetch_bdp_api(
        self,
        tickers: list[str],
        fields: list[str],
    ) -> pd.DataFrame:
        """Fetch reference data using Bloomberg API."""
        import blpapi

        service = self._session.getService("//blp/refdata")
        request = service.createRequest("ReferenceDataRequest")

        for ticker in tickers:
            request.getElement("securities").appendValue(ticker)
        for field in fields:
            request.getElement("fields").appendValue(field)

        self._session.sendRequest(request)

        all_data = []

        while True:
            event = self._session.nextEvent(500)

            for msg in event:
                if msg.hasElement("securityData"):
                    security_data = msg.getElement("securityData")

                    for i in range(security_data.numValues()):
                        security = security_data.getValueAsElement(i)
                        ticker = security.getElementAsString("security")
                        field_data = security.getElement("fieldData")

                        row_data = {"asset_id": ticker}
                        for field in fields:
                            if field_data.hasElement(field):
                                row_data[field.lower()] = field_data.getElementValue(field)

                        all_data.append(row_data)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        return pd.DataFrame(all_data)

    def _generate_synthetic_reference_data(
        self,
        tickers: list[str],
        fields: list[str],
    ) -> pd.DataFrame:
        """Generate synthetic reference data."""
        np.random.seed(42)

        all_data = []

        for ticker in tickers:
            row = {"asset_id": ticker}

            for field in fields:
                field_lower = field.lower()
                if "book" in field_lower or "pbr" in field_lower:
                    row[field_lower] = np.random.uniform(0.5, 5.0)
                elif "pe" in field_lower or "earning" in field_lower:
                    row[field_lower] = np.random.uniform(5, 50)
                elif "roe" in field_lower:
                    row[field_lower] = np.random.uniform(0.05, 0.30)
                elif "debt" in field_lower:
                    row[field_lower] = np.random.uniform(0.1, 2.0)
                elif "cap" in field_lower:
                    row[field_lower] = np.random.uniform(1e11, 1e13)  # 100B to 10T KRW
                elif "share" in field_lower or "out" in field_lower:
                    row[field_lower] = np.random.randint(1e7, 1e9)
                elif "foreign" in field_lower:
                    row[field_lower] = np.random.uniform(0.1, 0.6)
                elif "inst" in field_lower:
                    row[field_lower] = np.random.uniform(0.2, 0.7)
                else:
                    row[field_lower] = np.random.uniform(0, 100)

            all_data.append(row)

        return pd.DataFrame(all_data)

    def fetch_index_members(
        self,
        index_ticker: str,
        date: str | None = None,
    ) -> list[str]:
        """
        Fetch index constituent members.

        Args:
            index_ticker: Bloomberg index ticker
            date: As-of date (None = current)

        Returns:
            List of member tickers
        """
        # Try cache
        cache_key = f"members_{index_ticker}_{date or 'current'}"
        cache_path = self._get_cache_path("members", cache_key)

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df["asset_id"].tolist()

        # Try Bloomberg API
        if self._connected:
            try:
                members = self._fetch_index_members_api(index_ticker, date)
                if members:
                    df = pd.DataFrame({"asset_id": members})
                    save_parquet(df, cache_path)
                    return members
            except Exception as e:
                logger.error(f"Bloomberg API error: {e}")

        # Return synthetic members
        logger.warning(f"Generating synthetic index members for {index_ticker}")
        return self._generate_synthetic_members(index_ticker)

    def _fetch_index_members_api(
        self,
        index_ticker: str,
        date: str | None = None,
    ) -> list[str]:
        """Fetch index members using Bloomberg API."""
        import blpapi

        service = self._session.getService("//blp/refdata")
        request = service.createRequest("ReferenceDataRequest")

        request.getElement("securities").appendValue(index_ticker)
        request.getElement("fields").appendValue("INDX_MEMBERS")

        if date:
            overrides = request.getElement("overrides")
            override = overrides.appendElement()
            override.setElement("fieldId", "END_DATE_OVERRIDE")
            override.setElement("value", date.replace("-", ""))

        self._session.sendRequest(request)

        members = []

        while True:
            event = self._session.nextEvent(500)

            for msg in event:
                if msg.hasElement("securityData"):
                    security_data = msg.getElement("securityData")
                    for i in range(security_data.numValues()):
                        security = security_data.getValueAsElement(i)
                        field_data = security.getElement("fieldData")
                        if field_data.hasElement("INDX_MEMBERS"):
                            member_data = field_data.getElement("INDX_MEMBERS")
                            for j in range(member_data.numValues()):
                                member = member_data.getValueAsElement(j)
                                members.append(member.getElementAsString("Member Ticker and Exchange Code"))

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        return members

    def _generate_synthetic_members(self, index_ticker: str) -> list[str]:
        """Generate synthetic index members."""
        # Sample Korean stock tickers
        kospi200_samples = [
            "005930 KS Equity",  # Samsung Electronics
            "000660 KS Equity",  # SK Hynix
            "035420 KS Equity",  # NAVER
            "005380 KS Equity",  # Hyundai Motor
            "051910 KS Equity",  # LG Chem
            "006400 KS Equity",  # Samsung SDI
            "035720 KS Equity",  # Kakao
            "000270 KS Equity",  # Kia
            "028260 KS Equity",  # Samsung C&T
            "105560 KS Equity",  # KB Financial
            "055550 KS Equity",  # Shinhan Financial
            "012330 KS Equity",  # Hyundai Mobis
            "066570 KS Equity",  # LG Electronics
            "003550 KS Equity",  # LG
            "017670 KS Equity",  # SK Telecom
            "018260 KS Equity",  # Samsung SDS
            "034730 KS Equity",  # SK
            "096770 KS Equity",  # SK Innovation
            "032830 KS Equity",  # Samsung Life
            "086790 KS Equity",  # Hana Financial
        ]

        if "kospi" in index_ticker.lower() or "200" in index_ticker:
            return kospi200_samples
        else:
            return kospi200_samples[:10]


class DataLoader:
    """
    High-level data loader that combines multiple data sources.

    Provides a unified interface for loading price, fundamental, and flow data.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        fetcher: BloombergDataFetcher | None = None,
    ):
        """
        Initialize data loader.

        Args:
            config_path: Path to universe configuration
            fetcher: Bloomberg data fetcher instance
        """
        self.fetcher = fetcher or BloombergDataFetcher()
        self.config = None

        if config_path:
            self.config = load_yaml(config_path)

    def load_price_data(
        self,
        tickers: list[str] | None = None,
        start_date: str = "2018-01-01",
        end_date: str | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load price data for specified tickers.

        Args:
            tickers: List of tickers (None = use config universe)
            start_date: Start date
            end_date: End date (None = today)
            fields: Price fields to fetch

        Returns:
            DataFrame with columns [date, asset_id, open, high, low, close, volume, ...]
        """
        if tickers is None and self.config:
            tickers = self._get_universe_tickers()

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if fields is None:
            fields = ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_VOLUME", "TURNOVER"]

        df = self.fetcher.fetch_historical_data(
            tickers=tickers,
            fields=fields,
            start_date=start_date,
            end_date=end_date,
        )

        # Standardize column names
        column_map = {
            "px_last": "close",
            "px_open": "open",
            "px_high": "high",
            "px_low": "low",
            "px_volume": "volume",
        }

        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        return df

    def load_fundamental_data(
        self,
        tickers: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load fundamental data for specified tickers.

        Args:
            tickers: List of tickers (None = use config universe)
            fields: Fundamental fields to fetch

        Returns:
            DataFrame with columns [asset_id, pbr, per, roe, ...]
        """
        if tickers is None and self.config:
            tickers = self._get_universe_tickers()

        if fields is None:
            fields = [
                "PX_TO_BOOK_RATIO",
                "PE_RATIO",
                "DVD_PAYOUT_RATIO",
                "RETURN_COM_EQY",
                "TOT_DEBT_TO_TOT_EQY",
                "CUR_MKT_CAP",
            ]

        df = self.fetcher.fetch_reference_data(
            tickers=tickers,
            fields=fields,
        )

        # Standardize column names
        column_map = {
            "px_to_book_ratio": "pbr",
            "pe_ratio": "per",
            "dvd_payout_ratio": "dividend_yield",
            "return_com_eqy": "roe",
            "tot_debt_to_tot_eqy": "debt_to_equity",
            "cur_mkt_cap": "market_cap",
        }

        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        return df

    def load_index_data(
        self,
        index_ticker: str = "KOSPI2 Index",
        start_date: str = "2018-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Load index price data.

        Args:
            index_ticker: Bloomberg index ticker
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with index prices
        """
        return self.load_price_data(
            tickers=[index_ticker],
            start_date=start_date,
            end_date=end_date,
        )

    def _get_universe_tickers(self) -> list[str]:
        """Get tickers from configuration."""
        tickers = []

        if self.config and "universe" in self.config:
            universe = self.config["universe"]

            # Get index members
            if "benchmarks" in universe:
                for name, ticker in universe["benchmarks"].items():
                    members = self.fetcher.fetch_index_members(ticker)
                    tickers.extend(members)

            # Add ETFs
            if "etfs" in universe:
                for etf in universe["etfs"]:
                    tickers.append(etf["ticker"])

        return list(set(tickers))


# Convenience function
def create_data_fetcher(
    use_bloomberg: bool = True,
    cache_dir: str | Path | None = None,
) -> BloombergDataFetcher:
    """
    Create and initialize a data fetcher.

    Args:
        use_bloomberg: Whether to try connecting to Bloomberg
        cache_dir: Cache directory

    Returns:
        Initialized BloombergDataFetcher
    """
    fetcher = BloombergDataFetcher(cache_dir=cache_dir)

    if use_bloomberg:
        fetcher.connect()

    return fetcher
