#!/usr/bin/env python
"""
Bloomberg Data Fetch Script

Directly fetch data from Bloomberg without complex imports.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd
try:
    import blpapi
except ImportError:
    blpapi = None  # type: ignore[assignment]

# Bloomberg session setup
if blpapi is not None:
    SESSION_OPTIONS = blpapi.SessionOptions()
    SESSION_OPTIONS.setServerHost("localhost")
    SESSION_OPTIONS.setServerPort(8194)
else:
    SESSION_OPTIONS = None

DEFAULT_MIN_MARKET_CAP = 1e11  # KRW 100 billion
DEFAULT_MIN_TURNOVER = 1e8     # KRW 100 million


def _extract_ticker_code(text: str) -> str:
    """Extract 6-digit Korean stock code from any ticker text."""
    match = re.search(r"(\d{6})", str(text or ""))
    return match.group(1) if match else ""


def _infer_market_from_text(text: str) -> str:
    """Infer KOSPI/KOSDAQ from ticker text."""
    upper = str(text or "").upper()
    if " KQ" in upper or "KOSDAQ" in upper:
        return "KOSDAQ"
    return "KOSPI"


def _to_bbg_equity_ticker(code: str, market: str) -> str:
    """Convert code + market to Bloomberg equity ticker."""
    suffix = "KQ" if market == "KOSDAQ" else "KS"
    return f"{code} {suffix} Equity"


def _normalize_member_tickers(members: list[str]) -> list[str]:
    """Normalize index member strings to Bloomberg equity ticker format."""
    normalized = []
    seen: set[tuple[str, str]] = set()

    for raw in members:
        code = _extract_ticker_code(raw)
        if not code:
            continue
        market = _infer_market_from_text(raw)
        key = (code, market)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(_to_bbg_equity_ticker(code, market))

    return normalized


def _resolve_market_cap_threshold(min_market_cap: float, caps: pd.Series) -> float:
    """
    Resolve market-cap threshold scale.

    Some local datasets store market cap in million KRW, so we adapt the threshold
    if the observed scale is clearly smaller than KRW units.
    """
    clean_caps = pd.to_numeric(caps, errors="coerce").dropna()
    clean_caps = clean_caps[clean_caps > 0]
    if clean_caps.empty:
        return float(min_market_cap)

    p95 = float(clean_caps.quantile(0.95))
    if min_market_cap >= 1e10 and p95 < (min_market_cap / 1_000):
        return float(min_market_cap) / 1_000_000

    return float(min_market_cap)


def build_local_dynamic_universe(
    local_prices_path: Path,
    n_stocks: int,
    min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
    min_turnover: float = DEFAULT_MIN_TURNOVER,
) -> list[str]:
    """
    Build KOSPI+KOSDAQ universe from local prices.parquet with market-cap filter.
    """
    if not local_prices_path.exists():
        print(f"Local prices file not found: {local_prices_path}")
        return []

    print(f"Building dynamic universe from local prices: {local_prices_path}")
    prices = pd.read_parquet(local_prices_path)
    if prices.empty or "date" not in prices.columns or "ticker" not in prices.columns:
        print("Local prices schema is invalid or empty")
        return []

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date", "ticker"])
    if prices.empty:
        return []

    latest = prices[prices["date"] == prices["date"].max()].copy()
    latest["ticker_code"] = latest["ticker"].map(_extract_ticker_code)
    latest = latest[latest["ticker_code"] != ""]
    latest["market"] = latest["ticker"].map(_infer_market_from_text)

    market_cap = pd.to_numeric(latest.get("market_cap"), errors="coerce")
    market_cap_fund = pd.to_numeric(latest.get("market_cap_fund"), errors="coerce")
    latest["market_cap_resolved"] = market_cap.fillna(market_cap_fund)

    if latest["market_cap_resolved"].notna().any():
        cap_threshold = _resolve_market_cap_threshold(
            min_market_cap, latest["market_cap_resolved"]
        )
        latest = latest[latest["market_cap_resolved"] >= cap_threshold]

    if min_turnover > 0 and "turnover" in latest.columns:
        turnover = pd.to_numeric(latest["turnover"], errors="coerce")
        if turnover.notna().any():
            latest = latest[turnover >= float(min_turnover)]

    latest = latest.sort_values(
        ["market_cap_resolved", "turnover", "volume"],
        ascending=False,
        na_position="last",
    )
    latest = latest.drop_duplicates(subset=["ticker_code"], keep="first")
    latest = latest.head(n_stocks)

    market_mix = latest["market"].value_counts().to_dict()
    print(
        f"Dynamic universe selected: {len(latest)} tickers, "
        f"market mix={market_mix}"
    )

    return [
        _to_bbg_equity_ticker(code=row["ticker_code"], market=row["market"])
        for _, row in latest.iterrows()
    ]


def create_session():
    """Create Bloomberg session."""
    if blpapi is None or SESSION_OPTIONS is None:
        raise ModuleNotFoundError(
            "blpapi is not installed. Install Bloomberg Python API before fetching data."
        )

    session = blpapi.Session(SESSION_OPTIONS)
    if not session.start():
        raise ConnectionError("Failed to start Bloomberg session")
    if not session.openService("//blp/refdata"):
        raise ConnectionError("Failed to open //blp/refdata service")
    return session


def fetch_historical_data(
    session: blpapi.Session,
    tickers: list[str],
    fields: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch historical data from Bloomberg.
    """
    refdata_service = session.getService("//blp/refdata")
    request = refdata_service.createRequest("HistoricalDataRequest")

    for ticker in tickers:
        request.getElement("securities").appendValue(ticker)

    for field in fields:
        request.getElement("fields").appendValue(field)

    request.set("periodicitySelection", "DAILY")
    request.set("startDate", start_date.replace("-", ""))
    request.set("endDate", end_date.replace("-", ""))
    request.set("nonTradingDayFillOption", "ACTIVE_DAYS_ONLY")

    session.sendRequest(request)

    records = []

    while True:
        event = session.nextEvent(5000)

        for msg in event:
            if msg.hasElement("securityData"):
                security_data = msg.getElement("securityData")
                ticker = security_data.getElementAsString("security")

                # Clean up ticker name
                clean_ticker = _extract_ticker_code(ticker)
                market = _infer_market_from_text(ticker)

                if security_data.hasElement("fieldData"):
                    field_data_array = security_data.getElement("fieldData")

                    for i in range(field_data_array.numValues()):
                        field_data = field_data_array.getValueAsElement(i)

                        record = {
                            "ticker": clean_ticker,
                            "market": market,
                            "date": pd.Timestamp(field_data.getElementAsDatetime("date")),
                        }

                        for field in fields:
                            try:
                                if field_data.hasElement(field):
                                    record[field] = field_data.getElementAsFloat(field)
                            except:
                                pass

                        records.append(record)

        if event.eventType() == blpapi.Event.RESPONSE:
            break

    df = pd.DataFrame(records)
    return df


def get_index_members_bds(session: blpapi.Session, index_ticker: str) -> list[str]:
    """Get index members using BDS request."""
    refdata_service = session.getService("//blp/refdata")
    request = refdata_service.createRequest("ReferenceDataRequest")

    request.getElement("securities").appendValue(index_ticker)
    request.getElement("fields").appendValue("INDX_MEMBERS")

    session.sendRequest(request)

    members = []

    while True:
        event = session.nextEvent(5000)

        for msg in event:
            print(f"  Message: {msg.messageType()}")

            if msg.hasElement("securityData"):
                security_data_array = msg.getElement("securityData")

                for i in range(security_data_array.numValues()):
                    security_data = security_data_array.getValueAsElement(i)

                    if security_data.hasElement("fieldData"):
                        field_data = security_data.getElement("fieldData")

                        if field_data.hasElement("INDX_MEMBERS"):
                            members_data = field_data.getElement("INDX_MEMBERS")
                            print(f"  Found {members_data.numValues()} members")

                            for j in range(members_data.numValues()):
                                try:
                                    member = members_data.getValueAsElement(j)
                                    # Try different field names
                                    for field_name in ["Member Ticker and Exchange Code", "Ticker", "MEMBER_TICKER_AND_EXCH_CODE"]:
                                        if member.hasElement(field_name):
                                            ticker = member.getElementAsString(field_name)
                                            members.append(ticker)
                                            break
                                except Exception as e:
                                    print(f"  Error parsing member {j}: {e}")

        if event.eventType() == blpapi.Event.RESPONSE:
            break

    return members


def generate_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Generate features from price data."""
    feature_records = []

    for ticker in prices_df["ticker"].unique():
        ticker_data = prices_df[prices_df["ticker"] == ticker].sort_values("date")

        if len(ticker_data) < 60:
            continue

        closes = ticker_data["PX_LAST"].values
        dates = ticker_data["date"].values

        for i in range(60, len(ticker_data)):
            window_close = closes[i - 60:i + 1]

            # Avoid division by zero
            if closes[i - 5] <= 0 or closes[i - 21] <= 0 or closes[i - 60] <= 0:
                continue

            ret_5d = (closes[i] - closes[i - 5]) / closes[i - 5]
            ret_21d = (closes[i] - closes[i - 21]) / closes[i - 21]
            ret_60d = (closes[i] - closes[i - 60]) / closes[i - 60]

            # Volatility
            try:
                log_returns_5 = np.diff(np.log(window_close[-6:]))
                vol_5d = np.std(log_returns_5) if len(log_returns_5) > 0 else 0
                log_returns_21 = np.diff(np.log(window_close[-22:]))
                vol_21d = np.std(log_returns_21) if len(log_returns_21) > 0 else 0
            except:
                vol_5d = 0
                vol_21d = 0

            sma_5 = np.mean(window_close[-5:])
            sma_21 = np.mean(window_close[-21:])
            sma_60 = np.mean(window_close)

            # RSI
            price_changes = np.diff(window_close[-15:])
            gains = np.maximum(price_changes, 0)
            losses = np.maximum(-price_changes, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
            rsi = 100 - 100 / (1 + avg_gain / max(avg_loss, 1e-10))

            feature_records.append({
                "date": dates[i],
                "ticker": ticker,
                "ret_5d": ret_5d,
                "ret_21d": ret_21d,
                "ret_60d": ret_60d,
                "vol_5d": vol_5d,
                "vol_21d": vol_21d,
                "close_to_sma5": closes[i] / sma_5 - 1 if sma_5 > 0 else 0,
                "close_to_sma21": closes[i] / sma_21 - 1 if sma_21 > 0 else 0,
                "close_to_sma60": closes[i] / sma_60 - 1 if sma_60 > 0 else 0,
                "rsi_14": rsi,
            })

    return pd.DataFrame(feature_records)


def generate_labels(prices_df: pd.DataFrame, forward_days: int = 21) -> pd.DataFrame:
    """Generate labels from price data."""
    label_records = []

    for ticker in prices_df["ticker"].unique():
        ticker_data = prices_df[prices_df["ticker"] == ticker].sort_values("date")

        if len(ticker_data) < forward_days + 60:
            continue

        closes = ticker_data["PX_LAST"].values
        dates = ticker_data["date"].values

        for i in range(60, len(ticker_data) - forward_days):
            if closes[i] <= 0:
                continue
            fwd_return = (closes[i + forward_days] - closes[i]) / closes[i]

            label_records.append({
                "date": dates[i],
                "ticker": ticker,
                "y_reg": fwd_return,
                "y_cls": 1 if fwd_return > 0 else 0,
            })

    return pd.DataFrame(label_records)


def get_member_universe_from_indexes(
    session: blpapi.Session,
    index_tickers: list[str],
) -> list[str]:
    """Fetch and normalize member universe from one or more Bloomberg indexes."""
    members_raw: list[str] = []

    for index_ticker in index_tickers:
        print(f"Trying to fetch {index_ticker} members...")
        index_members = get_index_members_bds(session, index_ticker)
        print(f"  {index_ticker}: {len(index_members)} raw members")
        members_raw.extend(index_members)

    normalized = _normalize_member_tickers(members_raw)
    print(f"Combined normalized members: {len(normalized)}")
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Fetch Bloomberg data")
    parser.add_argument(
        "--index",
        type=str,
        default="KOSPI Index,KOSDAQ Index",
        help='Comma-separated index tickers (e.g., "KOSPI Index,KOSDAQ Index")',
    )
    parser.add_argument("--n-stocks", type=int, default=100, help="Number of stocks")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start date")
    parser.add_argument("--end-date", type=str, default="2024-12-31", help="End date")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument(
        "--local-prices",
        type=str,
        default="data/processed/prices.parquet",
        help="Local prices parquet for dynamic universe fallback/filter",
    )
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=DEFAULT_MIN_MARKET_CAP,
        help="Minimum market cap (KRW basis; auto-scaled if local data uses smaller units)",
    )
    parser.add_argument(
        "--min-turnover",
        type=float,
        default=DEFAULT_MIN_TURNOVER,
        help="Minimum turnover for local dynamic universe filter",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Bloomberg Data Fetch")
    print(f"Period: {args.start_date} to {args.end_date}")
    print("=" * 60)

    # Connect to Bloomberg
    print("Connecting to Bloomberg...")
    session = create_session()
    print("Connected.")

    # 1) Try index-member universe first (KOSPI + KOSDAQ)
    index_list = [s.strip() for s in args.index.split(",") if s.strip()]
    members = get_member_universe_from_indexes(session, index_list) if index_list else []

    # 2) Build local dynamic universe as fallback and/or supplement
    dynamic_local = build_local_dynamic_universe(
        local_prices_path=Path(args.local_prices),
        n_stocks=max(args.n_stocks * 2, args.n_stocks),
        min_market_cap=args.min_market_cap,
        min_turnover=args.min_turnover,
    )

    if members:
        print("Using index member universe as primary source")
        tickers = members[:args.n_stocks]

        # Fill shortage with local dynamic universe
        if len(tickers) < args.n_stocks:
            seen_codes = {_extract_ticker_code(t) for t in tickers}
            for t in dynamic_local:
                code = _extract_ticker_code(t)
                if code in seen_codes:
                    continue
                tickers.append(t)
                seen_codes.add(code)
                if len(tickers) >= args.n_stocks:
                    break
    else:
        print("No index members found. Falling back to local dynamic universe.")
        tickers = dynamic_local[:args.n_stocks]

    if not tickers:
        session.stop()
        raise RuntimeError("No tickers resolved. Check Bloomberg index symbols or local prices file.")

    print(f"Will fetch data for {len(tickers)} tickers")
    print(f"First 5 tickers: {tickers[:5]}")

    # Fetch price data
    print(f"\nFetching price data...")

    price_fields = ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_VOLUME"]

    # Fetch in batches
    batch_size = 10
    all_prices = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size}: {batch[0][:10]}...")

        try:
            batch_prices = fetch_historical_data(
                session, batch, price_fields,
                args.start_date, args.end_date
            )
            if not batch_prices.empty:
                all_prices.append(batch_prices)
                print(f"    Fetched {len(batch_prices)} records")
            else:
                print(f"    No data for this batch")
        except Exception as e:
            print(f"    Error: {e}")

    if not all_prices:
        print("\nNo price data fetched! Check Bloomberg connection.")
        session.stop()
        return

    prices_df = pd.concat(all_prices, ignore_index=True)
    print(f"\nTotal price records: {len(prices_df)}")
    print(f"Unique tickers: {prices_df['ticker'].nunique()}")
    print(f"Date range: {prices_df['date'].min()} to {prices_df['date'].max()}")

    # Add close/open columns
    if "PX_LAST" in prices_df.columns:
        prices_df["close"] = prices_df["PX_LAST"]
    if "PX_OPEN" in prices_df.columns:
        prices_df["open"] = prices_df["PX_OPEN"]
    else:
        prices_df["open"] = prices_df.get("close", prices_df["PX_LAST"])

    # Generate features
    print("\nGenerating features...")
    features_df = generate_features(prices_df)
    print(f"Generated {len(features_df)} feature records")

    # Generate labels
    print("Generating labels...")
    labels_df = generate_labels(prices_df)
    print(f"Generated {len(labels_df)} labels")

    # Save data
    print("\nSaving data...")

    prices_df.to_parquet(output_dir / "prices.parquet", index=False)
    features_df.to_parquet(output_dir / "features.parquet", index=False)
    labels_df.to_parquet(output_dir / "labels.parquet", index=False)

    # Save summary
    summary = {
        "run_date": datetime.now().isoformat(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_tickers": prices_df["ticker"].nunique(),
        "n_price_records": len(prices_df),
        "n_feature_records": len(features_df),
        "n_label_records": len(labels_df),
        "tickers": list(prices_df["ticker"].unique()),
    }

    import yaml
    with open(output_dir / "etl_summary.yaml", "w") as f:
        yaml.dump(summary, f)

    session.stop()

    print("=" * 60)
    print("Data fetch complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - prices.parquet: {len(prices_df)} records")
    print(f"  - features.parquet: {len(features_df)} records")
    print(f"  - labels.parquet: {len(labels_df)} records")
    print("=" * 60)


if __name__ == "__main__":
    main()
