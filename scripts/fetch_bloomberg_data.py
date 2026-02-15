#!/usr/bin/env python
"""
Bloomberg Data Fetch Script

Directly fetch data from Bloomberg without complex imports.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import blpapi

# Bloomberg session setup
SESSION_OPTIONS = blpapi.SessionOptions()
SESSION_OPTIONS.setServerHost("localhost")
SESSION_OPTIONS.setServerPort(8194)


def create_session():
    """Create Bloomberg session."""
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
                clean_ticker = ticker.replace(" KS Equity", "").replace(" Equity", "")

                if security_data.hasElement("fieldData"):
                    field_data_array = security_data.getElement("fieldData")

                    for i in range(field_data_array.numValues()):
                        field_data = field_data_array.getValueAsElement(i)

                        record = {
                            "ticker": clean_ticker,
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


# KOSPI 200 major tickers (Korean stock codes)
KOSPI200_TICKERS = [
    "005930",  # Samsung Electronics
    "000660",  # SK Hynix
    "035420",  # Naver
    "005380",  # Hyundai Motor
    "051910",  # LG Chem
    "006400",  # Samsung SDI
    "035720",  # Kakao
    "068270",  # Celltrion
    "028260",  # Samsung C&T
    "012330",  # Hyundai Mobis
    "055550",  # Shinhan Financial
    "105560",  # KB Financial
    "096770",  # SK Innovation
    "003670",  # Posco Holdings
    "066570",  # LG Electronics
    "086790",  # Hana Financial
    "034730",  # SK
    "017670",  # SK Telecom
    "015760",  # Korea Electric Power
    "032830",  # Samsung Life
    "003550",  # LG
    "018260",  # Samsung SDS
    "000270",  # Kia
    "033780",  # KT&G
    "009150",  # Samsung Electro-Mechanics
    "316140",  # Woori Financial
    "010950",  # S-Oil
    "090430",  # Amorepacific
    "011170",  # Lotte Chem
    "036570",  # NCsoft
    "030200",  # KT
    "000810",  # Samsung Fire
    "034020",  # Doosan Enerbility
    "005490",  # Posco
    "009540",  # Hyundai Heavy
    "267250",  # HD Hyundai
    "010130",  # Korea Zinc
    "018880",  # Hanon Systems
    "047050",  # Daewoo E&C
    "011780",  # Kumho Petrochemical
    "024110",  # Industrial Bank
    "323410",  # Kakao Bank
    "259960",  # Krafton
    "352820",  # Hive
    "003490",  # Korean Air
    "097950",  # CJ CheilJedang
    "000720",  # Hyundai E&C
    "207940",  # Samsung Biologics
    "034220",  # LG Display
    "010140",  # Samsung Heavy
    "021240",  # Woongjin Coway
    "028050",  # Samsung Engineering
    "005830",  # DB Insurance
    "032640",  # LG Uplus
    "011200",  # HMM
    "006800",  # Mirae Asset
    "139480",  # E-Mart
    "161390",  # Hankook Tire
    "001040",  # CJ
    "004020",  # Hyundai Steel
    "088350",  # Hanwha Life
    "009830",  # Hanwha Solution
    "000880",  # Hanwha
    "078930",  # GS
    "036460",  # Korea Gas
    "051900",  # LG H&H
    "004170",  # Shinsegae
    "010620",  # Hyundai Mipo
    "003410",  # Ssangyong C&E
    "006360",  # GS E&C
    "042660",  # Daewoo Shipbuilding
    "002790",  # Amore G
    "004990",  # Lotte
    "000100",  # Yuhan
    "011070",  # LG Innotek
    "271560",  # Orion
    "008770",  # Hotel Shilla
    "003620",  # SK Networks
    "002380",  # KCC
    "004800",  # Hyosung
    "241560",  # Doosan Bobcat
    "120110",  # Kolon Industries
    "329180",  # HD Hyundai Marine
    "069500",  # KODEX 200
    "114090",  # GKL
    "005940",  # NH Investment
    "047810",  # Korea Aerospace
    "010060",  # OCI Holdings
    "071050",  # Korea Investment
    "006280",  # Green Cross
    "128940",  # Hanmi Pharm
    "000150",  # Doosan
    "012450",  # Hanwha Aerospace
    "034730",  # SK Inc
    "000210",  # Daelim Industrial
    "028670",  # Pan Ocean
    "079550",  # LIG Nex1
    "004370",  # Nongshim
    "005850",  # Samsung Card
    "039490",  # Kiwoom Securities
]


def main():
    parser = argparse.ArgumentParser(description="Fetch Bloomberg data")
    parser.add_argument("--index", type=str, default="KOSPI200 Index", help="Index ticker")
    parser.add_argument("--n-stocks", type=int, default=100, help="Number of stocks")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start date")
    parser.add_argument("--end-date", type=str, default="2024-12-31", help="End date")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")

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

    # Try to get index members from Bloomberg
    print(f"Trying to fetch {args.index} members...")
    members = get_index_members_bds(session, args.index)

    if len(members) == 0:
        print("Using predefined KOSPI200 tickers...")
        # Use predefined tickers
        tickers = [f"{t} KS Equity" for t in KOSPI200_TICKERS[:args.n_stocks]]
    else:
        print(f"Found {len(members)} members from Bloomberg")
        tickers = [f"{t} Equity" if " Equity" not in t else t for t in members[:args.n_stocks]]

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
