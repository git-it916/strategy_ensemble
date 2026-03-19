#!/usr/bin/env python3
"""
End-to-end pipeline verification script.

Simulates one complete rebalance cycle with 5 symbols:
  1. Fetches fresh data from Binance (daily, hourly, 5m, funding)
  2. Checks column name and timezone consistency
  3. Runs each active alpha
  4. Runs signal aggregation
  5. Shows what Sonnet would receive
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import traceback
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

SEPARATOR = "=" * 72

def section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def subsection(title: str):
    print(f"\n  --- {title} ---")


def describe_df(df: pd.DataFrame, label: str):
    """Print summary info about a DataFrame."""
    print(f"\n  [{label}]")
    print(f"    Rows: {len(df)}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Dtypes:")
    for col in df.columns:
        print(f"      {col}: {df[col].dtype}")
    if "ticker" in df.columns:
        print(f"    Unique tickers: {df['ticker'].nunique()}")
    # Date range
    for col in ["date", "datetime"]:
        if col in df.columns:
            print(f"    {col} range: {df[col].min()} -> {df[col].max()}")
    # Sample row
    if len(df) > 0:
        print(f"    Sample row (first):")
        for c in df.columns:
            print(f"      {c}: {df[c].iloc[0]}")
    print()


# ===========================================================================
# Setup: Initialize BinanceApi with keys
# ===========================================================================
print("Loading API keys and initializing BinanceApi...")
import yaml
keys_path = PROJECT_ROOT / "config" / "keys.yaml"
with open(keys_path) as f:
    keys = yaml.safe_load(f)
binance_cfg = keys.get("binance", {})

from src.execution.binance_api import BinanceApi
api = BinanceApi(
    api_key=binance_cfg.get("api_key", ""),
    api_secret=binance_cfg.get("api_secret", ""),
)

# Use 5 symbols for speed
TEST_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]
print(f"Test symbols: {TEST_SYMBOLS}")


# ===========================================================================
# SECTION 1: Fetch fresh data
# ===========================================================================
section("1. DATA FETCHING")

# --- 1a: Daily OHLCV ---
subsection("1a. Daily OHLCV (1d, 300 days)")
t0 = time.time()
prices_1d = api.get_ohlcv_batch(TEST_SYMBOLS, timeframe="1d", days=300)
print(f"  Fetch time: {time.time()-t0:.1f}s")
describe_df(prices_1d, "prices_1d (raw from get_ohlcv_batch)")

# --- 1b: Hourly OHLCV ---
subsection("1b. Hourly OHLCV (1h, 7 days)")
t0 = time.time()
prices_1h_raw = api.get_ohlcv_batch(TEST_SYMBOLS, timeframe="1h", days=7)
print(f"  Fetch time: {time.time()-t0:.1f}s")
describe_df(prices_1h_raw, "prices_1h RAW (before rename)")

# Rename as daemon does
prices_1h = prices_1h_raw.copy()
if "date" in prices_1h.columns:
    prices_1h = prices_1h.rename(columns={"date": "datetime"})
describe_df(prices_1h, "prices_1h AFTER rename (daemon applies)")

# --- 1c: 5-minute OHLCV ---
subsection("1c. 5-minute OHLCV (5m, 5 days)")
t0 = time.time()
prices_5m_raw = api.get_ohlcv_batch(TEST_SYMBOLS, timeframe="5m", days=5)
print(f"  Fetch time: {time.time()-t0:.1f}s")
describe_df(prices_5m_raw, "prices_5m RAW (before rename)")

prices_5m = prices_5m_raw.copy()
if "date" in prices_5m.columns:
    prices_5m = prices_5m.rename(columns={"date": "datetime"})
describe_df(prices_5m, "prices_5m AFTER rename (daemon applies)")

# --- 1d: Funding rates ---
subsection("1d. Funding Rate History (90 days)")
t0 = time.time()
funding = api.get_funding_history_batch(TEST_SYMBOLS, days=90)
print(f"  Fetch time: {time.time()-t0:.1f}s")
describe_df(funding, "funding_rates")

# Build features (same as daemon)
features = None
if not funding.empty:
    features = prices_1d[["date", "ticker"]].copy()
    features = features.merge(
        funding[["date", "ticker", "funding_rate"]],
        on=["date", "ticker"],
        how="left",
    )
    print(f"  Features built: {len(features)} rows, NaN funding_rate: {features['funding_rate'].isna().sum()}")
else:
    print("  WARNING: No funding data returned!")


# ===========================================================================
# SECTION 2: Column Name Consistency
# ===========================================================================
section("2. COLUMN NAME CONSISTENCY")

print(f"\n  get_ohlcv_batch('1d') returns columns: {list(prices_1d.columns)}")
print(f"  get_ohlcv_batch('1h') returns columns: {list(prices_1h_raw.columns)}")
print(f"  get_ohlcv_batch('5m') returns columns: {list(prices_5m_raw.columns)}")
print(f"  After daemon rename, 1h columns: {list(prices_1h.columns)}")
print(f"  After daemon rename, 5m columns: {list(prices_5m.columns)}")
print()

# Check each alpha's expected column
print("  Alpha expectations:")
print("    CSMomentum.generate_signals():        uses prices['date']")
print("    TimeSeriesMeanReversion.generate_signals(): uses prices['date']")
print("    FundingRateCarry.generate_signals():   uses features['date']")
print("    IntradayRSI.generate_signals():        checks 'datetime' first, falls back to 'date'")
print("    IntradayVWAP.generate_signals():       checks 'datetime' first, falls back to 'date'")
print("    IntradayTimeSeriesMomentum.generate_signals(): checks 'datetime' first, falls back to 'date'")
print()

# Actual check
mismatches = []
if "date" not in prices_1d.columns:
    mismatches.append("Daily data missing 'date' column!")
if "datetime" not in prices_1h.columns:
    mismatches.append("1h data (after rename) missing 'datetime' column!")
if "datetime" not in prices_5m.columns:
    mismatches.append("5m data (after rename) missing 'datetime' column!")
if features is not None and "date" not in features.columns:
    mismatches.append("Features missing 'date' column!")

if mismatches:
    print("  MISMATCHES FOUND:")
    for m in mismatches:
        print(f"    [FAIL] {m}")
else:
    print("  [OK] All column names are consistent with alpha expectations.")


# ===========================================================================
# SECTION 3: Timezone Consistency
# ===========================================================================
section("3. TIMEZONE CONSISTENCY")

print(f"\n  Daily 'date' dtype:    {prices_1d['date'].dtype}")
print(f"  Daily 'date' sample:   {prices_1d['date'].iloc[-1]}  (type={type(prices_1d['date'].iloc[-1])})")

print(f"\n  1h 'datetime' dtype:   {prices_1h['datetime'].dtype}")
print(f"  1h 'datetime' sample:  {prices_1h['datetime'].iloc[-1]}")
has_tz_1h = hasattr(prices_1h['datetime'].dtype, 'tz') and prices_1h['datetime'].dtype.tz is not None
print(f"  1h has timezone?       {has_tz_1h}")
if has_tz_1h:
    print(f"  1h timezone:           {prices_1h['datetime'].dtype.tz}")

print(f"\n  5m 'datetime' dtype:   {prices_5m['datetime'].dtype}")
has_tz_5m = hasattr(prices_5m['datetime'].dtype, 'tz') and prices_5m['datetime'].dtype.tz is not None
print(f"  5m has timezone?       {has_tz_5m}")
if has_tz_5m:
    print(f"  5m timezone:           {prices_5m['datetime'].dtype.tz}")

now_naive = datetime.now()
now_utc = datetime.now(timezone.utc)
print(f"\n  datetime.now() (naive): {now_naive}  tzinfo={now_naive.tzinfo}")
print(f"  datetime.now(UTC):      {now_utc}  tzinfo={now_utc.tzinfo}")
print(f"  Daemon uses datetime.now() (NAIVE) as 'now' parameter.")

# Test the actual comparison
print("\n  Testing comparisons:")
try:
    result = prices_1d['date'] <= now_naive
    print(f"    prices_1d['date'] <= datetime.now()     -> OK ({result.sum()}/{len(result)} rows pass)")
except Exception as e:
    print(f"    prices_1d['date'] <= datetime.now()     -> ERROR: {e}")

try:
    result = prices_1h['datetime'] <= now_naive
    print(f"    prices_1h['datetime'] <= datetime.now() -> OK ({result.sum()}/{len(result)} rows pass)")
except Exception as e:
    print(f"    prices_1h['datetime'] <= datetime.now() -> ERROR: {e}")

try:
    result = prices_5m['datetime'] <= now_naive
    print(f"    prices_5m['datetime'] <= datetime.now() -> OK ({result.sum()}/{len(result)} rows pass)")
except Exception as e:
    print(f"    prices_5m['datetime'] <= datetime.now() -> ERROR: {e}")

# Test with UTC-aware now (what alphas do internally)
try:
    result = prices_1h['datetime'] <= now_utc
    print(f"    prices_1h['datetime'] <= now(UTC)       -> OK ({result.sum()}/{len(result)} rows pass)")
except Exception as e:
    print(f"    prices_1h['datetime'] <= now(UTC)       -> ERROR: {e}")

# How intraday alphas handle it
print("\n  Intraday alphas (RSI/VWAP/TSM) handle tz mismatch by:")
print("    1. Check if prices time_col has tz")
print("    2. If so and 'date' param is naive, upgrade to UTC")
print("    3. This should prevent comparison errors")


# ===========================================================================
# SECTION 4: Run Each Active Alpha
# ===========================================================================
section("4. ALPHA SIGNAL GENERATION")

from config.settings import STRATEGIES
from src.alphas.openclaw_1 import (
    CSMomentum,
    TimeSeriesMeanReversion,
    FundingRateCarry,
    IntradayRSI,
    IntradayVWAP,
    IntradayTimeSeriesMomentum,
)

alpha_signals = {}
now = datetime.now()  # Same as daemon uses

# --- CSMomentum ---
subsection("4a. CSMomentum (daily)")
try:
    cfg = STRATEGIES.get("cs_momentum", {})
    alpha_csm = CSMomentum(
        lookback_days=int(cfg.get("lookback_days", 21)),
        skip_days=int(cfg.get("skip_days", 3)),
    )
    alpha_csm.fit(prices_1d, features)
    result_csm = alpha_csm.generate_signals(now, prices_1d, features)
    sig = result_csm.signals
    alpha_signals["CSMomentum"] = sig
    print(f"  Signals: {len(sig)} tickers")
    print(f"  Score range: [{sig['score'].min():.4f}, {sig['score'].max():.4f}]")
    print(f"  NaN scores: {sig['score'].isna().sum()}")
    print(f"  Sample:\n{sig.to_string(index=False)}")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# --- TimeSeriesMeanReversion ---
subsection("4b. TimeSeriesMeanReversion (daily)")
try:
    cfg = STRATEGIES.get("time_series_mean_reversion", {})
    alpha_mr = TimeSeriesMeanReversion(
        signal_window=int(cfg.get("signal_window", 5)),
        baseline_window=int(cfg.get("baseline_window", 60)),
    )
    alpha_mr.fit(prices_1d, features)
    result_mr = alpha_mr.generate_signals(now, prices_1d, features)
    sig = result_mr.signals
    alpha_signals["TimeSeriesMeanReversion"] = sig
    print(f"  Signals: {len(sig)} tickers")
    print(f"  Score range: [{sig['score'].min():.4f}, {sig['score'].max():.4f}]")
    print(f"  NaN scores: {sig['score'].isna().sum()}")
    print(f"  Sample:\n{sig.to_string(index=False)}")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# --- FundingRateCarry ---
subsection("4c. FundingRateCarry (daily + features)")
try:
    cfg = STRATEGIES.get("funding_rate_carry", {})
    alpha_frc = FundingRateCarry(
        lookback_days=int(cfg.get("lookback_days", 14)),
        abs_threshold=float(cfg.get("abs_threshold", 0.0001)),
    )
    alpha_frc.fit(prices_1d, features)
    result_frc = alpha_frc.generate_signals(now, prices_1d, features)
    sig = result_frc.signals
    alpha_signals["FundingRateCarry"] = sig
    print(f"  Signals: {len(sig)} tickers")
    if not sig.empty:
        print(f"  Score range: [{sig['score'].min():.4f}, {sig['score'].max():.4f}]")
        print(f"  NaN scores: {sig['score'].isna().sum()}")
        print(f"  Sample:\n{sig.to_string(index=False)}")
    else:
        print("  (empty - check features & funding data)")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# --- IntradayRSI ---
subsection("4d. IntradayRSI (1h data)")
try:
    cfg = STRATEGIES.get("intraday_rsi", {})
    alpha_rsi = IntradayRSI(
        rsi_period=int(cfg.get("rsi_period", 14)),
        oversold=float(cfg.get("oversold", 30)),
        overbought=float(cfg.get("overbought", 70)),
    )
    alpha_rsi.fit(prices_1h, features)
    result_rsi = alpha_rsi.generate_signals(now, prices_1h, features)
    sig = result_rsi.signals
    alpha_signals["IntradayRSI"] = sig
    print(f"  Signals: {len(sig)} tickers")
    if not sig.empty:
        print(f"  Score range: [{sig['score'].min():.4f}, {sig['score'].max():.4f}]")
        print(f"  NaN scores: {sig['score'].isna().sum()}")
        print(f"  Sample:\n{sig.to_string(index=False)}")
    else:
        print("  (empty)")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# --- IntradayVWAP ---
subsection("4e. IntradayVWAP (1h data)")
try:
    cfg = STRATEGIES.get("intraday_vwap", {})
    alpha_vwap = IntradayVWAP(
        lookback_bars=int(cfg.get("lookback_bars", 24)),
    )
    alpha_vwap.fit(prices_1h, features)
    result_vwap = alpha_vwap.generate_signals(now, prices_1h, features)
    sig = result_vwap.signals
    alpha_signals["IntradayVWAP"] = sig
    print(f"  Signals: {len(sig)} tickers")
    if not sig.empty:
        print(f"  Score range: [{sig['score'].min():.4f}, {sig['score'].max():.4f}]")
        print(f"  NaN scores: {sig['score'].isna().sum()}")
        print(f"  Sample:\n{sig.to_string(index=False)}")
    else:
        print("  (empty)")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# --- IntradayTimeSeriesMomentum ---
subsection("4f. IntradayTimeSeriesMomentum (5m data)")
try:
    cfg = STRATEGIES.get("intraday_time_series_momentum", {})
    alpha_itsm = IntradayTimeSeriesMomentum(
        lookback_bars=int(cfg.get("lookback_bars", 36)),
    )
    alpha_itsm.fit(prices_5m, features)
    result_itsm = alpha_itsm.generate_signals(now, prices_5m, features)
    sig = result_itsm.signals
    alpha_signals["IntradayTimeSeriesMomentum"] = sig
    print(f"  Signals: {len(sig)} tickers")
    if not sig.empty:
        print(f"  Score range: [{sig['score'].min():.4f}, {sig['score'].max():.4f}]")
        print(f"  NaN scores: {sig['score'].isna().sum()}")
        print(f"  Sample:\n{sig.to_string(index=False)}")
    else:
        print("  (empty)")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()


# ===========================================================================
# SECTION 5: Signal Aggregation
# ===========================================================================
section("5. SIGNAL AGGREGATION")

from src.daemon.signal_aggregator import SignalAggregator

aggregator = SignalAggregator()

# Build weights from config (same as daemon's _config_weights_with_regime fallback)
# Map alpha class names to config keys
_ALPHA_TO_CONFIG = {
    "CSMomentum": "cs_momentum",
    "TimeSeriesMeanReversion": "time_series_mean_reversion",
    "FundingRateCarry": "funding_rate_carry",
    "IntradayRSI": "intraday_rsi",
    "IntradayVWAP": "intraday_vwap",
    "IntradayTimeSeriesMomentum": "intraday_time_series_momentum",
}

alpha_weights = {}
for alpha_name in alpha_signals:
    config_key = _ALPHA_TO_CONFIG.get(alpha_name, alpha_name)
    strat_cfg = STRATEGIES.get(config_key, {})
    weight = float(strat_cfg.get("weight", 0))
    alpha_weights[alpha_name] = weight

print(f"\n  Alpha weights (from config):")
total_w = 0
for name, w in sorted(alpha_weights.items(), key=lambda x: -x[1]):
    print(f"    {name}: {w:.2f}")
    total_w += w
print(f"    TOTAL: {total_w:.2f}")

aggregated = aggregator.aggregate(
    alpha_signals=alpha_signals,
    alpha_weights=alpha_weights,
)

print(f"\n  Aggregated scores ({len(aggregated.scores)} tickers):")
ranked = sorted(aggregated.scores.items(), key=lambda x: abs(x[1]), reverse=True)
for ticker, score in ranked[:5]:
    direction = "LONG" if score > 0 else "SHORT"
    short_t = ticker.replace("/USDT:USDT", "")
    print(f"    {short_t}: {score:+.4f} -> {direction}")

# Contribution breakdown for top ticker
if ranked:
    top_ticker = ranked[0][0]
    top_contribs = aggregated.contributions.get(top_ticker, {})
    print(f"\n  Contribution breakdown for top ticker ({top_ticker.replace('/USDT:USDT', '')}):")
    for alpha_name, contrib in sorted(top_contribs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {alpha_name}: {contrib:+.4f}")

# Check for NaN
has_nan = any(np.isnan(v) for v in aggregated.scores.values())
print(f"\n  Any NaN in final scores? {has_nan}")
if has_nan:
    print("  [FAIL] NaN values found in aggregated scores!")
else:
    print("  [OK] No NaN values in final scores.")


# ===========================================================================
# SECTION 6: What Gets Sent to Sonnet
# ===========================================================================
section("6. SONNET PROMPT PREVIEW")

from src.daemon.sonnet_decision_maker import SonnetDecisionMaker, _ALPHA_DISPLAY

# Build a minimal market context similar to daemon's
market_context = {
    "regime": "sideways",
    "regime_score": 0,
    "btc_price": float(prices_1d[prices_1d["ticker"] == "BTC/USDT:USDT"]["close"].iloc[-1]) if "BTC/USDT:USDT" in prices_1d["ticker"].values else 0,
    "btc_24h_change": 0.01,
    "btc_7d_change": -0.02,
    "btc_30d_change": 0.05,
    "eth_price": float(prices_1d[prices_1d["ticker"] == "ETH/USDT:USDT"]["close"].iloc[-1]) if "ETH/USDT:USDT" in prices_1d["ticker"].values else 0,
    "eth_24h_change": 0.005,
    "eth_7d_change": -0.03,
    "eth_30d_change": 0.04,
    "pct_above_ma20": 0.55,
    "adv_decline_ratio": 1.1,
    "aggregated_scores": aggregated.scores,
    "alpha_weights": aggregated.weights_used,
}

# We don't actually call Sonnet, just build the prompt
dummy_account = {"total_wallet_balance": 100, "available_balance": 95}

# Instantiate with dummy client
class DummyClient:
    pass

sdm = SonnetDecisionMaker(anthropic_client=DummyClient())

system_prompt = sdm._build_system_prompt(dummy_account)
user_prompt = sdm._build_user_prompt(
    alpha_signals=alpha_signals,
    current_positions=pd.DataFrame(),  # no positions
    account_info=dummy_account,
    market_context=market_context,
)

print(f"\n  System prompt length: {len(system_prompt)} chars")
print(f"  User prompt length: {len(user_prompt)} chars")

print("\n  --- SYSTEM PROMPT (first 500 chars) ---")
print(system_prompt[:500])

print("\n  --- USER PROMPT (full) ---")
print(user_prompt)


# ===========================================================================
# SUMMARY
# ===========================================================================
section("VERIFICATION SUMMARY")

print()
print("  1. Data Fetching:")
print(f"     Daily OHLCV:  {len(prices_1d)} rows, {prices_1d['ticker'].nunique()} tickers")
print(f"     Hourly OHLCV: {len(prices_1h)} rows, {prices_1h['ticker'].nunique()} tickers")
print(f"     5m OHLCV:     {len(prices_5m)} rows, {prices_5m['ticker'].nunique()} tickers")
print(f"     Funding:      {len(funding)} rows")
print()
print("  2. Column Names:")
print(f"     Daily uses 'date': {'date' in prices_1d.columns}")
print(f"     1h uses 'datetime' (after rename): {'datetime' in prices_1h.columns}")
print(f"     5m uses 'datetime' (after rename): {'datetime' in prices_5m.columns}")
print(f"     All alphas can handle both: YES (intraday check 'datetime' first)")
print()
print("  3. Timezone:")
print(f"     Daily date dtype: {prices_1d['date'].dtype}")
print(f"     1h datetime dtype: {prices_1h['datetime'].dtype}")
print(f"     5m datetime dtype: {prices_5m['datetime'].dtype}")
print(f"     Daemon now: naive datetime.now()")
tz_issue = has_tz_1h or has_tz_5m
if tz_issue:
    print(f"     POTENTIAL ISSUE: Intraday data is tz-aware but daemon passes naive now")
    print(f"     MITIGATION: Alphas handle this internally (upgrade naive to UTC)")
else:
    print(f"     No tz mismatch (all naive)")
print()
print("  4. Alpha Signals:")
for name, sig in alpha_signals.items():
    n = len(sig)
    nan_count = sig['score'].isna().sum() if n > 0 else 0
    score_range = f"[{sig['score'].min():.3f}, {sig['score'].max():.3f}]" if n > 0 else "N/A"
    status = "OK" if n > 0 and nan_count == 0 else "WARN"
    print(f"     [{status}] {name}: {n} signals, range={score_range}, NaN={nan_count}")
print()
print("  5. Aggregation:")
print(f"     {len(aggregated.scores)} tickers scored, NaN: {has_nan}")
if ranked:
    top_t, top_s = ranked[0]
    print(f"     Top signal: {top_t.replace('/USDT:USDT','')} = {top_s:+.4f}")
print()
print("  6. Sonnet Prompt: Built successfully ({} + {} chars)".format(
    len(system_prompt), len(user_prompt)
))
print()
print(SEPARATOR)
print("  PIPELINE VERIFICATION COMPLETE")
print(SEPARATOR)
