"""
Sonnet Decision Maker

Calls Claude Sonnet every rebalance cycle to make final trading decisions.
Receives raw alpha signals + market context, returns structured positions
with stop-loss and take-profit levels.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import SONNET_DECISION, TRADING, STRATEGIES

logger = logging.getLogger(__name__)

# Dedicated decision log directory
DECISION_LOG_DIR = Path(__file__).parent.parent.parent / "logs" / "sonnet_decisions"
DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """\
You are a crypto futures portfolio manager for a small account on Binance USDT-M perpetual futures.
You are the PRIMARY decision maker. Use all available data — quantitative signals, derivatives data, price action, funding rates, market regime — to make intelligent trading decisions.

POSITION RULES:
- Maximum {max_positions} simultaneous positions
- Each position weight: {min_weight:.0%} to {max_weight:.0%} of portfolio
- Every position MUST have stop_loss_pct (between {min_sl:.0%} and {max_sl:.0%}) and take_profit_pct (between {min_tp:.0%} and {max_tp:.0%})
- Minimum order notional on Binance is $5. With current balance and {leverage}x leverage, minimum weight is {min_weight_for_notional:.0%}

HOW TO USE SIGNALS:
- The aggregated_score combines 7 alpha signals: momentum (cs_momentum, ts_momentum, intraday_tsm), carry (funding_rate), and mean-reversion brakes (RSI, VWAP, mean_reversion).
- Scores above {entry_threshold:.2f} indicate a potential entry. Use this as a STARTING POINT, not a mandate.
- You MUST cross-check signals against the full context before acting:
  * Funding rates: negative funding = shorts pay longs. Shorting into negative funding means YOU pay carry AND risk a short squeeze.
  * Price momentum: if 24h return is strongly positive and you see a SHORT signal, consider whether it's a squeeze or genuine reversal.
  * OI + L/S ratio: crowded positioning (extreme L/S ratio + high OI) increases squeeze risk.
  * Regime + breadth: if 85% of coins are above MA20, the market is bullish — be cautious with shorts.
  * Volatility: high vol (>100% annualized) means wider SL needed and higher risk of stop-outs.
- You CAN and SHOULD skip or reduce trades when the context contradicts the signal.
- If a ticker was recently stopped out, ask yourself WHY before re-entering the same direction.

RISK SITUATIONS TO WATCH:
- SHORT SQUEEZE: negative funding + rising price + high short OI → avoid shorting
- LONG SQUEEZE: positive funding + falling price + high long OI → avoid longing
- TREND FIGHTING: mean-reversion signals against a strong multi-day trend → usually loses
- REPEATED STOP-OUTS: if a ticker was stopped out recently, the signal may be wrong

SIGNAL REFERENCE:
- Entry consideration: |aggregated_score| >= {entry_threshold:.2f}
- Close/flip consideration: opposite |aggregated_score| >= {reverse_threshold:.2f}
- These are guidelines, not absolute rules. Your judgment overrides when context warrants it.

EXISTING POSITION RULES:
- You MUST include EVERY existing position in your output with action HOLD, CLOSE, or updated LONG/SHORT
- Do NOT omit any existing position — omitted positions become unmanaged
- If a position is profitable, let it run. Do NOT close to switch to another coin
- PATIENCE: hold at least 30-60 minutes unless signals fully reversed
- NEVER close and re-open the same ticker in the same cycle — just HOLD

OUTPUT FORMAT (JSON only, no markdown, no explanation outside JSON):
{{
  "positions": [
    {{"ticker": "BTC/USDT:USDT", "action": "LONG", "weight": 0.25, "stop_loss_pct": -0.05, "take_profit_pct": 0.10, "reasoning": "brief reason"}}
  ],
  "market_assessment": "one line market view",
  "risk_note": "any risk warnings"
}}

action must be one of: LONG, SHORT, CLOSE, HOLD
ticker must use exchange format: SYMBOL/USDT:USDT (e.g. BTC/USDT:USDT)
"""


@dataclass
class PositionDecision:
    """Sonnet's decision for a single position."""
    ticker: str
    action: str              # LONG, SHORT, CLOSE, HOLD
    weight: float            # Portfolio weight (0.0 to 0.30)
    stop_loss_pct: float     # e.g., -0.05 means -5% from entry
    take_profit_pct: float   # e.g., +0.10 means +10% from entry
    reasoning: str


@dataclass
class SonnetDecision:
    """Complete trading decision from Sonnet."""
    positions: list[PositionDecision]
    market_assessment: str
    risk_note: str
    raw_response: str
    timestamp: datetime


# Alpha class name → human-readable config key
_ALPHA_DISPLAY = {
    "CSMomentum": "cs_momentum",
    "TimeSeriesMomentum": "time_series_momentum",
    "TimeSeriesMeanReversion": "time_series_mean_reversion",
    "PriceVolumeDivergence": "pv_divergence",
    "VolumeMomentum": "volume_momentum",
    "LowVolatilityAnomaly": "low_volatility_anomaly",
    "FundingRateCarry": "funding_rate_carry",
    "RSIReversalAlpha": "rsi_reversal",
    "VolatilityBreakoutAlpha": "vol_breakout",
    "IntradayRSI": "intraday_rsi",
    "IntradayTimeSeriesMomentum": "intraday_time_series_momentum",
    "IntradayVWAP": "intraday_vwap",
}


class SonnetDecisionMaker:
    """
    Calls Claude Sonnet to make portfolio trading decisions.

    Input: raw alpha signals + market context + current positions
    Output: structured PositionDecisions with SL/TP levels
    """

    def __init__(self, anthropic_client, model: str | None = None):
        self.client = anthropic_client
        self.model = model or SONNET_DECISION["model"]
        self._call_count = 0

    def make_decision(
        self,
        alpha_signals: dict[str, pd.DataFrame],
        current_positions: pd.DataFrame | None,
        account_info: dict,
        market_context: dict,
        managed_positions: list | None = None,
        recently_closed: list | None = None,
        features_summary: dict | None = None,
    ) -> SonnetDecision:
        """Call Sonnet and parse structured trading decision."""
        system_prompt = self._build_system_prompt(account_info)
        user_prompt = self._build_user_prompt(
            alpha_signals, current_positions, account_info, market_context,
            managed_positions=managed_positions,
            recently_closed=recently_closed,
            features_summary=features_summary,
        )

        logger.info(f"Calling Sonnet ({self.model})...")

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=SONNET_DECISION.get("max_tokens", 1024),
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=60.0,  # 60 second timeout to prevent daemon hang
            )

            raw = self._extract_text(message)
            self._call_count += 1
            logger.info(f"Sonnet call #{self._call_count} complete")

            data = self._parse_json(raw)
            positions = self._parse_positions(data.get("positions", []))
            positions = self._validate_positions(positions)

            decision = SonnetDecision(
                positions=positions,
                market_assessment=data.get("market_assessment", ""),
                risk_note=data.get("risk_note", ""),
                raw_response=raw,
                timestamp=datetime.now(),
            )

            # Log decision to file
            self._log_decision(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=raw,
                decision=decision,
            )

            return decision

        except Exception as e:
            logger.error(f"Sonnet decision failed: {e}")
            self._log_decision(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response="",
                error=str(e),
            )
            return SonnetDecision(
                positions=[],
                market_assessment="ERROR",
                risk_note=str(e),
                raw_response="",
                timestamp=datetime.now(),
            )

    def _build_system_prompt(self, account_info: dict | None = None) -> str:
        leverage = TRADING.get("max_leverage", 3.0)
        balance = float((account_info or {}).get("total_wallet_balance", 0))
        min_notional = TRADING.get("min_notional_usdt", 5)
        # Dynamic minimum weight based on actual balance
        if balance > 0 and leverage > 0:
            min_weight_for_notional = min_notional / (balance * leverage)
        else:
            min_weight_for_notional = 0.25  # fallback
        return SYSTEM_PROMPT.format(
            max_positions=TRADING.get("max_positions", 3),
            min_weight=TRADING.get("min_position_weight", 0.05),
            max_weight=TRADING.get("max_position_weight", 0.30),
            min_sl=SONNET_DECISION.get("tightest_stop_loss_pct", -0.02),
            max_sl=SONNET_DECISION.get("loosest_stop_loss_pct", -0.08),
            min_tp=SONNET_DECISION.get("smallest_take_profit_pct", 0.03),
            max_tp=SONNET_DECISION.get("largest_take_profit_pct", 0.15),
            leverage=leverage,
            min_weight_for_notional=min_weight_for_notional,
            entry_threshold=SONNET_DECISION.get("entry_score_threshold", 0.25),
            reverse_threshold=SONNET_DECISION.get("reverse_score_threshold", 0.20),
        )

    def _build_user_prompt(
        self,
        alpha_signals: dict[str, pd.DataFrame],
        current_positions: pd.DataFrame | None,
        account_info: dict,
        market_context: dict,
        managed_positions: list | None = None,
        recently_closed: list | None = None,
        features_summary: dict | None = None,
    ) -> str:
        lines = []
        now = datetime.now()
        entry_threshold = SONNET_DECISION.get("entry_score_threshold", 0.25)
        reverse_threshold = SONNET_DECISION.get("reverse_score_threshold", 0.20)

        agg_scores = market_context.get("aggregated_scores", {})
        ranked_by_strength = sorted(agg_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        top_ranked = [t for t, _ in ranked_by_strength[:6]]

        held_tickers: list[str] = []
        if current_positions is not None and not current_positions.empty:
            held_tickers = [
                str(t) for t in current_positions["ticker"].tolist()
                if str(t)
            ]

        # Focus prompt context on top candidates + currently held symbols.
        focus_tickers = list(dict.fromkeys(top_ranked + held_tickers))[:8]
        focus_set = set(focus_tickers)

        # ── Section 1: Trade Performance (feedback loop) ──
        perf = market_context.get("trade_performance")
        if perf:
            lines.append("# Your Recent Performance")
            d = perf.get("last_24h", {})
            if d.get("trades", 0) > 0:
                lines.append(
                    f"  Last 24h: {d['trades']} trades, "
                    f"{d.get('wins', 0)}W/{d.get('losses', 0)}L, "
                    f"net=${d.get('net_pnl', 0):+.2f} | "
                    f"Avg hold={d.get('avg_hold_min', 0):.0f}min"
                )
            w = perf.get("last_7d", {})
            if w.get("trades", 0) > 0:
                lines.append(
                    f"  Last 7d: {w['trades']} trades, "
                    f"{w.get('wins', 0)}W/{w.get('losses', 0)}L, "
                    f"net=${w.get('net_pnl', 0):+.2f} | "
                    f"Win rate={w.get('win_rate', 0):.0%}"
                )

        lines.append("\n# Decision Constraints")
        lines.append(
            f"  Entry gate: |aggregated_score| >= {entry_threshold:.2f}"
        )
        lines.append(
            f"  Reverse/CLOSE gate: opposite |aggregated_score| >= {reverse_threshold:.2f}"
        )
        lines.append(
            "  Timeframe roles: daily(20d)=direction, 5m(36)=timing-only"
        )

        # ── Section 2: Current Positions (with hold duration) ──
        lines.append("\n# Current Positions")
        managed_lookup: dict[str, Any] = {}
        if managed_positions:
            for mp in managed_positions:
                managed_lookup[mp.ticker] = mp

        if current_positions is not None and not current_positions.empty:
            for _, row in current_positions.iterrows():
                ticker = str(row.get("ticker", "?"))
                short_ticker = ticker.replace("/USDT:USDT", "")
                side = str(row.get("side", "?")).upper()
                size = float(row.get("size", 0))
                entry = float(row.get("entry_price", 0))
                pnl = float(row.get("unrealized_pnl", 0))
                pos_line = (
                    f"  {short_ticker} {side} size={size:.4f} "
                    f"entry=${entry:,.2f} pnl=${pnl:+.2f}"
                )
                # Add hold duration + SL/TP if managed
                mp = managed_lookup.get(ticker)
                if mp:
                    try:
                        opened_at = datetime.fromisoformat(mp.entry_time)
                        hold_min = (now - opened_at).total_seconds() / 60
                        pos_line += f" opened={hold_min:.0f}min ago"
                    except (ValueError, TypeError):
                        pass
                    sl_dist = (mp.stop_loss_price / entry - 1) if entry > 0 else 0
                    tp_dist = (mp.take_profit_price / entry - 1) if entry > 0 else 0
                    pos_line += (
                        f" | SL=${mp.stop_loss_price:,.2f}({sl_dist:+.1%})"
                        f" TP=${mp.take_profit_price:,.2f}({tp_dist:+.1%})"
                    )
                lines.append(pos_line)
        else:
            lines.append("  None")

        # ── Section 3: Recent Trades (anti-churn context) ──
        if recently_closed:
            lines.append("\n# Recent Trades (last 30min)")
            for cp in recently_closed[-5:]:  # last 5 max
                try:
                    closed_at = datetime.fromisoformat(cp.close_time)
                    opened_at = datetime.fromisoformat(cp.entry_time)
                    mins_ago = (now - closed_at).total_seconds() / 60
                    held_min = (closed_at - opened_at).total_seconds() / 60
                    short_t = cp.ticker.replace("/USDT:USDT", "")
                    lines.append(
                        f"  CLOSED {short_t} {cp.side} {mins_ago:.0f}min ago, "
                        f"held {held_min:.0f}min ({cp.close_reason})"
                    )
                except (ValueError, TypeError):
                    continue

        # ── Section 4: Market Context (BTC + ETH + Breadth) ──
        regime = market_context.get("regime", "unknown")
        regime_score = market_context.get("regime_score", 0)
        btc_price = market_context.get("btc_price", 0)
        btc_24h = market_context.get("btc_24h_change", 0)
        btc_7d = market_context.get("btc_7d_change", 0)
        btc_30d = market_context.get("btc_30d_change", 0)
        eth_price = market_context.get("eth_price", 0)
        eth_24h = market_context.get("eth_24h_change", 0)
        eth_7d = market_context.get("eth_7d_change", 0)
        eth_30d = market_context.get("eth_30d_change", 0)
        pct_ma20 = market_context.get("pct_above_ma20", 0.5)
        adv_dec = market_context.get("adv_decline_ratio", 1.0)

        lines.append(f"\n# Market Context")
        lines.append(
            f"  BTC ${btc_price:,.0f} | "
            f"24h={btc_24h:+.1%}, 7d={btc_7d:+.1%}, 30d={btc_30d:+.1%}"
        )
        if eth_price > 0:
            lines.append(
                f"  ETH ${eth_price:,.0f} | "
                f"24h={eth_24h:+.1%}, 7d={eth_7d:+.1%}, 30d={eth_30d:+.1%}"
            )
        lines.append(
            f"  Breadth: {pct_ma20:.0%} above MA20, A/D={adv_dec:.2f}"
        )
        lines.append(f"  Regime: {regime} (score={regime_score:+d})")

        # ── Section 5: Derivatives Sentiment (OI + Long/Short ratio) ──
        oi_data = market_context.get("open_interest", {})
        ls_data = market_context.get("long_short_ratio", {})
        if oi_data or ls_data:
            lines.append("\n# Derivatives Sentiment")
            all_tickers = sorted(
                set(list(oi_data.keys()) + list(ls_data.keys())),
                key=lambda t: abs(oi_data.get(t, {}).get("change_24h", 0)),
                reverse=True,
            )
            for ticker_sym in all_tickers[:10]:
                short_t = ticker_sym.replace("/USDT:USDT", "")
                parts = [short_t + ":"]
                oi = oi_data.get(ticker_sym, {})
                if oi:
                    oi_val = oi.get("value", 0)
                    oi_chg = oi.get("change_24h", 0)
                    if oi_val >= 1_000_000:
                        parts.append(f"OI=${oi_val / 1e6:.1f}M({oi_chg:+.0%})")
                    else:
                        parts.append(f"OI=${oi_val:,.0f}({oi_chg:+.0%})")
                ls = ls_data.get(ticker_sym, {})
                if ls:
                    ratio = ls.get("ratio", 1.0)
                    label = "longs crowded" if ratio > 1.5 else "shorts crowded" if ratio < 0.67 else "balanced"
                    parts.append(f"L/S={ratio:.2f} ({label})")
                if len(parts) > 1:
                    lines.append(f"  {' | '.join(parts)}")

        # ── Section 6: Price Context (volatility, RSI per ticker) ──
        if features_summary:
            lines.append("\n# Price Context (top candidates)")
            top_tickers = [t for t in focus_tickers if t in features_summary]
            for ticker_sym in top_tickers:
                feat = features_summary[ticker_sym]
                short_t = ticker_sym.replace("/USDT:USDT", "")
                price = feat.get("price", 0)
                ret_1d = feat.get("ret_1d", 0)
                vol_5d = feat.get("vol_5d", 0)
                vol_20d = feat.get("vol_20d", 0)
                rsi = feat.get("rsi_14", 50)
                lines.append(
                    f"  {short_t}: ${price:,.4g} | "
                    f"24h={ret_1d:+.1%} | "
                    f"vol_5d={vol_5d:.0%} vol_20d={vol_20d:.0%} | "
                    f"RSI={rsi:.0f}"
                )

        # ── Section 7: Alpha Signals (only active strategies) ──
        # Only show signals that contribute to the aggregated score
        active_strategies = {
            v for k, v in _ALPHA_DISPLAY.items()
            if STRATEGIES.get(v, {}).get("enabled") and STRATEGIES.get(v, {}).get("weight", 0) > 0
        }
        lines.append("\n# Alpha Signals — active only (score: -1=bearish, +1=bullish)")
        ticker_scores: dict[str, dict[str, float]] = {}

        for alpha_name, signals_df in alpha_signals.items():
            display = _ALPHA_DISPLAY.get(alpha_name, alpha_name)
            if display not in active_strategies:
                continue  # Skip disabled/zero-weight alphas
            for _, row in signals_df.iterrows():
                ticker = str(row["ticker"])
                if focus_set and ticker not in focus_set:
                    continue
                score = float(row.get("score", 0))
                if abs(score) > 0.05:
                    if ticker not in ticker_scores:
                        ticker_scores[ticker] = {}
                    ticker_scores[ticker][display] = round(score, 2)

        for ticker in focus_tickers:
            scores = ticker_scores.get(ticker)
            if not scores:
                continue
            score_str = ", ".join(f"{k}: {v:+.2f}" for k, v in scores.items())
            short_ticker = ticker.replace("/USDT:USDT", "")
            lines.append(f"  {short_ticker}: {{{score_str}}}")

        # ── Section 8: Aggregated Scores ──
        if agg_scores:
            lines.append("\n# AGGREGATED SCORES (combined signal: +LONG / -SHORT)")
            for ticker, score in ranked_by_strength[:6]:
                short_t = ticker.replace("/USDT:USDT", "")
                direction = "LONG" if score > 0 else "SHORT"
                lines.append(f"  {short_t}: {score:+.3f} → {direction}")

        # ── Section 8.5: Squeeze Risk Warnings ──
        squeeze_warnings = market_context.get("squeeze_warnings", {})
        if squeeze_warnings:
            lines.append("\n# ⚠️ SQUEEZE RISK DETECTED — review carefully before acting")
            for ticker, warning in squeeze_warnings.items():
                short_t = ticker.replace("/USDT:USDT", "")
                lines.append(f"  {short_t}: {warning}")

        # ── Section 9: Funding Rates ──
        funding_rates = market_context.get("funding_rates", {})
        if funding_rates:
            lines.append("\n# Funding Rates (8h cost: LONG pays if +, SHORT pays if -)")
            for ticker_sym in sorted(funding_rates, key=lambda t: abs(funding_rates[t]), reverse=True)[:5]:
                rate = funding_rates[ticker_sym]
                short_t = ticker_sym.replace("/USDT:USDT", "")
                annual = rate * 3 * 365 * 100
                lines.append(f"  {short_t}: {rate:+.4%} per 8h ({annual:+.1f}% annualized)")

        # ── Section 10: Defensive Alerts ──
        defensive_alerts = market_context.get("defensive_alerts", {})
        suppressed = market_context.get("suppressed_tickers", [])
        if defensive_alerts or suppressed:
            lines.append("\n# DANGER SIGNALS (defensive filters opposing momentum)")
            for ticker_sym, alerts in defensive_alerts.items():
                alert_str = ", ".join(f"{k}: {v:+.2f}" for k, v in alerts.items())
                lines.append(f"  {ticker_sym}: {{{alert_str}}}")
            if suppressed:
                lines.append(
                    f"\n  SUPPRESSED (DO NOT open new positions): "
                    f"{', '.join(suppressed)}"
                )

        # ── Section 11: Account ──
        balance = account_info.get("total_wallet_balance", 0)
        available = account_info.get("available_balance", 0)
        lines.append(f"\n# Account: ${balance:.2f} total, ${available:.2f} available")

        return "\n".join(lines)

    def _parse_positions(self, raw_positions: list[dict]) -> list[PositionDecision]:
        positions = []
        for p in raw_positions:
            try:
                positions.append(PositionDecision(
                    ticker=str(p["ticker"]),
                    action=str(p["action"]).upper(),
                    weight=float(p.get("weight", 0)),
                    stop_loss_pct=float(p.get("stop_loss_pct", SONNET_DECISION["default_stop_loss_pct"])),
                    take_profit_pct=float(p.get("take_profit_pct", SONNET_DECISION["default_take_profit_pct"])),
                    reasoning=str(p.get("reasoning", "")),
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid position: {p} — {e}")
        return positions

    def _validate_positions(self, positions: list[PositionDecision]) -> list[PositionDecision]:
        """Validate and clip positions to config bounds."""
        max_positions = TRADING.get("max_positions", 3)
        max_weight = TRADING.get("max_position_weight", 0.30)
        min_weight = TRADING.get("min_position_weight", 0.05)

        tightest_sl = SONNET_DECISION.get("tightest_stop_loss_pct", -0.02)
        loosest_sl = SONNET_DECISION.get("loosest_stop_loss_pct", -0.08)
        smallest_tp = SONNET_DECISION.get("smallest_take_profit_pct", 0.03)
        largest_tp = SONNET_DECISION.get("largest_take_profit_pct", 0.15)

        valid = []
        for p in positions:
            if p.action not in ("LONG", "SHORT", "CLOSE", "HOLD"):
                logger.warning(f"Invalid action '{p.action}' for {p.ticker}, skipping")
                continue

            if p.action == "CLOSE":
                valid.append(p)
                continue

            # Normalize SL/TP signs (Sonnet sometimes returns wrong signs)
            # SL must be negative, TP must be positive — regardless of LONG/SHORT
            p.stop_loss_pct = -abs(p.stop_loss_pct) if p.stop_loss_pct != 0 else tightest_sl
            p.take_profit_pct = abs(p.take_profit_pct) if p.take_profit_pct != 0 else smallest_tp

            # Clip SL/TP to allowed bounds
            # SL is negative: clamp between loosest (-0.08) and tightest (-0.02)
            p.stop_loss_pct = max(min(p.stop_loss_pct, tightest_sl), loosest_sl)
            # TP is positive: clamp between smallest (0.03) and largest (0.15)
            p.take_profit_pct = min(max(p.take_profit_pct, smallest_tp), largest_tp)

            if p.action == "HOLD":
                valid.append(p)
                continue

            # Clip weight (LONG/SHORT only)
            p.weight = min(max(p.weight, min_weight), max_weight)

            valid.append(p)

        # Limit to max_positions
        actionable = [p for p in valid if p.action in ("LONG", "SHORT")]
        if len(actionable) > max_positions:
            logger.warning(
                f"Sonnet returned {len(actionable)} positions, limiting to {max_positions}"
            )
            # Keep closes/holds + top N actionable
            non_actionable = [p for p in valid if p.action not in ("LONG", "SHORT")]
            valid = non_actionable + actionable[:max_positions]

        # Check total weight
        total_weight = sum(abs(p.weight) for p in valid if p.action in ("LONG", "SHORT"))
        if total_weight > 1.0:
            logger.warning(f"Total weight {total_weight:.2f} > 1.0, scaling down")
            scale = 1.0 / total_weight
            for p in valid:
                if p.action in ("LONG", "SHORT"):
                    p.weight *= scale

        return valid

    @staticmethod
    def _extract_text(message) -> str:
        for block in message.content:
            if block.type == "text":
                return block.text.strip()
        raise ValueError("No text in API response")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
        if match:
            return json.loads(match.group(1))

        last_close = raw.rfind("}")
        if last_close != -1:
            depth = 0
            for i in range(last_close, -1, -1):
                if raw[i] == "}":
                    depth += 1
                elif raw[i] == "{":
                    depth -= 1
                if depth == 0:
                    return json.loads(raw[i:last_close + 1])

        raise ValueError(f"Cannot parse JSON from Sonnet response: {raw[:200]}")

    def _log_decision(
        self,
        system_prompt: str,
        user_prompt: str,
        raw_response: str,
        decision: SonnetDecision | None = None,
        error: str | None = None,
    ) -> None:
        """Save full Sonnet input/output to logs/sonnet_decisions/YYYY-MM-DD.jsonl"""
        try:
            now = datetime.now()
            log_file = DECISION_LOG_DIR / f"{now.strftime('%Y-%m-%d')}.jsonl"

            entry = {
                "timestamp": now.isoformat(),
                "call_number": self._call_count,
                "model": self.model,
                "input": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
                "output": {
                    "raw_response": raw_response,
                },
            }

            if decision:
                entry["output"]["positions"] = [asdict(p) for p in decision.positions]
                entry["output"]["market_assessment"] = decision.market_assessment
                entry["output"]["risk_note"] = decision.risk_note

            if error:
                entry["error"] = error

            with open(log_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            logger.info(f"Decision logged to {log_file.name}")
        except Exception as e:
            logger.warning(f"Failed to log decision: {e}")

    @staticmethod
    def log_transaction(
        ticker: str,
        action: str,
        side: str,
        quantity: float,
        price: float,
        notional: float,
        status: str,
        reason: str = "",
    ) -> None:
        """Log a trade execution to logs/sonnet_decisions/transactions_YYYY-MM-DD.jsonl"""
        try:
            now = datetime.now()
            log_file = DECISION_LOG_DIR / f"transactions_{now.strftime('%Y-%m-%d')}.jsonl"

            entry = {
                "timestamp": now.isoformat(),
                "ticker": ticker,
                "action": action,
                "side": side,
                "quantity": quantity,
                "price": price,
                "notional": notional,
                "status": status,
                "reason": reason,
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log transaction: {e}")


def compute_trade_performance(log_dir: Path | None = None) -> dict:
    """Compute trade performance from transaction logs for Sonnet context.

    Returns dict with 'last_24h' and 'last_7d' sub-dicts containing:
        trades, wins, losses, net_pnl, avg_hold_min, win_rate
    """
    log_dir = log_dir or DECISION_LOG_DIR
    now = datetime.now()

    def _empty():
        return {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0, "avg_hold_min": 0, "win_rate": 0.0}

    result = {"last_24h": _empty(), "last_7d": _empty()}

    try:
        # Read transaction logs from last 7 days
        all_trades = []
        for i in range(7):
            day = now - timedelta(days=i)
            log_file = log_dir / f"transactions_{day.strftime('%Y-%m-%d')}.jsonl"
            if log_file.exists():
                with open(log_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                all_trades.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

        if not all_trades:
            return result

        for period_key, max_hours in [("last_24h", 24), ("last_7d", 168)]:
            cutoff = now - timedelta(hours=max_hours)
            period_trades = [
                t for t in all_trades
                if datetime.fromisoformat(t.get("timestamp", "2000-01-01")) >= cutoff
            ]

            if not period_trades:
                continue

            trades = len(period_trades)
            # Estimate PnL from notional and action
            # Trades with status "filled" that are CLOSE actions represent realized PnL
            net_pnl = 0.0
            wins = 0
            losses = 0
            for t in period_trades:
                pnl = float(t.get("pnl", 0))
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
                net_pnl += pnl

            result[period_key] = {
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "net_pnl": net_pnl,
                "avg_hold_min": 0,  # would need entry/exit time pairing
                "win_rate": wins / max(wins + losses, 1),
            }

    except Exception as e:
        logger.warning(f"Failed to compute trade performance: {e}")

    return result
