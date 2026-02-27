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
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import SONNET_DECISION, TRADING

logger = logging.getLogger(__name__)

# Dedicated decision log directory
DECISION_LOG_DIR = Path(__file__).parent.parent.parent / "logs" / "sonnet_decisions"
DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """\
You are a crypto futures portfolio manager for a small account on Binance USDT-M perpetual futures.

RULES:
- Maximum {max_positions} simultaneous positions
- Each position weight: {min_weight:.0%} to {max_weight:.0%} of portfolio
- Every position MUST have stop_loss_pct (between {min_sl:.0%} and {max_sl:.0%}) and take_profit_pct (between {min_tp:.0%} and {max_tp:.0%})
- Positive score = bullish signal, negative = bearish signal
- If no clear opportunity exists, return empty positions array (cash is a valid position)
- Prefer 1-2 high-conviction trades. With small capital, concentration beats diversification
- IMPORTANT: Minimum order notional on Binance is $5. With your current balance and {leverage}x leverage, minimum weight is {min_weight_for_notional:.0%}. Do NOT pick coins at lower weight — the order will fail
- Only pick coins where weight × balance × leverage > $5
- Consider regime: in trending markets favor momentum, in ranging markets favor mean reversion
- Be consistent: do NOT flip between HOLD and CLOSE on the same position every cycle. Commit to a direction

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
    "TimeSeriesMomentum": "ts_momentum",
    "TimeSeriesMeanReversion": "ts_mean_reversion",
    "PriceVolumeDivergence": "pv_divergence",
    "VolumeMomentum": "volume_momentum",
    "LowVolatilityAnomaly": "low_volatility",
    "FundingRateCarry": "funding_carry",
    "RSIReversalAlpha": "rsi_reversal",
    "VolatilityBreakoutAlpha": "vol_breakout",
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
    ) -> SonnetDecision:
        """Call Sonnet and parse structured trading decision."""
        system_prompt = self._build_system_prompt(account_info)
        user_prompt = self._build_user_prompt(
            alpha_signals, current_positions, account_info, market_context,
            managed_positions=managed_positions,
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
        )

    def _build_user_prompt(
        self,
        alpha_signals: dict[str, pd.DataFrame],
        current_positions: pd.DataFrame | None,
        account_info: dict,
        market_context: dict,
        managed_positions: list | None = None,
    ) -> str:
        lines = []

        # Section 1: Alpha signals matrix
        lines.append("# Alpha Signals (score: -1=bearish, +1=bullish)")
        ticker_scores: dict[str, dict[str, float]] = {}

        for alpha_name, signals_df in alpha_signals.items():
            display = _ALPHA_DISPLAY.get(alpha_name, alpha_name)
            for _, row in signals_df.iterrows():
                ticker = str(row["ticker"])
                score = float(row.get("score", 0))
                if abs(score) > 0.05:
                    if ticker not in ticker_scores:
                        ticker_scores[ticker] = {}
                    ticker_scores[ticker][display] = round(score, 2)

        # Sort by number of signals (most coverage first)
        for ticker in sorted(ticker_scores, key=lambda t: -len(ticker_scores[t])):
            scores = ticker_scores[ticker]
            score_str = ", ".join(f"{k}: {v:+.2f}" for k, v in scores.items())
            short_ticker = ticker.replace("/USDT:USDT", "")
            lines.append(f"  {short_ticker}: {{{score_str}}}")

        # Section 2: Market context
        regime = market_context.get("regime", "unknown")
        btc_24h = market_context.get("btc_24h_change", 0)
        btc_7d = market_context.get("btc_7d_change", 0)
        btc_30d = market_context.get("btc_30d_change", 0)
        btc_price = market_context.get("btc_price", 0)
        lines.append(
            f"\n# Market: BTC ${btc_price:,.0f} | "
            f"24h={btc_24h:+.1%}, 7d={btc_7d:+.1%}, 30d={btc_30d:+.1%} | "
            f"regime={regime}"
        )

        # Section 2.5: Funding rates (cost of holding positions)
        funding_rates = market_context.get("funding_rates", {})
        if funding_rates:
            lines.append("\n# Funding Rates (8h cost: LONG pays if +, SHORT pays if -)")
            for ticker_sym in sorted(funding_rates, key=lambda t: abs(funding_rates[t]), reverse=True)[:10]:
                rate = funding_rates[ticker_sym]
                short_t = ticker_sym.replace("/USDT:USDT", "")
                annual = rate * 3 * 365 * 100  # 3 payments/day × 365 days × 100%
                lines.append(f"  {short_t}: {rate:+.4%} per 8h ({annual:+.1f}% annualized)")

        # Section 3: Current positions with SL/TP levels
        lines.append("\n# Current Positions")
        # Build lookup of managed positions for SL/TP info
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
                # Add SL/TP if managed
                mp = managed_lookup.get(ticker)
                if mp:
                    current_price = entry + pnl / max(size, 1e-8) if size > 0 else entry
                    sl_dist = (mp.stop_loss_price / entry - 1) if entry > 0 else 0
                    tp_dist = (mp.take_profit_price / entry - 1) if entry > 0 else 0
                    pos_line += (
                        f" | SL=${mp.stop_loss_price:,.2f}({sl_dist:+.1%})"
                        f" TP=${mp.take_profit_price:,.2f}({tp_dist:+.1%})"
                    )
                lines.append(pos_line)
        else:
            lines.append("  None")

        # Section 4: Account
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

        valid = []
        for p in positions:
            if p.action not in ("LONG", "SHORT", "CLOSE", "HOLD"):
                logger.warning(f"Invalid action '{p.action}' for {p.ticker}, skipping")
                continue

            if p.action in ("CLOSE", "HOLD"):
                valid.append(p)
                continue

            # Clip weight
            p.weight = min(max(p.weight, min_weight), max_weight)

            # Clip SL/TP to allowed bounds
            tightest_sl = SONNET_DECISION.get("tightest_stop_loss_pct", -0.02)
            loosest_sl = SONNET_DECISION.get("loosest_stop_loss_pct", -0.08)
            # SL is negative: clamp between loosest (-0.08) and tightest (-0.02)
            p.stop_loss_pct = max(min(p.stop_loss_pct, tightest_sl), loosest_sl)

            smallest_tp = SONNET_DECISION.get("smallest_take_profit_pct", 0.03)
            largest_tp = SONNET_DECISION.get("largest_take_profit_pct", 0.15)
            # TP is positive: clamp between smallest (0.03) and largest (0.15)
            p.take_profit_pct = min(max(p.take_profit_pct, smallest_tp), largest_tp)

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
