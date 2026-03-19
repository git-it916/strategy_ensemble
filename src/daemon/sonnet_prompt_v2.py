"""
Sonnet Prompt V2 Builder

Enhanced prompt sections for Sonnet decision maker when using v2 alphas.
Provides richer, more structured information:
    - Alpha quality summary (trailing IC, hit rate, decay)
    - Signal agreement matrix
    - Confidence-adjusted scores
    - Contribution breakdown with context
"""

from __future__ import annotations

import logging
from typing import Any

from src.daemon.enhanced_signal_aggregator import EnhancedAggregatedSignal

logger = logging.getLogger(__name__)


def build_alpha_quality_section(
    alpha_quality: dict[str, dict[str, float]] | None,
) -> str:
    """
    Build alpha quality summary section.

    Args:
        alpha_quality: {alpha_name: {ic_5d, hit_rate_5d, decay_hours, ...}}
    """
    if not alpha_quality:
        return ""

    lines = ["# Alpha Signal Quality (trailing 7-day)"]

    total_ic = 0
    n_alphas = 0
    for name, metrics in alpha_quality.items():
        ic = metrics.get("ic_5d", 0)
        hr = metrics.get("hit_rate_5d", 0)
        decay = metrics.get("decay_hours", 24)

        # Only show if noteworthy
        if abs(ic) > 0.01 or hr > 0.55 or hr < 0.45:
            quality = _ic_label(ic)
            lines.append(
                f"  {name}: IC={ic:.3f} ({quality}), "
                f"hit={hr:.0%}, decay={decay:.0f}h"
            )
        total_ic += ic
        n_alphas += 1

    if n_alphas > 0:
        avg_ic = total_ic / n_alphas
        overall = _ic_label(avg_ic)
        lines.append(f"  Overall signal quality: {overall.upper()} (avg IC={avg_ic:.3f})")

    return "\n".join(lines)


def build_signal_agreement_section(
    agg: EnhancedAggregatedSignal,
    focus_tickers: list[str],
) -> str:
    """
    Build signal agreement section.

    Shows how many alphas agree on direction for each focus ticker.
    """
    if not agg.agreement:
        return ""

    lines = ["# Signal Agreement"]
    for ticker in focus_tickers:
        if ticker not in agg.agreement:
            continue
        n_agree, n_total, direction = agg.agreement[ticker]
        short_t = ticker.replace("/USDT:USDT", "")

        if n_total == 0:
            continue

        if n_agree == n_total:
            strength = "unanimous"
        elif n_agree >= n_total * 0.7:
            strength = "strong consensus"
        elif n_agree >= n_total * 0.5:
            strength = "moderate"
        else:
            strength = "mixed — proceed with caution"

        lines.append(
            f"  {short_t}: {n_agree}/{n_total} alphas agree {direction} ({strength})"
        )

    return "\n".join(lines) if len(lines) > 1 else ""


def build_confidence_adjusted_scores_section(
    agg: EnhancedAggregatedSignal,
    focus_tickers: list[str],
) -> str:
    """
    Build section showing raw vs confidence-adjusted scores.
    """
    if not agg.scores or not agg.effective_scores:
        return ""

    lines = ["# AGGREGATED SCORES (with confidence adjustment)"]

    ranked = sorted(
        agg.effective_scores.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    for ticker, eff_score in ranked[:6]:
        short_t = ticker.replace("/USDT:USDT", "")
        raw_score = agg.scores.get(ticker, 0)
        conf = agg.confidence.get(ticker, 1.0)
        direction = "LONG" if eff_score > 0 else "SHORT"

        line = f"  {short_t}: {raw_score:+.3f}"
        if abs(conf - 1.0) > 0.05:
            line += f" (confidence: {conf:.2f}) → effective: {eff_score:+.3f}"
            if conf < 0.5:
                line += " ← LOW CONFIDENCE"
        else:
            line += f" → {direction}"

        lines.append(line)

    if agg.vol_confidence < 0.9:
        lines.append(
            f"  Vol regime modifier: {agg.vol_confidence:.2f} "
            f"(high volatility → reduced confidence)"
        )

    return "\n".join(lines)


def build_contribution_breakdown_section(
    agg: EnhancedAggregatedSignal,
    focus_tickers: list[str],
    alpha_metadata: dict[str, dict] | None = None,
) -> str:
    """
    Build detailed contribution breakdown per ticker.

    Shows which alphas contributed most and their reasoning.
    """
    if not agg.contributions:
        return ""

    lines = ["# Score Breakdown (top candidates)"]

    for ticker in focus_tickers[:4]:
        contribs = agg.contributions.get(ticker, {})
        raw = agg.raw_scores.get(ticker, {})
        if not contribs:
            continue

        eff_score = agg.effective_scores.get(ticker, 0)
        short_t = ticker.replace("/USDT:USDT", "")
        lines.append(f"  {short_t} ({eff_score:+.3f}):")

        # Sort by absolute contribution
        sorted_contribs = sorted(
            contribs.items(), key=lambda x: abs(x[1]), reverse=True
        )

        total_abs = sum(abs(c) for _, c in sorted_contribs) or 1
        for alpha_name, contribution in sorted_contribs[:5]:
            pct = abs(contribution) / total_abs * 100
            raw_score = raw.get(alpha_name, 0)
            lines.append(
                f"    {alpha_name}: {contribution:+.3f} ({pct:.0f}%) "
                f"[raw={raw_score:+.2f}]"
            )

    return "\n".join(lines) if len(lines) > 1 else ""


def build_v2_sections(
    agg: EnhancedAggregatedSignal,
    focus_tickers: list[str],
    alpha_quality: dict[str, dict[str, float]] | None = None,
    alpha_metadata: dict[str, dict] | None = None,
) -> str:
    """
    Build all v2-specific prompt sections as a single string.

    This replaces sections 7 (Alpha Signals) and 8 (Aggregated Scores)
    from the v1 prompt with richer, more structured info.
    """
    sections = []

    # Alpha quality (only if tracking is enabled)
    quality_section = build_alpha_quality_section(alpha_quality)
    if quality_section:
        sections.append(quality_section)

    # Signal agreement
    agreement_section = build_signal_agreement_section(agg, focus_tickers)
    if agreement_section:
        sections.append(agreement_section)

    # Confidence-adjusted scores (replaces simple aggregated scores)
    scores_section = build_confidence_adjusted_scores_section(agg, focus_tickers)
    if scores_section:
        sections.append(scores_section)

    # Contribution breakdown
    contrib_section = build_contribution_breakdown_section(
        agg, focus_tickers, alpha_metadata
    )
    if contrib_section:
        sections.append(contrib_section)

    return "\n\n".join(sections)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _ic_label(ic: float) -> str:
    """Human-readable IC quality label."""
    if ic > 0.08:
        return "strong"
    elif ic > 0.04:
        return "good"
    elif ic > 0.01:
        return "moderate"
    elif ic > -0.02:
        return "weak"
    else:
        return "negative"
