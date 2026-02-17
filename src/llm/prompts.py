"""
LLM Prompt Templates

Structured prompts for all LLM tasks.
All prompts instruct the model to output JSON with reasoning logs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# Signal Generation (Qwen2.5:32B)
# =============================================================================

SIGNAL_SYSTEM_PROMPT = """\
You are a Korean equity market quantitative analyst.
You analyze market data and generate trading signals for KOSPI200 stocks.

You MUST respond in valid JSON format with this exact structure:
{
  "market_context": "brief market analysis",
  "regime_assessment": "bull|bear|sideways",
  "confidence": 0.85,
  "signals": [
    {"ticker": "005930", "score": 0.7, "side": "long", "reason": "1-sentence reason based on provided data"}
  ]
}

Rules:
- score range: -1.0 (strong sell) to +1.0 (strong buy), 0.0 = neutral
- side: "long" for positive score, "short" for negative score
- Only include stocks with |score| >= 0.2 (skip neutral stocks)
- reason: 1 sentence explaining why, referencing actual data provided (PER, PBR, RSI, momentum etc.)
- Do NOT invent data not in the prompt. Only use the provided stock data.
- confidence: 0.0 to 1.0, your overall confidence in the analysis"""

SIGNAL_USER_TEMPLATE = """\
Date: {date}
Stocks: {n_stocks}

=== Market Context ===
{market_context}

=== Stock Data (from DB) ===
{stock_data}

=== Feature Summary ===
{feature_summary}

=== Current Regime ===
{regime_info}

Regime assessment + full position recommendation needed.
For each signal, provide side (long/short) and a 1-sentence reason referencing the actual data above."""


# =============================================================================
# Ensemble Orchestration (DeepSeek R1:32B)
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a senior portfolio manager overseeing multiple quantitative strategies for Korean equities.

You receive signals from 8 alpha strategies (4 rule-based, 3 ML, 1 LLM-based) and must decide which signals to trust and how to weight them.

You MUST respond in valid JSON format with this exact structure:
{
  "reasoning": {
    "market_assessment": "current market conditions analysis in Korean",
    "strategy_evaluation": {
      "strategy_name": {"trust_level": 0.8, "rationale": "why trust/distrust"}
    },
    "portfolio_logic": "overall portfolio construction reasoning",
    "risk_considerations": "key risks being managed"
  },
  "final_signals": [
    {"ticker": "005930", "score": 0.65, "primary_driver": "strategy_name", "rationale": "reason"}
  ],
  "strategy_weights": {"strategy_name": 0.15},
  "confidence": 0.80,
  "position_sizing_notes": "any special sizing considerations"
}

Rules:
- Think step by step (chain of thought)
- Evaluate each strategy's recent performance
- Consider current market regime
- Identify conflicts between strategies and resolve them
- score range: -1.0 to +1.0
- Only include stocks with |score| >= 0.15
- Your reasoning should be thorough and in Korean"""

ORCHESTRATOR_USER_TEMPLATE = """\
날짜: {date}
시장 레짐: {regime}

=== 전략별 시그널 ===
{strategy_signals}

=== 전략 최근 성과 ===
{strategy_performance}

=== 전략 상관관계 ===
{strategy_correlations}

=== 시장 컨텍스트 ===
{market_context}

=== 리스크 제약 ===
- 최대 개별 비중: {max_position_weight:.0%}
- 최대 보유 종목: {max_positions}개
- 최대 허용 손실: {max_drawdown:.0%}

모든 전략 시그널을 평가하고 최종 포트폴리오 시그널을 생성하세요.
각 전략을 신뢰/불신하는 이유를 설명하세요.
JSON 형식으로 final_signals, strategy_weights, 상세 reasoning을 반환하세요."""


# =============================================================================
# Prompt Builders
# =============================================================================

def build_signal_prompt(
    date: datetime,
    prices: pd.DataFrame,
    features: pd.DataFrame | None,
    regime: str | None,
    top_k: int = 50,
) -> str:
    """Build the signal generation prompt for Qwen2.5:32B."""
    stock_data = format_stock_data_for_prompt(prices, features, top_k)
    feature_summary = _build_feature_summary(features, date) if features is not None else "피처 데이터 없음"
    market_context = _build_market_context_from_prices(prices)
    regime_info = regime if regime else "미확인"

    return SIGNAL_USER_TEMPLATE.format(
        n_stocks=min(top_k, prices["ticker"].nunique()),
        date=str(date.date()) if hasattr(date, "date") else str(date),
        market_context=market_context,
        stock_data=stock_data,
        feature_summary=feature_summary,
        regime_info=regime_info,
    )


def build_orchestrator_prompt(
    date: datetime,
    strategy_signals: dict[str, pd.DataFrame],
    strategy_performance: dict[str, dict],
    strategy_correlations: pd.DataFrame | None,
    market_context: dict[str, Any],
    regime: str | None,
    risk_constraints: dict[str, Any],
) -> str:
    """Build the ensemble orchestration prompt for DeepSeek R1:32B."""
    signals_text = format_strategy_signals_for_prompt(strategy_signals)
    perf_text = _format_performance(strategy_performance)
    corr_text = _format_correlations(strategy_correlations)
    context_text = _format_market_context(market_context)

    return ORCHESTRATOR_USER_TEMPLATE.format(
        date=str(date.date()) if hasattr(date, "date") else str(date),
        regime=regime or "미확인",
        strategy_signals=signals_text,
        strategy_performance=perf_text,
        strategy_correlations=corr_text,
        market_context=context_text,
        max_position_weight=risk_constraints.get("max_position_weight", 0.1),
        max_positions=risk_constraints.get("max_positions", 20),
        max_drawdown=risk_constraints.get("max_drawdown", 0.15),
    )


# =============================================================================
# Formatting Helpers
# =============================================================================

def format_stock_data_for_prompt(
    prices: pd.DataFrame,
    features: pd.DataFrame | None,
    top_k: int = 50,
) -> str:
    """Format stock data into compact tabular text for LLM.

    Includes price momentum, valuation (PER/PBR/ROE), and RSI from actual DB.
    """
    # Get latest data per ticker
    latest = prices.sort_values("date").groupby("ticker").tail(1).copy()

    # Select most active stocks by volume
    if "volume" in latest.columns:
        latest = latest.nlargest(top_k, "volume")
    else:
        latest = latest.head(top_k)

    # Merge fundamental data if available
    if features is not None and not features.empty:
        fund_latest = features.sort_values("date").groupby("ticker").tail(1)
        fund_cols = ["ticker"]
        for col in ("per", "pbr", "ev_ebitda", "roe"):
            if col in fund_latest.columns:
                fund_cols.append(col)
        if len(fund_cols) > 1:
            latest = latest.merge(
                fund_latest[fund_cols], on="ticker", how="left", suffixes=("", "_fund"),
            )

    # Build header — include valuation columns if available
    header = "ticker | close | chg_1d | chg_5d | volume | PER | PBR | ROE | RSI"
    lines = [header, "-" * len(header)]

    for _, row in latest.iterrows():
        ticker = row["ticker"]
        close = row.get("close", 0)

        # Calculate returns from prices
        ticker_prices = prices[prices["ticker"] == ticker].sort_values("date")
        if len(ticker_prices) >= 5:
            closes = ticker_prices["close"].values
            chg_1d = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
            chg_5d = (closes[-1] / closes[-5] - 1) * 100 if len(closes) >= 5 else 0
        else:
            chg_1d = chg_5d = 0

        vol = row.get("volume", 0)

        # Valuation — prefer fundamental data, fallback to market_daily columns
        per = row.get("per", row.get("pe_ratio", None))
        pbr = row.get("pbr", row.get("pb_ratio", None))
        roe = row.get("roe", None)
        rsi = row.get("rsi_14", None)

        per_s = f"{per:.1f}" if pd.notna(per) else "-"
        pbr_s = f"{pbr:.2f}" if pd.notna(pbr) else "-"
        roe_s = f"{roe:.1f}" if pd.notna(roe) else "-"
        rsi_s = f"{rsi:.0f}" if pd.notna(rsi) else "-"

        lines.append(
            f"{ticker} | {close:,.0f} | {chg_1d:+.1f}% | {chg_5d:+.1f}% "
            f"| {vol:,.0f} | {per_s} | {pbr_s} | {roe_s} | {rsi_s}"
        )

    return "\n".join(lines[:top_k + 2])  # header + separator + data


def format_strategy_signals_for_prompt(
    strategy_signals: dict[str, pd.DataFrame],
) -> str:
    """Format all strategy signals into readable text."""
    parts = []

    for name, signals in strategy_signals.items():
        if signals.empty:
            parts.append(f"\n[{name}] 시그널 없음")
            continue

        top = signals.nlargest(5, "score")
        bottom = signals.nsmallest(3, "score")

        lines = [f"\n[{name}] (총 {len(signals)}개 종목)"]
        lines.append("  Top 5: " + ", ".join(
            f"{r['ticker']}({r['score']:+.3f})"
            for _, r in top.iterrows()
        ))
        if not bottom.empty and bottom.iloc[0]["score"] < 0:
            lines.append("  Bottom 3: " + ", ".join(
                f"{r['ticker']}({r['score']:+.3f})"
                for _, r in bottom.iterrows()
            ))

        parts.append("\n".join(lines))

    return "\n".join(parts)


def _build_market_context_from_prices(prices: pd.DataFrame) -> str:
    """Build market context summary from price data."""
    dates = prices["date"].unique()
    if len(dates) < 20:
        return "데이터 부족"

    # Market-level stats from cross-section
    latest_date = prices["date"].max()
    recent = prices[prices["date"] >= latest_date - pd.Timedelta(days=30)]

    # Average return
    daily_returns = recent.groupby("date")["close"].mean().pct_change().dropna()
    if len(daily_returns) == 0:
        return "수익률 데이터 부족"

    avg_ret = daily_returns.mean() * 100
    vol = daily_returns.std() * np.sqrt(252) * 100
    cumul = (1 + daily_returns).prod() - 1
    n_up = (daily_returns > 0).sum()
    n_total = len(daily_returns)

    return (
        f"최근 30일 평균 일일수익률: {avg_ret:+.2f}%, "
        f"변동성(연율): {vol:.1f}%, "
        f"누적수익률: {cumul * 100:+.1f}%, "
        f"상승일수: {n_up}/{n_total}일"
    )


def _build_feature_summary(features: pd.DataFrame, date: datetime) -> str:
    """Build feature summary for the latest date."""
    if features is None or features.empty:
        return "피처 데이터 없음"

    latest = features[features["date"] <= pd.Timestamp(date)]
    if latest.empty:
        return "해당 날짜 피처 없음"

    latest_date = latest["date"].max()
    snapshot = latest[latest["date"] == latest_date]

    # Summary statistics for key features
    summary_cols = [
        c for c in snapshot.columns
        if c not in ("date", "ticker") and snapshot[c].dtype in ("float64", "int64")
    ]

    if not summary_cols:
        return "수치형 피처 없음"

    parts = []
    for col in summary_cols[:10]:  # Limit to 10 features
        vals = snapshot[col].dropna()
        if len(vals) > 0:
            parts.append(f"{col}: mean={vals.mean():.3f}, std={vals.std():.3f}")

    return "\n".join(parts)


def _format_performance(perf: dict[str, dict]) -> str:
    """Format strategy performance metrics."""
    if not perf:
        return "성과 데이터 없음"

    lines = []
    for name, metrics in perf.items():
        sharpe = metrics.get("sharpe", 0)
        win_rate = metrics.get("win_rate", 0)
        mean_ret = metrics.get("mean_return", 0)
        lines.append(
            f"  {name}: Sharpe={sharpe:.2f}, 승률={win_rate:.1%}, "
            f"평균수익={mean_ret:.4f}"
        )
    return "\n".join(lines)


def _format_correlations(corr: pd.DataFrame | None) -> str:
    """Format correlation matrix."""
    if corr is None or corr.empty:
        return "상관관계 데이터 없음"
    return corr.to_string(float_format=lambda x: f"{x:.2f}")


def _format_market_context(ctx: dict[str, Any]) -> str:
    """Format market context dict."""
    if not ctx:
        return "시장 컨텍스트 없음"
    return "\n".join(f"  {k}: {v}" for k, v in ctx.items())
