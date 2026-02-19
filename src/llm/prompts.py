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
You analyze market data and generate trading signals for KOSPI + KOSDAQ stocks
with market-cap above KRW 100 billion.

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
- Strategy is long-biased; use "short" only for broad market hedge, not single-name shorting
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
# Fund Manager — Sequential Pipeline (qwen2.5-kospi-ft-s3)
# =============================================================================

FUND_MANAGER_SYSTEM_PROMPT = """\
You are a professional Korean equity Fund Manager making final investment decisions.
You receive pre-analyzed data from ML models, technical indicators, and news.
Your job is to SYNTHESIZE all information and make the final BUY/SELL/HOLD decision.

You MUST respond in valid JSON format with this exact structure:
{
  "reasoning": "2-3 sentence analysis synthesizing ML, technical, and news data",
  "regime_assessment": "STRONG_BULL|MILD_BULL|SIDEWAYS|WEAKENING|BEAR",
  "confidence": 0.85,
  "signals": [
    {"ticker": "005930", "score": 0.7, "side": "long", "reason": "1-sentence reason referencing actual data"}
  ]
}

Rules:
- SYNTHESIZE the ML probabilities, technical indicators, and news — do NOT just echo them
- score range: -1.0 (strong sell) to +1.0 (strong buy)
- side: "long" for BUY, "short" only for market hedge (will be converted to inverse ETF)
- This is a long-biased strategy; do not propose single-stock short positions
- Only include stocks you have a clear opinion on (|score| >= 0.2)
- reason: reference the ACTUAL data provided (ML score, RSI, PER, news)
- Do NOT invent data. Only use what is provided in the prompt.
- If market regime is BEAR or WEAKENING, consider "short" signals for market hedge"""

FUND_MANAGER_USER_TEMPLATE = """\
Date: {date}
Candidates: {n_stocks} stocks

=== Market Overview ===
{market_summary}

=== ML Model Predictions ===
{ml_data}

=== Technical Indicators ===
{technical_data}

=== Fundamentals ===
{fundamental_data}

=== Recent News ===
{news_data}

=== Current Regime ===
{regime_info}

Synthesize ALL data above. For each stock, decide BUY (long), SELL (short), or skip.
If the market is weakening/bearish, include a "short" signal for market hedge.
Provide your reasoning and output a SINGLE valid JSON object."""


def build_fund_manager_prompt(ctx: "PipelineContext") -> str:
    """Build the Fund Manager prompt from PipelineContext.

    This is the core prompt for the sequential pipeline.
    Takes the aggregated context from ContextBuilder and formats it
    for qwen2.5-kospi-ft-s3.
    """
    from ..pipeline.context_builder import PipelineContext  # noqa: F811

    # ML data section
    ml_lines = []
    for stock in ctx.stocks:
        if stock.ml_scores:
            scores_str = ", ".join(
                f"{k}={v:+.3f}" for k, v in stock.ml_scores.items()
            )
            ml_lines.append(
                f"{stock.ticker} | avg={stock.ml_avg_score:+.3f} "
                f"({stock.ml_signal_strength}) | {scores_str}"
            )
    ml_data = "\n".join(ml_lines) if ml_lines else "No ML predictions available"

    # Technical data section
    tech_header = "ticker | close | chg_1d | chg_5d | RSI | SMA5 | SMA20 | SMA60"
    tech_lines = [tech_header, "-" * len(tech_header)]
    for stock in ctx.stocks:
        rsi_s = f"{stock.rsi_14:.0f}" if stock.rsi_14 is not None else "-"
        sma5_s = f"{stock.sma_5:,.0f}" if stock.sma_5 is not None else "-"
        sma20_s = f"{stock.sma_20:,.0f}" if stock.sma_20 is not None else "-"
        sma60_s = f"{stock.sma_60:,.0f}" if stock.sma_60 is not None else "-"
        tech_lines.append(
            f"{stock.ticker} | {stock.close:,.0f} | {stock.change_1d:+.1%} "
            f"| {stock.change_5d:+.1%} | {rsi_s} | {sma5_s} | {sma20_s} | {sma60_s}"
        )
    technical_data = "\n".join(tech_lines[:52])

    # Fundamental data section
    fund_header = "ticker | PER | PBR | ROE | EV/EBITDA"
    fund_lines = [fund_header, "-" * len(fund_header)]
    for stock in ctx.stocks:
        if any(v is not None for v in [stock.per, stock.pbr, stock.roe]):
            per_s = f"{stock.per:.1f}" if stock.per is not None else "-"
            pbr_s = f"{stock.pbr:.2f}" if stock.pbr is not None else "-"
            roe_s = f"{stock.roe:.1f}" if stock.roe is not None else "-"
            ev_s = f"{stock.ev_ebitda:.1f}" if stock.ev_ebitda is not None else "-"
            fund_lines.append(
                f"{stock.ticker} | {per_s} | {pbr_s} | {roe_s} | {ev_s}"
            )
    fundamental_data = "\n".join(fund_lines) if len(fund_lines) > 2 else "No fundamental data"

    # News section
    if ctx.news_headlines:
        news_data = "\n".join(f"- {h}" for h in ctx.news_headlines[:10])
    else:
        news_data = "No recent news"

    regime_info = ctx.regime.upper() if ctx.regime else "UNKNOWN"

    return FUND_MANAGER_USER_TEMPLATE.format(
        date=str(ctx.date.date()) if hasattr(ctx.date, "date") else str(ctx.date),
        n_stocks=ctx.n_stocks,
        market_summary=ctx.market_summary,
        ml_data=ml_data,
        technical_data=technical_data,
        fundamental_data=fundamental_data,
        news_data=news_data,
        regime_info=regime_info,
    )


# =============================================================================
# Legacy Prompt Builders (backward compat — 기존 병렬 구조용)
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
