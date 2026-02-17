"""
Context Builder — Sequential Pipeline의 핵심 모듈

ML 확률, 기술적 지표, 펀더멘털, 텔레그램 뉴스, 실시간 가격을
하나의 구조화된 컨텍스트로 통합하여 LLM(Fund Manager)에게 전달.

Architecture:
    ML Scores ─┐
    Technicals ─┼──→ ContextBuilder.build() ──→ LLM Prompt
    News       ─┤
    Real-time  ─┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..alphas.base_alpha import BaseAlpha, AlphaResult

logger = logging.getLogger(__name__)


@dataclass
class StockContext:
    """단일 종목의 통합 컨텍스트."""
    ticker: str
    name: str = ""
    # 가격 정보
    close: float = 0.0
    change_1d: float = 0.0
    change_5d: float = 0.0
    volume: int = 0
    # ML 예측
    ml_scores: dict[str, float] = field(default_factory=dict)
    ml_avg_score: float = 0.0
    ml_signal_strength: str = ""  # High / Medium / Low / Neutral
    # 기술적 지표
    rsi_14: float | None = None
    sma_5: float | None = None
    sma_20: float | None = None
    sma_60: float | None = None
    # 펀더멘털
    per: float | None = None
    pbr: float | None = None
    roe: float | None = None
    ev_ebitda: float | None = None
    # 뉴스
    news_summary: str = ""
    news_sentiment: str = ""  # Positive / Negative / Neutral

    @property
    def signal_strength(self) -> str:
        """ML 평균 스코어 기반 신호 강도."""
        s = abs(self.ml_avg_score)
        if s >= 0.6:
            return "High"
        elif s >= 0.3:
            return "Medium"
        elif s > 0.05:
            return "Low"
        return "Neutral"


@dataclass
class PipelineContext:
    """파이프라인 전체 컨텍스트."""
    date: datetime
    stocks: list[StockContext]
    regime: str | None = None
    market_summary: str = ""
    news_headlines: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_stocks(self) -> int:
        return len(self.stocks)


class ContextBuilder:
    """
    순차 파이프라인의 Step 2-3: 데이터 수집 및 컨텍스트 구성.

    ML 모델들을 "데이터 소스"로 사용 (주문을 내지 않음).
    기술적 지표, 펀더멘털, 뉴스를 결합하여 LLM에 전달할 컨텍스트 생성.
    """

    def __init__(
        self,
        ml_strategies: dict[str, BaseAlpha] | None = None,
        technical_strategies: dict[str, BaseAlpha] | None = None,
        regime_classifier: Any | None = None,
        news_collector: Any | None = None,
    ):
        self.ml_strategies = ml_strategies or {}
        self.technical_strategies = technical_strategies or {}
        self.regime_classifier = regime_classifier
        self.news_collector = news_collector

    def build(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        news_messages: list[dict] | None = None,
        top_k: int = 50,
    ) -> PipelineContext:
        """
        전체 컨텍스트 구성.

        Args:
            date: 시그널 생성 날짜
            prices: 가격 데이터
            features: 펀더멘털 데이터
            news_messages: 텔레그램 뉴스 메시지 리스트
            top_k: LLM에 전달할 최대 종목 수

        Returns:
            PipelineContext with all data aggregated
        """
        logger.info(f"Building context for {date}")

        # 1. ML 모델 스코어 수집 (데이터 소스로만 사용)
        ml_signals = self._collect_ml_signals(date, prices, features)

        # 2. 기술적 지표 스코어 수집
        technical_signals = self._collect_technical_signals(date, prices, features)

        # 3. 레짐 감지
        regime = self._detect_regime(date, prices, features)

        # 4. 뉴스 수집
        news_headlines, ticker_news = self._process_news(news_messages)

        # 5. 종목별 컨텍스트 통합
        stocks = self._merge_stock_contexts(
            date=date,
            prices=prices,
            features=features,
            ml_signals=ml_signals,
            technical_signals=technical_signals,
            ticker_news=ticker_news,
            top_k=top_k,
        )

        # 6. 시장 요약
        market_summary = self._build_market_summary(prices, regime)

        ctx = PipelineContext(
            date=date,
            stocks=stocks,
            regime=regime,
            market_summary=market_summary,
            news_headlines=news_headlines,
            metadata={
                "ml_strategies_used": list(ml_signals.keys()),
                "technical_strategies_used": list(technical_signals.keys()),
                "n_news": len(news_headlines),
            },
        )

        logger.info(
            f"Context built: {ctx.n_stocks} stocks, regime={regime}, "
            f"{len(news_headlines)} news items"
        )
        return ctx

    def _collect_ml_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None,
    ) -> dict[str, pd.DataFrame]:
        """ML 전략들로부터 확률/스코어 수집 (주문 X, 데이터만)."""
        results = {}
        for name, strategy in self.ml_strategies.items():
            if not strategy.is_fitted:
                logger.warning(f"ML strategy {name} not fitted, skipping")
                continue
            try:
                result = strategy.generate_signals(date, prices, features)
                if not result.signals.empty:
                    results[name] = result.signals
                    logger.debug(
                        f"  ML {name}: {len(result.signals)} signals"
                    )
            except Exception as e:
                logger.error(f"ML strategy {name} failed: {e}")
        return results

    def _collect_technical_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None,
    ) -> dict[str, pd.DataFrame]:
        """기술적/펀더멘털 전략 스코어 수집."""
        results = {}
        for name, strategy in self.technical_strategies.items():
            if not strategy.is_fitted:
                logger.warning(f"Technical strategy {name} not fitted, skipping")
                continue
            try:
                result = strategy.generate_signals(date, prices, features)
                if not result.signals.empty:
                    results[name] = result.signals
            except Exception as e:
                logger.error(f"Technical strategy {name} failed: {e}")
        return results

    def _detect_regime(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None,
    ) -> str | None:
        """시장 레짐 감지."""
        if self.regime_classifier is not None:
            try:
                return self.regime_classifier.predict(date, prices, features)
            except Exception as e:
                logger.error(f"Regime classifier failed: {e}")

        # 기본 레짐 감지 (가격 기반)
        try:
            latest = prices.sort_values("date").groupby("ticker")["close"].last()
            recent = prices[prices["date"] >= prices["date"].max() - pd.Timedelta(days=20)]
            earliest = recent.sort_values("date").groupby("ticker")["close"].first()
            common = latest.index.intersection(earliest.index)
            if len(common) < 5:
                return None
            median_ret = (latest[common] / earliest[common] - 1).median()
            if median_ret > 0.03:
                return "bull"
            elif median_ret < -0.03:
                return "bear"
            return "sideways"
        except Exception:
            return None

    def _process_news(
        self, messages: list[dict] | None
    ) -> tuple[list[str], dict[str, str]]:
        """뉴스 메시지 처리 → 헤드라인 + 종목별 뉴스."""
        if not messages:
            return [], {}

        headlines = []
        ticker_news: dict[str, list[str]] = {}

        for msg in messages[-30:]:  # 최근 30개
            text = msg.get("text", "").strip()
            if not text:
                continue
            # 헤드라인: 첫 줄만
            headline = text.split("\n")[0][:200]
            headlines.append(headline)

        # 종목별 뉴스 매핑은 텍스트에서 티커 추출 필요 (간소화)
        ticker_news_summary = {
            k: "\n".join(v[:3])
            for k, v in ticker_news.items()
        }

        return headlines[-10:], ticker_news_summary  # 최근 10개 헤드라인

    def _merge_stock_contexts(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None,
        ml_signals: dict[str, pd.DataFrame],
        technical_signals: dict[str, pd.DataFrame],
        ticker_news: dict[str, str],
        top_k: int = 50,
    ) -> list[StockContext]:
        """모든 데이터를 종목별로 통합."""
        # 최신 가격 데이터
        prices_filtered = prices[prices["date"] <= pd.Timestamp(date)]
        latest = prices_filtered.sort_values("date").groupby("ticker").tail(1).copy()

        # 거래량 기준 상위 종목 (ML 스코어가 있는 종목 우선)
        ml_tickers = set()
        for df in ml_signals.values():
            if "ticker" in df.columns:
                ml_tickers.update(df["ticker"].tolist())

        # ML 신호가 있는 종목 + 거래량 상위 종목
        if "volume" in latest.columns:
            vol_top = set(latest.nlargest(top_k * 2, "volume")["ticker"].tolist())
        else:
            vol_top = set(latest["ticker"].tolist()[:top_k * 2])

        candidate_tickers = list(ml_tickers | vol_top)[:top_k]

        # 펀더멘털 merge
        fund_data = {}
        if features is not None and not features.empty:
            fund_latest = features.sort_values("date").groupby("ticker").tail(1)
            for _, row in fund_latest.iterrows():
                t = row.get("ticker")
                if t:
                    fund_data[t] = row.to_dict()

        # 종목별 컨텍스트 생성
        stocks = []
        for ticker in candidate_tickers:
            ticker_prices = prices_filtered[
                prices_filtered["ticker"] == ticker
            ].sort_values("date")

            if ticker_prices.empty:
                continue

            last_row = ticker_prices.iloc[-1]
            close = last_row.get("close", 0)

            # 수익률 계산
            closes = ticker_prices["close"].values
            chg_1d = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0
            chg_5d = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0

            # ML 스코어 수집
            ml_scores = {}
            for strat_name, sig_df in ml_signals.items():
                match = sig_df[sig_df["ticker"] == ticker]
                if not match.empty:
                    ml_scores[strat_name] = float(match.iloc[0]["score"])

            ml_avg = np.mean(list(ml_scores.values())) if ml_scores else 0.0

            # 기술적 지표 (가격 데이터에서)
            rsi_14 = last_row.get("rsi_14")
            sma_5 = last_row.get("sma_5")
            sma_20 = last_row.get("sma_20")
            sma_60 = last_row.get("sma_60")

            # 펀더멘털
            fund = fund_data.get(ticker, {})

            ctx = StockContext(
                ticker=ticker,
                close=float(close),
                change_1d=float(chg_1d),
                change_5d=float(chg_5d),
                volume=int(last_row.get("volume", 0)),
                ml_scores=ml_scores,
                ml_avg_score=float(ml_avg),
                rsi_14=_safe_float(rsi_14),
                sma_5=_safe_float(sma_5),
                sma_20=_safe_float(sma_20),
                sma_60=_safe_float(sma_60),
                per=_safe_float(fund.get("per", last_row.get("pe_ratio"))),
                pbr=_safe_float(fund.get("pbr", last_row.get("pb_ratio"))),
                roe=_safe_float(fund.get("roe")),
                ev_ebitda=_safe_float(fund.get("ev_ebitda")),
                news_summary=ticker_news.get(ticker, ""),
            )
            ctx.ml_signal_strength = ctx.signal_strength
            stocks.append(ctx)

        # ML 평균 스코어 절대값 기준 정렬 (가장 강한 신호 먼저)
        stocks.sort(key=lambda s: abs(s.ml_avg_score), reverse=True)

        return stocks[:top_k]

    def _build_market_summary(
        self, prices: pd.DataFrame, regime: str | None
    ) -> str:
        """시장 전체 요약 생성."""
        try:
            latest_date = prices["date"].max()
            recent = prices[
                prices["date"] >= latest_date - pd.Timedelta(days=30)
            ]
            daily_returns = (
                recent.groupby("date")["close"].mean().pct_change().dropna()
            )
            if len(daily_returns) == 0:
                return "Insufficient data"

            avg_ret = daily_returns.mean() * 100
            vol = daily_returns.std() * np.sqrt(252) * 100
            cumul = ((1 + daily_returns).prod() - 1) * 100
            n_up = (daily_returns > 0).sum()
            n_total = len(daily_returns)

            regime_str = regime.upper() if regime else "UNKNOWN"

            return (
                f"Regime: {regime_str} | "
                f"30d Avg Daily Return: {avg_ret:+.2f}% | "
                f"Annualized Vol: {vol:.1f}% | "
                f"30d Cumulative: {cumul:+.1f}% | "
                f"Up Days: {n_up}/{n_total}"
            )
        except Exception:
            return "Market data unavailable"


def _safe_float(val: Any) -> float | None:
    """NaN-safe float conversion."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None
