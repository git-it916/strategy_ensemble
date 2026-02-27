import pandas as pd
import numpy as np
from typing import Any
from src.alphas.base_alpha import BaseAlpha, AlphaResult

class CSMomentum(BaseAlpha):
    """
    Cross-Sectional Momentum 전략
    로직: lookback 기간의 수익률에서 최근 skip_days를 제외한 순수 모멘텀 측정.
    크립토 시장에 맞게 기본 60일 lookback, 5일 skip (단기 반전 방지).
    데이터: 일봉 가격 데이터
    """

    def __init__(self, name="CSMomentum", lookback_days=60, skip_days=5):
        super().__init__(name)
        if skip_days >= lookback_days:
            raise ValueError(
                f"skip_days ({skip_days}) must be < lookback_days ({lookback_days})"
            )
        self.lookback_days = lookback_days
        self.skip_days = skip_days

    def fit(self, prices: pd.DataFrame, features: pd.DataFrame = None, labels: pd.DataFrame = None) -> dict[str, Any]:
        self.is_fitted = True
        self._fit_date = pd.Timestamp.now()
        return {"status": "fitted", "lookback_days": self.lookback_days, "skip_days": self.skip_days}

    def generate_signals(self, date: pd.Timestamp, prices: pd.DataFrame, features: pd.DataFrame = None) -> AlphaResult:
        """
        시그널 생성
        prices: 전체 기간 데이터가 들어오더라도 date 이전만 사용하도록 필터링
        """
        # Look-ahead bias 방지: date 이전 데이터만 사용
        past_prices = prices[prices['date'] <= date].copy()

        # 각 종목별로 모멘텀 계산
        # 12개월 전 가격과 1개월 전 가격의 변화율
        tickers = past_prices['ticker'].unique()
        mom_list = []

        for ticker in tickers:
            ticker_data = past_prices[past_prices['ticker'] == ticker].sort_values('date')

            if len(ticker_data) < self.lookback_days:
                continue

            price_start = ticker_data.iloc[-self.lookback_days]['close']
            price_end = ticker_data.iloc[-self.skip_days]['close']

            if price_start == 0 or pd.isna(price_start) or pd.isna(price_end):
                continue

            momentum = (price_end - price_start) / price_start
            mom_list.append({'ticker': ticker, 'momentum': momentum})

        if not mom_list:
            return AlphaResult(date=date, signals=pd.DataFrame(columns=['ticker', 'score']))

        mom_df = pd.DataFrame(mom_list)

        # 스코어링: 상대적 순위를 -1 ~ 1로 정규화
        mom_df['rank'] = mom_df['momentum'].rank(pct=True)
        mom_df['score'] = (mom_df['rank'] - 0.5) * 2

        return AlphaResult(
            date=date,
            signals=mom_df[['ticker', 'score']],
            metadata={'n_tickers': len(mom_df)}
        )

    def _get_extra_state(self) -> dict:
        return {"lookback_days": self.lookback_days, "skip_days": self.skip_days}

    def _restore_extra_state(self, state: dict) -> None:
        self.lookback_days = state.get("lookback_days", 60)
        self.skip_days = state.get("skip_days", 5)
