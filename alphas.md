# Alpha Strategies 상세 문서

## 목차

1. [전체 아키텍처](#1-전체-아키텍처)
2. [데이터 파이프라인](#2-데이터-파이프라인)
3. [Rule-Based Alphas](#3-rule-based-alphas)
   - 3.1 RSI Reversal
   - 3.2 Volatility Breakout
   - 3.3 Value F-Score
   - 3.4 Sentiment Long
4. [ML Alphas](#4-ml-alphas)
   - 4.1 Return Prediction (XGBoost)
   - 4.2 Intraday Pattern (LightGBM)
   - 4.3 Volatility Forecast (XGBoost)
   - 4.4 Regime Classifier (RandomForest)
5. [피처 엔지니어링](#5-피처-엔지니어링)
6. [라벨 엔지니어링](#6-라벨-엔지니어링)
7. [앙상블 통합](#7-앙상블-통합)
8. [실행 방법](#8-실행-방법)

---

## 1. 전체 아키텍처

```
src/alphas/
├── base_alpha.py              # 모든 알파의 추상 베이스 클래스
├── technical/
│   ├── rsi_reversal.py        # RSI 평균회귀 (룰기반)
│   └── vol_breakout.py        # 변동성 돌파 (룰기반)
├── fundamental/
│   ├── value_f_score.py       # Piotroski F-Score (룰기반)
│   └── sentiment_long.py      # 모멘텀+퀄리티 (룰기반)
└── ml/
    ├── base_ml_alpha.py       # ML 알파 베이스 (scaler, model 관리)
    ├── return_prediction.py   # 수익률 예측 (XGBoost)
    ├── intraday_pattern.py    # 인트라데이 패턴 (LightGBM)
    ├── volatility_forecast.py # 변동성 예측 (XGBoost)
    └── regime_classifier.py   # 시장 레짐 분류 (RandomForest)
```

### 클래스 계층 구조

```
BaseAlpha (ABC)
├── fit(prices, features, labels) → dict
├── generate_signals(date, prices, features) → AlphaResult
├── save_state(path) / load_state(path)
│
├── RSIReversalAlpha          ← 룰기반, fit()은 형식적
├── VolatilityBreakoutAlpha   ← 룰기반, fit()은 형식적
├── ValueFScoreAlpha          ← 룰기반, 펀더멘털 데이터 필요
├── SentimentLongAlpha        ← 룰기반, 모멘텀+퀄리티
│
└── BaseMLAlpha               ← ML 전용 확장
    ├── _build_model() → ML model (abstract)
    ├── fit() → StandardScaler + model.fit()
    ├── generate_signals() → model.predict() → score
    │
    ├── ReturnPredictionAlpha  (XGBRegressor)
    ├── IntradayPatternAlpha   (LGBMRegressor)
    └── VolatilityForecastAlpha(XGBRegressor, score 반전)

RegimeClassifier              ← BaseAlpha 상속하지 않음 (별도 인터페이스)
├── fit(market_features, regime_labels)
├── predict(market_features, date) → "bull" / "bear" / "sideways"
└── predict_proba() → {"bull": 0.5, "bear": 0.2, "sideways": 0.3}
```

### AlphaResult 구조

모든 알파의 `generate_signals()`가 반환하는 통일된 출력 형식:

```python
@dataclass
class AlphaResult:
    date: datetime
    signals: pd.DataFrame  # 반드시 asset_id, score 컬럼 포함
    metadata: dict          # 전략별 부가 정보
```

- `score > 0`: 매수 신호 (클수록 강한 확신)
- `score < 0`: 매도/회피 신호
- `score = 0`: 중립

---

## 2. 데이터 파이프라인

### 입력 데이터

| 데이터 | 형식 | 위치 | 스키마 |
|--------|------|------|--------|
| 일봉 OHLCV | Parquet | `data/processed/prices.parquet` | date, asset_id, open, high, low, close, volume |
| 분봉 데이터 | Parquet | `data/processed/minute_bars.parquet` | datetime, asset_id, open, high, low, close, volume |
| 피처 | Parquet | `data/processed/features.parquet` | date, asset_id, ret_1d, rsi_14, ... |
| 시장 피처 | Parquet | `data/processed/market_features.parquet` | date, market_ret, advance_decline_ratio, ... |
| 수익률 라벨 | Parquet | `data/processed/labels_return.parquet` | date, asset_id, y_reg (향후 5일 수익률) |
| 변동성 라벨 | Parquet | `data/processed/labels_volatility.parquet` | date, asset_id, y_reg (향후 5일 실현변동성) |
| 레짐 라벨 | Parquet | `data/processed/labels_regime.parquet` | date, y_reg (0/1/2), regime (bear/sideways/bull) |
| 인트라데이 라벨 | Parquet | `data/processed/labels_intraday.parquet` | date, asset_id, y_reg (익일 시가 갭) |

### 파이프라인 흐름

```
[1단계] 데이터 로드
data/raw/ (Bloomberg CSV)
    │
    ├── ParquetConverter (src/etl/converter.py)
    │   └── CSV → Parquet 변환, 컬럼명 매핑 (PX_LAST→close 등)
    │
    └── DataCleaner (src/etl/cleaner.py)
        └── 결측치 처리, 아웃라이어 제거, 음수 가격 제거

[2단계] 피처/라벨 빌드 (--build-features 옵션)
prices.parquet
    │
    ├── FeatureEngineer (src/etl/feature_engineer.py)
    │   ├── build_daily_features()   → features.parquet
    │   ├── build_intraday_features()→ (분봉→일별 집계)
    │   └── build_market_features()  → market_features.parquet
    │
    └── LabelEngineer (src/etl/label_engineer.py)
        ├── build_return_labels()    → labels_return.parquet
        ├── build_volatility_labels()→ labels_volatility.parquet
        ├── build_regime_labels()    → labels_regime.parquet
        └── build_intraday_labels()  → labels_intraday.parquet

[3단계] 학습
scripts/2_train_ensemble.py
    │
    ├── 룰기반 알파: fit()은 is_fitted=True만 설정 (파라미터 고정)
    ├── ML 알파: fit() → scaler.fit_transform(X) → model.fit(X, y)
    ├── RegimeClassifier: 별도 학습 → regime_classifier.joblib 저장
    │
    └── ModelManager → models/weights/*.joblib + registry.yaml

[4단계] 시그널 생성 + 트레이딩
scripts/3_run_trading.py
    │
    ├── RegimeClassifier.predict() → regime
    ├── EnsembleAgent.generate_signals(date, prices, features, regime)
    │   ├── 각 알파 → AlphaResult
    │   ├── 가중 평균 (성과 기반 + 레짐 조절)
    │   └── EnsembleSignal 반환
    │
    └── Allocator → 포지션 비중 → OrderManager → KIS API 주문
```

---

## 3. Rule-Based Alphas

### 3.1 RSI Reversal (`src/alphas/technical/rsi_reversal.py`)

**전략 유형**: 평균회귀 (Mean Reversion)

**핵심 로직**:
- RSI(14)가 30 아래 → 과매도 → 매수 신호 (score > 0)
- RSI(14)가 70 위 → 과매수 → 매도 신호 (score < 0)
- 중립 구간(30~70) → score = 0

**스코어 계산**:
```
과매도: score = (oversold - rsi) / oversold          # 0 ~ 1
과매수: score = -(rsi - overbought) / (100 - overbought)  # -1 ~ 0
중립:   score = 0
```

**파라미터** (`config/settings.py`):
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| rsi_period | 14 | RSI 계산 기간 |
| oversold | 30 | 과매도 임계값 |
| overbought | 70 | 과매수 임계값 |

**RSI 계산** (`_calculate_rsi()`):
- 최근 rsi_period+1개 종가에서 일일 변동(diff) 계산
- 상승분 평균(avg_gain)과 하락분 평균(avg_loss) 산출
- RS = avg_gain / avg_loss
- RSI = 100 - 100/(1+RS)

**데이터 요구사항**: 일봉 close만 있으면 동작. features에 `rsi_14` 컬럼이 있으면 사전계산된 값 사용.

**적합한 시장 환경**: 횡보장, 저변동성 장세

**metadata 출력**: `n_oversold`, `n_overbought` (과매도/과매수 종목 수)

---

### 3.2 Volatility Breakout (`src/alphas/technical/vol_breakout.py`)

**전략 유형**: 모멘텀 (Momentum / Breakout)

**핵심 로직**:
- 현재 종가가 최근 N일 최고가를 돌파하면 매수
- 현재 종가가 최근 N일 최저가를 이탈하면 매도
- 돌파 크기를 변동성(ATR-like)으로 나눠 정규화
- 거래량 확인(volume confirm)으로 신호 강도 조절

**스코어 계산**:
```
상향 돌파: score = min((close - highest_high) / atr, 3.0)
하향 이탈: score = -min((lowest_low - close) / atr, 3.0)
돌파 없음: score = (position_in_range - 0.5) * 0.5  # 약한 방향 편향

거래량 확인 (volume_ratio > 1.5일 때):
  score *= min(volume_ratio, 2.0)  # 최대 2배 부스트
```

**파라미터**:
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| lookback | 20 | 과거 범위 계산 기간 |
| breakout_threshold | 1.5 | ATR 배수 기준 (현재 스코어 계산에서 간접 사용) |
| volume_confirm | True | 거래량 확인 활성화 |

**변동성 계산**: `atr = std(hist_closes) * sqrt(252/lookback)` — 전통적 ATR이 아닌 표준편차 기반 연환산

**거래량 컬럼**: `PX_VOLUME` 또는 `volume` 자동 탐지

**적합한 시장 환경**: 추세장, 고변동성 장세

**metadata 출력**: `n_breakouts_up`, `n_breakouts_down`

---

### 3.3 Value F-Score (`src/alphas/fundamental/value_f_score.py`)

**전략 유형**: 밸류 (Value / Fundamental)

**핵심 로직**:
- Piotroski F-Score (간소화 버전)로 펀더멘털 품질 평가
- 높은 F-Score + 낮은 PBR 종목에 매수 신호
- 낮은 F-Score 종목에 매도 신호

**F-Score 계산** (현재 간소화된 4개 항목, 총 7점 만점):
| 항목 | 조건 | 점수 |
|------|------|------|
| ROE 양수 | `roe > 0` | +2 |
| 낮은 부채 | `debt_to_equity < 1` | +2 |
| 적정 PER | `0 < pe_ratio < 20` | +2 |
| 시가총액 | `market_cap > 1000억` | +1 |

**스코어 계산**:
```python
if f_score >= min_f_score and pb_ratio < max_pb_ratio:
    score = (f_score - min_f_score + 1) / 5  # 0~1 정규화
    score *= (max_pb_ratio - pb_ratio) / max_pb_ratio  # 낮은 PBR일수록 부스트
elif f_score < 3:
    score = -0.3  # 강한 매도 신호
else:
    score = 0.0
```

**파라미터**:
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| min_f_score | 5 | 매수 시그널 최소 F-Score |
| max_pb_ratio | 3.0 | PBR 상한 (밸류 필터) |

**데이터 요구사항**: features에 `roe`, `debt_to_equity`, `pe_ratio`, `market_cap`, `pb_ratio` 컬럼 필요. 없으면 빈 시그널 반환.

**적합한 시장 환경**: 전 구간. 특히 소형주에서 효과적.

---

### 3.4 Sentiment Long (`src/alphas/fundamental/sentiment_long.py`)

**전략 유형**: 모멘텀 + 퀄리티 (Quality Momentum)

**핵심 로직**:
- 가격 모멘텀(60일)과 펀더멘털 품질을 결합
- 모멘텀이 양수 + 펀더멘털 양호 = 강한 매수
- 역모멘텀(-0.3 이하) 종목은 시그널 50% 감쇄

**스코어 계산**:
```python
# 1. 가격 모멘텀 (60일)
momentum = (current_price - past_price) / past_price
momentum_score = clip(momentum * 5, -1, 1)

# 2. 퀄리티 스코어 (0~1)
quality_score = 0
if roe > 0.15:      quality_score += 0.4
elif roe > 0.10:    quality_score += 0.2
if debt_eq < 0.5:   quality_score += 0.3
elif debt_eq < 1.0: quality_score += 0.1
if market_cap > 1조: quality_score += 0.3

# 3. 가중 결합
combined = 0.6 * momentum_score + 0.4 * quality_score

# 4. 역모멘텀 페널티
if momentum_score < -0.3:
    combined *= 0.5
```

**파라미터**:
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| momentum_weight | 0.6 | 모멘텀 비중 |
| quality_weight | 0.4 | 퀄리티 비중 |
| momentum_lookback | 60 | 모멘텀 계산 기간 (거래일) |

**적합한 시장 환경**: 상승장, 퀄리티 랠리

---

## 4. ML Alphas

### 공통 구조 (BaseMLAlpha)

모든 ML 알파는 `BaseMLAlpha`를 상속하며 다음을 자동으로 처리:

**fit() 흐름**:
```python
1. features.merge(labels, on=["date", "asset_id"])  # 피처-라벨 정합
2. X = merged[self.feature_columns].values           # 피처 추출
3. NaN 행 제거
4. StandardScaler().fit_transform(X)                  # 피처 정규화
5. self._build_model().fit(X_scaled, y)               # 모델 학습
```

**generate_signals() 흐름**:
```python
1. features[date <= target_date]                      # lookahead 방지
2. groupby("asset_id").last()                         # 종목별 최신 피처
3. scaler.transform(X)                                # 동일 스케일러 적용
4. model.predict(X_scaled)                            # 예측
5. AlphaResult(signals=DataFrame(asset_id, score))    # 결과 반환
```

**state 저장**: model + scaler + feature_columns를 하나의 joblib 파일에 번들

---

### 4.1 Return Prediction (`src/alphas/ml/return_prediction.py`)

**목적**: 향후 N일(기본 5일) 주식 수익률 예측

**모델**: XGBRegressor

**역할**: "어떤 종목을 살 것인가" — 가장 직접적인 매매 시그널

**피처 목록** (16개):
```
일봉 테크니컬 (14개):
├── ret_1d              1일 수익률
├── ret_5d              5일 수익률
├── ret_20d             20일 수익률
├── ret_60d             60일 수익률
├── ma_ratio_5          close / MA(5) - 1
├── ma_ratio_20         close / MA(20) - 1
├── ma_ratio_60         close / MA(60) - 1
├── rsi_14              RSI(14)
├── bb_pct_b            Bollinger Band %B (0~1)
├── macd                MACD (EMA12 - EMA26)
├── macd_signal         MACD Signal (MACD의 EMA9)
├── vol_5d              5일 실현변동성 (연환산)
├── vol_20d             20일 실현변동성 (연환산)
└── volume_ratio_20d    거래량 / 거래량MA(20)

분봉 파생 (2개):
├── intraday_vol        일중 분봉 수익률의 표준편차
└── open_close_gap      시가 갭 ((시가-전일종가)/전일종가)
```

**라벨**: `y_reg = 향후 5일 수익률` (pct_change(5).shift(-5))

**모델 하이퍼파라미터** (`config/settings.py`):
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| n_estimators | 500 | 부스팅 라운드 수 |
| max_depth | 6 | 트리 최대 깊이 |
| learning_rate | 0.05 | 학습률 |
| subsample | 0.8 | 행 샘플링 비율 |
| colsample_bytree | 0.8 | 열 샘플링 비율 |
| min_child_weight | 10 | 리프 노드 최소 가중치 |
| reg_alpha | 0.1 | L1 정규화 |
| reg_lambda | 1.0 | L2 정규화 |

**스코어 해석**:
- score > 0: 향후 상승 예상 → 매수
- score < 0: 향후 하락 예상 → 회피
- 절대값이 클수록 강한 확신

---

### 4.2 Intraday Pattern (`src/alphas/ml/intraday_pattern.py`)

**목적**: 분봉 마이크로스트럭처 패턴에서 익일 수익률 예측

**모델**: LGBMRegressor

**역할**: "어떤 종목을 살 것인가" — 분봉 데이터만의 고유 엣지

**학습하는 패턴들**:
- 장 초반 30분 강한 모멘텀 → 종가까지 지속 여부
- 비정상적 거래량 집중 → 정보 기반 거래(informed trading) 감지
- 높은 분봉 수익률 왜도/첨도 → 레짐 전환 시그널

**피처 목록** (17개):
```
인트라데이 마이크로스트럭처 (13개):
├── intraday_vol            일중 변동성 (분봉 ret의 std * sqrt(n))
├── bar_return_skew         분봉 수익률 왜도 (skewness)
├── bar_return_kurtosis     분봉 수익률 첨도 (kurtosis)
├── large_bar_count         큰 움직임 분봉 수 (|ret| > 2σ)
├── large_bar_ratio         large_bar_count / 전체 분봉 수
├── ret_first_30min         장 시작 30분 수익률 (09:00~10:00)
├── ret_last_30min          장 마감 30분 수익률 (14:30~15:30)
├── price_range_am          오전 고저 범위 / 시가
├── price_range_pm          오후 고저 범위 / 시가
├── vwap_deviation          종가 vs VWAP 괴리율
├── volume_concentration    상위 30분봉 거래량 / 전체 거래량
├── volume_profile_morning  오전 거래량 / 전체 거래량
└── intraday_realized_vol   일중 실현변동성 (연환산)

일봉 컨텍스트 (4개):
├── ret_1d                  전일 수익률
├── vol_20d                 20일 변동성
├── volume_ratio_20d        거래량 비율
└── rsi_14                  RSI(14)
```

**라벨**: `y_reg = (익일시가 - 금일종가) / 금일종가` (next_open_gap)

**모델 하이퍼파라미터**:
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| n_estimators | 400 | 부스팅 라운드 수 |
| max_depth | 5 | 트리 최대 깊이 |
| learning_rate | 0.05 | 학습률 |
| subsample | 0.7 | 행 샘플링 (과적합 방지 강화) |
| colsample_bytree | 0.7 | 열 샘플링 |
| min_child_samples | 20 | 리프 노드 최소 샘플 수 |

**분봉 데이터 전처리** (`FeatureEngineer.build_intraday_features()`):
- 분봉 원본을 직접 모델에 넣지 않음
- 종목×일 단위로 집계하여 일별 1row로 변환
- 시간 윈도우: 오전(~10:00), 점심(11:30~13:00), 오후(14:30~)
- 최소 10개 분봉이 있어야 해당 날짜 피처 생성

**LightGBM 선택 이유**: 분봉 집계 데이터는 카디널리티가 높고 피처 간 상호작용이 중요. LightGBM은 XGBoost 대비 학습 속도가 빠르고, histogram-based splitting이 노이즈가 많은 분봉 피처에 적합.

---

### 4.3 Volatility Forecast (`src/alphas/ml/volatility_forecast.py`)

**목적**: 향후 N일(기본 5일) 실현변동성 예측

**모델**: XGBRegressor

**역할**: "얼마나 살 것인가" — 리스크 알파. 포지션 사이징에 사용.

**다른 알파와의 차이점**:
- 다른 알파: "이 종목을 사라/팔아" → 방향성 시그널
- 이 알파: "이 종목을 많이/적게 사라" → 규모 시그널
- `generate_signals()`에서 score를 반전 (`score = -predicted_vol`)
- 변동성 낮은 종목 → score 높음 → 큰 포지션 허용

**피처 목록** (15개):
```
변동성 관련 (8개):
├── vol_5d              5일 실현변동성 (연환산)
├── vol_20d             20일 실현변동성
├── vol_of_vol          변동성의 변동성 (vol_20d의 20일 std)
├── vol_ratio_5_20      단기/장기 변동성 비율 (vol 평균회귀 특성)
├── parkinson_vol       Parkinson 변동성 (고저가 기반, 더 효율적 추정치)
├── garman_klass_vol    Garman-Klass 변동성 (OHLC 전부 활용)
├── ret_abs_ma5         |수익률| 5일 이동평균
└── range_ratio_ma20    (고가-저가)/종가의 20일 평균

수익률/가격 (3개):
├── ret_1d              1일 수익률
├── ret_5d              5일 수익률
└── range_ratio         당일 (고가-저가)/종가

거래량 (1개):
└── volume_ratio_20d    거래량 이평 비율

분봉 기반 (3개) ← 변동성 예측의 핵심
├── intraday_realized_vol   분봉에서 계산한 일중 실현변동성
├── intraday_vol            일중 변동성
└── large_bar_ratio         큰 움직임 분봉 비율
```

**라벨**: `y_reg = 향후 5일 실현변동성 (연환산)`
```python
y_reg = daily_return.shift(-5).rolling(5).std() * sqrt(252)
```

**추가 메서드** — `predict_volatility(date, features)`:
- `generate_signals()`과 달리 원본 변동성 예측값을 반환 (반전 없음)
- Allocator가 직접 `predicted_vol` 컬럼을 사용하여 포지션 사이징 가능
- 반환값: `DataFrame(asset_id, predicted_vol)`
- `predicted_vol`의 하한값: 0.01 (음수 방지)

**분봉 데이터가 핵심인 이유**:
- HAR-RV(Heterogeneous Autoregressive Realized Volatility) 모델의 핵심 원리
- 일봉으로 계산한 변동성은 하루에 1개 관측치
- 분봉으로 계산하면 하루에 ~360개 관측치 → 훨씬 정확한 변동성 추정
- `intraday_realized_vol`이 feature importance에서 가장 높을 것으로 예상

**metadata 출력**: `type: "risk_alpha"`

---

### 4.4 Regime Classifier (`src/alphas/ml/regime_classifier.py`)

**목적**: 현재 시장이 상승장/횡보장/하락장 중 어디인지 분류

**모델**: RandomForestClassifier

**역할**: "전략 가중치를 어떻게 조절할 것인가" — 메타 전략

**BaseAlpha를 상속하지 않는 이유**:
- 입력 단위가 다름: 종목×일이 아니라 시장×일 (1 row per date)
- 출력 형태가 다름: 종목별 score가 아니라 단일 문자열("bull"/"bear"/"sideways")
- 별도의 save()/load() 인터페이스 사용

**피처 목록** (9개, 모두 시장 레벨):
```
├── market_ret              당일 시장 평균 수익률
├── market_ret_5d           5일 누적 시장 수익률
├── market_ret_20d          20일 누적 시장 수익률
├── market_vol_20d          20일 시장 변동성 (연환산)
├── cross_sectional_vol     종목 간 수익률 분산 (횡단면 변동성)
├── advance_decline_ratio   상승종목수 / 하락종목수
├── pct_above_ma20          20일 이평선 위 종목 비율
├── market_breadth          52주 신고가 수 - 52주 신저가 수
└── volume_trend            전체 거래량 / 거래량MA(20) - 1
```

**라벨 생성** (`LabelEngineer.build_regime_labels()`):
```python
forward_ret = 향후 20일 시장 누적 수익률
if forward_ret > 0.03:   → 2 (bull)
if forward_ret < -0.03:  → 0 (bear)
else:                     → 1 (sideways)
```

**모델 하이퍼파라미터**:
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| n_estimators | 300 | 트리 수 |
| max_depth | 8 | 트리 최대 깊이 |
| min_samples_leaf | 20 | 리프 최소 샘플 |
| class_weight | balanced | 클래스 불균형 자동 보정 |

**레짐별 전략 가중치 조절** (`config/settings.py`의 `regime_preferences`):

| 전략 | Bull 배수 | Bear 배수 | Sideways 배수 |
|------|-----------|-----------|---------------|
| rsi_reversal | 0.7 | 1.3 | 1.5 |
| vol_breakout | 1.5 | 0.5 | — |
| value_f_score | — | 1.2 | 1.0 |
| sentiment_long | 1.3 | — | — |
| return_prediction | 1.3 | 0.8 | 1.0 |
| intraday_pattern | 1.0 | — | 1.3 |
| volatility_forecast | — | 1.5 | — |

적용 방식: `EnsembleAgent._apply_regime_adjustment(regime)`에서 해당 레짐의 배수를 현재 가중치에 곱한 후 정규화.

**사용 흐름**:
```python
# 학습 시
classifier = RegimeClassifier(config=REGIME_CLASSIFIER)
classifier.fit(market_features, regime_labels)
classifier.save(MODELS_DIR / "weights" / "regime_classifier.joblib")

# 추론 시
classifier = RegimeClassifier.load(path)
regime = classifier.predict(market_features, date=today)  # "bull"
proba = classifier.predict_proba(market_features, date=today)
# → {"bear": 0.15, "sideways": 0.25, "bull": 0.60}

ensemble.generate_signals(today, prices, features, regime=regime)
```

---

## 5. 피처 엔지니어링

`src/etl/feature_engineer.py` — `FeatureEngineer` 클래스

### 5.1 일봉 피처 (`build_daily_features()`)

입력: `prices.parquet` (date, asset_id, open, high, low, close, volume)

| 카테고리 | 피처명 | 계산 방식 |
|----------|--------|-----------|
| **수익률** | ret_1d | close.pct_change(1) |
| | ret_5d | close.pct_change(5) |
| | ret_20d | close.pct_change(20) |
| | ret_60d | close.pct_change(60) |
| **이동평균** | ma_ratio_5 | close / MA(5) - 1 |
| | ma_ratio_20 | close / MA(20) - 1 |
| | ma_ratio_60 | close / MA(60) - 1 |
| **오실레이터** | rsi_14 | RSI(14) - Wilder 방식 |
| | bb_pct_b | (close - BB_lower) / BB_width, clipped 0~1 |
| | macd | EMA(12) - EMA(26) |
| | macd_signal | MACD의 EMA(9) |
| **변동성** | vol_5d | log_ret_1d.rolling(5).std() * sqrt(252) |
| | vol_20d | log_ret_1d.rolling(20).std() * sqrt(252) |
| | vol_of_vol | vol_20d.rolling(20).std() |
| | vol_ratio_5_20 | vol_5d / vol_20d |
| | parkinson_vol | sqrt(mean(ln(H/L)^2) / (4*ln2)), 20일 window |
| | garman_klass_vol | 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2, 20일 |
| **거래량** | volume_ratio_20d | volume / volume.rolling(20).mean() |
| **기타** | ret_abs_ma5 | abs(ret_1d).rolling(5).mean() |
| | range_ratio | (high - low) / close |
| | range_ratio_ma20 | range_ratio.rolling(20).mean() |

모든 피처는 `groupby("asset_id")` 내에서 계산하여 종목 간 데이터 오염 방지.

### 5.2 분봉 피처 (`build_intraday_features()`)

입력: 분봉 DataFrame (datetime, asset_id, open, high, low, close, volume)

시간대 정의:
- 장 초반: 09:00 ~ 10:00
- 오전: ~ 11:30
- 점심: 11:30 ~ 13:00
- 오후: 13:00 ~
- 장 마감: 14:30 ~ 15:30

| 피처명 | 계산 방식 |
|--------|-----------|
| intraday_vol | bar_ret.std() * sqrt(n_bars) |
| bar_return_skew | bar_ret.skew() |
| bar_return_kurtosis | bar_ret.kurtosis() |
| large_bar_count | count(abs(bar_ret) > 2*std) |
| large_bar_ratio | large_bar_count / n_bars |
| ret_first_30min | morning_last_close / morning_first_open - 1 |
| ret_last_30min | afternoon_last_close / afternoon_first_open - 1 |
| price_range_am | (am_high - am_low) / am_first_close |
| price_range_pm | (pm_high - pm_low) / pm_first_close |
| vwap_deviation | last_close / VWAP - 1 |
| volume_concentration | top30분_volume / total_volume |
| volume_profile_morning | morning_volume / total_volume |
| intraday_realized_vol | bar_ret.std() * sqrt(252 * n_bars) |

최소 분봉 수: 10개 미만이면 해당 종목×일 건너뜀.

### 5.3 시장 피처 (`build_market_features()`)

입력: 전 종목 일봉 데이터

출력: 날짜별 1행 (종목 축 없음)

| 피처명 | 계산 방식 |
|--------|-----------|
| market_ret | 전 종목 수익률 평균 |
| market_ret_5d | market_ret.rolling(5).sum() |
| market_ret_20d | market_ret.rolling(20).sum() |
| market_vol_20d | market_ret.rolling(20).std() * sqrt(252) |
| cross_sectional_vol | 종목 수익률들의 std (날짜별 횡단면) |
| advance_decline_ratio | 상승종목수 / 하락종목수 |
| pct_above_ma20 | (close > MA20)인 종목 비율 |
| market_breadth | 52주 신고가 수 - 52주 신저가 수 |
| volume_trend | total_volume / total_volume.rolling(20).mean() - 1 |

---

## 6. 라벨 엔지니어링

`src/etl/label_engineer.py` — `LabelEngineer` 클래스

모든 라벨은 `y_reg` 컬럼을 가지며, 이것이 ML 모델의 학습 타깃.

### 6.1 수익률 라벨 (`build_return_labels()`)

```python
y_reg = close.pct_change(horizon).shift(-horizon)
# 기본 horizon = 5 (5 거래일 = 약 1주)
```

- 단위: 종목×일
- 범위: 보통 -0.15 ~ +0.15
- NaN: 마지막 horizon일은 라벨 없음 (미래 데이터 부재)

### 6.2 변동성 라벨 (`build_volatility_labels()`)

```python
daily_ret = close.pct_change()
y_reg = daily_ret.shift(-horizon).rolling(horizon).std() * sqrt(252)
# 기본 horizon = 5
```

- 단위: 종목×일
- 범위: 보통 0.05 ~ 1.0 (연환산)
- 음수 불가 (변동성은 항상 양수)

### 6.3 레짐 라벨 (`build_regime_labels()`)

```python
market_avg_close = prices.groupby("date")["close"].mean()
market_ret = market_avg_close.pct_change()
forward_cumret = market_ret.shift(-1).rolling(horizon).apply(lambda x: (1+x).prod()-1)

if forward_cumret > 0.03:   y_reg = 2  # bull
elif forward_cumret < -0.03: y_reg = 0  # bear
else:                         y_reg = 1  # sideways
```

- 단위: 날짜 (종목 축 없음, 시장 전체)
- 기본 horizon = 20 거래일 (약 1개월)
- threshold 설정: `config/settings.py`의 `REGIME_CLASSIFIER`

### 6.4 인트라데이 라벨 (`build_intraday_labels()`)

```python
# next_open_gap (기본)
y_reg = (next_day_open - today_close) / today_close

# 또는 next_close_ret
y_reg = close.pct_change().shift(-1)
```

- 단위: 종목×일
- `next_open_gap`은 오버나이트 갭을 예측 → 장중 패턴과의 관계 학습

---

## 7. 앙상블 통합

### EnsembleAgent (`src/ensemble/agent.py`)

**시그널 결합 방식**:
```python
for each asset:
    weighted_score = Σ (strategy_weight[i] * strategy_score[i])
    final_score = weighted_score / Σ weights
```

**가중치 업데이트** (`_update_weights()`):
1. ScoreBoard에서 최근 21일 성과 조회
2. 전략별 Sharpe Ratio 계산 (하한 0.01)
3. Softmax-like 가중치: `perf_weight = sharpe / Σ sharpes`
4. 베이스 가중치와 블렌딩: `w = 0.5 * base_w + 0.5 * perf_w`
5. 레짐 조절: `w *= regime_multiplier`
6. 정규화: `w = w / Σ w`

### ScoreBoard (`src/ensemble/score_board.py`)

전략별 성과 추적:
- 일별 수익률 기록 (최대 504일 = 2년)
- Rolling Sharpe, 승률, 누적수익률 계산
- 전략 간 상관행렬 계산
- 최대 낙폭(MDD) 추적

### Allocator (`src/ensemble/allocator.py`)

시그널 → 포트폴리오 비중 변환:

| Allocator | 방식 | 용도 |
|-----------|------|------|
| TopKAllocator | Top K 균등 비중 | 단순, 안정적 |
| ScoreWeightedAllocator | 스코어 비례 비중 | 확신도 반영 |
| RiskParityAllocator | 역변동성 비중 | 리스크 균형 |
| BlackLittermanAllocator | 스코어/변동성 비중 | 이론적 최적 |

**VolatilityForecastAlpha와의 연동**:
- `RiskParityAllocator`가 과거 변동성 대신 `predict_volatility()`의 예측값을 사용하면
  미래 지향적 리스크 관리 가능

---

## 8. 실행 방법

### 전체 워크플로우 (피처 빌드 → 학습)

```bash
# prices.parquet에서 피처/라벨 자동 생성 후 전체 학습
python scripts/2_train_ensemble.py --build-features --train-end 2023-12-31
```

### 룰기반만 학습

```bash
python scripts/2_train_ensemble.py --strategies rsi_reversal,vol_breakout,value_f_score,sentiment_long
```

### ML만 학습

```bash
python scripts/2_train_ensemble.py --strategies return_prediction,intraday_pattern,volatility_forecast
```

### 특정 조합 학습

```bash
python scripts/2_train_ensemble.py --strategies rsi_reversal,return_prediction,volatility_forecast
```

### 트레이딩 실행

```bash
python scripts/3_run_trading.py
```

### 저장 구조

```
models/
├── weights/
│   ├── rsi_reversal.joblib
│   ├── vol_breakout.joblib
│   ├── return_prediction.joblib     # model + scaler + feature_columns 번들
│   ├── intraday_pattern.joblib
│   ├── volatility_forecast.joblib
│   ├── regime_classifier.joblib     # 별도 저장 (BaseAlpha 아님)
│   └── ensemble.joblib              # 가중치, score_board, 성과 이력
├── history/
│   └── 20260209_093000/             # 타임스탬프별 스냅샷
│       ├── weights/
│       └── registry.yaml
└── registry.yaml                     # 현재 모델 메타데이터 + 버전
```
