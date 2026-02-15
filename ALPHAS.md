# Alpha Strategies - 전략 상세 가이드

본 문서는 strategy_ensemble 시스템에 구현된 7가지 알파 전략의 아이디어와 구현 세부사항을 다룹니다.

---

## 📊 전략 개요

| 전략명 | 유형 | 타겟 | 적합 시장 | 가중치 |
|--------|------|------|-----------|--------|
| RSI Reversal | Rule-Based | 평균회귀 | 횡보장, 저변동성 | 0.25 |
| Volatility Breakout | Rule-Based | 추세추종 | 추세장, 고변동성 | 0.25 |
| Value F-Score | Rule-Based | 가치투자 | 전천후 | 0.25 |
| Sentiment Long | Rule-Based | 모멘텀 | 상승장, 퀄리티 랠리 | 0.25 |
| Return Prediction | ML (XGBoost) | 수익률 예측 | 데이터 충분시 | 0.20 |
| Intraday Pattern | ML (LightGBM) | 단기 패턴 | 일중 데이터 있을 때 | 0.15 |
| Volatility Forecast | ML (XGBoost) | 리스크 관리 | 포지션 사이징 | 0.10 |

---

## 1. RSI Reversal Alpha

### 🎯 **핵심 아이디어**
RSI(Relative Strength Index)가 극단값에 도달했을 때 평균 회귀를 노리는 전략

### 📐 **구현 로직**

```python
if RSI < 30:
    signal = BUY (과매도)
    score = (30 - RSI) / 30  # 더 과매도일수록 높은 점수

elif RSI > 70:
    signal = SELL (과매수)
    score = (RSI - 70) / 30  # 더 과매수일수록 낮은 점수

else:
    score = 0.5 - abs(RSI - 50) / 100  # 중립
```

### 📊 **사용 지표**
- **RSI(14)**: 14일 상대강도지수
- **계산식**: `100 - (100 / (1 + RS))`
  - RS = 평균 상승폭 / 평균 하락폭

### ✅ **장점**
- 명확한 매매 시그널
- 과매수/과매도 구간 명확
- 구현 단순, 백테스트 용이
- 횡보장에서 효과적

### ❌ **단점**
- 강한 추세장에서 실패 (계속 과매수/과매도 유지)
- 30/70 임계값이 시장마다 다를 수 있음
- 단독 사용시 손실 가능성
- 거짓 신호 빈번

### 🎲 **적합한 시장**
- ✅ 횡보장 (Range-bound)
- ✅ 저변동성 구간
- ✅ 정상 거래량
- ❌ 강한 추세장 (Trending)
- ❌ 고변동성 구간

### ⚙️ **파라미터**
```python
rsi_period = 14          # RSI 계산 기간
oversold = 30            # 과매도 임계값
overbought = 70          # 과매수 임계값
```

---

## 2. Volatility Breakout Alpha

### 🎯 **핵심 아이디어**
가격이 과거 변동성 범위를 돌파할 때 추세 시작으로 판단하여 진입

### 📐 **구현 로직**

```python
# 1. ATR (Average True Range) 계산
ATR = rolling_mean(max(high - low, abs(high - prev_close), abs(low - prev_close)))

# 2. 돌파 여부 확인
upper_band = high_20d + ATR * threshold
lower_band = low_20d - ATR * threshold

if close > upper_band and volume > avg_volume * 1.5:
    signal = BUY (상승 돌파)
    score = (close - upper_band) / ATR

elif close < lower_band and volume > avg_volume * 1.5:
    signal = SELL (하락 돌파)
    score = -(lower_band - close) / ATR

# 3. 거래량 확인 (false breakout 필터링)
if volume_confirmation:
    score *= volume_ratio
```

### 📊 **사용 지표**
- **ATR(20)**: 20일 평균 진폭
- **Bollinger Band**: 20일 이동평균 ± 2 표준편차
- **Volume Ratio**: 현재 거래량 / 20일 평균 거래량

### ✅ **장점**
- 추세 초기 진입 가능
- 거래량 확인으로 신뢰도 향상
- 변동성 정규화로 종목간 비교 가능
- 강한 추세장에서 높은 수익

### ❌ **단점**
- False breakout 빈번 (가짜 돌파)
- 횡보장에서 손실 누적
- 늦은 청산으로 이익 반납 가능
- 슬리피지 큼 (돌파 시점 경쟁)

### 🎲 **적합한 시장**
- ✅ 추세장 (Trending)
- ✅ 고변동성 구간
- ✅ 거래량 충분한 종목
- ❌ 횡보장
- ❌ 저변동성 구간

### ⚙️ **파라미터**
```python
lookback = 20               # 돌파 기준 기간
breakout_threshold = 1.5    # ATR 배수
volume_confirm = True       # 거래량 확인 여부
```

### 💡 **개선 아이디어**
- Donchian Channel 활용
- 시간대별 돌파 강도 차별화
- 섹터 모멘텀과 결합

---

## 3. Value F-Score Alpha

### 🎯 **핵심 아이디어**
Piotroski F-Score를 활용한 재무적으로 건강하고 저평가된 종목 선별

### 📐 **F-Score 계산 (0~9점)**

#### **수익성 (Profitability) - 4점**
1. **ROA > 0**: 양의 자산수익률 (1점)
2. **CFO > 0**: 양의 영업현금흐름 (1점)
3. **ΔROA > 0**: ROA 증가 (1점)
4. **CFO > Net Income**: 현금흐름 > 당기순이익 (회계 품질) (1점)

#### **레버리지/유동성 (Leverage) - 3점**
5. **ΔDebt < 0**: 장기부채 감소 (1점)
6. **ΔCurrent Ratio > 0**: 유동비율 증가 (1점)
7. **No Dilution**: 주식 희석 없음 (1점)

#### **운영 효율성 (Operating Efficiency) - 2점**
8. **ΔGross Margin > 0**: 매출총이익률 증가 (1점)
9. **ΔAsset Turnover > 0**: 자산회전율 증가 (1점)

### 📐 **신호 생성 로직**

```python
# 1. F-Score 계산
f_score = sum([
    int(roe > 0),
    int(cfo > 0),
    int(roe > roe_prev),
    int(cfo > net_income),
    int(debt < debt_prev),
    int(current_ratio > current_ratio_prev),
    int(shares_outstanding <= shares_prev),
    int(gross_margin > gross_margin_prev),
    int(asset_turnover > asset_turnover_prev),
])

# 2. 가치 필터
is_value = (pbr < max_pb_ratio) and (per < max_pe_ratio)

# 3. 점수 부여
if f_score >= 7 and is_value:
    score = 1.0  # Strong BUY
elif f_score >= 5 and is_value:
    score = 0.6  # BUY
elif f_score <= 3:
    score = 0.0  # Avoid
else:
    score = 0.5  # Neutral
```

### 📊 **필요 데이터**
#### **재무제표**
- ROA (Return on Assets)
- Operating Cash Flow
- Net Income
- Gross Margin
- Asset Turnover
- Long-term Debt
- Current Ratio
- Shares Outstanding

#### **밸류에이션**
- PBR (Price to Book Ratio)
- PER (Price to Earnings Ratio)

### ✅ **장점**
- 학계에서 검증된 전략 (Piotroski 2000)
- 시장 중립적 (전천후 작동)
- 재무 건전성 기반 (안정적)
- 중소형주에서 특히 효과적
- 장기 투자 적합

### ❌ **단점**
- 분기별 재무제표 딜레이 (정보 지연)
- 대형주에서 효과 낮음 (이미 효율적)
- 단기 수익률 낮음
- 데이터 품질에 민감
- 회계 조작에 취약

### 🎲 **적합한 시장**
- ✅ 모든 시장 상황 (전천후)
- ✅ 중소형주
- ✅ 저유동성 종목
- ✅ 장기 투자
- ❌ 초단기 트레이딩

### ⚙️ **파라미터**
```python
min_f_score = 5        # 최소 F-Score 기준
max_pb_ratio = 3.0     # 최대 PBR (가치 필터)
max_pe_ratio = 20.0    # 최대 PER
```

### 📚 **참고 논문**
- Piotroski, J. D. (2000). "Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers"

---

## 4. Sentiment Long Alpha

### 🎯 **핵심 아이디어**
가격 모멘텀과 재무 품질을 결합한 퀄리티 모멘텀 전략

### 📐 **구현 로직**

```python
# 1. 가격 모멘텀 점수
price_momentum = (
    ret_60d * 0.5 +      # 3개월 수익률
    ret_20d * 0.3 +      # 1개월 수익률
    ret_5d * 0.2         # 1주 수익률
)
momentum_score = rank_normalize(price_momentum)

# 2. 품질 점수
quality_score = (
    rank_normalize(roe) * 0.4 +           # ROE
    rank_normalize(oper_margin) * 0.3 +   # 영업이익률
    rank_normalize(eps_growth) * 0.3      # EPS 성장률
)

# 3. 최종 점수
score = (
    momentum_score * momentum_weight +    # 기본 0.6
    quality_score * quality_weight        # 기본 0.4
)

# 4. 필터링
if score > 0.7 and volume_ratio > 0.8:
    signal = STRONG_BUY
elif score < 0.3:
    signal = AVOID
```

### 📊 **사용 지표**

#### **모멘텀 지표**
- `ret_60d`: 60일 수익률 (3개월)
- `ret_20d`: 20일 수익률 (1개월)
- `ret_5d`: 5일 수익률 (1주)

#### **품질 지표**
- `ROE`: 자기자본이익률
- `oper_margin`: 영업이익률
- `eps_growth`: EPS 성장률
- `revenue_growth`: 매출 성장률

### ✅ **장점**
- 단순 모멘텀보다 안정적
- 품질 필터로 함정 종목 제거
- 상승장에서 강력한 성과
- 기관/외국인 선호 종목 포착

### ❌ **단점**
- 하락장 초기 큰 손실 (추세 반전 늦음)
- 밸류에이션 무시 (고평가 위험)
- 군중 심리 추종 (crowded trade)
- 단기 변동성 높음

### 🎲 **적합한 시장**
- ✅ 상승장 (Bull Market)
- ✅ 퀄리티 랠리
- ✅ 기관 매수 우위
- ❌ 하락장 초기
- ❌ 로테이션 장세

### ⚙️ **파라미터**
```python
momentum_weight = 0.6      # 모멘텀 가중치
quality_weight = 0.4       # 품질 가중치
momentum_lookback = 60     # 모멘텀 계산 기간
min_quality_score = 0.5    # 최소 품질 점수
```

### 💡 **개선 아이디어**
- 뉴스 센티먼트 추가
- 애널리스트 의견 통합
- 소셜 미디어 감성 분석

---

## 5. Return Prediction Alpha (ML)

### 🎯 **핵심 아이디어**
XGBoost를 활용하여 기술적/펀더멘털 피처로부터 향후 N일 수익률 예측

### 📐 **ML 파이프라인**

```
입력(X): features.parquet
    ↓
[Feature Selection]
    - 기술적 지표 (RSI, MACD, MA 등)
    - 거래량 지표
    - 변동성 지표
    - (선택) 일중 패턴
    ↓
[XGBoost Regressor]
    - Target: 5일 후 수익률
    - Loss: MSE (Mean Squared Error)
    - Objective: 수익률 최대화
    ↓
출력(y_pred): 예상 수익률
    ↓
[Score Normalization]
score = rank_normalize(y_pred)
```

### 📊 **사용 피처 (29개)**

#### **수익률 피처 (5개)**
```python
ret_1d, ret_5d, ret_20d, ret_60d
log_ret_1d
```

#### **이동평균 피처 (4개)**
```python
ma_ratio_5, ma_ratio_10, ma_ratio_20, ma_ratio_60
# ma_ratio = (close / MA) - 1
```

#### **기술적 지표 (4개)**
```python
rsi_14              # RSI
bb_pct_b            # Bollinger %B
macd, macd_signal   # MACD
```

#### **변동성 피처 (6개)**
```python
vol_5d, vol_20d            # 단기/중기 변동성
vol_of_vol                  # 변동성의 변동성
vol_ratio_5_20             # 변동성 비율
parkinson_vol              # High-Low 기반
garman_klass_vol           # OHLC 기반
```

#### **거래량 피처 (2개)**
```python
volume_ratio_20d           # 거래량 비율
turnover                   # 회전율
```

#### **일중 패턴 (선택, 8개)**
```python
intraday_vol               # 일중 변동성
open_close_gap             # 시가-종가 갭
vwap_deviation             # VWAP 괴리
volume_concentration       # 거래량 집중도
ret_first_30min            # 초반 30분 수익률
ret_last_30min             # 마지막 30분 수익률
price_range_am/pm          # 오전/오후 변동폭
```

### 🔧 **XGBoost 하이퍼파라미터**

```python
n_estimators = 500         # 트리 개수
max_depth = 6              # 트리 깊이
learning_rate = 0.05       # 학습률
subsample = 0.8            # 샘플 비율
colsample_bytree = 0.8     # 피처 비율
min_child_weight = 10      # 최소 자식 가중치
reg_alpha = 0.1            # L1 정규화
reg_lambda = 1.0           # L2 정규화
```

### 📈 **학습 프로세스**

```python
# 1. 데이터 준비
X_train = features[features['date'] <= train_end]
y_train = labels_return[labels['date'] <= train_end]

# 2. 시계열 검증 (Purged K-Fold)
for fold in range(5):
    train_idx, val_idx = get_purged_fold(fold)

    # 3. 학습
    model.fit(X_train[train_idx], y_train[train_idx])

    # 4. 검증
    y_pred = model.predict(X_train[val_idx])
    ic = calculate_ic(y_pred, y_train[val_idx])

    # 5. 조기 종료
    if ic < 0.01:
        break

# 6. 전체 데이터로 재학습
model.fit(X_train, y_train)
```

### ✅ **장점**
- 비선형 패턴 학습 가능
- 피처 중요도 분석 가능
- 과적합 방지 (정규화)
- 다양한 시장 상황 대응

### ❌ **단점**
- 데이터 충분히 필요 (최소 2년+)
- 미래 레짐 변화 대응 어려움
- 블랙박스 (해석 어려움)
- 계산 비용 높음
- 리밸런싱 빈도 제한

### 🎲 **적합한 시장**
- ✅ 충분한 학습 데이터
- ✅ 안정적인 시장 구조
- ✅ 낮은 거래비용
- ❌ 급격한 레짐 체인지
- ❌ 극단적 시장 이벤트

### ⚙️ **파라미터**
```python
horizon = 5                # 예측 기간 (5일)
min_ic = 0.02             # 최소 Information Coefficient
max_corr = 0.9            # 피처간 최대 상관계수
feature_selection = True   # 자동 피처 선택
```

### 📊 **성능 지표**
- **IC (Information Coefficient)**: 예측과 실제 수익률 상관계수
  - IC > 0.05: 우수
  - IC > 0.03: 양호
  - IC < 0.01: 사용 불가
- **Rank IC**: 순위 상관계수 (더 robust)
- **Hit Rate**: 방향 예측 정확도

---

## 6. Intraday Pattern Alpha (ML)

### 🎯 **핵심 아이디어**
LightGBM으로 일중 미세구조(Microstructure) 패턴을 학습하여 단기 수익률 예측

### 📐 **핵심 가설**

```
가설 1: 장 초반 강한 매수 → 종가까지 지속
가설 2: 비정상적 거래량 집중 → 정보거래 존재
가설 3: 일중 변동성 패턴 → 다음날 방향 예측 가능
가설 4: VWAP 괴리도 → 기관 포지션 파악
```

### 📊 **일중 피처 (18개)**

#### **시간대별 수익률 (4개)**
```python
ret_first_30min       # 09:00-09:30 수익률
ret_last_30min        # 14:30-15:00 수익률
ret_morning           # 오전(09:00-12:00) 수익률
ret_afternoon         # 오후(12:00-15:00) 수익률
```

#### **변동성 패턴 (4개)**
```python
intraday_vol              # 일중 실현 변동성
intraday_realized_vol     # 분봉 수익률 표준편차
price_range_am            # 오전 변동폭
price_range_pm            # 오후 변동폭
```

#### **거래량 패턴 (5개)**
```python
volume_concentration      # 거래량 집중도 (Herfindahl Index)
volume_profile_morning    # 오전 거래량 비중
volume_profile_afternoon  # 오후 거래량 비중
large_bar_count          # 큰 거래 발생 횟수
large_bar_ratio          # 큰 거래 비율
```

#### **미세구조 지표 (5개)**
```python
bar_return_skew       # 분봉 수익률 왜도
bar_return_kurtosis   # 분봉 수익률 첨도
vwap_deviation        # VWAP 대비 종가 괴리
price_impact          # 거래량당 가격 변화
bid_ask_spread_proxy  # 스프레드 추정치
```

### 🔧 **LightGBM 하이퍼파라미터**

```python
n_estimators = 400         # 트리 개수
max_depth = 5              # 트리 깊이 (XGBoost보다 얕음)
learning_rate = 0.05       # 학습률
subsample = 0.7            # 샘플 비율
colsample_bytree = 0.7     # 피처 비율
min_child_samples = 20     # 최소 샘플 수
reg_alpha = 0.1            # L1 정규화
reg_lambda = 1.0           # L2 정규화
```

### 📈 **예측 타겟**

```python
# Option 1: 다음날 종가 수익률
y = (close_t+1 / close_t) - 1

# Option 2: 다음날 시가 갭
y = (open_t+1 / close_t) - 1

# Option 3: 다음날 VWAP 수익률 (현재 사용)
y = (vwap_t+1 / close_t) - 1
```

### ✅ **장점**
- 고빈도 정보 활용
- 정보거래자 행동 포착
- 단기 예측력 우수
- 시장 미세구조 반영

### ❌ **단점**
- 일중 데이터 필수 (분봉)
- 데이터 용량 큼
- 계산 비용 높음
- 슬리피지에 민감
- 거래비용 영향 큼

### 🎲 **적합한 시장**
- ✅ 일중 데이터 있을 때
- ✅ 유동성 높은 종목
- ✅ 낮은 거래비용
- ✅ 빠른 실행 가능
- ❌ 저유동성 종목
- ❌ 높은 거래비용

### ⚙️ **파라미터**
```python
prediction_horizon = 1     # 예측 기간 (1일)
bar_interval = '1min'      # 분봉 간격
min_bars = 100            # 최소 분봉 수
max_spread_bps = 20       # 최대 스프레드
```

### 💡 **실전 활용**
```python
# 09:00 장 시작 후
signals_open = intraday_model.predict(yesterday_patterns)

# 14:30 장 마감 전
signals_close = intraday_model.predict(today_patterns)

# 두 신호 결합
final_signal = signals_open * 0.3 + signals_close * 0.7
```

---

## 7. Volatility Forecast Alpha (ML)

### 🎯 **핵심 아이디어**
향후 변동성을 예측하여 리스크 관리 및 포지션 사이징에 활용

**⚠️ 중요**: 이 알파는 수익률이 아닌 **변동성(리스크)**을 예측합니다!

### 📐 **사용 목적**

```python
# 1. 포지션 사이징 (역변동성 가중)
position_size[ticker] = capital * (1 / predicted_vol[ticker])

# 2. 리스크 패리티
weight[ticker] = (1 / predicted_vol[ticker]) / sum(1 / predicted_vol)

# 3. 리스크 조정 수익률
risk_adjusted_return = expected_return / predicted_vol

# 4. 동적 레버리지
leverage = target_vol / portfolio_predicted_vol
```

### 📊 **변동성 피처 (15개)**

#### **실현 변동성 (4개)**
```python
vol_5d              # 5일 실현 변동성
vol_20d             # 20일 실현 변동성
vol_of_vol          # 변동성의 변동성
vol_ratio_5_20      # 단기/중기 변동성 비율
```

#### **OHLC 기반 변동성 추정 (2개)**
```python
parkinson_vol       # Parkinson 추정치
                    # sqrt(1/(4ln2) * ln(High/Low)^2)

garman_klass_vol    # Garman-Klass 추정치
                    # 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2
```

#### **수익률 기반 (3개)**
```python
ret_abs_ma5         # 5일 절대 수익률 이동평균
ret_1d, ret_5d      # 단기 수익률 (변동성 예측에 유용)
```

#### **가격 범위 (2개)**
```python
range_ratio         # (High - Low) / Close
range_ratio_ma20    # 20일 평균 range ratio
```

#### **거래량 (1개)**
```python
volume_ratio_20d    # 거래량과 변동성 상관 있음
```

#### **일중 변동성 (3개)**
```python
intraday_realized_vol   # 일중 실현 변동성
intraday_vol            # 일중 변동폭
large_bar_ratio         # 큰 변동 발생 빈도
```

### 🔧 **XGBoost 설정**

```python
n_estimators = 500
max_depth = 5              # 얕게 (변동성은 단순 패턴)
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
min_child_weight = 20      # 높게 (과적합 방지)
reg_alpha = 0.1
reg_lambda = 1.0
```

### 📈 **예측 타겟**

```python
# 향후 5일 실현 변동성 (연율화)
y = std(returns[t+1:t+6]) * sqrt(252)
```

### 🔄 **신호 생성 (특이점!)**

```python
# 일반 알파와 반대: 낮은 변동성 = 높은 점수
predicted_vol = model.predict(X)

score = -predicted_vol  # 부호 반전!
# or
score = 1 / predicted_vol  # 역수
```

**왜 반전?**
- 변동성이 낮을수록 → 안전 → 더 많이 보유 가능
- 변동성이 높을수록 → 위험 → 포지션 축소

### ✅ **장점**
- 리스크 관리 필수 도구
- 변동성 예측은 수익률보다 쉬움 (autocorrelation 높음)
- 포트폴리오 전체 리스크 제어
- 드로다운 감소

### ❌ **단점**
- 단독 사용 불가 (수익률 예측 아님)
- 극단적 이벤트 예측 실패
- VIX 급등시 과소추정
- 과거 의존성 높음

### 🎲 **적용 방법**

#### **방법 1: 포지션 사이징**
```python
# 각 종목에 동일한 리스크 할당
for ticker in universe:
    target_risk = 0.02  # 2% 리스크
    predicted_vol = vol_model.predict(ticker)
    position_size[ticker] = target_risk / predicted_vol
```

#### **방법 2: 필터링**
```python
# 고변동성 종목 제외
if predicted_vol > threshold:
    exclude_from_portfolio(ticker)
```

#### **방법 3: 신호 조정**
```python
# 다른 알파 신호를 변동성으로 조정
adjusted_signal = raw_signal / predicted_vol
```

### ⚙️ **파라미터**
```python
prediction_horizon = 5     # 예측 기간
min_vol = 0.05            # 최소 변동성 (5%)
max_vol = 1.0             # 최대 변동성 (100%)
vol_floor = 0.1           # 변동성 하한
```

### 📊 **평가 지표**
- **RMSE**: 예측 오차
- **Hit Rate**: 변동성 증감 방향 정확도
- **Rank Correlation**: 순위 상관계수

---

## 🎭 Ensemble Integration

### **레짐별 전략 가중치 조정**

```python
regime_preferences = {
    "bull": {  # 상승장
        "vol_breakout": 1.5,         # 돌파 전략 강화
        "sentiment_long": 1.3,        # 모멘텀 강화
        "return_prediction": 1.3,
        "rsi_reversal": 0.7,          # 평균회귀 약화
        "value_f_score": 0.8,
    },
    "bear": {  # 하락장
        "rsi_reversal": 1.3,          # 평균회귀 강화
        "value_f_score": 1.2,         # 가치투자 강화
        "volatility_forecast": 1.5,   # 리스크 관리 강화
        "vol_breakout": 0.5,          # 돌파 전략 약화
        "sentiment_long": 0.6,
    },
    "sideways": {  # 횡보장
        "rsi_reversal": 1.4,          # 평균회귀 최대
        "value_f_score": 1.1,
        "vol_breakout": 0.6,          # 돌파 최소
        "sentiment_long": 0.7,
    },
}
```

### **동적 가중치 업데이트**

```python
# 최근 21일 성과 기반
for strategy in strategies:
    recent_ic = calculate_ic(
        predictions=strategy.signals[-21:],
        actual_returns=actual_returns[-21:]
    )

    # IC가 높을수록 가중치 증가
    dynamic_weight[strategy] = base_weight[strategy] * (1 + recent_ic)

# 정규화
total = sum(dynamic_weight.values())
final_weights = {k: v/total for k, v in dynamic_weight.items()}
```

---

## 📊 전략 조합 예시

### **보수적 포트폴리오**
```python
strategies = [
    ValueFScoreAlpha(weight=0.4),
    RSIReversalAlpha(weight=0.3),
    VolatilityForecastAlpha(weight=0.3),  # 리스크 관리
]
```

### **공격적 포트폴리오**
```python
strategies = [
    VolatilityBreakoutAlpha(weight=0.3),
    SentimentLongAlpha(weight=0.3),
    ReturnPredictionAlpha(weight=0.4),
]
```

### **균형 포트폴리오 (기본)**
```python
strategies = [
    # Rule-based
    RSIReversalAlpha(weight=0.20),
    VolatilityBreakoutAlpha(weight=0.20),
    ValueFScoreAlpha(weight=0.15),
    SentimentLongAlpha(weight=0.15),
    # ML-based
    ReturnPredictionAlpha(weight=0.20),
    IntradayPatternAlpha(weight=0.10),
]
```

---

## 🔬 백테스트 권장사항

### **최소 데이터 요구사항**
- **Rule-based 전략**: 1년+ (252 거래일)
- **ML 전략**: 3년+ (756 거래일)
- **Regime Classifier**: 5년+ (여러 사이클 필요)

### **검증 방법**
```python
# Walk-forward validation
for train_end in pd.date_range('2020-12-31', '2023-12-31', freq='3M'):
    # 학습
    train_data = data[data['date'] <= train_end]
    ensemble.fit(train_data)

    # 테스트 (다음 3개월)
    test_start = train_end + timedelta(days=1)
    test_end = train_end + timedelta(days=90)
    test_data = data[(data['date'] >= test_start) &
                     (data['date'] <= test_end)]

    # 성과 측정
    signals = ensemble.generate_signals(test_data)
    returns = backtest(signals, test_data)

    metrics[train_end] = evaluate(returns)
```

---

## 🚀 다음 단계

이 알파 전략들을 기반으로:

1. **학습 실행**: `python scripts/2_train_ensemble.py --build-features`
2. **백테스트**: `python scripts/3_run_trading.py`
3. **성과 분석**: 각 전략별 IC, Sharpe Ratio 확인
4. **최적화**: 저성과 전략 제외 또는 파라미터 조정
5. **실전 배포**: 검증 완료 후 라이브 트레이딩

---

## 📚 참고문헌

- Piotroski, J. D. (2000). "Value Investing: The Use of Historical Financial Statement Information"
- Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and Selling Losers"
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
