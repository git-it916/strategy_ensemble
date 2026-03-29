# Sniper V2

**멀티 심볼 기술적 트레이딩 봇** — BTC(EMA 크로스), SOL(EMA 크로스), XRP(펀딩 역추세).
앙상블 데몬(`run_daemon.py`)과 **완전히 독립** 동작.
60개월(5년) 그리드서치 최적화 완료 (2026-03-26).

---

## 1. 아키텍처

V1(Confluence 10점 + HTF 필터)에서 V2로 리팩터링:
- **Confluence Score 제거** — 10개 지표 투표 대신 심볼별 최적 파라미터(`SymbolConfig`)
- **멀티 심볼** — BTC + SOL + XRP 동시 실행 (asyncio 병렬)
- **전략 분리** — EMA 크로스(BTC/SOL) vs 펀딩 역추세(XRP) 별도 엔진
- **RL 데이터 축적** — 매 15분봉 시장 상태 자동 기록 (강화학습 준비)

---

## 2. 코드 구조

```
src/sniper_v2/
    config.py            ← SymbolConfig (BTC/SOL/XRP 각각)
    indicators.py        ← EMA, RSI, ATR, ADX, SWING (공통 지표)
    strategy.py          ← SniperV2 (EMA 크로스 엔진)
    funding_strategy.py  ← FundingContrarianSniper (펀딩 z-score 역추세)
    rl_config.py         ← RLConfig (RL 로깅/학습 설정)
    rl_logger.py         ← RLStateLogger (state/action/reward 기록)

scripts/
    run_sniper_v2.py     ← 데몬 러너 (멀티 심볼 비동기)
    backtest_sniper_v2.py ← 최적화된 백테스트 (지표 1회 계산)
    gridsearch_sniper_v2.py ← SL/TP/PP 파라미터 그리드서치
    backtest_xrp_rolling.py ← XRP 월간 walk-forward 최적화
    convert_rl_logs.py   ← JSONL → Parquet 변환 (RL Phase 2 준비)

logs/
    sniper_v2/{btc,sol,xrp}/  ← 기존 거래 로그 (JSONL)
    rl/states/{btc,sol,xrp}/  ← RL 상태 로그 (매 15분봉)
    rl/trades/{btc,sol,xrp}/  ← RL 거래 로그 (ENTRY/EXIT + full state)
```

---

## 3. 실행 방법

```bash
# BTC 단독
python scripts/run_sniper_v2.py

# SOL 단독
python scripts/run_sniper_v2.py --symbol SOL/USDT:USDT

# 3코인 동시 (BTC + SOL + XRP)
python scripts/run_sniper_v2.py --multi

# 모의 실행
python scripts/run_sniper_v2.py --multi --dry-run
```

---

## 4. 심볼별 전략 + 파라미터

### 4-1. BTC — EMA 크로스 (SWING SL)

60개월 그리드서치 결과: **PF 1.02, 잔고 $866** (vs 구 PF 0.87, $514)

| 파라미터 | 구 값 | 신 값 | 변경 이유 |
|----------|-------|-------|-----------|
| ema_fast / slow / trend | 20 / 50 / 200 | 20 / 50 / 200 | 유지 (EMA 자체는 최적) |
| rsi_bull / rsi_bear | 60 / 40 | 60 / 40 | 유지 |
| sl_method | SWING_5 | **SWING_15** | 5봉=1.25h → 15봉=3.75h, 노이즈 SL 대폭 감소 |
| tp1 / tp2 / tp3 RR | 1.0 / 2.0 / 3.0 | **1.5 / 2.5 / 4.0** | 타겟 확대, 승리 시 수익 극대화 |
| pp_trigger / pp_exit | 1.0R / 0.3R | **1.5R / 0.3R** | 더 오래 홀딩 후 수익보호 발동 |
| vol_filter / adx_min | 999 / 0 | 999 / 0 | 필터 없음 유지 |
| leverage / balance_ratio | 3x / 30% | 3x / 30% | 유지 |

**진입 조건**: EMA(20) 골든크로스 EMA(50) + close > EMA(200) + RSI > 60 (롱 기준)

### 4-2. SOL — EMA 크로스 (SWING SL + 강한 필터)

60개월 그리드서치 결과: **PF 1.57, 잔고 $1,189** (vs 구 PF 0.89, $640)

| 파라미터 | 구 값 | 신 값 | 변경 이유 |
|----------|-------|-------|-----------|
| ema_fast / slow / trend | 8 / 40 / 150 | 8 / 40 / 150 | 유지 |
| rsi_bull / rsi_bear | 65 / 35 | 65 / 35 | 유지 |
| sl_method | SWING_3 | SWING_3 | 유지 (SOL은 빠른 스윙이 적합) |
| adx_min | 25 | **30** | **이것 하나로 PF 0.89→1.57.** 약한 추세 거래 제거 |
| cooldown_bars | 10 | 10 | 유지 |
| tp1 / tp2 / tp3 RR | 1.25 / 3.38 / 5.5 | 1.25 / 3.38 / 5.5 | 유지 (이미 넓은 타겟) |
| leverage / balance_ratio | 3x / 30% | 3x / 30% | 유지 |

**진입 조건**: EMA(8) 골든크로스 EMA(40) + close > EMA(150) + RSI > 65 + **ADX >= 30**

### 4-3. XRP — 펀딩 역추세 (ATR SL)

60개월 그리드서치 결과: **PF 1.41, 잔고 $1,349, Calmar +1.55** (vs 구 PF 0.81, $797)

| 파라미터 | 구 값 | 신 값 | 변경 이유 |
|----------|-------|-------|-----------|
| price_confirm_z | 0.575 | **0.8** | 가격 확인 강화 → 가짜 시그널 필터 (핵심 변경) |
| tp1 / tp2 / tp3 RR | 1.5 / 3.0 / 5.0 | **2.0 / 3.5 / 6.0** | 타겟 확대, 승리 시 수익 극대화 |
| profit_protect | ON (1.5R→0.5R) | **OFF** | TP/Trailing에 맡김 (PP OFF가 PF 1위) |
| z_threshold | 1.125 | 1.125 | 유지 (그리드서치에서 최적) |
| sl_atr_mult | 2.0 | 2.0 | 유지 |
| funding_lookback | 75 | 75 | 유지 |

**진입 조건**: 펀딩z > 1.125 + 가격z > 0.8 → 숏 (과열 역추세) | 펀딩z < -1.125 + 가격z < -0.8 → 롱 (과냉 역추세)

---

## 5. 전략 로직

### 5-1. EMA 크로스 전략 (BTC, SOL)

```
매 15분봉 완성:
    ├─ 지표 계산: EMA(fast/slow/trend), RSI, ATR, ADX, SWING
    ├─ 쿨다운 체크 (SL 후 N봉 대기)
    ├─ 필터: Vol Filter, ADX >= adx_min
    ├─ 진입 조건: EMA 크로스 + 추세 확인 + RSI 확인
    └─ SL/TP 계산: SWING low/high → TP = Risk × RR

5초마다:
    ├─ 수익보호: peak_r >= pp_trigger이면 cur_r <= pp_exit 시 청산
    ├─ TP 체크: TP1→TP2→TP3 순서, 트레일링 스탑 이동
    └─ SL 체크: trail_price 또는 sl_price 이하 시 청산
```

### 5-2. 펀딩 역추세 전략 (XRP)

```
매 15분봉 완성:
    ├─ 펀딩비 z-score: (현재 - 75개 평균) / 표준편차
    ├─ 가격 z-score: 4h/12h/24h 수익률 / 변동성 (평균)
    ├─ 시그널: 펀딩z > 1.125 & 가격z > 0.8 → 숏 (역추세)
    ├─ 강도 필터: strength >= 0.3
    └─ SL/TP: ATR×2.0 기반

5초마다:
    ├─ TP 체크: TP1(2R)→TP2(3.5R)→TP3(6R), 트레일링
    └─ SL 체크: ATR×2.0 이탈 시 청산
```

### 5-3. 포지션 관리 (공통)

**트레일링 흐름 (롱 예시):**
```
진입 시:    스탑 = SL (SWING low 또는 entry - ATR×mult)
TP1 도달 → 스탑 = Entry (본전)
TP2 도달 → 스탑 = TP1
TP3 도달 → 스탑 = TP2
되돌림 → 스탑 터치 → TRAIL_TP{N}_HIT 청산
```

**수익보호 (PP):**
```
BTC: peak_r >= 1.5R이면 → cur_r <= 0.3R 시 PP_1.5R 청산
SOL: PP 없음
XRP: PP 없음 (OFF)
```

---

## 6. 60개월 백테스트 결과 (2021.04 ~ 2026.03)

### 최적화 후

| 코인 | 전략 | 거래 | 승률 | PF | 총 PnL | 최종 잔고 | Calmar |
|------|------|------|------|-----|--------|----------|--------|
| BTC | EMA 20/50/200 + SWING_15 | 660 | 36.7% | 1.02 | +28.5% | $861 | -0.04 |
| SOL | EMA 8/40/150 + ADX>=30 | 122 | 20.5% | 1.57 | +394.4% | $1,189 | +0.07 |
| XRP | Funding Z + PZ=0.8 | 87 | 26.4% | 1.41 | +148.1% | $1,349 | +0.28 |

### 최적화 전 (비교)

| 코인 | PF | 총 PnL | 최종 잔고 | 변화 |
|------|-----|--------|----------|------|
| BTC | 0.87 | -170.5% | $514 | → **$861 (+$347)** |
| SOL | 0.89 | -87.5% | $640 | → **$1,189 (+$549)** |
| XRP | 0.81 | -62.8% | $797 | → **$1,349 (+$552)** |

### SL 문제의 근본 원인과 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| BTC SL 과다 (-1,062%) | SWING_5 = 1.25h, 노이즈에 걸림 | SWING_15 = 3.75h |
| SOL SL 과다 (-650%) | 약한 추세(ADX<30)에서 진입 | ADX>=30 필터 추가 |
| XRP SL 과다 (-331%) | 가격 확인 부족 (PZ=0.575) | PZ=0.8, PP OFF |
| PP 조기 청산 | 1.0R에서 수익보호 발동 | 1.5R 이상 또는 OFF |
| TP 너무 좁음 | 1R/2R/3R | 1.5R/2.5R/4R (BTC), 2R/3.5R/6R (XRP) |

### XRP 롤링 리프레시 결과

매월 6개월 훈련창으로 파라미터 재최적화하는 walk-forward 백테스트:

| 항목 | 고정 파라미터 | 롤링 리프레시 |
|------|-------------|-------------|
| 거래 | 97건 | 153건 |
| PF | 0.81 | **1.11** |
| 총 PnL | -62.8% | **+69.3%** |
| 최종 잔고 | $797 | **$1,039** |

---

## 7. 청산 종류

| 사유 | 설명 | 일반 손익 |
|------|------|-----------|
| SL | 스탑로스 도달 | 손실 (-1R) |
| TRAIL_TP1_HIT | TP1 후 본전 근처에서 되돌림 | 소폭 (0~+0.5R) |
| TRAIL_TP2_HIT | TP2 후 TP1 수준에서 되돌림 | 중간 수익 (+1~2R) |
| TRAIL_TP3_HIT | TP3 후 TP2 수준에서 되돌림 | 큰 수익 (+2~3R) |
| REVERSE | 반대 시그널 발생 → 방향 전환 | 손익 혼재 (평균 양수) |
| PP_{N}R | 수익보호 발동 (최고 N×R 도달 후 하락) | 소폭 양수 |

---

## 8. 텔레그램 알림

### 진입

```
🟢 LONG XRP 💰 FUNDING

진입 사유:
펀딩 과냉 + 가격 과냉 → 역추세
펀딩: -0.000147 (z=-1.52, 임계=1.125)
가격Z: -0.92 (임계=0.800)
강도: 68% | RSI=47.0
SL: ATR×2.0 (1.46%)

포지션:
Entry: 1.3651
SL: 1.3545 (-0.78%)
TP1: 1.3862 (+1.55%)
TP2: 1.4021
TP3: 1.4285
Risk: 0.0106 | 3x | 잔고 30%
```

### 청산

```
❌ CLOSE XRP — 손절 (SL 도달)

PnL: -2.33%
최고 미실현: 0.3R (+0.70%)
1.3651 → 1.3545
보유: 95분 | LONG
```

---

## 9. RL 강화학습 로드맵

### Phase 1: 데이터 축적 (현재 ~ 6개월)

매 15분봉마다 시장 상태를 자동 기록:

| 로그 | 경로 | 내용 | 빈도 |
|------|------|------|------|
| 상태 | `logs/rl/states/{sym}/` | 지표 + 포지션 + 계좌 + HOLD 사유 | 96회/일/심볼 |
| 거래 | `logs/rl/trades/{sym}/` | ENTRY/EXIT + full state + reward | 거래당 2건 |

**Feature Vector (25D EMA / 31D Funding):**

```
[0]  (ema_fast - ema_slow) / atr     — EMA 스프레드
[1]  (close - ema_trend) / atr        — 추세 위치
[2]  (close - ema_fast) / atr         — 단기 EMA 대비
[3]  rsi / 100                        — RSI [0,1]
[4]  adx / 100                        — ADX [0,1]
[5]  atr / close                      — 상대 변동성
[6]  atr / atr_sma - 1                — 변동성 레짐
[7-8]  swing 거리 / atr               — 지지/저항
[9-10] bull_cross, bear_cross         — 0/1
[11-15] 최근 5봉 수익률               — raw returns
[16-20] 최근 5봉 거래량 비율          — vol / sma
[21-24] 포지션 (보유, 방향, 미실현R, 보유기간)
[25-30] (XRP) 펀딩z, 가격z×3, 강도, 펀딩비
```

**예상 데이터**: 6개월 → 180K 상태, 600 거래 에피소드

### Phase 2: 오프라인 RL (3~6개월 후)

- **Environment**: Gymnasium 호환, 히스토리 리플레이
- **State**: 25D/31D feature vector
- **Action**: Discrete(5) — HOLD, ENTER_LONG, ENTER_SHORT, EXIT, REVERSE
- **Reward**: R-multiple (PnL/risk) + 드로다운 패널티 + 홀딩 비용
- **Algorithm**: CQL (Conservative Q-Learning) — 오프라인 데이터에 최적, Q값 과대평가 방지

### Phase 3: 온라인 배포 (검증 후)

- RL이 규칙 시스템을 **대체하지 않고 보조** (Advisory Layer)
- 규칙 + RL 동의 → 풀사이즈 | RL만 진입 → 스킵 (규칙이 안전장치)
- SL은 규칙 기반 그대로 유지 (RL이 SL 제거 불가)
- 일일 손실 5% 초과 시 RL 비활성화
- 점진적 롤아웃: 10% → 50% → 100%

---

## 10. 앙상블과의 비교

| 항목 | Ensemble (`run_daemon.py`) | Sniper V2 (`run_sniper_v2.py`) |
|------|---------------------------|-------------------------------|
| 대상 | 19코인 스캔 | **BTC + SOL + XRP** (각각 최적화) |
| 판단 | 10알파 → 앙상블 → Sonnet AI | EMA 크로스 / 펀딩 z-score |
| AI | Sonnet이 최종 결정 | **순수 기술적 지표 (AI 없음)** |
| 주기 | 5분 리밸런스 | 15분봉 + 5초 SL/TP 체크 |
| RL | 없음 | **Phase 1 데이터 축적 중** |
| 동시 실행 | 가능 (독립) | 가능 (독립) |

같은 바이낸스 계정 사용 시 포지션 겹칠 수 있음 → 잔고 분리 권장.

---

## 11. 주요 스크립트 사용법

```bash
# 백테스트 (3코인 60개월)
python scripts/backtest_sniper_v2.py --all --months 60

# 그리드서치 (SL 중심)
python scripts/gridsearch_sniper_v2.py --symbol BTC --months 60

# XRP 롤링 리프레시
python scripts/backtest_xrp_rolling.py --months 60 --train-months 6

# RL 로그 변환 (JSONL → Parquet)
python scripts/convert_rl_logs.py
```
