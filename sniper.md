# Precision Sniper

TradingView Pine Script "Precision Sniper [WillyAlgoTrader]"를 Python으로 포팅한 **단일 코인 전용** 트레이딩 봇.
기존 앙상블 데몬(`run_daemon.py`)과 **완전히 독립적**으로 동작한다.

---

## 코드 위치

| 파일 | 설명 |
|---|---|
| `src/sniper/config.py` | 모든 파라미터 (심볼, EMA, TP/SL, ATR 등) |
| `src/sniper/indicators.py` | 지표 계산 (EMA, RSI, MACD, ATR, ADX/DMI, VWAP) |
| `src/sniper/strategy.py` | 전략 엔진 (Confluence Scoring + Entry + TP/SL/Trailing) |
| `scripts/run_sniper.py` | 데몬 러너 (ccxt 바이낸스 + 텔레그램 알림) |
| `logs/sniper/` | 거래 로그 (JSONL, 날짜별) |

공유 인프라: `config/keys.yaml` (API 키), `src/notifications/telegram_notifier.py`, `src/execution/time_sync.py`

---

## 실행 방법

```bash
# BTC 기본 (5분봉 + 1h HTF 필터)
python scripts/run_sniper.py

# 심볼 변경
python scripts/run_sniper.py --symbol ETH/USDT:USDT
python scripts/run_sniper.py --symbol SOL/USDT:USDT

# 타임프레임 변경
python scripts/run_sniper.py --tf 15m --htf 4h

# 모의 실행 (주문 없이 시그널만 확인)
python scripts/run_sniper.py --dry-run

# 조합
python scripts/run_sniper.py --symbol ETH/USDT:USDT --tf 15m --htf 4h --dry-run
```

---

## 전략 로직

### 1. 진입 조건 (4가지 모두 충족)

| 조건 | 설명 |
|---|---|
| EMA Cross | EMA(9)가 EMA(21) 상향/하향 돌파 |
| Momentum | close > EMA(9) & EMA(21) (롱) 또는 반대 (숏) |
| RSI Filter | RSI(13) < 75 (롱) 또는 > 25 (숏) — 과매수/과매도 차단 |
| Confluence Score | 10점 만점 중 **5점 이상** 달성 |

### 2. Confluence Scoring (최대 10점)

| 항목 | 점수 | 롱 조건 | 숏 조건 |
|---|---|---|---|
| EMA 정배열 | 1.0 | fast > slow | fast < slow |
| 트렌드 방향 | 1.0 | close > EMA(55) | close < EMA(55) |
| RSI 모멘텀 | 1.0 | 50 < RSI < 75 | 25 < RSI < 50 |
| MACD 히스토그램 | 1.0 | hist > 0 | hist < 0 |
| MACD 크로스 | 1.0 | MACD > Signal | MACD < Signal |
| VWAP 위치 | 1.0 | close > VWAP | close < VWAP |
| 거래량 확인 | 1.0 | vol > SMA(20) × 1.2 | 동일 |
| ADX 트렌드 강도 | 1.0 | ADX > 20 & DI+ > DI- | ADX > 20 & DI- > DI+ |
| HTF 바이어스 | **1.5** | 1h EMA fast > slow | 1h EMA fast < slow |
| 가격 위치 | 0.5 | close > EMA(9) | close < EMA(9) |

### 3. 리스크 관리

```
SL = ATR(14) × 1.5  (또는 스윙 로우/하이 기반 — 더 가까운 쪽)
TP1 = Entry ± Risk × 1.0  (1:1 R:R)
TP2 = Entry ± Risk × 2.0  (1:2 R:R)
TP3 = Entry ± Risk × 3.0  (1:3 R:R)
```

### 4. 트레일링 스탑

| 이벤트 | 트레일링 스탑 이동 |
|---|---|
| TP1 도달 | SL → **손익분기점** (Entry) |
| TP2 도달 | SL → **TP1** |
| TP3 도달 | SL → **TP2** |

최소 1:1 수익은 확보하면서 추세가 계속되면 더 먹는 구조.

### 5. Anti-Duplicate

같은 방향으로 연속 진입하지 않음. 롱 시그널 → 롱 진입 후, 다음 롱 시그널은 무시.
숏 시그널이 나와야 방향이 리셋된다.

---

## 파라미터 조정

모든 파라미터는 `src/sniper/config.py`에서 관리:

```python
# 공격적으로 변경 (더 많은 시그널)
MIN_CONFLUENCE_SCORE = 3
SL_ATR_MULT = 1.0

# 보수적으로 변경 (더 적은 시그널, 높은 승률)
MIN_CONFLUENCE_SCORE = 7
SL_ATR_MULT = 2.0

# 스캘핑 (1분봉)
SNIPER_TIMEFRAME = "1m"
SNIPER_HTF = "15m"
```

---

## 로그 구조

`logs/sniper/YYYY-MM-DD.jsonl` — 한 줄 JSON, ENTRY/EXIT 이벤트:

```json
{"event": "ENTRY", "symbol": "BTC/USDT:USDT", "direction": "LONG", "price": 85000.00, "score": 7.5, "sl": 84800.00, "tp1": 85200.00, "tp2": 85400.00, "tp3": 85600.00}
{"event": "EXIT", "symbol": "BTC/USDT:USDT", "direction": "LONG", "entry_price": 85000.00, "exit_price": 85400.00, "reason": "TRAIL_TP2_HIT", "pnl_pct": 1.41, "tp1_hit": true, "tp2_hit": true}
```

---

## 앙상블 데몬과의 관계

| | Ensemble (`run_daemon.py`) | Sniper (`run_sniper.py`) |
|---|---|---|
| 대상 | 19개 코인 스캔 | **단일 코인** |
| 판단 | 10개 알파 → 앙상블 → Sonnet AI | EMA cross + Confluence scoring |
| 진입 | 스코어 임계값 + 필터 | 4조건 동시 충족 |
| 청산 | SL/TP/trailing + Sonnet | ATR 기반 TP1-3 + trailing |
| AI 의존 | Sonnet이 최종 결정 | **순수 기술적 지표** |
| 동시 실행 | 가능 (서로 독립) | 가능 (서로 독립) |

두 데몬은 같은 바이낸스 계정을 사용하므로, 동시 실행 시 포지션이 겹칠 수 있다.
별도 서브계정이나 잔고 분리를 권장.
