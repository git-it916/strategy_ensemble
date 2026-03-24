"""
Sniper V2 — Configuration

Grid Search 최적화 결과 (28,350 + 2,160 + 123 조합 테스트).
BTC 15m, 12개월 백테스트 기준 최적 파라미터.

Baseline (PF 1.72, DD -31%) → V2 (PF 2.91, DD -11%)
"""

# === 대상 심볼 ===
SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "15m"

# === EMA 설정 ===
EMA_FAST = 18
EMA_SLOW = 40
EMA_TREND = 200

# === RSI 필터 ===
RSI_LEN = 14
RSI_BULL = 55       # 롱: RSI > 55 (모멘텀 확인)
RSI_BEAR = 40       # 숏: RSI < 40

# === SL (ATR 기반) ===
ATR_LEN = 14
SL_ATR_MULT = 1.5

# === TP (R:R 비율) ===
TP1_RR = 1.5        # TP1 = 1.5R (본전이 아닌 최소 수익 확보)
TP2_RR = 2.75       # TP2 = 2.75R
TP3_RR = 4.0        # TP3 = 4R (기존 3R보다 넓게)

# === 필터 (DD -31% → -11%의 핵심) ===
VOL_FILTER = 2.0        # ATR > ATR_SMA(42) × 2.0 이면 진입 금지 (급락/급등 차단)
VOL_FILTER_SMA = 42     # ATR 평균 산출 기간
ADX_LEN = 14
ADX_MIN = 20            # ADX < 20 이면 진입 금지 (횡보장 차단)
COOLDOWN_BARS = 20      # SL 후 20봉(5시간) 대기 (whipsaw 방지)

# === 트레일링 ===
USE_TRAILING = True     # TP1 → 본전, TP2 → TP1, TP3 → TP2

# === 실행 ===
LEVERAGE = 3
BALANCE_USAGE_RATIO = 0.95
CHECK_INTERVAL_SEC = 5

# === 워밍업 ===
WARMUP_BARS = 215       # EMA(200) + 15봉 여유

# === 로그 ===
LOG_DIR = "logs/sniper_v2"
