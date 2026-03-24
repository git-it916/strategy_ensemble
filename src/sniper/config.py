"""
Precision Sniper — Configuration

Pine Script 원본의 모든 파라미터를 여기서 관리.
"""

# === 대상 심볼 ===
SNIPER_SYMBOL = "SOL/USDT:USDT"
SNIPER_TIMEFRAME = "15m"         # 기본 캔들 타임프레임
SNIPER_HTF = "4h"                # 상위 타임프레임 (트렌드 필터)

# === EMA 설정 ===
EMA_FAST_LEN = 9
EMA_SLOW_LEN = 21
EMA_TREND_LEN = 55

# === 엔트리 엔진 ===
MIN_CONFLUENCE_SCORE = 8         # 최소 컨플루언스 점수 (max 10)
RSI_LEN = 13
RSI_OB = 75                      # 과매수 기준
RSI_OS = 25                      # 과매도 기준

# === MACD ===
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# === ADX ===
ADX_LEN = 14
ADX_STRONG = 20                  # 강한 트렌드 기준

# === 거래량 ===
VOL_SMA_LEN = 20
VOL_ABOVE_MULT = 1.2             # 평균 대비 1.2배 이상이면 거래량 확인

# === 리스크 관리 ===
ATR_LEN = 14
SL_ATR_MULT = 1.5                # SL = ATR × 1.5
TP1_RR = 1.0                     # TP1 = 1:1 R:R
TP2_RR = 2.0                     # TP2 = 1:2 R:R
TP3_RR = 3.0                     # TP3 = 1:3 R:R

USE_TRAILING = True              # 트레일링 스탑 활성화
USE_STRUCTURE_SL = True          # 구조적 SL (스윙 하이/로우 기반)
SWING_LOOKBACK = 10              # 스윙 탐색 봉 수

# === 실행 ===
LEVERAGE = 3
BALANCE_USAGE_RATIO = 0.95       # 잔고의 95% 사용
CHECK_INTERVAL_SEC = 5           # 포지션 체크 간격

# === 워밍업 ===
WARMUP_BARS = 60                 # 최소 60봉 이후 시그널 생성

# === 로그 ===
LOG_DIR = "logs/sniper"
