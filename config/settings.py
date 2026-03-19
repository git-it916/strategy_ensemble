"""
Global Settings — CLAUDE.md 섹션 12 기준

모든 매직 넘버는 여기에. 코드에 하드코딩하지 않는다.
"""

from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# === Universe ===
UNIVERSE_SIZE = 20
MIN_24H_VOLUME_USDT = 50_000_000  # $50M 이상만

# 화이트리스트: 시가총액 상위 메이저 코인만 허용
# 이 목록에 없는 코인은 거래량이 아무리 높아도 제외 (펌프 방지)
COIN_WHITELIST = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX",
    "LINK", "DOT", "SUI", "NEAR", "APT", "ARB", "OP",
    "ATOM", "UNI", "FIL", "LTC", "TIA", "INJ", "SEI",
    "FET", "RENDER", "TAO", "HYPE", "AAVE", "MKR",
    "TON", "TRX", "MATIC", "ETC",
]

# 블랙리스트: 화이트리스트에 있더라도 추가로 제외할 코인
BLACKLIST = [
    "BTCDOM", "DEFI",  # 지수
]

# === Alpha Weights (합계 = 1.0) ===
# 인트라데이 중심: 실시간/5분/1시간 데이터 알파 78%, 일봉 배경 22%
ALPHA_WEIGHTS = {
    # --- 인트라데이 (5분~1시간, 매 사이클 갱신) ---
    "MomentumMultiScale": 0.25,        # 5분봉 멀티스케일 (핵심)
    "IntradayVWAPV2": 0.20,            # 1시간봉 VWAP 이탈
    "OrderbookImbalance": 0.18,        # 실시간 오더북
    "IntradayRSIV2": 0.15,             # 1시간봉 RSI
    # --- 배경 컨텍스트 (일봉, 4시간 갱신) ---
    "FundingCarryEnhanced": 0.08,      # 펀딩 방향 참고
    "MomentumComposite": 0.06,         # 20일 추세 참고
    "MeanReversionMultiHorizon": 0.05, # 장기 z-score 참고
    "DerivativesSentiment": 0.03,      # OI/LS 참고
    # --- 비활성 ---
    "SpreadMomentum": 0.00,
    "VolatilityRegime": 0.00,
}

# === 진입 임계값 (롱/숏 비대칭) ===
LONG_ENTRY_THRESHOLD = 0.14            # 백테스트 기준 최적 (승률 50%, ROI+)
SHORT_ENTRY_THRESHOLD = -0.16          # 숏은 약간 엄격

# === SL/TP (롱/숏 비대칭) ===
LONG_SL_PCT = -0.035                   # -0.05→-0.035 (x3=-10.5%, 큰 손실 방지)
LONG_TP_PCT = 0.10
SHORT_SL_PCT = -0.025                  # -0.03→-0.025
SHORT_TP_PCT = 0.08

# === 트레일링 ===
TRAILING_ACTIVATION_PCT = 0.025
TRAILING_DISTANCE_PCT = 0.015

# === 레버리지 ===
LEVERAGE = 3

# === 타이밍 ===
REBALANCE_INTERVAL_SEC = 300        # 5분
SLTP_CHECK_INTERVAL_SEC = 5
ORDERBOOK_INTERVAL_SEC = 60
DAILY_REFRESH_INTERVAL_SEC = 14400  # 4시간
UNIVERSE_REFRESH_INTERVAL_SEC = 3600  # 1시간
MIN_HOLD_MINUTES = 15              # 90→15 (인트라데이: 3사이클 최소 확인)
COOLDOWN_MINUTES = 20              # 60→20 (같은 코인 재진입 쿨다운)
SWITCH_COOLDOWN_MINUTES = 15       # 60→15
SWITCH_SAME_DIR_GAP = 0.10         # 새 스케일 기준
SWITCH_REVERSE_SCORE_DROP = 0.08   # 새 스케일 기준
ENTRY_CONFIRM_CYCLES = 2           # 3→2 (10분 확인, 인트라데이 속도)
ENTRY_REQUIRE_RISING = True        # 2사이클간 스코어 상승 요구
ENTRY_MIN_SCORE_INCREASE = 0.01    # 0.015→0.01 (2사이클이므로 완화)
# 청산 — 점진적 시그널 감쇠 (인트라데이: 빠른 반응)
FADE_THRESHOLD = 0.08              # 진입 임계값의 ~57%
FADE_DURATION_MIN = 15             # 30→15 (인트라데이)
WEAK_THRESHOLD = 0.04              # 시그널 거의 소멸
WEAK_DURATION_MIN = 10             # 15→10
MAX_TRADES_PER_DAY = 6             # 4→6 (인트라데이 빈도 증가)
SWITCH_MIN_HOLD_MINUTES = 30       # 120→30 (인트라데이)
MAX_SAME_SYMBOL_LOSSES = 3         # 같은 코인 연속 N번 손실시 4시간 차단
SAME_SYMBOL_BAN_HOURS = 4

# === 일일 손실 한도 ===
DAILY_MAX_LOSS_PCT = -0.05

# === Stacking ===
STACKING_MIN_HOURS = 16  # 16시간 후 첫 학습 (192 사이클 × 10코인 = ~1920행)
STACKING_TRAIN_WINDOW_DAYS = 90
STACKING_RETRAIN_INTERVAL_DAYS = 30

# === 포지션 사이징 ===
BALANCE_USAGE_RATIO = 0.95  # 잔고의 95% 사용

# === Binance ===
BINANCE_RATE_LIMIT_SLEEP = 0.2
