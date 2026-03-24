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
# IC 기반 재구성 (2026-03-20):
# - MC(IC +0.093), MR(반전 +0.130), DS(IC +0.108) → 핵심 알파로 승격
# - MMS: IC≈0이지만 스케일 통일(*0.5 제거)로 실질 기여 2배 → 가중치 축소
# - RSI: 부호 반전(추세추종), 반전 후 IC +0.071
# - OB: 부호 반전(contrarian), 반전 후 IC +0.039
ALPHA_WEIGHTS = {
    # --- 핵심 (IC 검증됨, 높은 예측력) ---
    "MomentumComposite": 0.20,         # IC +0.093(1h), 최고 일관성 (6→20%)
    "FundingCarryEnhanced": 0.15,      # IC +0.044(1h), 구조적 edge (10→15%)
    "DerivativesSentiment": 0.12,      # IC +0.108(1h), OI/LS 기반 (3→12%)
    "MeanReversionMultiHorizon": 0.10, # 반전 후 IC +0.130(1h), 추세확인 (5→10%)
    # --- 보조 (조건부 유효) ---
    "IntradayRSIV2": 0.12,             # 반전 후 IC +0.071, 추세추종 (15→12%)
    "MomentumMultiScale": 0.12,        # IC≈0 but 스케일 2배 → 축소 (30→12%)
    "IntradayVWAPV2": 0.12,            # IC +0.012, 극단 이탈 시만 유효 (23→12%)
    "OrderbookImbalance": 0.07,        # 반전 후 IC +0.039, contrarian (8→7%)
    # --- 비활성 ---
    "SpreadMomentum": 0.00,
    "VolatilityRegime": 0.00,
}

# === 진입 임계값 (롱/숏 비대칭) ===
# MMS(12%)+MC(20%)=32%만 스케일 2배, 나머지 68% 동일 → 합성 ~1.3배 증가
LONG_ENTRY_THRESHOLD = 0.16            # 0.12×1.3≈0.16
SHORT_ENTRY_THRESHOLD = -0.18          # -0.14×1.3≈-0.18

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
SWITCH_SAME_DIR_GAP = 0.13         # 스케일 통일 반영 (0.10×1.3)
SWITCH_REVERSE_SCORE_DROP = 0.10   # 스케일 통일 반영 (0.08×1.3)
ENTRY_CONFIRM_CYCLES = 2           # 3→2 (10분 확인, 인트라데이 속도)
ENTRY_REQUIRE_RISING = True        # 2사이클간 스코어 상승 요구
ENTRY_MIN_SCORE_INCREASE = 0.01    # 0.015→0.01 (2사이클이므로 완화)
# 청산 — 점진적 시그널 감쇠 (인트라데이: 빠른 반응)
FADE_THRESHOLD = 0.09              # 진입 임계값(0.16)의 ~56%
FADE_DURATION_MIN = 15             # 30→15 (인트라데이)
WEAK_THRESHOLD = 0.045             # 시그널 거의 소멸
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

# === 과열 필터 ===
OVERHEAT_1H_PCT = 0.05                # 1시간 수익률 ±5% 이상이면 추격 진입 차단

# === 거래량 필터 ===
VOL_FILTER_BASELINE_BARS = 36         # 최근 3시간 (36개 5분봉) baseline
VOL_FILTER_THRESHOLD = 0.5           # baseline 대비 50% 미만이면 차단

# === Binance ===
BINANCE_RATE_LIMIT_SLEEP = 0.2
