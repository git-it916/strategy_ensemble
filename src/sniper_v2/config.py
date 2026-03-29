"""
Sniper V2 — Configuration (Multi-Symbol)

60개월 그리드서치 최적화 (2026-03):
BTC: SWING_15 + PP 1.5R→0.3R + TP 1.5/2.5/4 (PF 1.02, 잔고 $866)
SOL: SWING_3 + ADX>=30 (PF 1.57, 잔고 $1,189) — Calmar 기준 ATR×2.0도 유력
XRP: PZ=0.8 + PP OFF + TP 2/3.5/6 (PF 1.41, 잔고 $1,349)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolConfig:
    """심볼별 전략 파라미터."""
    # EMA
    ema_fast: int
    ema_slow: int
    ema_trend: int
    # RSI
    rsi_len: int = 14
    rsi_bull: float = 55
    rsi_bear: float = 40
    # SL
    atr_len: int = 14
    sl_method: str = "ATR"      # "ATR" or "SWING"
    sl_atr_mult: float = 1.5
    swing_lookback: int = 5
    # TP
    tp1_rr: float = 1.5
    tp2_rr: float = 2.75
    tp3_rr: float = 4.0
    # 필터
    vol_filter: float = 2.0
    vol_filter_sma: int = 42
    adx_len: int = 14
    adx_min: float = 20
    cooldown_bars: int = 20
    # 트레일링
    use_trailing: bool = True
    # 수익보호 (미실현 수익이 trigger R 이상 갔다가 exit R로 돌아오면 청산)
    profit_protect: bool = False
    pp_trigger: float = 1.0     # 수익보호 발동 기준 (R 단위)
    pp_exit: float = 0.3        # 수익보호 청산 기준 (R 단위)
    # 실행
    leverage: int = 3
    balance_ratio: float = 0.45
    # 워밍업
    warmup_bars: int = 215
    # 로그
    log_dir: str = "logs/sniper_v2"


# ══════════════════════════════════════
# 심볼별 최적 파라미터
# ══════════════════════════════════════

BTC_CONFIG = SymbolConfig(
    # EMA 20/50/200 (60개월 검증)
    ema_fast=20, ema_slow=50, ema_trend=200,
    # RSI 60/40
    rsi_bull=60, rsi_bear=40,
    # SWING_15 SL — 5→15: 노이즈 SL 대폭 감소
    sl_method="SWING", swing_lookback=15,
    # TP 1.5R / 2.5R / 4R — 타겟 확대
    tp1_rr=1.5, tp2_rr=2.5, tp3_rr=4.0,
    # 필터 없음
    vol_filter=999.0, adx_min=0, cooldown_bars=0,
    # 수익보호: 1.5R 이상 갔다가 0.3R로 돌아오면 청산 (1.0→1.5: 더 오래 홀딩)
    profit_protect=True, pp_trigger=1.5, pp_exit=0.3,
    # 잔고 30%
    balance_ratio=0.30,
    warmup_bars=215,
    log_dir="logs/sniper_v2/btc",
)

SOL_CONFIG = SymbolConfig(
    # EMA 8/40/150 — 빠른 크로스, 중간 트렌드 필터
    ema_fast=8, ema_slow=40, ema_trend=150,
    # RSI 65/35 — 강한 모멘텀에서만 진입
    rsi_bull=65, rsi_bear=35,
    # SWING_3 SL (최근 3봉 스윙)
    sl_method="SWING", swing_lookback=3,
    # TP 1.25R / 3.38R / 5.5R — 넓은 타겟
    tp1_rr=1.25, tp2_rr=3.38, tp3_rr=5.5,
    # 필터: VolFilter=2.0, ADX>=30 (25→30: 약한 추세 제거, PF 0.89→1.57), Cooldown=10
    vol_filter=2.0, adx_min=30, cooldown_bars=10,
    # 잔고 30%
    balance_ratio=0.30,
    warmup_bars=165,    # EMA(150) + 15
    log_dir="logs/sniper_v2/sol",
)

# XRP: Funding Contrarian (60개월 그리드서치 PF 1.41, Calmar +1.55)
# EMA 시그널이 아닌 펀딩비 z-score 역추세 전략
# SymbolConfig는 포지션 관리(SL/TP/trailing)에만 사용
# FundingConfig에서 price_confirm_z=0.8, tp=2.0/3.5/6.0 적용
XRP_CONFIG = SymbolConfig(
    # EMA — 미사용 (펀딩 전략), placeholder
    ema_fast=12, ema_slow=26, ema_trend=100,
    # RSI — 메타데이터 로깅용
    rsi_bull=70, rsi_bear=30,
    # SL: ATR×2.0 (FundingConfig에서 오버라이드)
    sl_method="ATR", sl_atr_mult=2.0,
    # TP: FundingConfig에서 오버라이드 (2.0/3.5/6.0)
    tp1_rr=2.0, tp2_rr=3.5, tp3_rr=6.0,
    # 필터: 쿨다운 5봉
    vol_filter=999.0, adx_min=0, cooldown_bars=5,
    # 수익보호 OFF — TP/Trailing에 맡김 (PP OFF가 PF 1.41로 최고)
    profit_protect=False, pp_trigger=1.5, pp_exit=0.5,
    # 잔고 30%
    balance_ratio=0.30,
    warmup_bars=100,
    log_dir="logs/sniper_v2/xrp",
)


# AVAX: 60개월 그리드서치 최적화 (2026-03-28, PF 기준)
# PF 2.02 | WR 29.8% | PnL +981% | MDD -44% | Calmar 11.45
# EMA 12/50/200 (BTC형 느린 크로스) + SWING_5 + ADX>=30 + TP 2/3.5/6
AVAX_CONFIG = SymbolConfig(
    # EMA 12/50/200 — BTC형 느린 크로스가 AVAX에서 최적
    ema_fast=12, ema_slow=50, ema_trend=200,
    # RSI 65/35 — 강한 모멘텀에서만 진입
    rsi_bull=65, rsi_bear=35,
    # SWING_5 SL (최근 5봉 스윙)
    sl_method="SWING", swing_lookback=5,
    # TP 2.0R / 3.5R / 6.0R — 공격적 타겟
    tp1_rr=2.0, tp2_rr=3.5, tp3_rr=6.0,
    # 필터: ADX>=30 (강한 추세만), Cooldown=5
    vol_filter=999.0, adx_min=30, cooldown_bars=5,
    # PP OFF
    profit_protect=False,
    # 잔고 30%
    balance_ratio=0.30,
    warmup_bars=215,    # EMA(200) + 15
    log_dir="logs/sniper_v2/avax",
)

# 심볼 → 설정 매핑
CONFIGS = {
    "BTC/USDT:USDT": BTC_CONFIG,
    "SOL/USDT:USDT": SOL_CONFIG,
    "XRP/USDT:USDT": XRP_CONFIG,
    "AVAX/USDT:USDT": AVAX_CONFIG,
}

# 펀딩 전략 사용 심볼 (EMA가 아닌 펀딩비 기반)
FUNDING_STRATEGY_SYMBOLS = {"XRP/USDT:USDT"}

# VWAP 모멘텀 전략 사용 심볼 (look-ahead bias 확인으로 비활성화)
VWAP_MOMENTUM_SYMBOLS: set[str] = set()

# 공통 설정
TIMEFRAME = "15m"
CHECK_INTERVAL_SEC = 5
