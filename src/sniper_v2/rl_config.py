"""
Sniper V2 — RL Configuration.

RL-ready 로깅 + 향후 학습을 위한 설정.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RLConfig:
    """RL 로깅 및 학습 설정."""

    # ── 로깅 ──
    log_base_dir: str = "logs/rl"
    log_version: int = 2

    # ── Feature 차원 ──
    n_features_ema: int = 25       # BTC/SOL: EMA 전략
    n_features_funding: int = 31   # XRP: 펀딩 전략 (EMA 25 + 펀딩 6)
    recent_candles: int = 5        # 최근 N봉 수익률/거래량

    # ── Reward 가중치 (Phase 2) ──
    reward_pnl_weight: float = 1.0
    reward_clean_win_bonus: float = 0.2    # MAE > -0.5R인 승리
    reward_drawdown_penalty: float = 0.3   # 깊은 역행 페널티
    reward_hold_cost_per_bar: float = -0.0001  # 보유 비용 (펀딩+기회비용)

    # ── 안전장치 (Phase 3) ──
    max_daily_loss_pct: float = -0.05      # 일일 손실 5% 초과 시 RL 비활성화
    rl_confidence_threshold: float = 0.7   # 이 이상일 때만 RL 액션 실행
    initial_rl_position_pct: float = 0.10  # 롤아웃 초기 RL 포지션 비율


RL_CONFIG = RLConfig()
