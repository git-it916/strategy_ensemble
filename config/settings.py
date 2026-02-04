"""
Global Settings

Trading parameters and configuration.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Database
DUCKDB_PATH = DATA_DIR / "quant.duckdb"

# =============================================================================
# Universe Settings
# =============================================================================
UNIVERSE = {
    "index": "KOSPI200",
    "min_market_cap": 1e11,  # 최소 시가총액 (1000억)
    "min_volume": 1e8,  # 최소 거래대금 (1억)
    "exclude_sectors": ["금융"],  # 제외 섹터
    "max_stocks": 100,  # 최대 종목 수
}

# =============================================================================
# Trading Parameters
# =============================================================================
TRADING = {
    # Position limits
    "max_position_weight": 0.1,  # 최대 개별 비중 10%
    "min_position_weight": 0.02,  # 최소 개별 비중 2%
    "max_positions": 20,  # 최대 보유 종목 수
    "min_trade_value": 100_000,  # 최소 거래금액 (10만원)

    # Risk limits
    "max_leverage": 1.0,  # 최대 레버리지
    "max_drawdown": 0.15,  # 최대 손실 허용 (15%)
    "daily_loss_limit": 0.03,  # 일일 손실 한도 (3%)

    # Execution
    "order_type": "limit",  # limit or market
    "slippage_bps": 10,  # 슬리피지 가정 (10bps)

    # Transaction costs (Korean market)
    "commission_bps": 1.5,  # 수수료 (0.015%)
    "tax_bps": 23.0,  # 거래세 (0.23%, 매도시)
}

# =============================================================================
# Strategy Settings
# =============================================================================
STRATEGIES = {
    "rsi_reversal": {
        "enabled": True,
        "weight": 0.25,
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
    },
    "vol_breakout": {
        "enabled": True,
        "weight": 0.25,
        "lookback": 20,
        "breakout_threshold": 1.5,
    },
    "value_f_score": {
        "enabled": True,
        "weight": 0.25,
        "min_f_score": 5,
        "max_pb_ratio": 3.0,
    },
    "sentiment_long": {
        "enabled": True,
        "weight": 0.25,
        "momentum_lookback": 60,
    },
}

# =============================================================================
# Ensemble Settings
# =============================================================================
ENSEMBLE = {
    "use_dynamic_weights": True,
    "performance_lookback": 21,  # 가중치 계산 기간
    "performance_blend": 0.5,  # 성과 기반 가중치 비율

    # Regime preferences
    "regime_preferences": {
        "bull": {
            "vol_breakout": 1.5,
            "sentiment_long": 1.3,
            "rsi_reversal": 0.7,
        },
        "bear": {
            "rsi_reversal": 1.3,
            "value_f_score": 1.2,
            "vol_breakout": 0.5,
        },
        "sideways": {
            "rsi_reversal": 1.5,
            "value_f_score": 1.0,
        },
    },
}

# =============================================================================
# Backtest Settings
# =============================================================================
BACKTEST = {
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100_000_000,  # 1억원
    "rebalance_frequency": "daily",
    "benchmark": "KOSPI",
}

# =============================================================================
# Schedule Settings
# =============================================================================
SCHEDULE = {
    "market_open": "09:00",
    "market_close": "15:30",
    "data_update": "08:30",  # 장 시작 전 데이터 업데이트
    "signal_generation": "09:00",  # 장 시작 시 신호 생성
    "rebalance_time": "09:10",  # 리밸런싱 실행
    "eod_report": "15:35",  # 장 마감 후 리포트
}

# =============================================================================
# Logging Settings
# =============================================================================
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_rotation": "1 day",
    "retention": "30 days",
}
