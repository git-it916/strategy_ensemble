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
    "index": "KOSPI_KOSDAQ",  # KOSPI + KOSDAQ 전종목
    "min_market_cap": 1e11,  # 최소 시가총액 (1000억)
    "min_volume": 1e8,  # 최소 거래대금 (1억)
    "exclude_sectors": ["금융"],  # 제외 섹터
    "max_stocks": 100,  # 최대 종목 수
}

# =============================================================================
# Inverse ETF Mapping (숏 대응용)
# =============================================================================
INVERSE_MAPPING = {
    "KOSPI": "252670",   # KODEX 200선물인버스2X
    "KOSDAQ": "251340",  # KODEX 코스닥150선물인버스
}
# 시장 헤지용 인버스 ETF 티커 집합 (주문 로직에서 식별용)
INVERSE_ETF_TICKERS = set(INVERSE_MAPPING.values())

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
    # --- ML Alphas ---
    "return_prediction": {
        "enabled": True,
        "weight": 0.2,
        "type": "ml",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
    },
    "intraday_pattern": {
        "enabled": True,
        "weight": 0.15,
        "type": "ml",
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
    },
    "volatility_forecast": {
        "enabled": True,
        "weight": 0.1,
        "type": "ml",
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
    },
    # --- LLM Alpha (순차 파이프라인에서 최종 판단 역할) ---
    "llm_alpha": {
        "enabled": True,
        "type": "llm",
        "model": "qwen2.5-kospi-ft-s3",
        "temperature": 0.3,
        "max_stocks_in_prompt": 50,
    },
}

# =============================================================================
# Regime Classifier Settings
# =============================================================================
REGIME_CLASSIFIER = {
    "enabled": True,
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "regime_horizon": 20,       # 레짐 판단 기간 (20 거래일)
    "bull_threshold": 0.03,     # 이 이상이면 상승장
    "bear_threshold": -0.03,    # 이 이하이면 하락장
}

# =============================================================================
# Ensemble Settings
# =============================================================================
ENSEMBLE = {
    "use_dynamic_weights": True,
    "performance_lookback": 21,  # 가중치 계산 기간
    "performance_blend": 0.5,  # 성과 기반 가중치 비율

    # LLM orchestrator (순차 파이프라인에서 Gemini가 담당)
    "use_llm_orchestrator": True,
    "llm_model": "gemini-2.5-flash",
    "llm_temperature": 0.3,
    "llm_timeout": 60.0,

    # Regime preferences
    "regime_preferences": {
        "bull": {
            "vol_breakout": 1.5,
            "sentiment_long": 1.3,
            "rsi_reversal": 0.7,
            "return_prediction": 1.3,
            "intraday_pattern": 1.0,
            "llm_alpha": 1.0,
        },
        "bear": {
            "rsi_reversal": 1.3,
            "value_f_score": 1.2,
            "vol_breakout": 0.5,
            "return_prediction": 0.8,
            "volatility_forecast": 1.5,
            "llm_alpha": 1.0,
        },
        "sideways": {
            "rsi_reversal": 1.5,
            "value_f_score": 1.0,
            "intraday_pattern": 1.3,
            "return_prediction": 1.0,
            "llm_alpha": 1.2,
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
# LLM Settings (Gemini + Ollama 병행)
# =============================================================================
LLM_CONFIG = {
    # Gemini (앙상블 오케스트레이션, 시장 분석)
    "gemini_model": "gemini-2.5-flash",
    "gemini_temperature": 0.3,
    "gemini_timeout": 60.0,
    # Ollama (파인튜닝 모델 - 전략 특화 시그널 생성)
    "ollama_url": "http://localhost:11434",
    "ollama_model": "qwen2.5-kospi-ft-s3",
    "ollama_temperature": 0.3,
    "ollama_timeout": 120.0,
    # 공통
    "max_retries": 2,
    "retry_delay": 5.0,
    "max_stocks_in_prompt": 50,
    "reasoning_log_dir": str(LOGS_DIR / "reasoning"),
}

# =============================================================================
# WebSocket Settings (KIS Real-time)
# =============================================================================
WEBSOCKET_CONFIG = {
    "candle_interval_minutes": 1,   # 1분봉 수집 (데이터 해상도)
    "signal_interval_minutes": 15,  # 15분마다 시그널 생성 (LLM 호출 주기)
    "auto_reconnect": True,
    "reconnect_delay": 5.0,
    "max_reconnect_attempts": 10,
    "heartbeat_interval": 30,
    "min_ticks_for_signal": 10,     # 최소 틱 수 미달 시 시그널 생성 건너뜀
}

# =============================================================================
# Pipeline Settings (순차 파이프라인 구조)
# =============================================================================
PIPELINE = {
    # Step 1: Universe filter
    "min_market_cap": 1e11,
    # Step 2: ML feature extraction (기존 ML 알파를 데이터 소스로 사용)
    "ml_strategies": [
        "return_prediction",
        "volatility_forecast",
        "intraday_pattern",
    ],
    # Step 3: Technical + fundamental 소스
    "technical_strategies": [
        "rsi_reversal",
        "vol_breakout",
        "value_f_score",
        "sentiment_long",
    ],
    # Step 4: LLM reasoning (최종 판단)
    "llm_model": "qwen2.5-kospi-ft-s3",
    # Step 5: Risk management
    "max_position_weight": 0.1,
    "max_positions": 20,
    # Step 6: Approval
    "require_human_approval": True,
    "approval_timeout_seconds": 300,  # 5분 대기
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
