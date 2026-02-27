"""
Global Settings

Trading parameters and configuration for Binance USDT-M perpetual futures.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# Binance Exchange Settings
# =============================================================================
BINANCE = {
    "keys_path": "config/keys.yaml",
    "testnet": False,
    "futures_ws_url": "wss://fstream.binance.com",
    "rate_limit_sleep": 0.2,
}

# =============================================================================
# Universe Settings  (Binance USDT-M Perpetual Futures)
# =============================================================================
UNIVERSE = {
    "exchange": "binance_futures",
    "quote_currency": "USDT",
    "top_n_by_volume": 10,
    "min_volume_usdt": 1e8,
    "exclude_symbols": [
        "BTCDOMUSDT", "DEFIUSDT", "BTCSTUSDT",
        # TradFi / commodity perps (require separate agreement)
        "XAGUSDT", "XAUUSDT",
        # Low-liquidity / micro-cap tokens prone to min-notional issues
        "POWERUSDT",
    ],
}

# =============================================================================
# Trading Parameters  (Futures / Long-Short)
# =============================================================================
TRADING = {
    "max_position_weight": 0.30,
    "min_position_weight": 0.05,
    "max_positions": 3,
    "min_notional_usdt": 5,
    "max_leverage": 3.0,
    "max_drawdown": 0.15,
    "daily_loss_limit": 0.03,
    "order_type": "market",
    "slippage_bps": 5,
    "commission_bps": 4.0,
    "funding_cost_per_8h": True,
}

# =============================================================================
# Strategy Settings
# =============================================================================
STRATEGIES = {
    # --- OpenClaw Alphas (Active) ---
    # Weights sum to 1.0 across all enabled strategies
    "cs_momentum": {
        "enabled": True,
        "weight": 0.15,
        "lookback_days": 60,
        "skip_days": 5,
    },
    "ts_momentum": {
        "enabled": True,
        "weight": 0.12,
        "lookback_days": 20,
    },
    "ts_mean_reversion": {
        "enabled": True,
        "weight": 0.08,
        "signal_window": 5,
        "baseline_window": 60,
    },
    "pv_divergence": {
        "enabled": True,
        "weight": 0.08,
        "lookback_days": 20,
    },
    "volume_momentum": {
        "enabled": True,
        "weight": 0.07,
        "lookback_days": 20,
    },
    "low_volatility_anomaly": {
        "enabled": True,
        "weight": 0.07,
        "lookback_days": 20,
    },
    "funding_rate_carry": {
        "enabled": True,
        "weight": 0.20,
        "lookback_days": 7,
    },
    "rsi_reversal": {
        "enabled": True,
        "weight": 0.12,
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
    },
    "vol_breakout": {
        "enabled": True,
        "weight": 0.11,
        "lookback": 20,
        "breakout_threshold": 1.5,
    },
    "value_f_score": {"enabled": False, "weight": 0.0},
    "sentiment_long": {"enabled": False, "weight": 0.0},
    "return_prediction": {"enabled": False, "weight": 0.0, "type": "ml"},
    "intraday_pattern": {"enabled": False, "weight": 0.0, "type": "ml"},
    "volatility_forecast": {"enabled": False, "weight": 0.0, "type": "ml"},
    "llm_alpha": {"enabled": False, "type": "llm"},
}

# =============================================================================
# Regime Classifier Settings
# =============================================================================
REGIME_CLASSIFIER = {
    "enabled": False,
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "regime_horizon": 20,
    "bull_threshold": 0.03,
    "bear_threshold": -0.03,
}

# =============================================================================
# Ensemble Settings
# =============================================================================
ENSEMBLE = {
    "use_dynamic_weights": True,
    "performance_lookback": 21,
    "performance_blend": 0.5,
    "use_llm_orchestrator": False,
    "regime_preferences": {
        "bull": {
            "cs_momentum": 1.5,
            "ts_momentum": 1.4,
            "vol_breakout": 1.3,
            "funding_rate_carry": 1.2,
            "ts_mean_reversion": 0.6,
        },
        "bear": {
            "ts_mean_reversion": 1.5,
            "funding_rate_carry": 1.4,
            "low_volatility_anomaly": 1.3,
            "cs_momentum": 0.6,
            "ts_momentum": 0.5,
        },
        "sideways": {
            "ts_mean_reversion": 1.3,
            "pv_divergence": 1.2,
            "funding_rate_carry": 1.1,
            "cs_momentum": 0.8,
        },
    },
}

# =============================================================================
# Backtest Settings
# =============================================================================
BACKTEST = {
    "start_date": "2022-01-01",
    "end_date": "2025-12-31",
    "initial_capital": 10_000,
    "rebalance_frequency": "daily",
    "benchmark": "BTC/USDT:USDT",
}

# =============================================================================
# Schedule Settings
# =============================================================================
SCHEDULE = {
    "market_open": "00:00",
    "market_close": "23:59",
    "data_update": "00:05",
    "signal_generation": "00:10",
    "rebalance_time": "00:15",
    "eod_report": "00:00",
}

# =============================================================================
# WebSocket Settings (Binance Real-time)
# =============================================================================
WEBSOCKET_CONFIG = {
    "candle_interval": "1m",
    "signal_interval_minutes": 5,
    "auto_reconnect": True,
    "reconnect_delay": 5.0,
    "max_reconnect_attempts": 10,
    "heartbeat_interval": 20,
    "min_ticks_for_signal": 10,
}

# =============================================================================
# Pipeline Settings
# =============================================================================
PIPELINE = {
    "top_n_symbols": 50,
    "active_strategies": [
        "cs_momentum",
        "ts_momentum",
        "ts_mean_reversion",
        "pv_divergence",
        "volume_momentum",
        "low_volatility_anomaly",
        "funding_rate_carry",
        "rsi_reversal",
        "vol_breakout",
    ],
    # Use TRADING values for consistency (override only for backtest if needed)
    "max_position_weight": TRADING["max_position_weight"],
    "max_positions": TRADING["max_positions"],
    "require_human_approval": True,
    "approval_timeout_seconds": 300,
}

# =============================================================================
# Daemon Settings (Unified: OpenClaw + Trading)
# =============================================================================
DAEMON = {
    "rebalance_interval_minutes": 5,
    "auto_research_interval_hours": 24,
    "trade_approval_timeout_seconds": 300,
    "max_proposal_positions": 10,
}

# =============================================================================
# Sonnet Decision Maker Settings
# =============================================================================
SONNET_DECISION = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "default_stop_loss_pct": -0.05,
    "default_take_profit_pct": 0.10,
    # SL bounds: tightest = closest to entry, loosest = farthest from entry
    "tightest_stop_loss_pct": -0.02,   # -2% (closest allowed SL)
    "loosest_stop_loss_pct": -0.08,    # -8% (farthest allowed SL)
    # TP bounds: smallest = minimum profit target, largest = max target
    "smallest_take_profit_pct": 0.03,  # +3% minimum TP
    "largest_take_profit_pct": 0.15,   # +15% maximum TP
    "sltp_check_interval_seconds": 5,
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
