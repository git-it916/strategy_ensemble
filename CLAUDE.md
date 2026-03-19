# Strategy Ensemble - Project Rules

## Language
- Always respond in Korean (한국어로 답변)

## Project Overview
- Binance USDT-M perpetual futures trading bot
- 10 alphas → signal aggregation → Sonnet decision → execution
- 3x leverage, asymmetric long/short thresholds
- Daemon: 5min rebalance cycle, 5s SL/TP check

## Tech Stack
- Python 3, conda environment
- ccxt (Binance), anthropic (Sonnet), telethon (Telegram)
- loguru logging, pydantic validation, duckdb
- Config: `config/settings.py` (all magic numbers here)
- Keys: `config/keys.yaml` (NEVER commit or display)

## Project Structure
```
config/          — settings.py (weights, thresholds), keys.yaml
src/daemon/      — unified_daemon.py, signal_aggregator.py, sonnet_decision_maker.py, sltp_monitor.py, position_store.py, telegram_notifier.py
src/alphas/      — base_alpha.py, v2/ (alpha implementations)
src/ensemble/    — stacking models
src/execution/   — order execution
scripts/         — run_daemon.py, backtest.py
logs/            — daemon logs
```

## Code Conventions
- All config values in `config/settings.py`, no hardcoded magic numbers
- Korean comments are fine (existing codebase uses Korean)
- Use loguru for logging, not print()
- Always handle NaN values in alpha calculations
- Test with `pytest` from project root

## Key Commands
- Run daemon: `python scripts/run_daemon.py`
- Run tests: `pytest tests/`
- Check logs: `tail -f logs/`

## Critical Rules
- NEVER modify `config/keys.yaml` or expose API keys
- NEVER change SL/TP values in position_store during HOLD decisions
- Always check for NaN before aggregating alpha scores
- Sonnet is the PRIMARY decision maker, not a rubber stamp
