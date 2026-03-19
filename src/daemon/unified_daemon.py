"""
Unified Daemon

Single-process daemon merging OpenClaw (research) and Trading Pipeline (execution).

Main loop:
  - Telegram command polling: every 5 seconds
  - Signal generation + rebalance: every 15 minutes
  - Lifecycle check: once per day
  - Auto research: every 24 hours (optional)
  - WebSocket: continuous 1m candle collection (separate thread)
"""

from __future__ import annotations

import logging
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
import yaml

from config.settings import (
    DAEMON,
    ENSEMBLE,
    GATES,
    MODELS_DIR,
    SONNET_DECISION,
    STRATEGIES,
    TRADING,
    UNIVERSE,
    WEBSOCKET_CONFIG,
)

# V2 alpha system imports (lazy-loaded via ALPHA_VERSION)
try:
    from config.settings import ALPHA_VERSION, STRATEGIES_V2, ENSEMBLE_V2
except ImportError:
    ALPHA_VERSION = "v1"
    STRATEGIES_V2 = {}
    ENSEMBLE_V2 = {}

from src.daemon.signal_aggregator import SignalAggregator, AggregatedSignal
from src.daemon.sonnet_decision_maker import SonnetDecisionMaker, SonnetDecision, PositionDecision, DECISION_LOG_DIR
from src.daemon.position_store import PositionStore
from src.daemon.sltp_monitor import SLTPMonitor
from src.daemon.trade_proposal import TradeProposalBuilder

logger = logging.getLogger(__name__)


class UnifiedDaemon:
    """
    Unified daemon combining OpenClaw research with live trading.

    Pipeline per rebalance cycle:
        1. Collect raw signals from 7 alphas (10 coins)
        2. Gather market context (BTC trend, regime)
        3. Call Sonnet with signals + context → structured decision
        4. Execute via Rebalancer
        5. Store SL/TP levels for continuous monitoring

    Between cycles:
        - SL/TP monitor checks prices every 5 seconds
        - Auto-closes positions when stop-loss or take-profit is hit
    """

    def __init__(
        self,
        dry_run: bool = False,
        require_approval: bool = False,
        enable_research: bool = False,
    ):
        self.dry_run = dry_run
        self.require_approval = require_approval
        self.enable_research = enable_research

        self._running = False
        self._paused = False
        self._processing = False

        self._last_rebalance: datetime | None = None
        self._last_lifecycle: datetime | None = None
        self._last_research: datetime | None = None
        self._last_data_refresh: datetime | None = None

        # Components (initialized in initialize())
        self.keys: dict = {}
        self.binance_api = None
        self.ws = None
        self.telegram = None
        self.registry = None
        self.tracker = None
        self.lifecycle = None
        self.claude_orchestrator = None
        self.leverage_calc = None
        self.rebalancer = None
        self.openclaw_daemon = None

        self.signal_aggregator = SignalAggregator()
        self.proposal_builder = TradeProposalBuilder()
        self.sonnet_decision_maker = None
        self.position_store = None
        self.sltp_monitor = None

        # Data
        self._prices: pd.DataFrame | None = None
        self._prices_1h: pd.DataFrame | None = None  # Hourly OHLCV for intraday alphas
        self._prices_5m: pd.DataFrame | None = None  # 5m OHLCV for intraday momentum
        self._features: pd.DataFrame | None = None
        self._universe: list[str] = []

        # Ensemble (trained model, optional)
        self._ensemble = None

        # openclaw_1 alpha instances
        self._base_alphas: dict[str, Any] = {}

        # V2 alpha system (initialized if ALPHA_VERSION == "v2")
        self._v2_alphas: dict[str, Any] = {}
        self._v2_alpha_categories: dict[str, str] = {}
        self._data_manager = None
        self._enhanced_aggregator = None

        # Config
        self._rebalance_interval = DAEMON.get("rebalance_interval_minutes", 30)

    # ==================================================================
    # Initialization
    # ==================================================================

    def initialize(self) -> None:
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("Unified Daemon — Initializing")
        logger.info(f"  Dry Run: {self.dry_run}")
        logger.info(f"  Approval: {self.require_approval}")
        logger.info(f"  Research: {self.enable_research}")
        logger.info(f"  Rebalance Interval: {self._rebalance_interval}min")
        logger.info("=" * 60)

        self._load_keys()
        self._init_binance_api()
        self._load_data()
        self._init_base_alphas()
        if ALPHA_VERSION == "v2":
            self._init_v2_alphas()
        self._init_ensemble()
        self._init_openclaw()
        self._init_rebalancer()
        self._init_sonnet()
        self._init_websocket()
        self._register_extra_commands()

        logger.info("All components initialized")

        self.telegram.send_message(
            "<b>Unified Daemon Started</b>\n\n"
            f"Universe: {len(self._universe)} symbols\n"
            f"Base alphas: {len(self._base_alphas)}\n"
            f"Decision: Sonnet ({SONNET_DECISION['model'].split('-')[1]})\n"
            f"SL/TP: Active (5s check)\n"
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}\n"
            f"Max positions: {TRADING.get('max_positions', 3)}"
        )

    def _load_keys(self) -> None:
        keys_path = Path(__file__).parent.parent.parent / "config" / "keys.yaml"
        if not keys_path.exists():
            raise RuntimeError("config/keys.yaml not found!")
        with open(keys_path) as f:
            self.keys = yaml.safe_load(f)

    def _init_binance_api(self) -> None:
        from src.execution.binance_api import BinanceApi

        binance_cfg = self.keys.get("binance", {})
        self.binance_api = BinanceApi(
            api_key=binance_cfg.get("api_key", ""),
            api_secret=binance_cfg.get("api_secret", ""),
        )
        logger.info("BinanceApi initialized")

    def _load_data(self) -> None:
        """Fetch historical data from Binance via ccxt."""
        n_symbols = UNIVERSE.get("top_n_by_volume", 50)
        exclude = UNIVERSE.get("exclude_symbols", [])
        lookback_days = 300

        logger.info(f"Fetching top {n_symbols} symbols by volume...")
        self._universe = self.binance_api.get_top_symbols(n=n_symbols, exclude=exclude)
        logger.info(f"Universe: {len(self._universe)} symbols")

        logger.info(f"Fetching {lookback_days}d OHLCV...")
        self._prices = self.binance_api.get_ohlcv_batch(
            self._universe, timeframe="1d", days=lookback_days,
        )
        logger.info(
            f"OHLCV: {len(self._prices)} rows, "
            f"{self._prices['ticker'].nunique()} symbols"
        )

        # Fetch 7 days of 1h candles for intraday alphas (RSI, VWAP)
        logger.info("Fetching 7d hourly OHLCV for intraday alphas...")
        try:
            self._prices_1h = self.binance_api.get_ohlcv_batch(
                self._universe, timeframe="1h", days=7,
            )
            if self._prices_1h is not None and not self._prices_1h.empty:
                # Rename 'date' to 'datetime' for intraday resolution
                if "date" in self._prices_1h.columns:
                    self._prices_1h = self._prices_1h.rename(columns={"date": "datetime"})
                logger.info(
                    f"Hourly OHLCV: {len(self._prices_1h)} rows, "
                    f"{self._prices_1h['ticker'].nunique()} symbols"
                )
            else:
                logger.warning("Hourly OHLCV fetch returned empty")
                self._prices_1h = None
        except Exception as e:
            logger.error(f"Failed to fetch hourly OHLCV: {e}")
            self._prices_1h = None

        # Fetch 5m candles for intraday momentum timing
        intraday_mom_cfg = STRATEGIES.get("intraday_time_series_momentum", {})
        intraday_mom_tf = intraday_mom_cfg.get("timeframe", "5m")
        intraday_mom_days = int(intraday_mom_cfg.get("history_days", 5))
        logger.info(
            f"Fetching {intraday_mom_days}d {intraday_mom_tf} OHLCV "
            "for intraday momentum..."
        )
        try:
            self._prices_5m = self.binance_api.get_ohlcv_batch(
                self._universe, timeframe=intraday_mom_tf, days=intraday_mom_days,
            )
            if self._prices_5m is not None and not self._prices_5m.empty:
                if "date" in self._prices_5m.columns:
                    self._prices_5m = self._prices_5m.rename(columns={"date": "datetime"})
                logger.info(
                    f"5m OHLCV: {len(self._prices_5m)} rows, "
                    f"{self._prices_5m['ticker'].nunique()} symbols"
                )
            else:
                logger.warning("5m OHLCV fetch returned empty")
                self._prices_5m = None
        except Exception as e:
            logger.error(f"Failed to fetch 5m OHLCV: {e}")
            self._prices_5m = None

        logger.info("Fetching 90d funding rates...")
        funding = self.binance_api.get_funding_history_batch(self._universe, days=90)

        if not funding.empty:
            self._features = self._prices[["date", "ticker"]].copy()
            self._features = self._features.merge(
                funding[["date", "ticker", "funding_rate"]],
                on=["date", "ticker"],
                how="left",
            )
            logger.info(f"Features: {len(self._features)} rows (with funding_rate)")
        else:
            self._features = None
            logger.warning("No funding rate data")

        self._last_data_refresh = datetime.now()

    def _refresh_data(self) -> None:
        """
        Re-fetch OHLCV and funding rate data from Binance.

        Called periodically (every 4 hours) so that alpha signals and regime
        detection use up-to-date price data instead of stale startup data.
        Only re-fetches prices and funding — does NOT re-detect universe
        or re-fit alphas (those are heavier operations).
        """
        logger.info("Refreshing price & funding data...")
        try:
            lookback_days = 300
            new_prices = self.binance_api.get_ohlcv_batch(
                self._universe, timeframe="1d", days=lookback_days,
            )
            if new_prices is not None and not new_prices.empty:
                self._prices = new_prices
                logger.info(
                    f"Prices refreshed: {len(self._prices)} rows, "
                    f"{self._prices['ticker'].nunique()} symbols"
                )
            else:
                logger.warning("Price refresh returned empty — keeping old data")

            # Refresh hourly OHLCV for intraday alphas
            try:
                new_1h = self.binance_api.get_ohlcv_batch(
                    self._universe, timeframe="1h", days=7,
                )
                if new_1h is not None and not new_1h.empty:
                    if "date" in new_1h.columns:
                        new_1h = new_1h.rename(columns={"date": "datetime"})
                    self._prices_1h = new_1h
                    logger.info(f"Hourly OHLCV refreshed: {len(self._prices_1h)} rows")
            except Exception as e:
                logger.error(f"Hourly OHLCV refresh failed: {e}")

            # Refresh 5m OHLCV for intraday momentum
            intraday_mom_cfg = STRATEGIES.get("intraday_time_series_momentum", {})
            intraday_mom_tf = intraday_mom_cfg.get("timeframe", "5m")
            intraday_mom_days = int(intraday_mom_cfg.get("history_days", 5))
            try:
                new_5m = self.binance_api.get_ohlcv_batch(
                    self._universe, timeframe=intraday_mom_tf, days=intraday_mom_days,
                )
                if new_5m is not None and not new_5m.empty:
                    if "date" in new_5m.columns:
                        new_5m = new_5m.rename(columns={"date": "datetime"})
                    self._prices_5m = new_5m
                    logger.info(f"5m OHLCV refreshed: {len(self._prices_5m)} rows")
            except Exception as e:
                logger.error(f"5m OHLCV refresh failed: {e}")

            # Refresh funding rates
            funding = self.binance_api.get_funding_history_batch(
                self._universe, days=90,
            )
            if not funding.empty and self._prices is not None:
                self._features = self._prices[["date", "ticker"]].copy()
                self._features = self._features.merge(
                    funding[["date", "ticker", "funding_rate"]],
                    on=["date", "ticker"],
                    how="left",
                )
                logger.info(f"Features refreshed: {len(self._features)} rows")

            self._last_data_refresh = datetime.now()

        except Exception as e:
            logger.error(f"Data refresh failed (keeping old data): {e}")

    def _refresh_intraday_data(self) -> None:
        """
        Refresh 1h and 5m OHLCV every rebalance cycle so intraday alphas
        (RSI, VWAP, intraday_tsm — 85% of weight) see fresh prices.
        Daily OHLCV and funding rates stay on the 4h schedule.
        """
        try:
            new_1h = self.binance_api.get_ohlcv_batch(
                self._universe, timeframe="1h", days=7,
            )
            if new_1h is not None and not new_1h.empty:
                if "date" in new_1h.columns:
                    new_1h = new_1h.rename(columns={"date": "datetime"})
                self._prices_1h = new_1h
                logger.info(f"1h refreshed: {len(self._prices_1h)} rows")
        except Exception as e:
            logger.warning(f"1h refresh failed (keeping old): {e}")

        try:
            intraday_cfg = STRATEGIES.get("intraday_time_series_momentum", {})
            tf = intraday_cfg.get("timeframe", "5m")
            days = int(intraday_cfg.get("history_days", 5))
            new_5m = self.binance_api.get_ohlcv_batch(
                self._universe, timeframe=tf, days=days,
            )
            if new_5m is not None and not new_5m.empty:
                if "date" in new_5m.columns:
                    new_5m = new_5m.rename(columns={"date": "datetime"})
                self._prices_5m = new_5m
                logger.info(f"5m refreshed: {len(self._prices_5m)} rows")
        except Exception as e:
            logger.warning(f"5m refresh failed (keeping old): {e}")

    def _should_refresh_data(self, now: datetime) -> bool:
        """Check if price data should be refreshed (every 4 hours)."""
        if self._last_data_refresh is None:
            return False  # Just loaded during init
        return (now - self._last_data_refresh) > timedelta(hours=4)

    def _init_base_alphas(self) -> None:
        """Load and fit openclaw_1 rule-based alphas."""
        from src.alphas.openclaw_1 import (
            CSMomentum,
            TimeSeriesMomentum,
            TimeSeriesMeanReversion,
            PriceVolumeDivergence,
            VolumeMomentum,
            LowVolatilityAnomaly,
            FundingRateCarry,
            IntradayRSI,
            IntradayTimeSeriesMomentum,
            IntradayVWAP,
        )

        alpha_classes = [
            CSMomentum,
            TimeSeriesMomentum,
            TimeSeriesMeanReversion,
            PriceVolumeDivergence,
            VolumeMomentum,
            LowVolatilityAnomaly,
            FundingRateCarry,
        ]

        # Intraday alphas
        intraday_classes = [IntradayRSI, IntradayVWAP, IntradayTimeSeriesMomentum]

        for cls in alpha_classes:
            try:
                if cls is CSMomentum:
                    cfg = STRATEGIES.get("cs_momentum", {})
                    alpha = cls(
                        lookback_days=int(cfg.get("lookback_days", 21)),
                        skip_days=int(cfg.get("skip_days", 3)),
                    )
                elif cls is TimeSeriesMomentum:
                    cfg = STRATEGIES.get("time_series_momentum", {})
                    alpha = cls(
                        lookback_days=int(cfg.get("lookback_days", 20)),
                    )
                elif cls is TimeSeriesMeanReversion:
                    cfg = STRATEGIES.get("time_series_mean_reversion", {})
                    alpha = cls(
                        signal_window=int(cfg.get("signal_window", 5)),
                        baseline_window=int(cfg.get("baseline_window", 60)),
                    )
                elif cls is PriceVolumeDivergence:
                    cfg = STRATEGIES.get("pv_divergence", {})
                    alpha = cls(
                        lookback_days=int(cfg.get("lookback_days", 20)),
                    )
                elif cls is VolumeMomentum:
                    cfg = STRATEGIES.get("volume_momentum", {})
                    alpha = cls(
                        lookback_days=int(cfg.get("lookback_days", 20)),
                    )
                elif cls is LowVolatilityAnomaly:
                    cfg = STRATEGIES.get("low_volatility_anomaly", {})
                    alpha = cls(
                        lookback_days=int(cfg.get("lookback_days", 20)),
                    )
                elif cls is FundingRateCarry:
                    cfg = STRATEGIES.get("funding_rate_carry", {})
                    alpha = cls(
                        lookback_days=int(cfg.get("lookback_days", 14)),
                        abs_threshold=float(cfg.get("abs_threshold", 0.0001)),
                    )
                else:
                    alpha = cls()
                alpha.fit(self._prices, self._features)
                self._base_alphas[alpha.name] = alpha
                logger.info(f"  Base alpha loaded: {alpha.name}")
            except Exception as e:
                logger.error(f"Failed to load alpha {cls.__name__}: {e}")

        for cls in intraday_classes:
            try:
                if cls is IntradayRSI:
                    cfg = STRATEGIES.get("intraday_rsi", {})
                    alpha = cls(
                        rsi_period=int(cfg.get("rsi_period", 14)),
                        oversold=float(cfg.get("oversold", 30)),
                        overbought=float(cfg.get("overbought", 70)),
                    )
                    fit_prices = self._prices_1h if self._prices_1h is not None else self._prices
                elif cls is IntradayVWAP:
                    cfg = STRATEGIES.get("intraday_vwap", {})
                    alpha = cls(
                        lookback_bars=int(cfg.get("lookback_bars", 24)),
                    )
                    fit_prices = self._prices_1h if self._prices_1h is not None else self._prices
                elif cls is IntradayTimeSeriesMomentum:
                    cfg = STRATEGIES.get("intraday_time_series_momentum", {})
                    alpha = cls(
                        lookback_bars=int(cfg.get("lookback_bars", 36)),
                        scale=float(cfg.get("scale", 5.0)),
                    )
                    fit_prices = self._prices_5m if self._prices_5m is not None else self._prices
                else:
                    alpha = cls()
                    fit_prices = self._prices
                alpha.fit(fit_prices, self._features)
                self._base_alphas[alpha.name] = alpha
                logger.info(f"  Intraday alpha loaded: {alpha.name}")
            except Exception as e:
                logger.error(f"Failed to load intraday alpha {cls.__name__}: {e}")

        logger.info(f"Base alphas: {len(self._base_alphas)} loaded")

    def _init_v2_alphas(self) -> None:
        """Load and fit v2 alpha suite — 10 alphas hardcoded."""
        from src.alphas.v2 import (
            MomentumMultiScale,
            MomentumComposite,
            FundingCarryEnhanced,
            MeanReversionMultiHorizon,
            IntradayVWAPV2,
            IntradayRSIV2,
            OrderbookImbalance,
            DerivativesSentiment,
            VolatilityRegime,
            SpreadMomentum,
        )
        from src.data.data_manager import DataManager
        from src.daemon.enhanced_signal_aggregator import EnhancedSignalAggregator

        # Initialize DataManager
        settings = {"UNIVERSE": UNIVERSE, "STRATEGIES": STRATEGIES}
        self._data_manager = DataManager(self.binance_api, settings)
        self._data_manager._prices_1d = self._prices
        self._data_manager._prices_1h = self._prices_1h
        self._data_manager._prices_5m = self._prices_5m
        self._data_manager._features = self._features
        self._data_manager._universe = self._universe
        self._data_manager._init_orderbook_collector()
        self._data_manager._last_daily_refresh = datetime.now()
        self._data_manager._last_intraday_refresh = datetime.now()

        self._enhanced_aggregator = EnhancedSignalAggregator()

        # ──────────────────────────────────────────────────────────
        # 10 alphas + 1 modifier — hardcoded, no config loop
        # ──────────────────────────────────────────────────────────
        self._v2_alphas: dict[str, Any] = {}
        self._v2_alpha_categories: dict[str, str] = {}
        self._v2_weights: dict[str, float] = {}

        def _register(alpha, category: str, weight: float):
            try:
                alpha.fit(self._prices, self._features)
                self._v2_alphas[alpha.name] = alpha
                self._v2_alpha_categories[alpha.name] = category
                self._v2_weights[alpha.name] = weight
                logger.info(f"  V2: {alpha.name} ({category}, w={weight:.0%})")
            except Exception as e:
                logger.error(f"V2 alpha init failed: {alpha.name}: {e}")

        # 1. MomentumMultiScale — 22% (primary intraday timing)
        _register(
            MomentumMultiScale(
                name="MomentumMultiScale",
                lookbacks=(6, 18, 36),
                weights=(0.35, 0.35, 0.30),
                scale=7.0,
            ),
            category="momentum", weight=0.22,
        )

        # 2. FundingCarryEnhanced — 18% (structural carry)
        _register(
            FundingCarryEnhanced(
                name="FundingCarryEnhanced",
                lookback_days=14,
                velocity_lookback=7,
            ),
            category="carry", weight=0.18,
        )

        # 3. MomentumComposite — 15% (absolute + relative + risk-adjusted)
        _register(
            MomentumComposite(
                name="MomentumComposite",
                lookback_days=20,
                skip_days=3,
            ),
            category="momentum", weight=0.15,
        )

        # 4. IntradayVWAPV2 — 10% (VWAP band mean-reversion)
        _register(
            IntradayVWAPV2(
                name="IntradayVWAPV2",
                lookback_bars=24,
                band_threshold=1.5,
            ),
            category="mean_reversion", weight=0.10,
        )

        # 5. IntradayRSIV2 — 8% (RSI overbought/oversold)
        _register(
            IntradayRSIV2(
                name="IntradayRSIV2",
                rsi_period=14,
                oversold=30.0,
                overbought=70.0,
            ),
            category="mean_reversion", weight=0.08,
        )

        # 6. DerivativesSentiment — 8% (OI-funding divergence)
        _register(
            DerivativesSentiment(
                name="DerivativesSentiment",
                oi_change_threshold=5.0,
                funding_threshold=0.0003,
            ),
            category="carry", weight=0.08,
        )

        # 7. MeanReversionMultiHorizon — 8% (multi-horizon z-score)
        _register(
            MeanReversionMultiHorizon(
                name="MeanReversionMultiHorizon",
                horizons=[(3, 20), (5, 60), (10, 120)],
            ),
            category="mean_reversion", weight=0.08,
        )

        # 8. OrderbookImbalance — 5% (미검증 → 보수적 시작)
        _register(
            OrderbookImbalance(
                name="OrderbookImbalance",
                scale=3.0,
            ),
            category="microstructure", weight=0.05,
        )

        # 9. SpreadMomentum — 3% (미검증 → 최소 시작)
        _register(
            SpreadMomentum(
                name="SpreadMomentum",
                min_snapshots=5,
                scale=2.0,
            ),
            category="microstructure", weight=0.03,
        )

        # 10. VolatilityRegime — 3% (confidence modifier)
        _register(
            VolatilityRegime(name="VolatilityRegime"),
            category="composite", weight=0.03,
        )

        total_w = sum(self._v2_weights.values())
        logger.info(
            f"V2 alphas: {len(self._v2_alphas)} loaded, "
            f"total weight={total_w:.2f}"
        )

        # Stacking meta-model 초기화
        alpha_names = list(self._v2_alphas.keys())
        self._enhanced_aggregator.init_stacking(
            alpha_names=alpha_names,
            forward_horizon=5,
            train_window_days=90,
            retrain_interval_days=30,
        )
        # 기존 학습 모델 로드 시도
        self._enhanced_aggregator.load_stacking()

    def _collect_v2_signals(self) -> tuple[dict, dict, float]:
        """
        Collect signals from v2 alphas using DataBundle.
        Uses generate_signals_v2_safe() for data_requirements validation.

        Returns:
            (alpha_signals, alpha_metadata, vol_confidence)
        """
        bundle = self._data_manager.get_bundle()
        now = datetime.now(timezone.utc)

        alpha_signals: dict[str, pd.DataFrame] = {}
        alpha_metadata: dict[str, dict] = {}
        vol_confidence = 1.0

        for name, alpha in self._v2_alphas.items():
            # generate_signals_v2_safe: 데이터 검증 + 에러 핸들링 내장
            result = alpha.generate_signals_v2_safe(now, bundle)
            if result.signals is not None and not result.signals.empty:
                alpha_signals[name] = result.signals
                alpha_metadata[name] = {
                    "confidence": result.confidence,
                    "regime_affinity": result.regime_affinity,
                    "signal_decay_hours": result.signal_decay_hours,
                }
                alpha_metadata[name].update(result.metadata)

                if name == "VolatilityRegime":
                    vol_confidence = result.confidence

        return alpha_signals, alpha_metadata, vol_confidence

    def _aggregate_v2_signals(
        self,
        alpha_signals: dict[str, pd.DataFrame],
        alpha_metadata: dict[str, dict],
        vol_confidence: float,
        regime: str = "sideways",
    ):
        """Aggregate v2 signals using EnhancedSignalAggregator."""
        # 하드코딩된 weights 사용 (config가 아닌 _init_v2_alphas에서 설정)
        regime_mults = ENSEMBLE_V2.get("regime_preferences", {}).get(regime, {})

        return self._enhanced_aggregator.aggregate(
            alpha_signals=alpha_signals,
            alpha_weights=self._v2_weights,
            alpha_metadata=alpha_metadata,
            alpha_categories=self._v2_alpha_categories,
            regime_multipliers=regime_mults,
            vol_confidence=vol_confidence,
        )

    def _init_ensemble(self) -> None:
        """Load trained ensemble model (optional)."""
        from src.ensemble_models import ModelManager

        manager = ModelManager(MODELS_DIR)
        try:
            self._ensemble = manager.load_ensemble()
            logger.info(f"Ensemble loaded: {list(self._ensemble.strategies.keys())}")
        except FileNotFoundError:
            logger.info("No trained ensemble found (optional)")
            self._ensemble = None

    def _init_openclaw(self) -> None:
        """Initialize OpenClaw subsystems directly (no duplicate data loading)."""
        import anthropic as anthropic_mod

        from src.openclaw.config import OPENCLAW_DATA_DIR, OPENCLAW_LOGS_DIR, GENERATED_ALPHAS_DIR
        from src.openclaw.registry.alpha_registry import AlphaRegistry
        from src.openclaw.registry.performance_tracker import PerformanceTracker
        from src.openclaw.registry.lifecycle_manager import LifecycleManager
        from src.openclaw.orchestrator.claude_orchestrator import ClaudeEnsembleOrchestrator
        from src.openclaw.orchestrator.weight_optimizer import WeightOptimizer
        from src.openclaw.execution.leverage_calculator import LeverageCalculator
        from src.openclaw.telegram.command_handler import OpenClawTelegramHandler
        from src.openclaw.telegram.formatters import TelegramFormatter

        # Ensure directories
        OPENCLAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        OPENCLAW_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        GENERATED_ALPHAS_DIR.mkdir(parents=True, exist_ok=True)

        # Anthropic client
        api_key = self.keys.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in config/keys.yaml")
        self.anthropic_client = anthropic_mod.Anthropic(api_key=api_key)

        # Core components
        self.registry = AlphaRegistry()
        self.tracker = PerformanceTracker()

        telegram_cfg = self.keys.get("telegram", {})
        self.telegram = OpenClawTelegramHandler(
            bot_token=telegram_cfg.get("bot_token"),
            chat_id=telegram_cfg.get("chat_id"),
            broadcast_chat_ids=telegram_cfg.get("broadcast_chat_ids", []),
        )
        self.formatter = TelegramFormatter()

        self.lifecycle = LifecycleManager(
            registry=self.registry,
            performance_tracker=self.tracker,
            notifier=self.telegram,
        )

        self.weight_optimizer = WeightOptimizer()
        self.claude_orchestrator = ClaudeEnsembleOrchestrator(
            anthropic_client=self.anthropic_client,
        )
        self.leverage_calc = LeverageCalculator()

        # Build OpenClawDaemon for research (lazy — shares our components)
        from src.openclaw.main import OpenClawDaemon
        self.openclaw_daemon = OpenClawDaemon(dry_run=self.dry_run)
        self.openclaw_daemon._running = True
        self.openclaw_daemon.anthropic_client = self.anthropic_client
        self.openclaw_daemon.binance_api = self.binance_api
        self.openclaw_daemon.registry = self.registry
        self.openclaw_daemon.tracker = self.tracker
        self.openclaw_daemon.telegram = self.telegram
        self.openclaw_daemon.formatter = self.formatter
        self.openclaw_daemon.lifecycle = self.lifecycle
        self.openclaw_daemon.weight_optimizer = self.weight_optimizer
        self.openclaw_daemon.claude_orchestrator = self.claude_orchestrator
        self.openclaw_daemon.leverage_calc = self.leverage_calc

        # Research components (lazy init — only loaded when /research is called)
        try:
            from src.openclaw.researcher.brave_search import BraveSearchClient
            from src.openclaw.researcher.idea_parser import IdeaParser
            from src.openclaw.researcher.code_generator import AlphaCodeGenerator
            from src.openclaw.researcher.code_validator import CodeValidator
            from src.openclaw.researcher.experiment_tracker import ExperimentTracker
            from src.openclaw.validator.quality_gates import QualityGateChecker
            from src.openclaw.validator.correlation_checker import CorrelationChecker
            from src.openclaw.validator.summary_builder import SummaryBuilder

            self.openclaw_daemon.brave_search = BraveSearchClient(
                api_key=self.keys.get("BRAVE_API_KEY")
            )
            self.openclaw_daemon.idea_parser = IdeaParser(self.anthropic_client)
            self.openclaw_daemon.code_generator = AlphaCodeGenerator(self.anthropic_client)
            self.openclaw_daemon.code_validator = CodeValidator()
            self.openclaw_daemon.experiment_tracker = ExperimentTracker()
            self.openclaw_daemon.quality_gates = QualityGateChecker()
            self.openclaw_daemon.correlation_checker = CorrelationChecker()
            self.openclaw_daemon.summary_builder = SummaryBuilder()
        except Exception as e:
            logger.warning(f"Research components init failed (non-critical): {e}")

        # Register OpenClaw Telegram commands
        self.telegram.register_command("research", self.openclaw_daemon._cmd_research)
        self.telegram.register_command("status", self.openclaw_daemon._cmd_status)
        self.telegram.register_command("kill", self.openclaw_daemon._cmd_kill)
        self.telegram.register_command("approve", self.openclaw_daemon._cmd_approve)
        self.telegram.register_command("reject", self.openclaw_daemon._cmd_reject)
        self.telegram.register_command("mutate", self.openclaw_daemon._cmd_mutate)

        # Chat handler
        try:
            from src.openclaw.telegram.chat_handler import ChatHandler
            chat_handler = ChatHandler(
                anthropic_client=self.anthropic_client,
                registry=self.registry,
                tracker=self.tracker,
            )
            self.telegram.set_chat_handler(chat_handler)
        except Exception as e:
            logger.warning(f"Chat handler init failed: {e}")

        logger.info(
            f"OpenClaw initialized: {self.registry.active_count} active, "
            f"{len(self.registry.get_paper())} paper, "
            f"{self.registry.total_count} total"
        )

    def _init_rebalancer(self) -> None:
        from src.openclaw.execution.rebalancer import Rebalancer

        self.rebalancer = Rebalancer(
            binance_api=self.binance_api,
            notifier=self.telegram,
            dry_run=self.dry_run,
        )
        logger.info("Rebalancer initialized")

    def _init_sonnet(self) -> None:
        """Initialize Sonnet decision maker, position store, and SL/TP monitor."""
        from src.openclaw.config import OPENCLAW_DATA_DIR

        self.sonnet_decision_maker = SonnetDecisionMaker(
            anthropic_client=self.anthropic_client,
            model=SONNET_DECISION.get("model"),
        )
        logger.info(f"Sonnet decision maker initialized ({SONNET_DECISION['model']})")

        self.position_store = PositionStore(
            path=OPENCLAW_DATA_DIR / "managed_positions.json",
        )
        logger.info(
            f"Position store: {len(self.position_store.get_active())} active positions"
        )

        self.sltp_monitor = SLTPMonitor(
            position_store=self.position_store,
            binance_api=self.binance_api,
            rebalancer=self.rebalancer,
            notifier=self.telegram,
            dry_run=self.dry_run,
        )
        logger.info("SL/TP monitor initialized (5s interval)")

    def _init_websocket(self) -> None:
        from src.execution.binance_websocket import BinanceWebSocket

        candle_interval = WEBSOCKET_CONFIG.get("candle_interval", "1m")
        self.ws = BinanceWebSocket(candle_interval=candle_interval)
        logger.info(f"WebSocket: {candle_interval} candles")

    def _register_extra_commands(self) -> None:
        """Register additional Telegram commands on top of OpenClaw's."""
        self.telegram.register_command("positions", self._cmd_positions)
        self.telegram.register_command("balance", self._cmd_balance)
        self.telegram.register_command("pause", self._cmd_pause)
        self.telegram.register_command("resume", self._cmd_resume)
        self.telegram.register_command("closeall", self._cmd_closeall)

    # ==================================================================
    # Main Loop
    # ==================================================================

    def run(self) -> None:
        """Main event loop."""
        self._running = True

        # Start WebSocket
        self.ws.start()
        for symbol in self._universe:
            self.ws.subscribe(symbol)
        logger.info(f"WebSocket subscribed: {len(self._universe)} symbols")

        try:
            while self._running:
                now = datetime.now()

                # 1. Poll Telegram commands
                events = self.telegram.poll_commands()
                self._handle_trade_callbacks(events)

                # 1.5. Check SL/TP triggers (every 5 seconds)
                if not self._paused and self.sltp_monitor:
                    try:
                        triggered = self.sltp_monitor.check_all()
                        for t in triggered:
                            logger.info(f"SL/TP triggered: {t}")
                            # Record cooldown so re-entry is blocked
                            ticker = t.get("ticker")
                            if ticker:
                                if not hasattr(self, "_close_cooldowns"):
                                    self._close_cooldowns = {}
                                self._close_cooldowns[ticker] = datetime.now()
                    except Exception as e:
                        logger.error(f"SL/TP monitor error: {e}")

                # 2. Rebalance cycle
                if not self._paused and self._should_rebalance(now):
                    if not self._processing:
                        self._processing = True
                        try:
                            self._run_rebalance_cycle(now)
                            self._last_rebalance = now
                        except Exception as e:
                            logger.error(f"Rebalance error: {e}\n{traceback.format_exc()}")
                            self.telegram.send_message(
                                f"<b>Rebalance Error</b>\n{str(e)[:200]}"
                            )
                        finally:
                            self._processing = False

                # 3. Daily lifecycle check
                if self._should_lifecycle(now):
                    try:
                        summary = self.lifecycle.daily_lifecycle_check()
                        self._last_lifecycle = now
                        logger.info(f"Lifecycle check: {summary}")
                    except Exception as e:
                        logger.error(f"Lifecycle error: {e}")

                # 4. Periodic data refresh (every 4 hours)
                if self._should_refresh_data(now):
                    try:
                        self._refresh_data()
                    except Exception as e:
                        logger.error(f"Data refresh error: {e}")

                # 5. Auto research (24h)
                if self.enable_research and self._should_research(now):
                    try:
                        self.telegram.send_message(
                            "🔬 <b>Auto research session starting...</b>"
                        )
                        self.openclaw_daemon.run_research_session()
                        self._last_research = now
                    except Exception as e:
                        logger.error(f"Research error: {e}")

                time.sleep(5)

        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.stop()

    def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down unified daemon...")
        self._running = False
        if self.ws:
            self.ws.stop()
        if self.openclaw_daemon:
            self.openclaw_daemon.stop()
        logger.info("Daemon stopped")

    # ==================================================================
    # Rebalance Cycle (Core Pipeline)
    # ==================================================================

    def _run_rebalance_cycle(self, now: datetime) -> None:
        """
        Rebalance cycle with Sonnet as decision maker.

        1. Collect raw alpha signals
        2. Gather market context
        3. Get current positions + P&L
        4. Call Sonnet → structured decision
        5. Send decision to Telegram
        6. Execute via Rebalancer
        7. Update PositionStore with SL/TP levels
        """
        logger.info(f"{'='*40} REBALANCE CYCLE {'='*40}")

        # -- Step 0: Refresh intraday data every cycle for fresh signals --
        self._refresh_intraday_data()

        # ================================================================
        # Step 1 + 1.5: 시그널 수집 + 통합 (v1/v2 분기)
        # ================================================================
        if ALPHA_VERSION == "v2" and self._v2_alphas:
            # ── V2: DataBundle → 10개 알파 → Stacking/가중합 ──
            if self._data_manager is not None:
                self._data_manager._prices_1d = self._prices
                self._data_manager._prices_1h = self._prices_1h
                self._data_manager._prices_5m = self._prices_5m
                self._data_manager._features = self._features
                self._data_manager.refresh_orderbooks()

            alpha_signals, alpha_metadata, vol_confidence = self._collect_v2_signals()
            if not alpha_signals:
                logger.info("No V2 signals generated")
                return
            logger.info(f"V2 signals from {len(alpha_signals)} alphas")

            # 레짐 감지 (기존 로직 재활용)
            regime = self._detect_regime() if hasattr(self, '_detect_regime') else "sideways"
            aggregated_v2 = self._aggregate_v2_signals(
                alpha_signals, alpha_metadata, vol_confidence, regime,
            )

            # Stacking 재학습 체크 (30일마다)
            if self._enhanced_aggregator is not None and self._prices is not None:
                self._enhanced_aggregator.retrain_stacking_if_needed(self._prices)

            # v1 aggregated 인터페이스로 변환 (하위 코드 호환)
            from src.daemon.signal_aggregator import AggregatedSignal
            aggregated = AggregatedSignal(
                scores=aggregated_v2.effective_scores,
                contributions=aggregated_v2.contributions,
                weights_used=aggregated_v2.weights_used,
            )
            # alpha_signals는 이미 {name: DataFrame} 형태
            alpha_entries = {}  # v2에서는 미사용

            method = aggregated_v2.ensemble_method
            logger.info(
                f"V2 aggregated ({method}): {len(aggregated.scores)} tickers, "
                f"top: {sorted(aggregated.scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]}"
            )
        else:
            # ── V1: 기존 가중합 경로 ──
            alpha_signals, alpha_entries = self._collect_all_signals(now)
            if not alpha_signals:
                logger.info("No signals generated from any source")
                return
            logger.info(f"Signals from {len(alpha_signals)} alphas")

            alpha_weights = self._config_weights_with_regime(alpha_signals)
            aggregated = self.signal_aggregator.aggregate(
                alpha_signals=alpha_signals,
                alpha_weights=alpha_weights,
            )

        logger.info(
            f"Aggregated: {len(aggregated.scores)} tickers, "
            f"top: {sorted(aggregated.scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]}"
        )

        # -- Step 1.6: Squeeze risk detection (informational for Sonnet) --
        # Detect squeeze conditions and pass as context — Sonnet decides whether to act
        squeeze_warnings = {}
        if self._features is not None and "funding_rate" in self._features.columns:
            latest_funding = self._features.sort_values("date").groupby("ticker").last()
            for ticker, score in list(aggregated.scores.items()):
                try:
                    fr = latest_funding.loc[ticker, "funding_rate"] if ticker in latest_funding.index else None
                    if fr is None or pd.isna(fr):
                        continue
                    # Get 24h price return
                    tkr_prices = self._prices[self._prices["ticker"] == ticker].sort_values("date") if self._prices is not None else None
                    if tkr_prices is None or len(tkr_prices) < 2:
                        continue
                    current_price = tkr_prices["close"].iloc[-1]
                    past_idx = max(0, len(tkr_prices) - 2)
                    past_price = tkr_prices["close"].iloc[past_idx]
                    if past_price <= 0:
                        continue
                    price_ret = (current_price / past_price) - 1

                    # Short squeeze risk: SHORT signal + negative funding + price rising
                    if score < 0 and fr < -0.0005 and price_ret > 0.05:
                        logger.warning(
                            f"SQUEEZE RISK: {ticker} SHORT score {score:.3f} "
                            f"(funding={fr:.4%}, 24h={price_ret:+.1%}) — Sonnet will decide"
                        )
                        squeeze_warnings[ticker] = f"SHORT SQUEEZE RISK: funding={fr:.4%} (shorts pay), price 24h={price_ret:+.1%}. Shorting is dangerous."
                    # Long squeeze risk: LONG signal + positive funding + price falling
                    elif score > 0 and fr > 0.0005 and price_ret < -0.05:
                        logger.warning(
                            f"SQUEEZE RISK: {ticker} LONG score {score:.3f} "
                            f"(funding={fr:.4%}, 24h={price_ret:+.1%}) — Sonnet will decide"
                        )
                        squeeze_warnings[ticker] = f"LONG SQUEEZE RISK: funding={fr:.4%} (longs pay), price 24h={price_ret:+.1%}. Longing is dangerous."
                except Exception as e:
                    logger.debug(f"Squeeze check failed for {ticker}: {e}")

        suppressed_tickers = set()
        defensive_scores: dict[str, dict[str, float]] = {}
        market_defensive_alerts = {}

        # -- Step 1.8: Signal-change detection — skip cycle if no positions and signals unchanged --
        # Only skip Sonnet when: (1) no positions held AND (2) signals unchanged
        # When holding positions, ALWAYS call Sonnet for ongoing management
        top5_key = tuple(
            sorted(aggregated.scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        )
        if not hasattr(self, "_last_signal_key"):
            self._last_signal_key = None
            self._signal_unchanged_count = 0

        has_positions = bool(
            self.position_store and self.position_store.get_active()
        )

        if top5_key == self._last_signal_key and not has_positions:
            self._signal_unchanged_count += 1
            if self._signal_unchanged_count < 3:  # Skip up to 2 cycles (10min) when in cash
                logger.info(
                    f"No positions & signals unchanged ({self._signal_unchanged_count}/3) — skipping Sonnet call. "
                    f"Top: {[t.replace('/USDT:USDT','') for t,_ in top5_key]}"
                )
                return
            else:
                logger.info("No positions & signals unchanged for 15min — forcing Sonnet check")
                self._signal_unchanged_count = 0
        else:
            self._signal_unchanged_count = 0
        self._last_signal_key = top5_key

        # -- Step 2: Market context --
        market_context = self._build_market_context()
        market_context["aggregated_scores"] = aggregated.scores
        market_context["alpha_weights"] = aggregated.weights_used
        if market_defensive_alerts:
            market_context["defensive_alerts"] = {
                t.replace("/USDT:USDT", ""): {k: round(v, 2) for k, v in d.items()}
                for t, d in market_defensive_alerts.items()
            }
        if suppressed_tickers:
            market_context["suppressed_tickers"] = [
                t.replace("/USDT:USDT", "") for t in suppressed_tickers
            ]
        if squeeze_warnings:
            market_context["squeeze_warnings"] = squeeze_warnings

        # -- Step 3: Current state --
        current_positions = self.binance_api.get_positions()
        account = self.binance_api.get_account()
        balance = float(account.get("total_wallet_balance", 0))

        # -- Step 3.5: Sync managed_positions with actual Binance positions --
        self._sync_position_store(current_positions)

        # -- Step 3.7: Build cooldown set --
        cooldown_tickers = set()
        if self.position_store:
            recently_closed = self.position_store.get_recently_closed(minutes=30)
            for cp in recently_closed:
                cooldown_tickers.add(cp.ticker)

        # -- Step 4: Rule-based decision (Sonnet 대체) --
        from src.daemon.rule_decision_maker import RuleDecisionMaker
        rule_maker = RuleDecisionMaker()
        decision = rule_maker.make_decision(
            effective_scores=aggregated.scores,
            contributions=aggregated.contributions,
            current_positions=current_positions,
            prices=self._prices,
            managed_positions=(
                self.position_store.get_active() if self.position_store else None
            ),
            cooldown_tickers=cooldown_tickers,
        )

        if not decision.positions:
            logger.info("Rule decision: no positions (staying in cash)")
            return

        # -- Step 5: Telegram 알림 (진입/청산) --
        from src.daemon.telegram_notifier import TelegramNotifier
        tg_notifier = self._get_telegram_notifier()

        for pos in decision.positions:
            if pos.action in ("LONG", "SHORT"):
                # 신규 진입 알림
                alpha_contribs = aggregated.contributions.get(pos.ticker, {})
                # 현재가 조회
                entry_price = self._get_current_price(pos.ticker)
                if entry_price > 0:
                    if pos.action == "LONG":
                        sl_price = entry_price * (1 + pos.stop_loss_pct)
                        tp_price = entry_price * (1 + pos.take_profit_pct)
                    else:
                        sl_price = entry_price * (1 - pos.stop_loss_pct)
                        tp_price = entry_price * (1 - pos.take_profit_pct)

                    if tg_notifier:
                        # 앙상블 메서드 정보
                        ensemble_method = "stacking" if hasattr(aggregated, 'ensemble_method') else "weighted_sum"
                        tg_notifier.send_entry(
                            coin=pos.ticker,
                            direction=pos.action,
                            entry_price=entry_price,
                            stop_loss_price=sl_price,
                            take_profit_price=tp_price,
                            alpha_contributions=alpha_contribs,
                            ensemble_method=ensemble_method,
                            ensemble_score=aggregated.scores.get(pos.ticker, 0),
                        )

        # 기존 텔레그램도 간단 알림
        proposal_text = self._format_rule_proposal(decision, balance)

        if self.dry_run:
            self.telegram.send_message(
                f"<b>[DRY RUN]</b>\n\n{proposal_text}"
            )
            logger.info("DRY RUN — no orders placed")
            return

        self.telegram.send_message(proposal_text)

        # -- Step 5.5: Approval gate --
        if self.require_approval:
            from config.settings import DAEMON
            timeout = DAEMON.get("trade_approval_timeout_seconds", 300)
            logger.info(f"Waiting for trade approval ({timeout}s timeout)...")
            self.telegram.send_message(
                "승인 대기 중... /approve 또는 /reject 를 입력하세요."
            )
            approved = self._wait_for_trade_approval(timeout=timeout)
            if not approved:
                logger.info("Trade REJECTED or timed out — skipping execution")
                self.telegram.send_message("거래가 거부되었거나 시간 초과되었습니다.")
                return
            logger.info("Trade APPROVED by user")
            self.telegram.send_message("승인됨 — 주문 실행 중...")

        # -- Step 5.8: Anti-churn filter --
        # Block CLOSE on positions younger than min_hold_minutes
        # Block LONG/SHORT on tickers closed within cooldown_minutes
        from config.settings import DAEMON
        min_hold_min = DAEMON.get("min_hold_minutes", 60)
        cooldown_min = DAEMON.get("cooldown_after_close_minutes", 60)
        now_dt = datetime.now()

        # Build lookup from managed positions
        managed_lookup = {}
        if self.position_store:
            for mp in self.position_store.get_active():
                managed_lookup[mp.ticker] = mp

        # Also track entry times from _position_open_times (survives managed_positions gaps)
        if not hasattr(self, "_position_open_times"):
            self._position_open_times = {}
        # Populate from managed_positions for any we haven't tracked yet
        for ticker, mp in managed_lookup.items():
            if ticker not in self._position_open_times:
                try:
                    self._position_open_times[ticker] = datetime.fromisoformat(mp.entry_time)
                except Exception:
                    pass

        # Build set of tickers currently held on Binance
        binance_held_tickers = set()
        if current_positions is not None and not current_positions.empty:
            for _, row in current_positions.iterrows():
                if abs(float(row.get("size", 0))) > 0:
                    binance_held_tickers.add(row.get("ticker", ""))

        # Signal gate prefetch (new entries only)
        signal_gate_cfg = GATES.get("signal_gate", {})
        signal_gate_enabled = bool(
            GATES.get("enabled", False) and signal_gate_cfg.get("enabled", False)
        )
        quote_volumes_24h: dict[str, float] = {}
        if signal_gate_enabled:
            new_entry_tickers = sorted({
                p.ticker for p in decision.positions
                if p.action in ("LONG", "SHORT") and p.ticker not in binance_held_tickers
            })
            if new_entry_tickers:
                quote_volumes_24h = self.binance_api.get_quote_volume_batch(new_entry_tickers)

        filtered_positions = []
        for pos in decision.positions:
            # --- Min hold check: block CLOSE if position too young ---
            if pos.action == "CLOSE" and pos.ticker in binance_held_tickers:
                open_time = self._position_open_times.get(pos.ticker)
                if open_time is not None:
                    held_min = (now_dt - open_time).total_seconds() / 60
                    if held_min < min_hold_min:
                        logger.info(
                            f"Anti-churn: blocking CLOSE {pos.ticker} "
                            f"(held {held_min:.0f}min < {min_hold_min}min). Forcing HOLD."
                        )
                        pos = PositionDecision(
                            ticker=pos.ticker, action="HOLD",
                            weight=pos.weight,
                            stop_loss_pct=pos.stop_loss_pct, take_profit_pct=pos.take_profit_pct,
                            reasoning=f"Anti-churn HOLD (held {held_min:.0f}min)",
                        )
                        filtered_positions.append(pos)
                        continue

            # --- Defensive emergency exit: override HOLD if filters scream danger ---
            if pos.action == "HOLD" and pos.ticker in defensive_scores:
                defenses = defensive_scores[pos.ticker]
                managed_mp = managed_lookup.get(pos.ticker)
                if managed_mp and defenses:
                    is_long = managed_mp.side == "LONG"
                    # Count how many filters strongly oppose the position direction
                    danger_count = 0
                    for d_score in defenses.values():
                        if is_long and d_score < -0.5:
                            danger_count += 1
                        elif not is_long and d_score > 0.5:
                            danger_count += 1
                    if danger_count >= 2:
                        logger.warning(
                            f"DEFENSIVE EXIT: {pos.ticker} — {danger_count} filters "
                            f"strongly opposing {managed_mp.side}. Forcing CLOSE."
                        )
                        pos = PositionDecision(
                            ticker=pos.ticker, action="CLOSE",
                            weight=0.0,
                            stop_loss_pct=pos.stop_loss_pct, take_profit_pct=pos.take_profit_pct,
                            reasoning=f"Defensive exit: {danger_count} filters opposing",
                        )
                        filtered_positions.append(pos)
                        continue

            # --- Cooldown check: block re-entry after recent close ---
            cooldowns = getattr(self, "_close_cooldowns", {})
            if pos.action in ("LONG", "SHORT") and pos.ticker in cooldowns:
                closed_at = cooldowns[pos.ticker]
                cooldown_elapsed = (now_dt - closed_at).total_seconds() / 60
                if cooldown_elapsed < cooldown_min:
                    logger.info(
                        f"Anti-churn: blocking {pos.action} {pos.ticker} "
                        f"(closed {cooldown_elapsed:.0f}min ago < {cooldown_min}min cooldown)"
                    )
                    continue

            # --- Suppressed ticker block: never open new positions on suppressed tickers ---
            if pos.action in ("LONG", "SHORT") and pos.ticker in suppressed_tickers:
                if pos.ticker not in binance_held_tickers:
                    logger.info(
                        f"Blocking new {pos.action} on suppressed ticker "
                        f"{pos.ticker.replace('/USDT:USDT', '')} — defensive filters oppose entry"
                    )
                    continue

            # --- Signal gate: volume surge + liquidity filter for NEW entries ---
            if (
                signal_gate_enabled
                and pos.action in ("LONG", "SHORT")
                and pos.ticker not in binance_held_tickers
            ):
                recent_bars = int(signal_gate_cfg.get("recent_volume_bars", 3))
                baseline_bars = int(signal_gate_cfg.get("baseline_volume_bars", 36))
                min_surge = float(signal_gate_cfg.get("volume_surge_min_ratio", 1.20))
                min_quote_vol = float(signal_gate_cfg.get("min_quote_volume_usdt", 0.0))

                surge_ratio = self._get_volume_surge_ratio(
                    pos.ticker,
                    recent_bars=recent_bars,
                    baseline_bars=baseline_bars,
                )
                quote_vol_24h = float(quote_volumes_24h.get(pos.ticker, 0.0))

                blocked_reasons = []
                if surge_ratio is not None and surge_ratio < min_surge:
                    blocked_reasons.append(
                        f"volume surge {surge_ratio:.2f} < {min_surge:.2f}"
                    )
                if quote_vol_24h > 0 and quote_vol_24h < min_quote_vol:
                    blocked_reasons.append(
                        f"24h quote vol ${quote_vol_24h:,.0f} < ${min_quote_vol:,.0f}"
                    )

                if blocked_reasons:
                    logger.info(
                        f"Signal gate blocked new {pos.action} "
                        f"{pos.ticker.replace('/USDT:USDT', '')}: "
                        + " | ".join(blocked_reasons)
                    )
                    continue

            # --- Same-ticker re-entry block: if Sonnet wants to CLOSE then re-OPEN same ticker ---
            if pos.action in ("LONG", "SHORT") and pos.ticker in binance_held_tickers:
                # Already holding this ticker — check if Sonnet also issued CLOSE for it
                close_in_batch = any(
                    p.action == "CLOSE" and p.ticker == pos.ticker
                    for p in decision.positions
                )
                if close_in_batch:
                    logger.info(
                        f"Anti-churn: blocking same-cycle CLOSE+{pos.action} for {pos.ticker}. "
                        f"Converting to HOLD."
                    )
                    pos = PositionDecision(
                        ticker=pos.ticker, action="HOLD",
                        weight=pos.weight,
                        stop_loss_pct=pos.stop_loss_pct, take_profit_pct=pos.take_profit_pct,
                        reasoning=f"Anti-churn: blocked same-cycle flip",
                    )
                    filtered_positions.append(pos)
                    continue

            filtered_positions.append(pos)

        decision.positions = filtered_positions

        # -- Step 6: Convert to target weights + execute --
        target_weights = {}
        leverage_per_symbol = {}
        for pos in decision.positions:
            if pos.action == "LONG":
                target_weights[pos.ticker] = pos.weight
            elif pos.action == "SHORT":
                target_weights[pos.ticker] = -pos.weight
            elif pos.action == "CLOSE":
                target_weights[pos.ticker] = 0.0
            # HOLD → don't change

            if pos.action in ("LONG", "SHORT", "CLOSE"):
                leverage_per_symbol[pos.ticker] = TRADING.get("max_leverage", 3.0)

        # -- Step 6.5: No-trade zone — suppress switches with insufficient edge --
        current_positions_map = {
            p["symbol"]: p for p in self.rebalancer._get_current_positions()
        }
        positions_to_remove = []
        for ticker, weight in list(target_weights.items()):
            if weight == 0.0 or ticker in current_positions_map:
                continue  # CLOSE or already holding — pass through

            # New entry: check if we're closing something to make room
            closes = [
                t for t, w in target_weights.items()
                if w == 0.0 and t in current_positions_map
            ]
            if not closes:
                continue  # Free slot, OK

            new_score = aggregated.scores.get(ticker, 0)
            for closed_ticker in closes:
                old_score = aggregated.scores.get(closed_ticker, 0)
                score_improvement = abs(new_score) - abs(old_score)
                min_edge = max(0.05, abs(old_score) * 0.10)
                if score_improvement < min_edge:
                    logger.info(
                        f"No-trade zone: blocking {ticker} (score {new_score:.3f}) "
                        f"replacing {closed_ticker} (score {old_score:.3f}), "
                        f"improvement {score_improvement:.3f} < {min_edge:.3f}"
                    )
                    positions_to_remove.append(ticker)
                    positions_to_remove.append(closed_ticker)
                    break

        for t in set(positions_to_remove):
            target_weights.pop(t, None)

        executed = []
        if target_weights:
            logger.info(f"Executing: {len(target_weights)} target positions")
            executed = self.rebalancer.rebalance(
                target_weights=target_weights,
                leverage_per_symbol=leverage_per_symbol,
                total_capital=balance,
            )
            logger.info(f"Rebalance complete: {len(executed)} orders")

            # Log each transaction
            for order in executed:
                SonnetDecisionMaker.log_transaction(
                    ticker=order.get("symbol", ""),
                    action=order.get("side", ""),
                    side=order.get("side", ""),
                    quantity=float(order.get("quantity", 0)),
                    price=float(order.get("price", 0)),
                    notional=float(order.get("notional", 0)),
                    status=order.get("status", ""),
                    reason=f"target_weight={order.get('target_weight', 0):.2f}, diff={order.get('diff', 0):+.2f}",
                )

        # -- Step 7: Update position store with SL/TP --
        # Build a map of fill prices from executed orders
        fill_prices = {}
        if target_weights:
            for order in executed:
                symbol = order.get("symbol", "")
                fill_price = float(order.get("price", 0))
                if symbol and fill_price > 0:
                    fill_prices[symbol] = fill_price

        for pos in decision.positions:
            if pos.action in ("LONG", "SHORT"):
                # Check if we already have the same-direction position
                # Sonnet sometimes outputs SHORT instead of HOLD for existing shorts
                existing = self.position_store.positions.get(pos.ticker)
                if existing and existing.side == pos.action:
                    # Already holding same direction — treat as HOLD, don't update
                    logger.info(
                        f"Ignoring duplicate {pos.action} for {pos.ticker} "
                        f"(already holding {existing.side} @ {existing.entry_price:.4f})"
                    )
                    continue

                try:
                    # Only store position if order was actually filled
                    price = fill_prices.get(pos.ticker, 0)
                    if price <= 0:
                        # No fill price = order failed or wasn't executed
                        logger.warning(
                            f"Skipping position store for {pos.ticker}: "
                            f"no fill price (order likely failed)"
                        )
                        continue
                    if price > 0:
                        self.position_store.upsert(
                            ticker=pos.ticker,
                            side=pos.action,
                            entry_price=price,
                            weight=pos.weight,
                            stop_loss_pct=pos.stop_loss_pct,
                            take_profit_pct=pos.take_profit_pct,
                            reasoning=pos.reasoning,
                        )
                        # Track open time for anti-churn
                        if not hasattr(self, "_position_open_times"):
                            self._position_open_times = {}
                        self._position_open_times[pos.ticker] = datetime.now()
                except Exception as e:
                    logger.error(f"Failed to store position {pos.ticker}: {e}")
            elif pos.action == "HOLD":
                # SL/TP are set once at entry and never changed on HOLD.
                # Sonnet returns slightly different SL/TP each cycle, causing
                # dangerous drift (e.g., SL widening from -5% to -8%).
                pass
            elif pos.action == "CLOSE":
                self.position_store.remove(pos.ticker, reason="sonnet_close")
                # Record cooldown timestamp
                if not hasattr(self, "_close_cooldowns"):
                    self._close_cooldowns = {}
                self._close_cooldowns[pos.ticker] = datetime.now()
                # Clean up open time tracking
                if hasattr(self, "_position_open_times"):
                    self._position_open_times.pop(pos.ticker, None)

        logger.info(f"{'='*40} CYCLE END {'='*40}")

    def _inject_hold_for_orphans(
        self, decision: SonnetDecision, managed: list | None
    ) -> SonnetDecision:
        """
        Safety net: if Sonnet omitted an existing managed position,
        auto-inject a HOLD decision so it doesn't become unmanaged.

        This prevents positions from staying open indefinitely when the
        LLM simply forgets to mention them in its response.
        """
        if not managed:
            return decision

        # Tickers that Sonnet already addressed
        decided_tickers = {p.ticker for p in decision.positions}

        orphans = [mp for mp in managed if mp.ticker not in decided_tickers]

        for mp in orphans:
            logger.warning(
                f"Orphan position detected: {mp.ticker} {mp.side} "
                f"not in Sonnet decision — auto-injecting HOLD"
            )
            decision.positions.append(
                PositionDecision(
                    ticker=mp.ticker,
                    action="HOLD",
                    weight=mp.target_weight,
                    stop_loss_pct=mp.stop_loss_pct,
                    take_profit_pct=mp.take_profit_pct,
                    reasoning=f"[AUTO-HOLD] Sonnet omitted this position",
                )
            )

        if orphans:
            self.telegram.send_message(
                f"<b>Orphan Safety Net</b>\n"
                f"Sonnet이 {len(orphans)}개 기존 포지션을 누락하여 "
                f"자동 HOLD 처리함:\n"
                + "\n".join(
                    f"  {mp.ticker} {mp.side}" for mp in orphans
                )
            )

        return decision

    def _sync_position_store(self, current_positions: pd.DataFrame) -> None:
        """
        Remove ghost entries from managed_positions that don't exist on Binance.

        Called every rebalance cycle to prevent managed_positions.json from
        accumulating stale entries that block new position entries.

        IMPORTANT: Only syncs when Binance API returned valid data (non-empty).
        If current_positions is empty, we can't distinguish "no positions" from
        "API call failed" — so we skip the sync to avoid removing real positions.
        """
        if not self.position_store:
            return

        managed = self.position_store.get_active()
        if not managed:
            return

        # Guard: skip sync only if API actually failed (returned None)
        if current_positions is None:
            logger.debug(
                "Ghost sync skipped: Binance API failed. "
                "Keeping managed entries intact."
            )
            return

        # Build set of tickers that actually have positions on Binance
        actual_tickers: set[str] = set()
        for _, row in current_positions.iterrows():
            size = abs(float(row.get("size", 0)))
            if size > 0:
                actual_tickers.add(row.get("ticker", ""))

        # No actual positions on Binance (API succeeded but no open positions)
        if not actual_tickers:
            # Binance responded with position rows (API worked) but all sizes are 0
            removed = []
            for mp in managed:
                self.position_store.remove(mp.ticker, reason="ghost_sync")
                removed.append(mp.ticker)
            if removed:
                logger.warning(
                    f"Ghost position sync: removed {len(removed)} stale entries "
                    f"(Binance confirms no open positions): "
                    f"{', '.join(t.replace('/USDT:USDT', '') for t in removed)}"
                )
            return

        # Normal case: Binance has some positions — remove managed entries not on Binance
        removed = []
        for mp in managed:
            if mp.ticker not in actual_tickers:
                self.position_store.remove(mp.ticker, reason="ghost_sync")
                removed.append(mp.ticker)

        if removed:
            logger.warning(
                f"Ghost position sync: removed {len(removed)} stale entries: "
                f"{', '.join(t.replace('/USDT:USDT', '') for t in removed)}"
            )

    def _build_market_context(self) -> dict:
        """Build market context dict for Sonnet prompt."""
        regime = self._detect_regime()

        try:
            btc_data = self._prices[
                self._prices["ticker"].str.startswith("BTC/")
            ].sort_values("date")
            close = btc_data["close"].values

            ctx = {
                "regime": regime,
                "btc_price": float(close[-1]) if len(close) > 0 else 0,
                "btc_24h_change": float(close[-1] / close[-2] - 1) if len(close) > 1 else 0,
                "btc_7d_change": float(close[-1] / close[-7] - 1) if len(close) > 7 else 0,
                "btc_30d_change": float(close[-1] / close[-30] - 1) if len(close) > 30 else 0,
            }
        except Exception as e:
            logger.warning(f"Market context build failed: {e}")
            ctx = {"regime": regime, "btc_price": 0, "btc_24h_change": 0, "btc_7d_change": 0, "btc_30d_change": 0}

        # Add ETH data and breadth from regime detection
        try:
            eth_data = self._prices[
                self._prices["ticker"].str.startswith("ETH/")
            ].sort_values("date")
            eth_close = eth_data["close"].values

            ctx["eth_price"] = float(eth_close[-1]) if len(eth_close) > 0 else 0
            ctx["eth_24h_change"] = float(eth_close[-1] / eth_close[-2] - 1) if len(eth_close) > 1 else 0
            ctx["eth_7d_change"] = float(eth_close[-1] / eth_close[-7] - 1) if len(eth_close) > 7 else 0
            ctx["eth_30d_change"] = float(eth_close[-1] / eth_close[-30] - 1) if len(eth_close) > 30 else 0
        except Exception as e:
            logger.debug(f"ETH context failed: {e}")
            ctx.update({"eth_price": 0, "eth_24h_change": 0, "eth_7d_change": 0, "eth_30d_change": 0})

        # Breadth data from _detect_regime()
        regime_details = getattr(self, "_regime_details", {})
        ctx["pct_above_ma20"] = regime_details.get("pct_above_ma20", 0.5)
        ctx["adv_decline_ratio"] = regime_details.get("adv_decline_ratio", 1.0)
        ctx["regime_score"] = regime_details.get("regime_score", 0)

        # Add current funding rates for held/candidate tickers
        try:
            funding_rates = {}
            if self._features is not None and "funding_rate" in self._features.columns:
                latest = self._features.sort_values("date").groupby("ticker").last()
                for ticker, row in latest.iterrows():
                    fr = row.get("funding_rate")
                    if fr is not None and not pd.isna(fr):
                        funding_rates[str(ticker)] = float(fr)
            ctx["funding_rates"] = funding_rates
        except Exception as e:
            logger.warning(f"Funding rates context failed: {e}")
            ctx["funding_rates"] = {}

        # Add derivatives sentiment (OI + Long/Short ratio) for top 10 symbols
        try:
            top_symbols = self._universe[:10] if self._universe else []
            if top_symbols:
                ctx["open_interest"] = self.binance_api.get_open_interest_batch(top_symbols)
                ctx["long_short_ratio"] = self.binance_api.get_long_short_ratio_batch(top_symbols)
            else:
                ctx["open_interest"] = {}
                ctx["long_short_ratio"] = {}
        except Exception as e:
            logger.warning(f"Derivatives sentiment fetch failed: {e}")
            ctx["open_interest"] = {}
            ctx["long_short_ratio"] = {}

        return ctx

    def _get_volume_surge_ratio(
        self,
        ticker: str,
        recent_bars: int = 3,
        baseline_bars: int = 36,
    ) -> float | None:
        """
        Compute recent/baseline volume ratio from 5m candles.

        Returns None when data is insufficient.
        """
        if self._prices_5m is None or self._prices_5m.empty:
            return None
        if recent_bars <= 0 or baseline_bars <= 0:
            return None

        time_col = "datetime" if "datetime" in self._prices_5m.columns else "date"
        tkr = (
            self._prices_5m[self._prices_5m["ticker"] == ticker]
            .sort_values(time_col)
            .tail(recent_bars + baseline_bars)
        )
        if len(tkr) < (recent_bars + baseline_bars):
            return None

        vols = pd.to_numeric(tkr["volume"], errors="coerce").dropna()
        if len(vols) < (recent_bars + baseline_bars):
            return None

        recent_mean = float(vols.tail(recent_bars).mean())
        baseline_series = vols.head(len(vols) - recent_bars).tail(baseline_bars)
        if baseline_series.empty:
            return None

        baseline_median = float(baseline_series.median())
        if baseline_median <= 0:
            return None

        return recent_mean / baseline_median

    def _build_features_summary(self) -> dict:
        """Extract per-ticker price/vol/RSI summary from daily OHLCV for Sonnet context."""
        summary = {}
        try:
            if self._prices is None or self._prices.empty:
                return summary

            import numpy as np

            for ticker, grp in self._prices.groupby("ticker"):
                grp = grp.sort_values("date")
                if len(grp) < 21:
                    continue
                closes = grp["close"].values
                price = float(closes[-1])
                ret_1d = float(closes[-1] / closes[-2] - 1) if len(closes) > 1 else 0.0

                # Annualized volatility
                log_rets = np.diff(np.log(closes[-21:]))  # last 20 days
                vol_20d = float(np.std(log_rets) * np.sqrt(252)) if len(log_rets) > 1 else 0.0
                log_rets_5 = np.diff(np.log(closes[-6:]))  # last 5 days
                vol_5d = float(np.std(log_rets_5) * np.sqrt(252)) if len(log_rets_5) > 1 else 0.0

                # RSI 14
                deltas = np.diff(closes[-15:])
                gains = np.where(deltas > 0, deltas, 0)
                losses_arr = np.where(deltas < 0, -deltas, 0)
                avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0
                avg_loss = float(np.mean(losses_arr)) if len(losses_arr) > 0 else 0
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1 + rs))

                summary[str(ticker)] = {
                    "price": price,
                    "ret_1d": ret_1d,
                    "vol_5d": vol_5d,
                    "vol_20d": vol_20d,
                    "rsi_14": rsi,
                }
        except Exception as e:
            logger.warning(f"Features summary build failed: {e}")

        return summary

    def _translate_to_korean(self, text: str) -> str:
        """Translate English text to Korean using Haiku for speed."""
        try:
            resp = self.anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": (
                        "다음 트레이딩 분석을 자연스러운 한국어로 번역해줘. "
                        "전문 용어(RSI, SL, TP, 롱, 숏 등)는 그대로 유지. "
                        "간결하게 번역하고, 원문의 의미를 정확히 전달해.\n\n"
                        f"{text}"
                    ),
                }],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Translation failed, using original: {e}")
            return text

    def _format_sonnet_proposal(self, decision: SonnetDecision, balance: float) -> str:
        """Format Sonnet's decision as a Telegram message (Korean)."""
        action_kr = {
            "LONG": "🟢 롱 진입",
            "SHORT": "🔴 숏 진입",
            "CLOSE": "⚪ 청산",
            "HOLD": "🔵 유지",
        }

        # Translate Sonnet's English fields to Korean
        market_kr = self._translate_to_korean(decision.market_assessment)
        risk_kr = self._translate_to_korean(decision.risk_note) if decision.risk_note else ""

        lines = [
            "<b>📊 AI 매매 결정</b>",
            f"시장: {market_kr}",
            f"잔고: ${balance:.2f}",
            "",
        ]

        for p in decision.positions:
            short_ticker = p.ticker.replace("/USDT:USDT", "")
            action_text = action_kr.get(p.action, p.action)
            reasoning_kr = self._translate_to_korean(p.reasoning)

            lines.append(f"  {action_text} {short_ticker} {p.weight:.0%}")
            if p.action in ("LONG", "SHORT"):
                lines.append(f"    손절: {p.stop_loss_pct:+.1%} | 익절: {p.take_profit_pct:+.1%}")
            lines.append(f"    {reasoning_kr}")

        if risk_kr:
            lines.append(f"\n⚠️ 위험: {risk_kr}")

        return "\n".join(lines)

    def _format_rule_proposal(self, decision, balance: float) -> str:
        """규칙 기반 결정을 텔레그램 메시지로 포맷."""
        action_kr = {
            "LONG": "🟢 롱 진입",
            "SHORT": "🔴 숏 진입",
            "CLOSE": "⚪ 청산",
            "HOLD": "🔵 유지",
        }
        lines = [
            "<b>📊 매매 결정 (규칙 기반)</b>",
            f"잔고: ${balance:.2f}",
            "",
        ]
        for p in decision.positions:
            short_ticker = p.ticker.replace("/USDT:USDT", "")
            action_text = action_kr.get(p.action, p.action)
            lines.append(f"  {action_text} {short_ticker} {p.weight:.0%}")
            if p.action in ("LONG", "SHORT"):
                lines.append(f"    손절: {p.stop_loss_pct:+.1%} | 익절: {p.take_profit_pct:+.1%}")
            lines.append(f"    {p.reasoning}")

        if decision.risk_note:
            lines.append(f"\n⚠️ {decision.risk_note}")

        return "\n".join(lines)

    def _get_telegram_notifier(self):
        """TelegramNotifier 인스턴스 반환 (없으면 None)."""
        if hasattr(self, "_tg_notifier"):
            return self._tg_notifier
        try:
            from src.daemon.telegram_notifier import TelegramNotifier
            keys = self.keys or {}
            tg_cfg = keys.get("telegram", {})
            bot_token = tg_cfg.get("bot_token", "")
            chat_id = str(tg_cfg.get("chat_id", ""))
            if not bot_token or not chat_id:
                # broadcast_chat_ids 사용
                chat_ids = tg_cfg.get("broadcast_chat_ids", [])
                if chat_ids:
                    chat_id = str(chat_ids[0])
            if bot_token and chat_id:
                self._tg_notifier = TelegramNotifier(
                    bot_token=bot_token, chat_id=chat_id
                )
                return self._tg_notifier
        except Exception as e:
            logger.debug(f"TelegramNotifier init failed: {e}")
        self._tg_notifier = None
        return None

    def _get_current_price(self, ticker: str) -> float:
        """티커의 현재 가격 반환."""
        try:
            if self._prices is not None and not self._prices.empty:
                tkr = self._prices[self._prices["ticker"] == ticker].sort_values("date")
                if not tkr.empty:
                    return float(tkr["close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    # ==================================================================
    # Signal Collection
    # ==================================================================

    def _collect_all_signals(
        self, now: datetime
    ) -> tuple[dict[str, pd.DataFrame], list]:
        """
        Collect signals from 3 sources:
          1. openclaw_1 base alphas (7)
          2. Registry live alphas (OpenClaw-generated)
          3. Trained ensemble model (optional)
        """
        alpha_signals: dict[str, pd.DataFrame] = {}
        alpha_entries = []

        # Intraday alpha names and their data sources
        _HOURLY_ALPHAS = {"IntradayRSI", "IntradayVWAP"}
        _FIVE_MIN_ALPHAS = {"IntradayTimeSeriesMomentum"}

        # Source 1: Base alphas
        for name, alpha in self._base_alphas.items():
            try:
                if name in _FIVE_MIN_ALPHAS:
                    if self._prices_5m is None:
                        logger.warning(f"Skipping {name}: 5m data not available")
                        continue
                    result = alpha.generate_signals(now, self._prices_5m, self._features)
                elif name in _HOURLY_ALPHAS:
                    if self._prices_1h is None:
                        logger.warning(f"Skipping {name}: 1h data not available")
                        continue
                    result = alpha.generate_signals(now, self._prices_1h, self._features)
                else:
                    result = alpha.generate_signals(now, self._prices, self._features)
                if result and not result.signals.empty:
                    alpha_signals[name] = result.signals
            except Exception as e:
                logger.error(f"Base alpha {name} signal error: {e}")

        # Source 2: Registry live alphas
        for entry in self.registry.get_active():
            if entry.name in alpha_signals:
                continue  # Don't duplicate base alphas
            try:
                instance = self.openclaw_daemon._load_alpha_instance(entry)
                if instance:
                    result = instance.generate_signals(now, self._prices, self._features)
                    if result and not result.signals.empty:
                        alpha_signals[entry.name] = result.signals
                        alpha_entries.append(entry)
            except Exception as e:
                logger.error(f"Registry alpha {entry.name} signal error: {e}")

        # Source 3: Trained ensemble (as a single "ensemble" alpha)
        if self._ensemble is not None:
            try:
                ensemble_result = self._ensemble.generate_signals(
                    date=now,
                    prices=self._prices,
                    features=self._features,
                )
                if not ensemble_result.signals.empty:
                    alpha_signals["trained_ensemble"] = ensemble_result.signals
            except Exception as e:
                logger.error(f"Ensemble signal error: {e}")

        return alpha_signals, alpha_entries

    def _get_alpha_weights(
        self,
        alpha_signals: dict[str, pd.DataFrame],
        alpha_entries: list,
    ) -> dict[str, float]:
        """
        Get alpha weights: risk-parity → config weights → regime adjustment.

        Priority:
          1. Risk-parity (if performance history exists)
          2. Config-defined weights from STRATEGIES (with regime adjustment)

        Regime adjustment prevents contradictory signals from canceling:
          - Bull: boost momentum, suppress mean reversion
          - Bear: boost mean reversion + carry, suppress momentum
          - Sideways: boost mean reversion + carry, moderate momentum
        """
        from src.openclaw.orchestrator.weight_optimizer import WeightOptimizer

        optimizer = WeightOptimizer()

        # Try risk-parity first (needs performance history)
        alpha_returns = {}
        for name in alpha_signals:
            try:
                returns = self.tracker.get_daily_returns(name)
                if returns is not None and len(returns) > 0:
                    alpha_returns[name] = returns
            except Exception:
                pass

        if len(alpha_returns) >= 2:
            try:
                weights = optimizer.risk_parity(alpha_returns)
                logger.info(f"Risk-parity weights: {weights}")
                return weights
            except Exception as e:
                logger.warning(f"Risk-parity failed: {e}")

        # Fallback: config-defined weights + regime adjustment
        weights = self._config_weights_with_regime(alpha_signals)
        return weights

    @staticmethod
    def _classify_asset(ret_7d: float, ret_20d: float) -> str:
        """Classify a single asset's trend as bull/bear/sideways."""
        if ret_7d > 0.03 and ret_20d > 0.05:
            return "bull"
        elif ret_7d < -0.03 and ret_20d < -0.05:
            return "bear"
        return "sideways"

    def _detect_regime(self) -> str:
        """
        Multi-signal regime detection using BTC + ETH + market breadth.

        Scoring: BTC (±2), ETH (±1), breadth (±1)
        bull >= +3, bear <= -3, else sideways

        Returns: 'bull', 'bear', or 'sideways'
        """
        try:
            if self._prices is None or self._prices.empty:
                return "sideways"

            score = 0
            signal_map = {"bull": 1, "bear": -1, "sideways": 0}
            details = {}

            # --- BTC signal (weight: 2) ---
            btc_data = self._prices[
                self._prices["ticker"].str.startswith("BTC/")
            ].sort_values("date")

            if len(btc_data) >= 10:
                btc_close = btc_data["close"].values
                btc_ret_7d = (btc_close[-1] / btc_close[-7] - 1) if len(btc_close) >= 7 else 0
                btc_ret_20d = (btc_close[-1] / btc_close[-20] - 1) if len(btc_close) >= 20 else 0
                btc_signal = self._classify_asset(btc_ret_7d, btc_ret_20d)
                score += signal_map[btc_signal] * 2
                details["BTC"] = f"{btc_ret_7d:+.1%}/{btc_ret_20d:+.1%}→{btc_signal}"
            else:
                btc_ret_7d = btc_ret_20d = 0

            # --- ETH signal (weight: 1) ---
            eth_data = self._prices[
                self._prices["ticker"].str.startswith("ETH/")
            ].sort_values("date")

            eth_ret_7d = eth_ret_20d = 0
            if len(eth_data) >= 10:
                eth_close = eth_data["close"].values
                eth_ret_7d = (eth_close[-1] / eth_close[-7] - 1) if len(eth_close) >= 7 else 0
                eth_ret_20d = (eth_close[-1] / eth_close[-20] - 1) if len(eth_close) >= 20 else 0
                eth_signal = self._classify_asset(eth_ret_7d, eth_ret_20d)
                score += signal_map[eth_signal]
                details["ETH"] = f"{eth_ret_7d:+.1%}/{eth_ret_20d:+.1%}→{eth_signal}"

            # --- Market breadth (weight: 1) ---
            pct_above_ma20 = 0.5
            adv_decline = 1.0
            try:
                latest_date = self._prices["date"].max()
                latest = self._prices[self._prices["date"] == latest_date].copy()
                if len(latest) > 5:
                    # pct above MA20
                    for ticker, grp in self._prices.groupby("ticker"):
                        grp = grp.sort_values("date")
                        if len(grp) >= 20:
                            ma20 = grp["close"].rolling(20).mean().iloc[-1]
                            last_close = grp["close"].iloc[-1]
                            latest.loc[latest["ticker"] == ticker, "_above_ma20"] = int(last_close > ma20)

                    if "_above_ma20" in latest.columns:
                        pct_above_ma20 = latest["_above_ma20"].mean()

                    # advance/decline ratio (1-day returns)
                    prev_date = self._prices["date"].unique()
                    prev_date = sorted(prev_date)
                    if len(prev_date) >= 2:
                        prev = self._prices[self._prices["date"] == prev_date[-2]]
                        merged = latest[["ticker", "close"]].merge(
                            prev[["ticker", "close"]], on="ticker", suffixes=("", "_prev")
                        )
                        if len(merged) > 0:
                            rets = merged["close"] / merged["close_prev"] - 1
                            n_adv = (rets > 0).sum()
                            n_dec = (rets < 0).sum()
                            adv_decline = n_adv / max(n_dec, 1)

                if pct_above_ma20 > 0.65:
                    score += 1
                elif pct_above_ma20 < 0.35:
                    score -= 1
                details["breadth"] = f"{pct_above_ma20:.0%} above MA20, A/D={adv_decline:.2f}"
            except Exception as e:
                logger.debug(f"Breadth calculation failed: {e}")

            # --- Final regime ---
            if score >= 3:
                regime = "bull"
            elif score <= -3:
                regime = "bear"
            else:
                regime = "sideways"

            # Store for market_context
            self._regime_details = {
                "btc_ret_7d": btc_ret_7d,
                "btc_ret_20d": btc_ret_20d,
                "eth_ret_7d": eth_ret_7d,
                "eth_ret_20d": eth_ret_20d,
                "pct_above_ma20": pct_above_ma20,
                "adv_decline_ratio": adv_decline,
                "regime_score": score,
            }

            detail_str = ", ".join(f"{k}: {v}" for k, v in details.items())
            logger.info(f"Regime: {regime} (score={score:+d}, {detail_str})")
            return regime

        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return "sideways"

    # Map alpha.name (class name) → config key (snake_case)
    _ALPHA_NAME_TO_CONFIG = {
        "CSMomentum": "cs_momentum",
        "TimeSeriesMomentum": "time_series_momentum",
        "TimeSeriesMeanReversion": "time_series_mean_reversion",
        "PriceVolumeDivergence": "pv_divergence",
        "VolumeMomentum": "volume_momentum",
        "LowVolatilityAnomaly": "low_volatility_anomaly",
        "FundingRateCarry": "funding_rate_carry",
        "IntradayRSI": "intraday_rsi",
        "IntradayTimeSeriesMomentum": "intraday_time_series_momentum",
        "IntradayVWAP": "intraday_vwap",
    }

    def _config_weights_with_regime(
        self, alpha_signals: dict[str, pd.DataFrame]
    ) -> dict[str, float]:
        """
        Use STRATEGIES config weights adjusted by current regime.

        Instead of equal 14% for all 7 alphas, uses the config-defined
        weights (e.g. funding_rate_carry=0.25, cs_momentum=0.20) and
        adjusts them based on the market regime to prevent contradictory
        signals from canceling each other out.
        """
        regime = self._detect_regime()
        regime_multipliers = ENSEMBLE.get("regime_preferences", {}).get(regime, {})

        raw_weights = {}
        for name in alpha_signals:
            # Map class name → config key
            config_key = self._ALPHA_NAME_TO_CONFIG.get(name, name)

            # Get base weight from config
            config_w = STRATEGIES.get(config_key, {}).get("weight", 0.10)

            # Apply regime multiplier (also uses config_key)
            multiplier = regime_multipliers.get(config_key, 1.0)
            raw_weights[name] = config_w * multiplier

        # Normalize to sum=1
        total = sum(raw_weights.values())
        if total <= 0:
            n = len(alpha_signals)
            return {name: 1.0 / n for name in alpha_signals}

        weights = {name: w / total for name, w in raw_weights.items()}

        logger.info(
            f"Config+regime weights ({regime}): "
            + ", ".join(f"{k}={v:.1%}" for k, v in sorted(weights.items(), key=lambda x: -x[1]))
        )

        return weights

    def _compute_leverage(
        self,
        alpha_entries: list,
        alpha_weights: dict[str, float],
    ) -> dict[str, float]:
        """Compute per-symbol leverage."""
        try:
            leverages = self.leverage_calc.calculate_per_alpha(
                alpha_entries, self.tracker
            )
            avg = self.leverage_calc.calculate_portfolio_leverage(
                leverages, alpha_weights
            )
        except Exception:
            avg = 1.0

        # Apply same leverage to all symbols
        return {t: avg for t in self._universe}

    def _get_current_weights(self) -> dict[str, float]:
        """Get current position weights from Binance."""
        try:
            positions = self.binance_api.get_positions()
            account = self.binance_api.get_account()
            balance = account.get("total_wallet_balance", 0)
            if balance <= 0 or positions.empty:
                return {}

            weights = {}
            for _, row in positions.iterrows():
                size = float(row.get("size", 0))
                entry_price = float(row.get("entry_price", 0))
                notional = size * entry_price
                side = row.get("side", "long")
                w = notional / balance
                if side == "short":
                    w = -w
                weights[row["ticker"]] = w
            return weights
        except Exception as e:
            logger.error(f"Failed to get current weights: {e}")
            return {}

    # ==================================================================
    # Approval Workflow
    # ==================================================================

    def _wait_for_trade_approval(self, timeout: int = 300) -> bool:
        """Wait for trade_approve / trade_reject callback."""
        deadline = time.time() + timeout

        while time.time() < deadline and self._running:
            events = self.telegram.poll_commands()
            for event in events:
                if event.get("type") == "callback":
                    data = event.get("data", "")
                    if data == "trade_approve":
                        return True
                    elif data == "trade_reject":
                        return False
            time.sleep(2)

        return False  # Timeout

    def _handle_trade_callbacks(self, events: list[dict]) -> None:
        """Handle trade approval callbacks from poll_commands results."""
        # Trade callbacks are handled in _wait_for_trade_approval
        # This catches callbacks that arrive outside the approval window
        pass

    # ==================================================================
    # Extra Telegram Commands
    # ==================================================================

    def _cmd_positions(self, args: str) -> None:
        """Handle /positions — show current Binance positions."""
        try:
            positions = self.binance_api.get_positions()
            if positions.empty:
                self.telegram.send_message("포지션 없음")
                return

            lines = ["<b>현재 포지션</b>\n"]
            for _, row in positions.iterrows():
                ticker = row["ticker"].replace("/USDT:USDT", "")
                side = row["side"].upper()
                size = row["size"]
                pnl = row.get("unrealized_pnl", 0)
                lev = row.get("leverage", 1)
                lines.append(
                    f"  {ticker} {side} {size:.4f} "
                    f"PnL: ${float(pnl):+,.2f} ({lev}x)"
                )

            self.telegram.send_message("\n".join(lines))
        except Exception as e:
            self.telegram.send_message(f"Error: {e}")

    def _cmd_balance(self, args: str) -> None:
        """Handle /balance — show account balance."""
        try:
            account = self.binance_api.get_account()
            total = account.get("total_wallet_balance", 0)
            available = account.get("available_balance", 0)
            upnl = account.get("total_unrealized_pnl", 0)
            self.telegram.send_message(
                f"<b>계좌 잔고</b>\n\n"
                f"총 자산: ${total:,.2f}\n"
                f"가용 잔고: ${available:,.2f}\n"
                f"미실현 PnL: ${upnl:+,.2f}"
            )
        except Exception as e:
            self.telegram.send_message(f"Error: {e}")

    def _cmd_pause(self, args: str) -> None:
        """Handle /pause — pause trading."""
        self._paused = True
        self.telegram.send_message("⏸ 매매 일시 중지됨")
        logger.info("Trading paused by user")

    def _cmd_resume(self, args: str) -> None:
        """Handle /resume — resume trading."""
        self._paused = False
        self.telegram.send_message("▶ 매매 재개")
        logger.info("Trading resumed by user")

    def _cmd_closeall(self, args: str) -> None:
        """Handle /closeall — close all positions."""
        self._paused = True
        self.telegram.send_message(
            "🔴 <b>전량 청산 시작</b>\n매매 자동 중지됨",
        )
        logger.info("Close all positions requested")

        try:
            results = self.rebalancer.close_all_positions()
            closed = [r for r in results if r.get("status") in ("closed", "dry_run_close")]
            errors = [r for r in results if r.get("status") == "error"]

            lines = [f"<b>청산 완료</b>\n"]
            for r in closed:
                lines.append(f"  {r['symbol']} — 청산됨")
            for r in errors:
                lines.append(f"  {r['symbol']} — 실패: {r.get('error', '?')}")

            if not results:
                lines.append("열린 포지션 없음")

            lines.append(f"\n매매 재개: /resume")
            self.telegram.send_message("\n".join(lines))
        except Exception as e:
            logger.error(f"Close all failed: {e}")
            self.telegram.send_message(f"청산 실패: {e}")

    # ==================================================================
    # Timing Helpers
    # ==================================================================

    def _should_rebalance(self, now: datetime) -> bool:
        if self._last_rebalance is None:
            return True  # First rebalance runs immediately
        elapsed = (now - self._last_rebalance).total_seconds()
        return elapsed >= self._rebalance_interval * 60

    def _should_lifecycle(self, now: datetime) -> bool:
        if self._last_lifecycle is None:
            # Delay first lifecycle check by 1 hour to avoid startup congestion
            self._last_lifecycle = now - timedelta(hours=23)
            return False
        return (now - self._last_lifecycle) > timedelta(hours=24)

    def _should_research(self, now: datetime) -> bool:
        if self._last_research is None:
            # Delay first research by 2 hours to avoid startup congestion
            self._last_research = now - timedelta(hours=22)
            return False
        return (now - self._last_research) > timedelta(hours=24)
