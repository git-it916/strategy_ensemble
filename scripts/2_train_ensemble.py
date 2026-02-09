#!/usr/bin/env python
"""
2. Train Ensemble Script

앙상블 모델 학습 및 전략 가중치 최적화.
ModelManager를 통해 구조화된 모델 저장 (joblib + registry.yaml).

Supports both rule-based and ML-based alpha strategies.
ML alphas require features.parquet and labels.parquet in data/processed/.

Usage:
    python scripts/2_train_ensemble.py
    python scripts/2_train_ensemble.py --strategies rsi_reversal,vol_breakout
    python scripts/2_train_ensemble.py --strategies return_prediction,intraday_pattern
    python scripts/2_train_ensemble.py --build-features
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    STRATEGIES,
    ENSEMBLE,
    BACKTEST,
    REGIME_CLASSIFIER,
)
from config.logging_config import setup_logging

logger = setup_logging("train_ensemble")


def load_data():
    """Load processed data."""
    prices_path = PROCESSED_DATA_DIR / "prices.parquet"
    features_path = PROCESSED_DATA_DIR / "features.parquet"
    labels_path = PROCESSED_DATA_DIR / "labels.parquet"

    # ML-specific label files
    labels_return_path = PROCESSED_DATA_DIR / "labels_return.parquet"
    labels_vol_path = PROCESSED_DATA_DIR / "labels_volatility.parquet"
    labels_regime_path = PROCESSED_DATA_DIR / "labels_regime.parquet"
    labels_intraday_path = PROCESSED_DATA_DIR / "labels_intraday.parquet"
    market_features_path = PROCESSED_DATA_DIR / "market_features.parquet"

    data = {}

    if prices_path.exists():
        data["prices"] = pd.read_parquet(prices_path)
        data["prices"]["date"] = pd.to_datetime(data["prices"]["date"])

    if features_path.exists():
        data["features"] = pd.read_parquet(features_path)
        data["features"]["date"] = pd.to_datetime(data["features"]["date"])

    if labels_path.exists():
        data["labels"] = pd.read_parquet(labels_path)
        data["labels"]["date"] = pd.to_datetime(data["labels"]["date"])

    # Load ML-specific labels (fallback to generic labels if not found)
    for name, path in [
        ("labels_return", labels_return_path),
        ("labels_volatility", labels_vol_path),
        ("labels_regime", labels_regime_path),
        ("labels_intraday", labels_intraday_path),
        ("market_features", market_features_path),
    ]:
        if path.exists():
            data[name] = pd.read_parquet(path)
            if "date" in data[name].columns:
                data[name]["date"] = pd.to_datetime(data[name]["date"])

    return data


def _get_strategy_config(name: str) -> dict:
    """Get strategy config, stripping keys that aren't constructor params."""
    config = STRATEGIES.get(name, {}).copy()
    config.pop("enabled", None)
    config.pop("weight", None)
    config.pop("type", None)
    return config


def initialize_strategies(strategy_names: list[str] | None = None):
    """Initialize alpha strategies (rule-based + ML)."""
    from src.alphas.technical import RSIReversalAlpha, VolatilityBreakoutAlpha
    from src.alphas.fundamental import ValueFScoreAlpha, SentimentLongAlpha
    from src.alphas.ml import (
        ReturnPredictionAlpha,
        IntradayPatternAlpha,
        VolatilityForecastAlpha,
    )

    strategies = []

    strategy_map = {
        # Rule-based
        "rsi_reversal": lambda: RSIReversalAlpha(**_get_strategy_config("rsi_reversal")),
        "vol_breakout": lambda: VolatilityBreakoutAlpha(**_get_strategy_config("vol_breakout")),
        "value_f_score": lambda: ValueFScoreAlpha(**_get_strategy_config("value_f_score")),
        "sentiment_long": lambda: SentimentLongAlpha(**_get_strategy_config("sentiment_long")),
        # ML-based
        "return_prediction": lambda: ReturnPredictionAlpha(
            config=_get_strategy_config("return_prediction"),
        ),
        "intraday_pattern": lambda: IntradayPatternAlpha(
            config=_get_strategy_config("intraday_pattern"),
        ),
        "volatility_forecast": lambda: VolatilityForecastAlpha(
            config=_get_strategy_config("volatility_forecast"),
        ),
    }

    if strategy_names is None:
        strategy_names = [
            name for name, config in STRATEGIES.items()
            if config.get("enabled", True)
        ]

    for name in strategy_names:
        if name in strategy_map:
            strategies.append(strategy_map[name]())
            logger.info(f"Initialized strategy: {name}")
        else:
            logger.warning(f"Unknown strategy: {name}")

    return strategies


def prepare_ml_labels(data: dict, train_end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """
    Prepare label DataFrames for ML alphas.

    ML alphas need specific label types. This resolves which label to use:
    1. If dedicated label file exists (labels_return.parquet), use it
    2. Otherwise fall back to generic labels.parquet

    Returns:
        Dict mapping label_type -> filtered DataFrame
    """
    ml_labels = {}

    for key in ["labels_return", "labels_volatility", "labels_intraday"]:
        if key in data:
            df = data[key]
            ml_labels[key] = df[df["date"] <= train_end]
        elif "labels" in data:
            # Fallback: generic labels (y_reg) used for all
            ml_labels[key] = data["labels"][data["labels"]["date"] <= train_end]

    if "labels_regime" in data:
        ml_labels["labels_regime"] = data["labels_regime"][
            data["labels_regime"]["date"] <= train_end
        ]

    return ml_labels


def train_regime_classifier(data: dict, train_end: pd.Timestamp):
    """Train the regime classifier separately (not a per-stock alpha)."""
    from src.alphas.ml import RegimeClassifier

    if not REGIME_CLASSIFIER.get("enabled", False):
        logger.info("Regime classifier disabled, skipping")
        return None

    if "market_features" not in data:
        logger.warning("No market_features found, skipping regime classifier")
        return None

    if "labels_regime" not in data:
        logger.warning("No regime labels found, skipping regime classifier")
        return None

    market_feat = data["market_features"][data["market_features"]["date"] <= train_end]
    regime_labels = data["labels_regime"][data["labels_regime"]["date"] <= train_end]

    classifier = RegimeClassifier(config=REGIME_CLASSIFIER)
    result = classifier.fit(market_feat, regime_labels)
    logger.info(f"Regime classifier: {result}")

    # Save
    regime_path = MODELS_DIR / "weights" / "regime_classifier.joblib"
    regime_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(regime_path)
    logger.info(f"Regime classifier saved -> {regime_path}")

    return classifier


def train_ensemble(
    strategies: list,
    data: dict,
    train_end_date: str,
) -> dict:
    """Train ensemble model (rule-based + ML strategies)."""
    from src.ensemble import EnsembleAgent

    logger.info(f"Training ensemble with {len(strategies)} strategies")

    # Filter training data
    train_end = pd.Timestamp(train_end_date)

    train_prices = data["prices"][data["prices"]["date"] <= train_end]
    train_features = data["features"][data["features"]["date"] <= train_end] if "features" in data else None

    # For ML alphas: resolve the correct label set per strategy type
    ml_labels = prepare_ml_labels(data, train_end)

    # Default labels for rule-based (and ML fallback)
    train_labels = data["labels"][data["labels"]["date"] <= train_end] if "labels" in data else None

    # ML alphas that need specific labels — fit them individually first
    from src.alphas.ml import BaseMLAlpha
    ml_strategies = [s for s in strategies if isinstance(s, BaseMLAlpha)]
    rule_strategies = [s for s in strategies if not isinstance(s, BaseMLAlpha)]

    fit_results = {}

    # Fit ML strategies with their specific labels
    label_map = {
        "return_prediction": "labels_return",
        "intraday_pattern": "labels_intraday",
        "volatility_forecast": "labels_volatility",
    }

    for strategy in ml_strategies:
        label_key = label_map.get(strategy.name, "labels_return")
        labels = ml_labels.get(label_key, train_labels)

        logger.info(f"Fitting ML strategy: {strategy.name} (labels: {label_key})")
        try:
            result = strategy.fit(train_prices, train_features, labels)
            fit_results[strategy.name] = {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Failed to fit {strategy.name}: {e}")
            fit_results[strategy.name] = {"status": "error", "error": str(e)}

    logger.info(f"Training data: {len(train_prices)} price records through {train_end_date}")

    # Create ensemble with all strategies
    ensemble = EnsembleAgent(
        strategies=strategies,
        config=ENSEMBLE,
    )

    # Fit rule-based strategies via ensemble (ML ones already fitted above)
    if rule_strategies:
        rule_fit = ensemble.fit(
            prices=train_prices,
            features=train_features,
            labels=train_labels,
        )
        fit_results.update(rule_fit)

    logger.info(f"Fit results: {list(fit_results.keys())}")

    # Train regime classifier
    regime_classifier = train_regime_classifier(data, train_end)

    return {
        "ensemble": ensemble,
        "fit_results": fit_results,
        "train_end_date": train_end_date,
        "regime_classifier": regime_classifier,
    }


def save_models(ensemble, fit_results: dict, data: dict, train_end_date: str):
    """Save trained models via ModelManager."""
    from src.ensemble_models import ModelManager

    manager = ModelManager(MODELS_DIR)

    # Build data metadata for lineage tracking
    data_meta = {
        "train_end": train_end_date,
        "n_price_rows": len(data.get("prices", [])),
        "n_feature_rows": len(data.get("features", [])),
        "n_label_rows": len(data.get("labels", [])),
        "data_hash": ModelManager.compute_data_hash(data["prices"]) if "prices" in data else None,
    }

    registry_path = manager.save_ensemble(ensemble, fit_results, data_meta)
    logger.info(f"Models saved. Registry: {registry_path}")
    logger.info(f"Model version: {manager.get_version()}")


def build_features_and_labels(data: dict):
    """
    Build features and labels from raw price data, then save to processed dir.

    Call with --build-features flag to regenerate before training.
    """
    from src.etl.feature_engineer import FeatureEngineer
    from src.etl.label_engineer import LabelEngineer

    prices = data["prices"]
    logger.info(f"Building features from {len(prices)} price rows...")

    # --- Features ---
    fe = FeatureEngineer()
    feature_sets = fe.build_all(prices)

    # Save combined (daily + intraday) features
    combined = feature_sets["combined"]
    combined.to_parquet(PROCESSED_DATA_DIR / "features.parquet", index=False)
    logger.info(f"Saved features: {combined.shape}")

    # Save market features (for regime classifier)
    market = feature_sets["market"]
    market.to_parquet(PROCESSED_DATA_DIR / "market_features.parquet", index=False)
    logger.info(f"Saved market features: {market.shape}")

    # Reload into data dict
    data["features"] = combined
    data["market_features"] = market

    # --- Labels ---
    le = LabelEngineer()
    all_labels = le.build_all(prices)

    for label_type, label_df in all_labels.items():
        path = PROCESSED_DATA_DIR / f"labels_{label_type}.parquet"
        label_df.to_parquet(path, index=False)
        data[f"labels_{label_type}"] = label_df
        logger.info(f"Saved labels_{label_type}: {len(label_df)} rows")

    # Also save default labels (return) as the generic labels.parquet
    all_labels["return"].to_parquet(
        PROCESSED_DATA_DIR / "labels.parquet", index=False
    )
    data["labels"] = all_labels["return"]

    logger.info("Feature and label engineering complete")
    return data


def main():
    parser = argparse.ArgumentParser(description="Train ensemble model")

    parser.add_argument(
        "--strategies",
        type=str,
        help="Comma-separated list of strategies to use",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2023-12-31",
        help="Training end date",
    )
    parser.add_argument(
        "--build-features",
        action="store_true",
        help="Rebuild features and labels from prices before training",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Ensemble Training Started")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    data = load_data()

    if "prices" not in data:
        logger.error("No price data found. Run 1_update_data.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(data['prices'])} price records")

    # Build features if requested
    if args.build_features:
        data = build_features_and_labels(data)

    # Initialize strategies
    strategy_names = args.strategies.split(",") if args.strategies else None
    strategies = initialize_strategies(strategy_names)

    if not strategies:
        logger.error("No strategies initialized")
        sys.exit(1)

    # Train ensemble
    result = train_ensemble(
        strategies=strategies,
        data=data,
        train_end_date=args.train_end,
    )

    # Save models
    save_models(result["ensemble"], result["fit_results"], data, args.train_end)

    # Print summary
    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info(f"Strategies: {list(result['ensemble'].strategies.keys())}")
    logger.info(f"Weights: {result['ensemble'].get_weights()}")
    if result.get("regime_classifier"):
        logger.info("Regime classifier: trained and saved")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
