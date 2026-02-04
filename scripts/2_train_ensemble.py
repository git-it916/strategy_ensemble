#!/usr/bin/env python
"""
2. Train Ensemble Script

앙상블 모델 학습 및 전략 가중치 최적화.

Usage:
    python scripts/2_train_ensemble.py
    python scripts/2_train_ensemble.py --strategies rsi_reversal,vol_breakout
"""

from __future__ import annotations

import argparse
import pickle
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
)
from config.logging_config import setup_logging

logger = setup_logging("train_ensemble")


def load_data():
    """Load processed data."""
    prices_path = PROCESSED_DATA_DIR / "prices.parquet"
    features_path = PROCESSED_DATA_DIR / "features.parquet"
    labels_path = PROCESSED_DATA_DIR / "labels.parquet"

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

    return data


def initialize_strategies(strategy_names: list[str] | None = None):
    """Initialize alpha strategies."""
    from src.alphas.technical import RSIReversalAlpha, VolatilityBreakoutAlpha
    from src.alphas.fundamental import ValueFScoreAlpha, SentimentLongAlpha

    strategies = []

    strategy_map = {
        "rsi_reversal": lambda: RSIReversalAlpha(**STRATEGIES.get("rsi_reversal", {})),
        "vol_breakout": lambda: VolatilityBreakoutAlpha(**STRATEGIES.get("vol_breakout", {})),
        "value_f_score": lambda: ValueFScoreAlpha(**STRATEGIES.get("value_f_score", {})),
        "sentiment_long": lambda: SentimentLongAlpha(**STRATEGIES.get("sentiment_long", {})),
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

    return strategies


def train_ensemble(
    strategies: list,
    data: dict,
    train_end_date: str,
) -> dict:
    """Train ensemble model."""
    from src.ensemble import EnsembleAgent

    logger.info(f"Training ensemble with {len(strategies)} strategies")

    # Filter training data
    train_end = pd.Timestamp(train_end_date)

    train_prices = data["prices"][data["prices"]["date"] <= train_end]
    train_features = data["features"][data["features"]["date"] <= train_end] if "features" in data else None
    train_labels = data["labels"][data["labels"]["date"] <= train_end] if "labels" in data else None

    logger.info(f"Training data: {len(train_prices)} price records through {train_end_date}")

    # Create ensemble
    ensemble = EnsembleAgent(
        strategies=strategies,
        config=ENSEMBLE,
    )

    # Fit all strategies
    fit_results = ensemble.fit(
        prices=train_prices,
        features=train_features,
        labels=train_labels,
    )

    logger.info(f"Fit results: {fit_results}")

    return {
        "ensemble": ensemble,
        "fit_results": fit_results,
        "train_end_date": train_end_date,
    }


def save_models(ensemble, fit_results: dict):
    """Save trained models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_dir = MODELS_DIR / "weights"
    weights_dir.mkdir(exist_ok=True)

    # Save ensemble
    ensemble_path = weights_dir / "ensemble.pkl"
    with open(ensemble_path, "wb") as f:
        pickle.dump(ensemble, f)
    logger.info(f"Saved ensemble to {ensemble_path}")

    # Save individual strategies
    for name, strategy in ensemble.strategies.items():
        strategy_path = weights_dir / f"{name}.pkl"
        with open(strategy_path, "wb") as f:
            pickle.dump(strategy, f)
        logger.info(f"Saved {name}")

    # Save training metadata
    import yaml
    meta = {
        "train_date": datetime.now().isoformat(),
        "strategies": list(ensemble.strategies.keys()),
        "weights": ensemble.get_weights(),
        "fit_results": {k: str(v) for k, v in fit_results.items()},
    }

    meta_path = MODELS_DIR / "training_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(meta, f)


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
    save_models(result["ensemble"], result["fit_results"])

    # Print summary
    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info(f"Strategies: {list(result['ensemble'].strategies.keys())}")
    logger.info(f"Weights: {result['ensemble'].get_weights()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
