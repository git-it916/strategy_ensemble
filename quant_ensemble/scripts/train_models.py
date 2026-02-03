#!/usr/bin/env python
"""
Model Training Script

Train signal models and ensemble.

Usage:
    python train_models.py --config config/backtest.yaml --data-dir data/processed
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common import get_logger
from signals.alpha import MomentumAlpha, MeanReversionAlpha, QualityAlpha
from signals.models.regime_nn import RegimeClassifier, RegimeNN
from signals.models.momentum_nn import MomentumNet
from ensemble import MoEEnsemble, StackingEnsemble, MetaLabeler, WeightedAverageEnsemble

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed data."""
    data_path = Path(data_dir)

    features_df = pd.read_parquet(data_path / "features.parquet")
    labels_df = pd.read_parquet(data_path / "labels.parquet")
    prices_df = pd.read_parquet(data_path / "prices.parquet")

    # Ensure datetime
    for df in [features_df, labels_df, prices_df]:
        df["date"] = pd.to_datetime(df["date"])

    return features_df, labels_df, prices_df


def train_models(
    config_path: str,
    data_dir: str,
    output_dir: str,
    ensemble_type: str = "moe",
) -> dict:
    """
    Train signal models.

    Args:
        config_path: Path to configuration file
        data_dir: Directory with processed data
        output_dir: Output directory for trained models
        ensemble_type: Type of ensemble (moe, stacking, weighted, meta)

    Returns:
        Training summary
    """
    logger.info("=" * 60)
    logger.info("Starting Model Training")
    logger.info(f"Ensemble type: {ensemble_type}")
    logger.info("=" * 60)

    config = load_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    features_df, labels_df, prices_df = load_data(data_dir)

    logger.info(f"Features: {len(features_df)} records, {len(features_df.columns)} columns")
    logger.info(f"Labels: {len(labels_df)} records")

    # Split train/test
    train_config = config.get("training", {})
    train_end_date = pd.Timestamp(train_config.get("train_end_date", "2023-06-30"))

    train_features = features_df[features_df["date"] <= train_end_date]
    train_labels = labels_df[labels_df["date"] <= train_end_date]

    logger.info(f"Training data: {len(train_features)} records through {train_end_date.date()}")

    # Initialize base models
    logger.info("Initializing base models...")

    model_config = config.get("models", {})

    # Rule-based alphas
    momentum_alpha = MomentumAlpha(model_config.get("momentum", {}))
    momentum_alpha.model_name = "momentum_alpha"

    reversion_alpha = MeanReversionAlpha(model_config.get("mean_reversion", {}))
    reversion_alpha.model_name = "mean_reversion_alpha"

    quality_alpha = QualityAlpha(model_config.get("quality", {}))
    quality_alpha.model_name = "quality_alpha"

    base_models = [momentum_alpha, reversion_alpha, quality_alpha]

    # Neural network models (if configured)
    if model_config.get("use_nn", False):
        logger.info("Adding neural network models...")

        nn_config = model_config.get("momentum_nn", {})
        momentum_nn = MomentumNet(nn_config)
        momentum_nn.model_name = "momentum_nn"
        base_models.append(momentum_nn)

    # Train base models
    logger.info("Training base models...")

    for model in base_models:
        logger.info(f"Training {model.model_name}...")
        try:
            result = model.fit(train_features, train_labels, train_config)
            logger.info(f"  {model.model_name} trained: {result}")
        except Exception as e:
            logger.error(f"  Failed to train {model.model_name}: {e}")

    # Initialize regime model for MoE
    regime_model = None
    if ensemble_type == "moe":
        logger.info("Training regime model...")
        regime_config = model_config.get("regime", {})
        regime_model = RegimeClassifier(regime_config)
        regime_model.model_name = "regime_classifier"

        try:
            regime_result = regime_model.fit(train_features, train_labels, train_config)
            logger.info(f"  Regime model trained: {regime_result}")
        except Exception as e:
            logger.error(f"  Failed to train regime model: {e}")
            regime_model = None

    # Create ensemble
    logger.info(f"Creating {ensemble_type} ensemble...")

    ensemble_config = config.get("ensemble", {})

    if ensemble_type == "moe" and regime_model is not None:
        ensemble = MoEEnsemble(
            base_models=base_models,
            regime_model=regime_model,
            config=ensemble_config,
        )
    elif ensemble_type == "stacking":
        ensemble = StackingEnsemble(
            base_models=base_models,
            config=ensemble_config,
        )
    elif ensemble_type == "meta":
        # Meta-labeling with best base model
        best_model = base_models[0]  # Use momentum as primary
        ensemble = MetaLabeler(
            base_model=best_model,
            config=ensemble_config,
        )
    else:
        ensemble = WeightedAverageEnsemble(
            base_models=base_models,
            config=ensemble_config,
        )

    ensemble.model_name = f"{ensemble_type}_ensemble"

    # Train ensemble
    logger.info("Training ensemble...")
    try:
        ensemble_result = ensemble.fit(train_features, train_labels, train_config)
        logger.info(f"Ensemble trained: {ensemble_result}")
    except Exception as e:
        logger.error(f"Failed to train ensemble: {e}")
        ensemble_result = {"error": str(e)}

    # Save models
    logger.info("Saving models...")

    # Save base models
    for model in base_models:
        model_path = output_path / f"{model.model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"  Saved {model.model_name}")

    # Save regime model
    if regime_model is not None:
        regime_path = output_path / "regime_model.pkl"
        with open(regime_path, "wb") as f:
            pickle.dump(regime_model, f)
        logger.info("  Saved regime_model")

    # Save ensemble
    ensemble_path = output_path / "ensemble.pkl"
    with open(ensemble_path, "wb") as f:
        pickle.dump(ensemble, f)
    logger.info("  Saved ensemble")

    # Training summary
    summary = {
        "run_date": datetime.now().isoformat(),
        "ensemble_type": ensemble_type,
        "train_end_date": str(train_end_date.date()),
        "n_train_records": len(train_features),
        "base_models": [m.model_name for m in base_models],
        "ensemble_result": ensemble_result,
        "output_dir": str(output_path),
    }

    summary_path = output_path / "training_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    logger.info("=" * 60)
    logger.info("Model Training Complete")
    logger.info(f"Models saved to: {output_path}")
    logger.info("=" * 60)

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train signal models")

    parser.add_argument(
        "--config",
        type=str,
        default="config/backtest.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with processed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        choices=["moe", "stacking", "weighted", "meta"],
        default="moe",
        help="Ensemble type",
    )

    args = parser.parse_args()

    try:
        summary = train_models(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            ensemble_type=args.ensemble,
        )
        print(f"\nTraining complete. Summary: {summary}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
