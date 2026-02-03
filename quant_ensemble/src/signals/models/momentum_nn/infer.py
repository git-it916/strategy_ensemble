"""
Momentum Neural Network Inference

Inference wrapper for trained momentum models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ....common import get_logger
from ...base import MLModel
from .model import MomentumClassifier, MomentumNN

logger = get_logger(__name__)


class MomentumNNModel(MLModel):
    """
    Inference wrapper for momentum neural network.

    Implements the SignalModel interface for use in the ensemble system.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize momentum NN model.

        Args:
            config: Model configuration
        """
        super().__init__(config)

        self.model_type = self.config.get("model_type", "regression")
        self.hidden_dims = self.config.get("hidden_dims", [64, 32])
        self.dropout = self.config.get("dropout", 0.3)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._feature_cols = []
        self._norm_params = {}

    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Train the model.

        Args:
            features_df: Training features
            labels_df: Training labels
            config: Training configuration

        Returns:
            Training artifact
        """
        from .train import MomentumNNTrainer

        config = config or {}

        trainer = MomentumNNTrainer(
            model_type=self.model_type,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            learning_rate=config.get("learning_rate", 1e-3),
            batch_size=config.get("batch_size", 256),
            n_epochs=config.get("n_epochs", 100),
            patience=config.get("patience", 10),
            device=self.device,
        )

        artifact = trainer.train(features_df, labels_df)

        # Store model and parameters
        self._model = trainer.model
        self._feature_cols = artifact["feature_cols"]
        self._norm_params = artifact["normalization_params"]
        self._artifact = artifact
        self.is_fitted = True

        return artifact

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions for a given date.

        Args:
            date: Prediction date
            features_df: Features for all assets

        Returns:
            Scores DataFrame
        """
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Filter to available feature columns
        available_cols = [c for c in self._feature_cols if c in features_df.columns]
        if not available_cols:
            logger.warning("No feature columns found for prediction")
            return pd.DataFrame({
                "date": [date] * len(features_df),
                "asset_id": features_df["asset_id"].values,
                "score": [0.5] * len(features_df),
                "confidence": [0.0] * len(features_df),
                "model_name": [self.model_name] * len(features_df),
            })

        # Prepare features
        X = features_df[available_cols].values
        X = np.nan_to_num(X, nan=0.0)

        # Normalize
        X = (X - self._norm_params["mean"][:len(available_cols)]) / \
            self._norm_params["std"][:len(available_cols)]

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Predict
        self._model.eval()
        with torch.no_grad():
            if self.model_type == "classification":
                scores = self._model.predict_score(X_tensor).cpu().numpy().flatten()
            else:
                scores = self._model(X_tensor).cpu().numpy().flatten()

        # Convert to percentile ranks
        scores = pd.Series(scores).rank(pct=True, na_option="keep").values

        # Calculate confidence
        confidence = len(available_cols) / len(self._feature_cols)

        return pd.DataFrame({
            "date": date,
            "asset_id": features_df["asset_id"].values,
            "score": scores,
            "confidence": confidence,
            "model_name": self.model_name,
        })

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model_name": self.model_name,
            "config": self.config,
            "model_type": self.model_type,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "feature_cols": self._feature_cols,
            "norm_params": self._norm_params,
            "model_state": {k: v.cpu() for k, v in self._model.state_dict().items()}
            if self._model else None,
            "is_fitted": self.is_fitted,
        }

        torch.save(save_dict, path)
        logger.info(f"Saved model to {path}")

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_dict = torch.load(path, map_location=self.device)

        self.model_name = save_dict["model_name"]
        self.config = save_dict["config"]
        self.model_type = save_dict["model_type"]
        self.hidden_dims = save_dict["hidden_dims"]
        self.dropout = save_dict["dropout"]
        self._feature_cols = save_dict["feature_cols"]
        self._norm_params = save_dict["norm_params"]
        self.is_fitted = save_dict["is_fitted"]

        if save_dict["model_state"]:
            input_dim = len(self._feature_cols)

            if self.model_type == "classification":
                self._model = MomentumClassifier(
                    input_dim=input_dim,
                    hidden_dims=self.hidden_dims,
                    dropout=self.dropout,
                )
            else:
                self._model = MomentumNN(
                    input_dim=input_dim,
                    hidden_dims=self.hidden_dims,
                    dropout=self.dropout,
                )

            self._model.load_state_dict(save_dict["model_state"])
            self._model = self._model.to(self.device)

        logger.info(f"Loaded model from {path}")
