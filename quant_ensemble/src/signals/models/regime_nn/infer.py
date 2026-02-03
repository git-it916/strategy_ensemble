"""
Regime Neural Network Inference

Inference wrapper for trained regime models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ....common import RegimeProbability, get_logger
from ...base import MLModel
from .model import RegimeNN

logger = get_logger(__name__)


class RegimeNNModel(MLModel):
    """
    Inference wrapper for regime neural network.

    Outputs regime probabilities for use in MoE ensemble.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.n_regimes = self.config.get("n_regimes", 2)
        self.hidden_dims = self.config.get("hidden_dims", [32, 16])
        self.dropout = self.config.get("dropout", 0.3)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.smoothing_alpha = self.config.get("smoothing_alpha", 0.2)

        self._model = None
        self._feature_cols = []
        self._norm_params = {}
        self._prev_probs = None

    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train the regime model."""
        from .train import RegimeNNTrainer

        config = config or {}
        trainer = RegimeNNTrainer(
            n_regimes=self.n_regimes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            device=self.device,
        )

        artifact = trainer.train(features_df, labels_df)

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
        Predict regime probabilities.

        Note: This returns regime probabilities, not asset scores.
        For use in MoE, call predict_regime() instead.

        Args:
            date: Prediction date
            features_df: Features (should contain market-wide features)

        Returns:
            DataFrame with regime probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        probs = self.predict_regime(date, features_df)

        # Return as DataFrame for consistency with SignalModel interface
        return pd.DataFrame({
            "date": [date],
            "regime_0_prob": [probs.probabilities.get("regime_0", 0.5)],
            "regime_1_prob": [probs.probabilities.get("regime_1", 0.5)],
            "dominant_regime": [probs.dominant_regime],
            "model_name": [self.model_name],
        })

    def predict_regime(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> RegimeProbability:
        """
        Predict regime probabilities for a given date.

        Args:
            date: Prediction date
            features_df: Features DataFrame

        Returns:
            RegimeProbability object
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Prepare features (aggregate to single observation)
        available_cols = [c for c in self._feature_cols if c in features_df.columns]

        if not available_cols:
            # Return uniform probabilities
            probs = {f"regime_{i}": 1.0 / self.n_regimes for i in range(self.n_regimes)}
            return RegimeProbability(date=date, probabilities=probs)

        # Aggregate features
        X = features_df[available_cols].mean().values.reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)
        X = (X - self._norm_params["mean"][:len(available_cols)]) / \
            self._norm_params["std"][:len(available_cols)]

        # Predict
        X_tensor = torch.FloatTensor(X).to(self.device)

        self._model.eval()
        with torch.no_grad():
            probs = self._model.predict_proba(X_tensor, calibrated=True)
            probs = probs.cpu().numpy().flatten()

        # Apply smoothing
        if self._prev_probs is not None:
            probs = self.smoothing_alpha * probs + (1 - self.smoothing_alpha) * self._prev_probs
            probs = probs / probs.sum()  # Renormalize

        self._prev_probs = probs.copy()

        # Create probability dict
        prob_dict = {f"regime_{i}": float(probs[i]) for i in range(self.n_regimes)}

        return RegimeProbability(date=date, probabilities=prob_dict)

    def reset_state(self) -> None:
        """Reset internal state (smoothing)."""
        self._prev_probs = None

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_name": self.model_name,
            "config": self.config,
            "n_regimes": self.n_regimes,
            "hidden_dims": self.hidden_dims,
            "feature_cols": self._feature_cols,
            "norm_params": self._norm_params,
            "model_state": {k: v.cpu() for k, v in self._model.state_dict().items()}
            if self._model else None,
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        save_dict = torch.load(path, map_location=self.device)

        self.model_name = save_dict["model_name"]
        self.config = save_dict["config"]
        self.n_regimes = save_dict["n_regimes"]
        self.hidden_dims = save_dict["hidden_dims"]
        self._feature_cols = save_dict["feature_cols"]
        self._norm_params = save_dict["norm_params"]
        self.is_fitted = save_dict["is_fitted"]

        if save_dict["model_state"]:
            input_dim = len(self._feature_cols)
            self._model = RegimeNN(
                input_dim=input_dim,
                n_regimes=self.n_regimes,
                hidden_dims=self.hidden_dims,
            )
            self._model.load_state_dict(save_dict["model_state"])
            self._model = self._model.to(self.device)
