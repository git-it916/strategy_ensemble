"""
Value Neural Network Inference

Inference wrapper for trained value models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ....common import get_logger
from ...base import MLModel
from .model import ValueClassifier, ValueNN

logger = get_logger(__name__)


class ValueNNModel(MLModel):
    """Inference wrapper for value neural network."""

    def __init__(self, config: dict[str, Any] | None = None):
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
        from .train import ValueNNTrainer

        config = config or {}
        trainer = ValueNNTrainer(
            model_type=self.model_type,
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
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        available_cols = [c for c in self._feature_cols if c in features_df.columns]
        if not available_cols:
            return pd.DataFrame({
                "date": [date] * len(features_df),
                "asset_id": features_df["asset_id"].values,
                "score": [0.5] * len(features_df),
                "confidence": [0.0] * len(features_df),
                "model_name": [self.model_name] * len(features_df),
            })

        X = features_df[available_cols].values
        X = np.nan_to_num(X, nan=0.0)
        X = (X - self._norm_params["mean"][:len(available_cols)]) / \
            self._norm_params["std"][:len(available_cols)]

        X_tensor = torch.FloatTensor(X).to(self.device)

        self._model.eval()
        with torch.no_grad():
            if self.model_type == "classification":
                scores = self._model.predict_score(X_tensor).cpu().numpy().flatten()
            else:
                scores = self._model(X_tensor).cpu().numpy().flatten()

        scores = pd.Series(scores).rank(pct=True, na_option="keep").values
        confidence = len(available_cols) / len(self._feature_cols)

        return pd.DataFrame({
            "date": date,
            "asset_id": features_df["asset_id"].values,
            "score": scores,
            "confidence": confidence,
            "model_name": self.model_name,
        })

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_name": self.model_name,
            "config": self.config,
            "model_type": self.model_type,
            "hidden_dims": self.hidden_dims,
            "feature_cols": self._feature_cols,
            "norm_params": self._norm_params,
            "model_state": {k: v.cpu() for k, v in self._model.state_dict().items()}
            if self._model else None,
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str | Path) -> None:
        save_dict = torch.load(path, map_location=self.device)

        self.model_name = save_dict["model_name"]
        self.config = save_dict["config"]
        self.model_type = save_dict["model_type"]
        self.hidden_dims = save_dict["hidden_dims"]
        self._feature_cols = save_dict["feature_cols"]
        self._norm_params = save_dict["norm_params"]
        self.is_fitted = save_dict["is_fitted"]

        if save_dict["model_state"]:
            input_dim = len(self._feature_cols)
            if self.model_type == "classification":
                self._model = ValueClassifier(input_dim=input_dim, hidden_dims=self.hidden_dims)
            else:
                self._model = ValueNN(input_dim=input_dim, hidden_dims=self.hidden_dims)
            self._model.load_state_dict(save_dict["model_state"])
            self._model = self._model.to(self.device)
