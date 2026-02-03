"""
Value Neural Network Training

Training logic with checkpointing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ....common import get_logger, set_seed
from .model import ValueClassifier, ValueNN

logger = get_logger(__name__)


class ValueNNTrainer:
    """Trainer for value neural network models."""

    def __init__(
        self,
        model_type: str = "regression",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        n_epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        seed: int = 42,
    ):
        self.model_type = model_type
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.feature_cols = []
        self.normalization_params = {}

    def train(
        self,
        train_features: pd.DataFrame,
        train_labels: pd.DataFrame,
        val_features: pd.DataFrame | None = None,
        val_labels: pd.DataFrame | None = None,
        feature_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Train the model."""
        set_seed(self.seed)

        # Identify value-related feature columns
        if feature_cols is None:
            exclude = {"date", "asset_id"}
            feature_cols = [
                c for c in train_features.columns
                if c not in exclude and any(
                    v in c.lower() for v in ["pbr", "per", "roe", "value", "debt", "dividend"]
                )
            ]
            # Fallback to all numeric columns if no value features found
            if not feature_cols:
                feature_cols = [
                    c for c in train_features.columns
                    if c not in exclude
                ]

        self.feature_cols = feature_cols

        # Prepare data
        X_train, y_train = self._prepare_data(train_features, train_labels)

        if val_features is not None and val_labels is not None:
            X_val, y_val = self._prepare_data(val_features, val_labels)
        else:
            n_val = int(len(X_train) * 0.1)
            X_val, y_val = X_train[-n_val:], y_train[-n_val:]
            X_train, y_train = X_train[:-n_val], y_train[:-n_val]

        # Initialize model
        input_dim = X_train.shape[1]

        if self.model_type == "classification":
            self.model = ValueClassifier(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            )
            criterion = nn.CrossEntropyLoss()
        else:
            self.model = ValueNN(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            )
            criterion = nn.MSELoss()

        self.model = self.model.to(self.device)

        # Data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train) if self.model_type != "classification"
            else torch.LongTensor(y_train.astype(int)),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val) if self.model_type != "classification"
            else torch.LongTensor(y_val.astype(int)),
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.n_epochs):
            # Train
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)

                if self.model_type == "classification":
                    loss = criterion(outputs, y_batch)
                else:
                    loss = criterion(outputs.squeeze(), y_batch.squeeze())

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)

                    if self.model_type == "classification":
                        loss = criterion(outputs, y_batch)
                    else:
                        loss = criterion(outputs.squeeze(), y_batch.squeeze())
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_state:
            self.model.load_state_dict(best_state)

        return {
            "model_state": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "model_type": self.model_type,
            "input_dim": input_dim,
            "hidden_dims": self.hidden_dims,
            "feature_cols": self.feature_cols,
            "normalization_params": self.normalization_params,
            "history": history,
        }

    def _prepare_data(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        df = features.merge(
            labels[["date", "asset_id", "y_reg", "y_cls"]],
            on=["date", "asset_id"],
            how="inner",
        )

        X = df[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0)

        if not self.normalization_params:
            self.normalization_params = {
                "mean": np.nanmean(X, axis=0),
                "std": np.nanstd(X, axis=0) + 1e-8,
            }

        X = (X - self.normalization_params["mean"]) / self.normalization_params["std"]

        if self.model_type == "classification":
            y = df["y_cls"].values
        else:
            y = df["y_reg"].values

        return X, y
