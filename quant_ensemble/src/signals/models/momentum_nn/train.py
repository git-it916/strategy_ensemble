"""
Momentum Neural Network Training

Training logic with checkpointing and walk-forward validation.
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
from tqdm import tqdm

from ....common import get_logger, save_pickle, set_seed
from .model import MomentumClassifier, MomentumNN

logger = get_logger(__name__)


class MomentumNNTrainer:
    """
    Trainer for momentum neural network models.

    Supports:
    - Regression training (MSE loss)
    - Ranking training (margin loss)
    - Classification training (cross-entropy loss)
    """

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
        """
        Initialize trainer.

        Args:
            model_type: 'regression', 'ranking', or 'classification'
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Learning rate
            batch_size: Batch size
            n_epochs: Maximum epochs
            patience: Early stopping patience
            device: Device to use ('cuda' or 'cpu')
            seed: Random seed
        """
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
        """
        Train the model.

        Args:
            train_features: Training features
            train_labels: Training labels
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)
            feature_cols: Feature columns to use

        Returns:
            Training artifact
        """
        set_seed(self.seed)

        # Identify feature columns
        if feature_cols is None:
            exclude = {"date", "asset_id"}
            feature_cols = [c for c in train_features.columns if c not in exclude]
        self.feature_cols = feature_cols

        # Prepare data
        X_train, y_train = self._prepare_data(train_features, train_labels)

        if val_features is not None and val_labels is not None:
            X_val, y_val = self._prepare_data(val_features, val_labels)
        else:
            # Split train data for validation
            n_val = int(len(X_train) * 0.1)
            X_val, y_val = X_train[-n_val:], y_train[-n_val:]
            X_train, y_train = X_train[:-n_val], y_train[:-n_val]

        # Initialize model
        input_dim = X_train.shape[1]

        if self.model_type == "classification":
            n_classes = 5  # Quintiles
            self.model = MomentumClassifier(
                input_dim=input_dim,
                n_classes=n_classes,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            )
            criterion = nn.CrossEntropyLoss()
        else:
            self.model = MomentumNN(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            )
            criterion = nn.MSELoss()

        self.model = self.model.to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train) if self.model_type != "classification"
            else torch.LongTensor(y_train.astype(int)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val) if self.model_type != "classification"
            else torch.LongTensor(y_val.astype(int)),
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.n_epochs):
            # Train
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)

                if self.model_type == "classification":
                    loss = criterion(outputs, y_batch)
                else:
                    loss = criterion(outputs.squeeze(), y_batch.squeeze())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = self.model(X_batch)

                    if self.model_type == "classification":
                        loss = criterion(outputs, y_batch)
                    else:
                        loss = criterion(outputs.squeeze(), y_batch.squeeze())

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.n_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

        # Create artifact
        artifact = {
            "model_state": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "model_type": self.model_type,
            "input_dim": input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "feature_cols": self.feature_cols,
            "normalization_params": self.normalization_params,
            "history": history,
            "best_val_loss": best_val_loss,
        }

        return artifact

    def _prepare_data(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        # Merge features and labels
        df = features.merge(
            labels[["date", "asset_id", "y_reg", "y_cls"]],
            on=["date", "asset_id"],
            how="inner",
        )

        # Extract features
        X = df[self.feature_cols].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Normalize features (save params for inference)
        if not self.normalization_params:
            self.normalization_params = {
                "mean": np.nanmean(X, axis=0),
                "std": np.nanstd(X, axis=0) + 1e-8,
            }

        X = (X - self.normalization_params["mean"]) / self.normalization_params["std"]

        # Extract labels
        if self.model_type == "classification":
            y = df["y_cls"].values
        else:
            y = df["y_reg"].values

        return X, y

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "feature_cols": self.feature_cols,
            "normalization_params": self.normalization_params,
            "config": {
                "model_type": self.model_type,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
            },
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


def train_momentum_nn(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    val_features: pd.DataFrame | None = None,
    val_labels: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convenience function to train momentum NN.

    Args:
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        config: Training configuration

    Returns:
        Training artifact
    """
    config = config or {}

    trainer = MomentumNNTrainer(
        model_type=config.get("model_type", "regression"),
        hidden_dims=config.get("hidden_dims"),
        dropout=config.get("dropout", 0.3),
        learning_rate=config.get("learning_rate", 1e-3),
        batch_size=config.get("batch_size", 256),
        n_epochs=config.get("n_epochs", 100),
        patience=config.get("patience", 10),
    )

    return trainer.train(train_features, train_labels, val_features, val_labels)
