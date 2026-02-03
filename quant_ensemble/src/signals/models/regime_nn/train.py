"""
Regime Neural Network Training

Training logic for regime classification models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ....common import get_logger, set_seed
from .model import RegimeNN

logger = get_logger(__name__)


class RegimeNNTrainer:
    """Trainer for regime neural network models."""

    def __init__(
        self,
        n_regimes: int = 2,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        n_epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        seed: int = 42,
    ):
        """
        Initialize trainer.

        Args:
            n_regimes: Number of regime classes
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Learning rate
            batch_size: Batch size
            n_epochs: Maximum epochs
            patience: Early stopping patience
            device: Device to use
            seed: Random seed
        """
        self.n_regimes = n_regimes
        self.hidden_dims = hidden_dims or [32, 16]
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
        train_labels: pd.DataFrame | None = None,
        val_features: pd.DataFrame | None = None,
        val_labels: pd.DataFrame | None = None,
        feature_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Train the regime model.

        For unsupervised regime detection, labels can be None.
        Labels should be regime indices (0, 1, ..., n_regimes-1).

        Args:
            train_features: Training features (market-wide)
            train_labels: Training labels (regime indices, optional)
            val_features: Validation features
            val_labels: Validation labels
            feature_cols: Feature columns to use

        Returns:
            Training artifact
        """
        set_seed(self.seed)

        # Identify regime-related feature columns
        if feature_cols is None:
            exclude = {"date", "asset_id"}
            # Look for market-wide features
            feature_cols = [
                c for c in train_features.columns
                if c not in exclude and any(
                    v in c.lower() for v in [
                        "market", "vol", "ret", "vix", "spread", "breadth", "corr"
                    ]
                )
            ]
            if not feature_cols:
                feature_cols = [c for c in train_features.columns if c not in exclude]

        self.feature_cols = feature_cols

        # Prepare data (aggregate to date level if needed)
        X_train = self._prepare_features(train_features)

        # Create pseudo-labels if not provided (based on volatility clustering)
        if train_labels is None:
            y_train = self._create_pseudo_labels(train_features)
        else:
            y_train = self._prepare_labels(train_labels)

        # Validation
        if val_features is not None:
            X_val = self._prepare_features(val_features)
            if val_labels is None:
                y_val = self._create_pseudo_labels(val_features)
            else:
                y_val = self._prepare_labels(val_labels)
        else:
            n_val = int(len(X_train) * 0.1)
            X_val, y_val = X_train[-n_val:], y_train[-n_val:]
            X_train, y_train = X_train[:-n_val], y_train[:-n_val]

        # Initialize model
        input_dim = X_train.shape[1]

        self.model = RegimeNN(
            input_dim=input_dim,
            n_regimes=self.n_regimes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )
        self.model = self.model.to(self.device)

        # Data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val),
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.n_epochs):
            # Train
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
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
                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

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

        # Calibrate temperature
        self.model.eval()
        with torch.no_grad():
            val_logits = self.model(torch.FloatTensor(X_val).to(self.device))
            self.model.calibrate(val_logits, torch.LongTensor(y_val).to(self.device))

        return {
            "model_state": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "n_regimes": self.n_regimes,
            "input_dim": input_dim,
            "hidden_dims": self.hidden_dims,
            "feature_cols": self.feature_cols,
            "normalization_params": self.normalization_params,
        }

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for training."""
        # Aggregate to date level (mean across assets)
        if "date" in df.columns and "asset_id" in df.columns:
            agg_df = df.groupby("date")[self.feature_cols].mean()
            X = agg_df.values
        else:
            X = df[self.feature_cols].values

        X = np.nan_to_num(X, nan=0.0)

        if not self.normalization_params:
            self.normalization_params = {
                "mean": np.nanmean(X, axis=0),
                "std": np.nanstd(X, axis=0) + 1e-8,
            }

        X = (X - self.normalization_params["mean"]) / self.normalization_params["std"]
        return X

    def _prepare_labels(self, labels_df: pd.DataFrame) -> np.ndarray:
        """Prepare regime labels."""
        if "regime" in labels_df.columns:
            return labels_df["regime"].values.astype(int)
        return np.zeros(len(labels_df), dtype=int)

    def _create_pseudo_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Create pseudo-labels based on volatility clustering."""
        # Find volatility column
        vol_col = None
        for col in df.columns:
            if "vol" in col.lower() and "market" in col.lower():
                vol_col = col
                break
            elif "vol" in col.lower():
                vol_col = col

        if vol_col is None:
            # Random labels
            return np.random.randint(0, self.n_regimes, len(df))

        # Aggregate to date level
        if "date" in df.columns:
            vol = df.groupby("date")[vol_col].mean().values
        else:
            vol = df[vol_col].values

        # Simple threshold-based clustering
        vol_median = np.nanmedian(vol)
        labels = (vol > vol_median).astype(int)

        return labels
