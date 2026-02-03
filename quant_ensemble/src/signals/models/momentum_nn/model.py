"""
Momentum Neural Network Architecture

Cross-sectional ranker for momentum features.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MomentumNN(nn.Module):
    """
    Neural network for momentum-based ranking.

    Architecture:
        - Input: Momentum features (returns at multiple horizons)
        - Hidden: FC layers with BatchNorm + Dropout
        - Output: Single score per asset

    The model learns non-linear combinations of momentum signals.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Initialize momentum neural network.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer (single score)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            scores: Output tensor of shape (batch_size, 1)
        """
        h = self.hidden_layers(x)
        scores = self.output_layer(h)
        return scores

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get hidden layer embeddings.

        Args:
            x: Input tensor

        Returns:
            embeddings: Hidden layer output
        """
        return self.hidden_layers(x)


class MomentumRanker(nn.Module):
    """
    Pairwise ranking model for momentum.

    Uses margin ranking loss to learn relative ordering.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        """
        Initialize ranking model.

        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.scorer = MomentumNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for pairwise comparison.

        Args:
            x1: First sample features
            x2: Second sample features

        Returns:
            (score1, score2): Scores for each sample
        """
        score1 = self.scorer(x1)
        score2 = self.scorer(x2)
        return score1, score2

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for single batch.

        Args:
            x: Input features

        Returns:
            scores: Predicted scores
        """
        return self.scorer(x)


class MomentumClassifier(nn.Module):
    """
    Quantile classification model for momentum.

    Predicts which quantile an asset belongs to based on future return.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 5,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        """
        Initialize classifier.

        Args:
            input_dim: Number of input features
            n_classes: Number of quantile classes
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.n_classes = n_classes

        # Build encoder
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Classification head
        self.classifier = nn.Linear(prev_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            logits: Class logits
        """
        h = self.encoder(x)
        logits = self.classifier(h)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            x: Input tensor

        Returns:
            probabilities: Softmax probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict ranking score from class probabilities.

        Score is the expected quantile (0 to 1).

        Args:
            x: Input tensor

        Returns:
            scores: Expected quantile scores
        """
        probs = self.predict_proba(x)
        # Expected value: sum(prob * class_index) / (n_classes - 1)
        class_indices = torch.arange(self.n_classes, device=x.device, dtype=x.dtype)
        expected = (probs * class_indices).sum(dim=-1) / (self.n_classes - 1)
        return expected.unsqueeze(-1)
