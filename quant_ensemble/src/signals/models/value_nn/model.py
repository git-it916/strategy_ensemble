"""
Value Neural Network Architecture

Mispricing scorer for fundamental features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNN(nn.Module):
    """
    Neural network for value-based ranking.

    Similar to MomentumNN but trained on fundamental features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden_layers(x)
        return self.output_layer(h)


class ValueClassifier(nn.Module):
    """Quantile classification model for value."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 5,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.n_classes = n_classes

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.classifier(h)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        class_indices = torch.arange(self.n_classes, device=x.device, dtype=x.dtype)
        expected = (probs * class_indices).sum(dim=-1) / (self.n_classes - 1)
        return expected.unsqueeze(-1)
