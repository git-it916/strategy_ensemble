"""
Regime Neural Network Architecture

Market regime classifier: risk-on vs risk-off.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeNN(nn.Module):
    """
    Neural network for market regime classification.

    Architecture:
        - Input: Market-wide features (volatility, returns, sentiment)
        - Hidden: FC layers with BatchNorm + Dropout
        - Output: Regime probabilities
    """

    def __init__(
        self,
        input_dim: int,
        n_regimes: int = 2,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        """
        Initialize regime classifier.

        Args:
            input_dim: Number of input features
            n_regimes: Number of regime classes (default: 2 for risk-on/risk-off)
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 16]

        self.input_dim = input_dim
        self.n_regimes = n_regimes

        # Encoder
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
        self.classifier = nn.Linear(prev_dim, n_regimes)

        # Temperature for calibration (Platt scaling)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            logits: Regime logits of shape (batch_size, n_regimes)
        """
        h = self.encoder(x)
        logits = self.classifier(h)
        return logits

    def predict_proba(self, x: torch.Tensor, calibrated: bool = True) -> torch.Tensor:
        """
        Predict regime probabilities.

        Args:
            x: Input tensor
            calibrated: Whether to apply temperature scaling

        Returns:
            probabilities: Softmax probabilities
        """
        logits = self.forward(x)

        if calibrated:
            logits = logits / self.temperature

        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict regime class.

        Args:
            x: Input tensor

        Returns:
            predicted_class: Predicted regime indices
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        n_iter: int = 100,
    ) -> None:
        """
        Calibrate temperature using Platt scaling.

        Args:
            val_logits: Validation logits
            val_labels: Validation labels
            lr: Learning rate for calibration
            n_iter: Number of optimization iterations
        """
        self.temperature.requires_grad = True
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_iter)

        def eval_fn():
            optimizer.zero_grad()
            loss = F.cross_entropy(val_logits / self.temperature, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        self.temperature.requires_grad = False


class RegimeHMM(nn.Module):
    """
    Neural network-based HMM for regime detection.

    Combines neural network emission model with HMM transition dynamics.
    """

    def __init__(
        self,
        input_dim: int,
        n_regimes: int = 2,
        hidden_dim: int = 32,
    ):
        """
        Initialize regime HMM.

        Args:
            input_dim: Number of input features
            n_regimes: Number of hidden states (regimes)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_regimes = n_regimes

        # Emission network: P(observation | state)
        self.emission_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes),
        )

        # Transition matrix parameters (log scale)
        self.log_transition = nn.Parameter(torch.zeros(n_regimes, n_regimes))

        # Initial state probabilities (log scale)
        self.log_initial = nn.Parameter(torch.zeros(n_regimes))

    def get_transition_matrix(self) -> torch.Tensor:
        """Get normalized transition matrix."""
        return F.softmax(self.log_transition, dim=-1)

    def get_initial_proba(self) -> torch.Tensor:
        """Get initial state probabilities."""
        return F.softmax(self.log_initial, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute emission probabilities.

        Args:
            x: Input tensor of shape (seq_len, batch_size, input_dim)
               or (batch_size, input_dim)

        Returns:
            emission_logits: Emission logits
        """
        return self.emission_net(x)

    def filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward filtering to compute P(state_t | observations_{1:t}).

        Args:
            x: Input sequence of shape (seq_len, input_dim)

        Returns:
            filtered_probs: Filtered state probabilities of shape (seq_len, n_regimes)
        """
        seq_len = x.shape[0]
        device = x.device

        # Get parameters
        transition = self.get_transition_matrix()
        initial = self.get_initial_proba()

        # Emission probabilities
        emission_logits = self.forward(x)
        emission_probs = F.softmax(emission_logits, dim=-1)

        # Forward pass (filtering)
        filtered = torch.zeros(seq_len, self.n_regimes, device=device)

        # t = 0
        filtered[0] = initial * emission_probs[0]
        filtered[0] = filtered[0] / filtered[0].sum()

        # t > 0
        for t in range(1, seq_len):
            pred = torch.matmul(filtered[t-1], transition)
            filtered[t] = pred * emission_probs[t]
            filtered[t] = filtered[t] / (filtered[t].sum() + 1e-8)

        return filtered


class RegimeGRU(nn.Module):
    """
    GRU-based regime classifier for sequential data.

    Uses recurrent architecture to capture temporal dynamics.
    """

    def __init__(
        self,
        input_dim: int,
        n_regimes: int = 2,
        hidden_dim: int = 32,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initialize GRU regime classifier.

        Args:
            input_dim: Number of input features
            n_regimes: Number of regime classes
            hidden_dim: GRU hidden dimension
            n_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_regimes = n_regimes
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(hidden_dim, n_regimes)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Initial hidden state

        Returns:
            (logits, hidden): Regime logits and final hidden state
        """
        output, hidden = self.gru(x, hidden)
        logits = self.classifier(output)
        return logits, hidden

    def predict_proba(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict regime probabilities.

        Args:
            x: Input tensor
            hidden: Initial hidden state

        Returns:
            probabilities: Softmax probabilities
        """
        logits, _ = self.forward(x, hidden)
        return F.softmax(logits, dim=-1)
