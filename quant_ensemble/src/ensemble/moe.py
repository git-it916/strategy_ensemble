"""
Mixture-of-Experts Ensemble

Regime-based gating of expert models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..common import RegimeProbability, get_logger
from ..signals.base import SignalModel
from .base import EnsembleModel

logger = get_logger(__name__)


class MoEEnsemble(EnsembleModel):
    """
    Mixture-of-Experts ensemble with regime-based gating.

    Process:
        1. Regime model outputs gate probabilities P(regime|date)
        2. Each expert model has a score S_i for each asset
        3. Final score = Σ P(regime_k) * w_{i,k} * S_i

    where w_{i,k} is the weight of expert i in regime k.
    """

    def __init__(
        self,
        base_models: list[SignalModel],
        regime_model: SignalModel,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize MoE ensemble.

        Args:
            base_models: List of expert models
            regime_model: Regime classification model
            config: Configuration with:
                - n_regimes: Number of regimes (default: 2)
                - smoothing_alpha: Regime probability smoothing (default: 0.2)
                - expert_weights: Optional pre-defined weights per regime
        """
        super().__init__(base_models, config)

        self.regime_model = regime_model
        self.n_regimes = self.config.get("n_regimes", 2)
        self.smoothing_alpha = self.config.get("smoothing_alpha", 0.2)

        # Expert weights per regime: dict[regime_idx, dict[model_name, weight]]
        self.expert_weights: dict[int, dict[str, float]] = {}
        self._prev_regime_probs = None

        # Initialize equal weights
        for k in range(self.n_regimes):
            self.expert_weights[k] = {
                m.model_name: 1.0 / len(base_models) for m in base_models
            }

    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fit the MoE ensemble.

        1. Fit the regime model
        2. Fit each expert model
        3. Learn regime-conditional weights based on performance

        Args:
            features_df: Training features
            labels_df: Training labels
            config: Training configuration

        Returns:
            Training artifact
        """
        config = config or {}

        # Fit regime model
        if not self.regime_model.is_fitted:
            logger.info("Fitting regime model...")
            self.regime_model.fit(features_df, labels_df, config)

        # Fit expert models
        for model in self.base_models:
            if not model.is_fitted:
                logger.info(f"Fitting expert: {model.model_name}")
                model.fit(features_df, labels_df, config)

        # Learn regime-conditional weights
        if self.config.get("learn_weights", True):
            logger.info("Learning regime-conditional expert weights...")
            self.expert_weights = self._learn_regime_weights(features_df, labels_df)

        self.is_fitted = True

        return {
            "type": "moe",
            "n_regimes": self.n_regimes,
            "expert_weights": self.expert_weights,
            "n_experts": len(self.base_models),
        }

    def _learn_regime_weights(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> dict[int, dict[str, float]]:
        """
        Learn optimal weights for each expert in each regime.

        Weights are based on regime-conditional IC (Information Coefficient).
        """
        # Get regime predictions
        regime_probs_list = []
        for date in features_df["date"].unique():
            date_features = features_df[features_df["date"] == date]
            regime_prob = self.regime_model.predict_regime(
                pd.Timestamp(date), date_features
            )
            regime_probs_list.append({
                "date": pd.Timestamp(date),
                **regime_prob.probabilities,
            })

        regime_df = pd.DataFrame(regime_probs_list)

        # Get predictions from each expert
        expert_preds = {}
        for model in self.base_models:
            preds = model.predict_batch(features_df)
            expert_preds[model.model_name] = preds

        # Calculate regime-conditional IC for each expert
        weights = {}

        for k in range(self.n_regimes):
            regime_col = f"regime_{k}"
            weights[k] = {}

            for model_name, preds in expert_preds.items():
                # Merge with labels and regime probs
                merged = preds.merge(
                    labels_df[["date", "asset_id", "y_reg"]],
                    on=["date", "asset_id"],
                    how="inner",
                )
                merged = merged.merge(regime_df, on="date", how="left")

                # Weight by regime probability
                if regime_col in merged.columns:
                    # Calculate weighted IC
                    merged["weight"] = merged[regime_col]
                    merged["weighted_product"] = merged["weight"] * merged["score"] * merged["y_reg"]

                    weighted_ic = (
                        merged.groupby("date")["weighted_product"].sum() /
                        merged.groupby("date")["weight"].sum()
                    ).mean()

                    weights[k][model_name] = max(weighted_ic, 0.01)
                else:
                    weights[k][model_name] = 1.0 / len(self.base_models)

            # Normalize weights
            total = sum(weights[k].values())
            weights[k] = {m: w / total for m, w in weights[k].items()}

        return weights

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate MoE predictions.

        Args:
            date: Prediction date
            features_df: Features for all assets

        Returns:
            Ensemble predictions
        """
        # Get regime probabilities
        regime_prob = self.regime_model.predict_regime(date, features_df)

        # Apply smoothing
        probs = np.array([
            regime_prob.probabilities.get(f"regime_{k}", 1.0 / self.n_regimes)
            for k in range(self.n_regimes)
        ])

        if self._prev_regime_probs is not None:
            probs = (
                self.smoothing_alpha * probs +
                (1 - self.smoothing_alpha) * self._prev_regime_probs
            )
            probs = probs / probs.sum()

        self._prev_regime_probs = probs.copy()

        # Get expert predictions
        predictions = {}
        for model in self.base_models:
            try:
                pred = model.predict(date, features_df)
                predictions[model.model_name] = pred
            except Exception as e:
                logger.warning(f"Expert {model.model_name} failed: {e}")

        # Combine with regime-weighted gating
        return self.combine_predictions(predictions, date, features_df, probs)

    def combine_predictions(
        self,
        predictions: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        features_df: pd.DataFrame | None = None,
        regime_probs: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Combine predictions using regime-based gating.

        Args:
            predictions: Expert predictions
            date: Prediction date
            features_df: Features (unused)
            regime_probs: Regime probabilities

        Returns:
            Combined predictions
        """
        if regime_probs is None:
            regime_probs = np.ones(self.n_regimes) / self.n_regimes

        # Calculate combined weights for each expert
        combined_weights = {}
        for model_name in predictions.keys():
            weight = 0.0
            for k in range(self.n_regimes):
                regime_weight = self.expert_weights.get(k, {}).get(model_name, 0.0)
                weight += regime_probs[k] * regime_weight
            combined_weights[model_name] = weight

        # Normalize
        total = sum(combined_weights.values())
        if total > 0:
            combined_weights = {k: v / total for k, v in combined_weights.items()}
        else:
            combined_weights = {k: 1.0 / len(predictions) for k in predictions}

        # Merge and weight predictions
        merged = None

        for model_name, pred in predictions.items():
            weight = combined_weights.get(model_name, 0.0)
            pred = pred.copy()
            pred["weighted_score"] = pred["score"] * weight

            if merged is None:
                merged = pred[["date", "asset_id", "weighted_score"]].copy()
            else:
                temp = pred[["date", "asset_id", "weighted_score"]]
                merged = merged.merge(temp, on=["date", "asset_id"], how="outer")
                merged["weighted_score"] = merged.filter(like="weighted_score").sum(axis=1)
                merged = merged[["date", "asset_id", "weighted_score"]]

        if merged is None:
            return pd.DataFrame({
                "date": date,
                "asset_id": features_df["asset_id"].values if features_df is not None else [],
                "score": 0.5,
                "confidence": 0.0,
                "model_name": self.model_name,
            })

        merged["score"] = merged["weighted_score"]
        merged["confidence"] = 1.0 - np.max(regime_probs)  # Lower confidence when regime is uncertain
        merged["model_name"] = self.model_name

        # Add regime info
        dominant_regime = int(np.argmax(regime_probs))
        merged["dominant_regime"] = dominant_regime
        merged["regime_prob"] = regime_probs[dominant_regime]

        return merged[["date", "asset_id", "score", "confidence", "model_name", "dominant_regime", "regime_prob"]]

    def get_regime_weights(self) -> pd.DataFrame:
        """Get expert weights by regime as DataFrame."""
        rows = []
        for k, weights in self.expert_weights.items():
            for model_name, weight in weights.items():
                rows.append({
                    "regime": k,
                    "expert": model_name,
                    "weight": weight,
                })
        return pd.DataFrame(rows)

    def reset_state(self) -> None:
        """Reset internal state (regime smoothing)."""
        self._prev_regime_probs = None
        if hasattr(self.regime_model, "reset_state"):
            self.regime_model.reset_state()
