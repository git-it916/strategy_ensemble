"""
Ensemble Base Interface

Base class for all ensemble methods.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import pandas as pd

from ..common import get_logger
from ..signals.base import SignalModel

logger = get_logger(__name__)


class EnsembleModel(SignalModel):
    """
    Base class for ensemble models.

    Ensembles combine multiple base model predictions into a single output.
    """

    def __init__(
        self,
        base_models: list[SignalModel],
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize ensemble.

        Args:
            base_models: List of base signal models to ensemble
            config: Ensemble configuration
        """
        super().__init__(config)
        self.base_models = base_models

    @abstractmethod
    def combine_predictions(
        self,
        predictions: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        features_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Combine predictions from base models.

        Args:
            predictions: Dict of model_name -> predictions DataFrame
            date: Prediction date
            features_df: Optional features for meta-learning

        Returns:
            Combined predictions DataFrame
        """
        pass

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions.

        Args:
            date: Prediction date
            features_df: Features for all assets

        Returns:
            Ensemble predictions DataFrame
        """
        # Get predictions from all base models
        predictions = {}

        for model in self.base_models:
            try:
                pred = model.predict(date, features_df)
                predictions[model.model_name] = pred
            except Exception as e:
                logger.warning(f"Model {model.model_name} failed: {e}")
                continue

        if not predictions:
            # Return neutral scores if all models fail
            return pd.DataFrame({
                "date": date,
                "asset_id": features_df["asset_id"].values,
                "score": [0.5] * len(features_df),
                "confidence": [0.0] * len(features_df),
                "model_name": self.model_name,
            })

        # Combine predictions
        return self.combine_predictions(predictions, date, features_df)


class SimpleAverageEnsemble(EnsembleModel):
    """
    Simple average ensemble.

    Takes the mean of all base model scores.
    """

    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fit base models."""
        for model in self.base_models:
            if not model.is_fitted:
                model.fit(features_df, labels_df, config)

        self.is_fitted = True
        return {"type": "simple_average", "n_models": len(self.base_models)}

    def combine_predictions(
        self,
        predictions: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        features_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Average all predictions."""
        # Merge all predictions
        merged = None

        for model_name, pred in predictions.items():
            pred = pred.copy()
            pred = pred.rename(columns={"score": f"score_{model_name}"})

            if merged is None:
                merged = pred[["date", "asset_id", f"score_{model_name}"]]
            else:
                merged = merged.merge(
                    pred[["date", "asset_id", f"score_{model_name}"]],
                    on=["date", "asset_id"],
                    how="outer",
                )

        # Calculate average
        score_cols = [c for c in merged.columns if c.startswith("score_")]
        merged["score"] = merged[score_cols].mean(axis=1)
        merged["confidence"] = merged[score_cols].notna().sum(axis=1) / len(score_cols)
        merged["model_name"] = self.model_name

        return merged[["date", "asset_id", "score", "confidence", "model_name"]]


class WeightedAverageEnsemble(EnsembleModel):
    """
    Weighted average ensemble.

    Uses predefined or learned weights for each model.
    """

    def __init__(
        self,
        base_models: list[SignalModel],
        weights: dict[str, float] | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize weighted ensemble.

        Args:
            base_models: List of base models
            weights: Dict of model_name -> weight (default: equal weights)
            config: Configuration
        """
        super().__init__(base_models, config)

        if weights is None:
            self.weights = {m.model_name: 1.0 / len(base_models) for m in base_models}
        else:
            self.weights = weights

    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fit base models and optionally learn weights."""
        for model in self.base_models:
            if not model.is_fitted:
                model.fit(features_df, labels_df, config)

        # Optionally learn weights from validation performance
        if self.config.get("learn_weights", False):
            self.weights = self._learn_weights(features_df, labels_df)

        self.is_fitted = True
        return {"type": "weighted_average", "weights": self.weights}

    def _learn_weights(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Learn optimal weights based on IC."""
        ics = {}

        for model in self.base_models:
            # Get predictions
            preds = model.predict_batch(features_df)

            # Merge with labels
            merged = preds.merge(
                labels_df[["date", "asset_id", "y_reg"]],
                on=["date", "asset_id"],
                how="inner",
            )

            # Calculate IC
            ic = merged.groupby("date").apply(
                lambda x: x["score"].corr(x["y_reg"])
            ).mean()

            ics[model.model_name] = max(ic, 0)  # Use positive ICs only

        # Normalize to weights
        total = sum(ics.values())
        if total > 0:
            return {k: v / total for k, v in ics.items()}
        else:
            return {m.model_name: 1.0 / len(self.base_models) for m in self.base_models}

    def combine_predictions(
        self,
        predictions: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        features_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Weighted average of predictions."""
        merged = None
        total_weight = 0.0

        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0.0)
            if weight == 0:
                continue

            pred = pred.copy()
            pred[f"weighted_score"] = pred["score"] * weight

            if merged is None:
                merged = pred[["date", "asset_id", "weighted_score"]]
            else:
                merged = merged.merge(
                    pred[["date", "asset_id", "weighted_score"]],
                    on=["date", "asset_id"],
                    how="outer",
                    suffixes=("", f"_{model_name}"),
                )
                # Sum weighted scores
                merged["weighted_score"] = merged.filter(like="weighted_score").sum(axis=1)
                merged = merged[["date", "asset_id", "weighted_score"]]

            total_weight += weight

        if merged is None:
            return pd.DataFrame({
                "date": [date],
                "asset_id": [None],
                "score": [0.5],
                "confidence": [0.0],
                "model_name": [self.model_name],
            })

        merged["score"] = merged["weighted_score"] / total_weight
        merged["confidence"] = total_weight
        merged["model_name"] = self.model_name

        return merged[["date", "asset_id", "score", "confidence", "model_name"]]
