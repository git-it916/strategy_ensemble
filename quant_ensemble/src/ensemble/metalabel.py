"""
Meta-Labeling Ensemble

Second-stage classifier: should we take this signal?
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from ..common import get_logger
from ..signals.base import SignalModel
from .base import EnsembleModel

logger = get_logger(__name__)


class MetaLabeler(EnsembleModel):
    """
    Meta-labeling second-stage classifier.

    Process:
        1. Base signal generates score/direction
        2. Meta-model predicts Pr(profitable | signal, features)
        3. Scale position by meta-prediction

    This filters out low-confidence trades and sizes positions based on confidence.
    """

    def __init__(
        self,
        base_model: SignalModel,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize meta-labeler.

        Args:
            base_model: Primary signal model
            config: Configuration with:
                - meta_model: Type of meta-model ('gbm', 'logistic')
                - threshold: Confidence threshold for taking trades
                - feature_cols: Features for meta-model
        """
        super().__init__([base_model], config)

        self.base_model = base_model
        self.threshold = self.config.get("threshold", 0.5)
        self.meta_model_type = self.config.get("meta_model", "gbm")

        self._meta_model = None
        self._scaler = StandardScaler()
        self._feature_cols = []

    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fit meta-labeler.

        1. Fit base model
        2. Generate base model predictions
        3. Create meta-labels (was the trade profitable?)
        4. Train meta-model to predict meta-labels

        Args:
            features_df: Training features
            labels_df: Training labels
            config: Training configuration

        Returns:
            Training artifact
        """
        config = config or {}

        # Fit base model
        if not self.base_model.is_fitted:
            logger.info("Fitting base model...")
            self.base_model.fit(features_df, labels_df, config)

        # Generate base model predictions
        logger.info("Generating base model predictions...")
        base_preds = self.base_model.predict_batch(features_df)

        # Create meta-labels
        logger.info("Creating meta-labels...")
        meta_df = self._create_meta_labels(base_preds, labels_df)

        if meta_df.empty:
            logger.warning("No meta-labels created")
            self.is_fitted = True
            return {"type": "metalabel", "status": "no_data"}

        # Prepare features for meta-model
        self._feature_cols = self.config.get("feature_cols", [])

        if not self._feature_cols:
            # Auto-select features
            exclude = {"date", "asset_id", "score", "confidence", "model_name", "y_reg", "meta_label"}
            self._feature_cols = [c for c in meta_df.columns if c not in exclude][:20]

        # Add base model score and confidence
        meta_features = ["base_score", "base_confidence"]
        if "base_score" not in meta_df.columns:
            meta_df["base_score"] = meta_df["score"]
            meta_df["base_confidence"] = meta_df.get("confidence", 1.0)

        all_feature_cols = meta_features + self._feature_cols
        available_cols = [c for c in all_feature_cols if c in meta_df.columns]

        X = meta_df[available_cols].values
        y = meta_df["meta_label"].values

        # Handle missing
        X = np.nan_to_num(X, nan=0.0)

        # Scale
        X = self._scaler.fit_transform(X)

        # Train meta-model
        logger.info(f"Training {self.meta_model_type} meta-model...")

        if self.meta_model_type == "gbm":
            self._meta_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            from sklearn.linear_model import LogisticRegression
            self._meta_model = LogisticRegression(random_state=42)

        self._meta_model.fit(X, y)

        self._feature_cols = available_cols
        self.is_fitted = True

        return {
            "type": "metalabel",
            "meta_model": self.meta_model_type,
            "n_features": len(self._feature_cols),
            "threshold": self.threshold,
        }

    def _create_meta_labels(
        self,
        base_preds: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create meta-labels for training.

        Meta-label = 1 if the trade would have been profitable
        """
        # Merge predictions with labels
        merged = base_preds.merge(
            labels_df[["date", "asset_id", "y_reg"]],
            on=["date", "asset_id"],
            how="inner",
        )

        if merged.empty:
            return pd.DataFrame()

        # Define trade direction based on score
        # High score = long, low score = short (or not trade)
        score_median = merged.groupby("date")["score"].transform("median")
        merged["direction"] = np.where(merged["score"] > score_median, 1, -1)

        # Meta-label: was the directional trade profitable?
        merged["trade_return"] = merged["direction"] * merged["y_reg"]
        merged["meta_label"] = (merged["trade_return"] > 0).astype(int)

        # Rename score for clarity
        merged["base_score"] = merged["score"]
        merged["base_confidence"] = merged.get("confidence", 1.0)

        return merged

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate meta-labeled predictions.

        Args:
            date: Prediction date
            features_df: Features for all assets

        Returns:
            Predictions with confidence scaling
        """
        # Get base model prediction
        base_pred = self.base_model.predict(date, features_df)

        if self._meta_model is None:
            # Return base predictions with neutral confidence
            base_pred["confidence"] = 0.5
            base_pred["model_name"] = self.model_name
            return base_pred

        # Prepare meta-features
        meta_df = base_pred.copy()
        meta_df["base_score"] = meta_df["score"]
        meta_df["base_confidence"] = meta_df.get("confidence", 1.0)

        # Add features
        for col in self._feature_cols:
            if col not in meta_df.columns and col in features_df.columns:
                meta_df[col] = features_df[col].values

        # Ensure all columns exist
        for col in self._feature_cols:
            if col not in meta_df.columns:
                meta_df[col] = 0.0

        X = meta_df[self._feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        X = self._scaler.transform(X)

        # Predict meta-label probability
        meta_prob = self._meta_model.predict_proba(X)[:, 1]

        # Scale score by meta-probability
        # Score is adjusted toward neutral (0.5) when meta-prob is low
        adjusted_score = meta_df["base_score"].values
        confidence_factor = meta_prob

        # Apply threshold: if below threshold, reduce confidence
        below_threshold = meta_prob < self.threshold
        confidence_factor[below_threshold] = meta_prob[below_threshold] / self.threshold * 0.5

        return pd.DataFrame({
            "date": date,
            "asset_id": meta_df["asset_id"].values,
            "score": adjusted_score,
            "confidence": confidence_factor,
            "meta_prob": meta_prob,
            "model_name": self.model_name,
        })

    def combine_predictions(
        self,
        predictions: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        features_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Not used for meta-labeling."""
        # Meta-labeler doesn't combine multiple predictions
        if predictions:
            return list(predictions.values())[0]
        return pd.DataFrame()
