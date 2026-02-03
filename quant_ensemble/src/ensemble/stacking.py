"""
Stacking Ensemble

Meta-model trained on out-of-fold base model scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..common import get_logger
from ..labels.leakage import create_purged_kfold
from ..signals.base import SignalModel
from .base import EnsembleModel

logger = get_logger(__name__)


class StackingEnsemble(EnsembleModel):
    """
    Stacking ensemble with out-of-fold predictions.

    ANTI-LEAKAGE: Uses OOF predictions to train meta-model.

    Process:
        1. Generate out-of-fold predictions from base models
        2. Concatenate OOF scores + optional features
        3. Train meta-model (Ridge regression by default)
    """

    def __init__(
        self,
        base_models: list[SignalModel],
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of base models
            config: Configuration with:
                - n_folds: Number of CV folds (default: 5)
                - meta_model: Type of meta-model (default: 'ridge')
                - use_features: Whether to include features in meta-model
                - label_horizon_days: For purging/embargo
        """
        super().__init__(base_models, config)

        self.n_folds = self.config.get("n_folds", 5)
        self.meta_model_type = self.config.get("meta_model", "ridge")
        self.use_features = self.config.get("use_features", False)
        self.label_horizon_days = self.config.get("label_horizon_days", 21)

        self._meta_model = None
        self._scaler = StandardScaler()
        self._meta_feature_cols = []

    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fit stacking ensemble.

        1. Generate OOF predictions from base models
        2. Train meta-model on OOF predictions

        Args:
            features_df: Training features
            labels_df: Training labels
            config: Training configuration

        Returns:
            Training artifact
        """
        config = config or {}

        # Generate OOF predictions
        logger.info("Generating out-of-fold predictions...")
        oof_preds = self._generate_oof_predictions(features_df, labels_df, config)

        if oof_preds.empty:
            logger.warning("No OOF predictions generated, falling back to simple average")
            self.is_fitted = True
            return {"type": "stacking", "status": "fallback"}

        # Merge OOF predictions with labels
        meta_df = oof_preds.merge(
            labels_df[["date", "asset_id", "y_reg"]],
            on=["date", "asset_id"],
            how="inner",
        )

        # Prepare meta-model features
        score_cols = [c for c in meta_df.columns if c.startswith("score_")]
        self._meta_feature_cols = score_cols.copy()

        if self.use_features:
            # Add subset of features
            feature_cols = [c for c in features_df.columns if c not in ["date", "asset_id"]]
            meta_df = meta_df.merge(features_df, on=["date", "asset_id"], how="left")
            self._meta_feature_cols.extend(feature_cols[:10])  # Limit features

        X_meta = meta_df[self._meta_feature_cols].values
        y_meta = meta_df["y_reg"].values

        # Handle missing values
        X_meta = np.nan_to_num(X_meta, nan=0.0)

        # Scale
        X_meta = self._scaler.fit_transform(X_meta)

        # Train meta-model
        logger.info(f"Training {self.meta_model_type} meta-model...")

        if self.meta_model_type == "ridge":
            self._meta_model = Ridge(alpha=1.0)
        else:
            self._meta_model = Ridge(alpha=1.0)  # Default

        self._meta_model.fit(X_meta, y_meta)

        # Refit all base models on full data
        logger.info("Refitting base models on full data...")
        for model in self.base_models:
            model.fit(features_df, labels_df, config)

        self.is_fitted = True

        return {
            "type": "stacking",
            "n_folds": self.n_folds,
            "meta_model": self.meta_model_type,
            "meta_feature_cols": self._meta_feature_cols,
            "n_base_models": len(self.base_models),
        }

    def _generate_oof_predictions(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Generate out-of-fold predictions from base models.

        ANTI-LEAKAGE: Uses purged K-fold CV.
        """
        all_oof = []

        # Create purged K-fold splits
        for fold_idx, (train_idx, val_idx) in enumerate(
            create_purged_kfold(
                features_df,
                n_splits=self.n_folds,
                label_horizon_days=self.label_horizon_days,
            )
        ):
            logger.info(f"Fold {fold_idx + 1}/{self.n_folds}")

            train_features = features_df.iloc[train_idx]
            train_labels = labels_df.iloc[train_idx] if len(labels_df) == len(features_df) else \
                labels_df[labels_df.index.isin(train_idx)]

            val_features = features_df.iloc[val_idx]

            # Train each base model on fold training data
            fold_preds = {"date": val_features["date"], "asset_id": val_features["asset_id"]}

            for model in self.base_models:
                # Clone and fit on fold
                model.fit(train_features, train_labels, config)

                # Predict on validation
                val_preds = model.predict_batch(val_features)
                fold_preds[f"score_{model.model_name}"] = val_preds.set_index(
                    ["date", "asset_id"]
                )["score"].reindex(
                    pd.MultiIndex.from_arrays([val_features["date"], val_features["asset_id"]])
                ).values

            all_oof.append(pd.DataFrame(fold_preds))

        if not all_oof:
            return pd.DataFrame()

        return pd.concat(all_oof, ignore_index=True)

    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate stacking predictions."""
        # Get predictions from all base models
        predictions = {}
        for model in self.base_models:
            try:
                pred = model.predict(date, features_df)
                predictions[model.model_name] = pred
            except Exception as e:
                logger.warning(f"Model {model.model_name} failed: {e}")

        return self.combine_predictions(predictions, date, features_df)

    def combine_predictions(
        self,
        predictions: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        features_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Combine using meta-model."""
        if self._meta_model is None:
            # Fallback to simple average
            merged = None
            for name, pred in predictions.items():
                if merged is None:
                    merged = pred[["date", "asset_id", "score"]].copy()
                else:
                    merged = merged.merge(pred[["date", "asset_id", "score"]], on=["date", "asset_id"], how="outer")

            if merged is None:
                return pd.DataFrame()

            score_cols = [c for c in merged.columns if c == "score" or c.endswith("_x") or c.endswith("_y")]
            merged["score"] = merged[score_cols].mean(axis=1)
            merged["confidence"] = 1.0
            merged["model_name"] = self.model_name
            return merged[["date", "asset_id", "score", "confidence", "model_name"]]

        # Build meta-features
        meta_features = pd.DataFrame({"date": date, "asset_id": features_df["asset_id"]})

        for model_name, pred in predictions.items():
            col_name = f"score_{model_name}"
            if col_name in self._meta_feature_cols:
                meta_features = meta_features.merge(
                    pred[["date", "asset_id", "score"]].rename(columns={"score": col_name}),
                    on=["date", "asset_id"],
                    how="left",
                )

        # Add features if configured
        if self.use_features and features_df is not None:
            for col in self._meta_feature_cols:
                if col not in meta_features.columns and col in features_df.columns:
                    meta_features[col] = features_df[col].values

        # Ensure all columns exist
        for col in self._meta_feature_cols:
            if col not in meta_features.columns:
                meta_features[col] = 0.0

        X = meta_features[self._meta_feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        X = self._scaler.transform(X)

        # Predict with meta-model
        scores = self._meta_model.predict(X)
        scores = pd.Series(scores).rank(pct=True, na_option="keep").values

        return pd.DataFrame({
            "date": date,
            "asset_id": meta_features["asset_id"].values,
            "score": scores,
            "confidence": 1.0,
            "model_name": self.model_name,
        })
