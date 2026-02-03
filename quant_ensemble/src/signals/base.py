"""
Signal Model Base Interface

All signal generators (alphas, ML models, ensembles) must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from ..common import get_logger, load_pickle, save_pickle

logger = get_logger(__name__)


class SignalModel(ABC):
    """
    Abstract base class for all signal generators.

    This is the core interface that ALL signals must implement:
    - Rule-based alphas (momentum, value, volatility, etc.)
    - ML/DL models (neural networks, gradient boosting, etc.)
    - Ensemble methods (stacking, MoE, etc.)

    CRITICAL: Models output `score` (for ranking), NOT direct return predictions.
    Scores should be cross-sectionally comparable.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize signal model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.model_name = self.config.get("name", self.__class__.__name__)
        self.is_fitted = False
        self._artifact: dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Train/fit the model.

        Args:
            features_df: Features with columns [date, asset_id, feature_1, ...]
            labels_df: Labels with columns [date, asset_id, y_reg, y_rank, ...]
            config: Additional training configuration

        Returns:
            artifact: Serializable dict containing model state
        """
        pass

    @abstractmethod
    def predict(
        self,
        date: pd.Timestamp,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions for a given date.

        Args:
            date: Prediction date
            features_df: Features for all assets at this date
                (columns: [date, asset_id, feature_1, ...])

        Returns:
            scores_df: DataFrame with columns:
                - date: Prediction date
                - asset_id: Asset identifier
                - score: Ranking score (cross-sectionally comparable)
                - confidence: Confidence in [0, 1]
                - model_name: Model identifier
        """
        pass

    def predict_batch(
        self,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions for multiple dates.

        Args:
            features_df: Features for multiple dates

        Returns:
            scores_df: Predictions for all dates
        """
        all_scores = []

        for date in features_df["date"].unique():
            date_features = features_df[features_df["date"] == date]
            scores = self.predict(pd.Timestamp(date), date_features)
            all_scores.append(scores)

        if not all_scores:
            return pd.DataFrame(columns=["date", "asset_id", "score", "confidence", "model_name"])

        return pd.concat(all_scores, ignore_index=True)

    def save(self, path: str | Path) -> None:
        """
        Save model artifact to disk.

        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model_name": self.model_name,
            "config": self.config,
            "is_fitted": self.is_fitted,
            "artifact": self._artifact,
        }

        save_pickle(artifact, path)
        logger.info(f"Saved model to {path}")

    def load(self, path: str | Path) -> None:
        """
        Load model artifact from disk.

        Args:
            path: Path to saved artifact
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        artifact = load_pickle(path)

        self.model_name = artifact["model_name"]
        self.config = artifact["config"]
        self.is_fitted = artifact["is_fitted"]
        self._artifact = artifact["artifact"]

        logger.info(f"Loaded model from {path}")

    def get_feature_importance(self) -> pd.DataFrame | None:
        """
        Get feature importance if available.

        Returns:
            DataFrame with columns [feature, importance] or None
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.model_name}', fitted={self.is_fitted})"


class RuleBasedAlpha(SignalModel):
    """
    Base class for rule-based alpha factors.

    Rule-based alphas don't require fitting - they compute scores
    directly from features using predefined rules.
    """

    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Rule-based alphas don't need fitting.

        This method is a no-op but marks the model as "fitted" for consistency.
        """
        self.is_fitted = True
        self._artifact = {"type": "rule_based"}
        return self._artifact

    def _create_score_df(
        self,
        date: pd.Timestamp,
        asset_ids: pd.Series,
        scores: pd.Series,
        confidences: pd.Series | float = 1.0,
    ) -> pd.DataFrame:
        """
        Helper to create standardized score DataFrame.

        Args:
            date: Prediction date
            asset_ids: Asset identifiers
            scores: Computed scores
            confidences: Confidence values

        Returns:
            Standardized score DataFrame
        """
        if isinstance(confidences, (int, float)):
            confidences = pd.Series([confidences] * len(asset_ids), index=asset_ids.index)

        return pd.DataFrame({
            "date": date,
            "asset_id": asset_ids.values,
            "score": scores.values,
            "confidence": confidences.values,
            "model_name": self.model_name,
        })


class MLModel(SignalModel):
    """
    Base class for ML/DL models.

    ML models require fitting on training data and produce predictions
    based on learned parameters.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._model = None
        self._feature_cols: list[str] = []
        self._normalizer_params: dict[str, Any] = {}

    def _get_feature_columns(self, features_df: pd.DataFrame) -> list[str]:
        """Get feature column names (excluding date and asset_id)."""
        exclude = {"date", "asset_id"}
        return [c for c in features_df.columns if c not in exclude]

    def _prepare_features(
        self,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare features for model input.

        Args:
            features_df: Raw features DataFrame

        Returns:
            Prepared features DataFrame
        """
        # Select only expected feature columns
        if self._feature_cols:
            available = [c for c in self._feature_cols if c in features_df.columns]
            return features_df[["date", "asset_id"] + available].copy()
        return features_df.copy()
