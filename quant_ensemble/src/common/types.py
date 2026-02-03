"""
Core Type Definitions - PROJECT CONSTITUTION

All modules must use these types for data exchange.
This ensures type safety and consistency across the entire system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================


class RebalanceFrequency(str, Enum):
    """Rebalancing frequency options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class AllocationMethod(str, Enum):
    """Portfolio allocation methods."""

    TOPK = "topk"
    SOFTMAX = "softmax"
    ZSCORE = "zscore"
    OPTIMIZATION = "optimization"


class OrderType(str, Enum):
    """Order types for execution."""

    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Order execution status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class RegimeType(str, Enum):
    """Market regime types."""

    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"


# =============================================================================
# Core Dataclasses
# =============================================================================


@dataclass
class FeatureRow:
    """
    Single row of features for one asset at one date.

    Attributes:
        date: The date of the observation
        asset_id: Unique identifier for the asset (Bloomberg ticker)
        features: Dictionary of feature name to value
    """

    date: pd.Timestamp
    asset_id: str
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "asset_id": self.asset_id,
            **self.features,
        }


@dataclass
class LabelRow:
    """
    Single label row for supervised learning.

    Attributes:
        date: The date of the observation
        asset_id: Unique identifier for the asset
        y_reg: Regression target (e.g., forward return)
        y_rank: Ranking target (1 to N)
        y_cls: Classification target (0/1 or multi-class)
        sample_weight: Sample importance weight
        purge_group: Group identifier for purging/embargo
    """

    date: pd.Timestamp
    asset_id: str
    y_reg: float
    y_rank: int
    y_cls: int
    sample_weight: float = 1.0
    purge_group: str = ""

    def __post_init__(self):
        if not self.purge_group:
            self.purge_group = f"{self.date.strftime('%Y%m%d')}_{self.asset_id}"


@dataclass
class SignalScore:
    """
    Signal output from any model/alpha.

    Attributes:
        date: The date of the signal
        asset_id: Unique identifier for the asset
        score: Ranking score (NOT raw return prediction)
        confidence: Confidence in [0, 1]
        model_name: Model identifier
    """

    date: pd.Timestamp
    asset_id: str
    score: float
    confidence: float = 1.0
    model_name: str = ""

    def __post_init__(self):
        # Clip confidence to [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class PortfolioWeight:
    """
    Portfolio position weight.

    Attributes:
        date: The date of the weight
        asset_id: Unique identifier for the asset
        weight: Position weight (typically -1 to 1)
        target_shares: Target number of shares (optional)
    """

    date: pd.Timestamp
    asset_id: str
    weight: float
    target_shares: int | None = None


@dataclass
class Trade:
    """
    A single trade execution.

    Attributes:
        date: Trade date
        asset_id: Asset identifier
        side: 'buy' or 'sell'
        shares: Number of shares
        price: Execution price
        cost: Transaction cost
        slippage: Slippage cost
    """

    date: pd.Timestamp
    asset_id: str
    side: str  # 'buy' or 'sell'
    shares: int
    price: float
    cost: float = 0.0
    slippage: float = 0.0

    @property
    def value(self) -> float:
        """Total trade value."""
        return self.shares * self.price

    @property
    def total_cost(self) -> float:
        """Total transaction cost including slippage."""
        return self.cost + self.slippage


@dataclass
class Order:
    """
    Order to be submitted to broker.

    Attributes:
        order_id: Unique order identifier
        asset_id: Asset identifier
        side: 'buy' or 'sell'
        order_type: Market or limit order
        shares: Number of shares
        limit_price: Limit price (for limit orders)
        status: Current order status
    """

    order_id: str
    asset_id: str
    side: str
    order_type: OrderType
    shares: int
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    filled_shares: int = 0
    filled_price: float | None = None


@dataclass
class RegimeProbability:
    """
    Regime probability output.

    Attributes:
        date: Date of the regime estimate
        probabilities: Dictionary of regime type to probability
        dominant_regime: Most likely regime
    """

    date: pd.Timestamp
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def dominant_regime(self) -> str:
        """Get the most likely regime."""
        if not self.probabilities:
            return RegimeType.NEUTRAL.value
        return max(self.probabilities, key=self.probabilities.get)

    def get_probability(self, regime: str) -> float:
        """Get probability for a specific regime."""
        return self.probabilities.get(regime, 0.0)


# =============================================================================
# Pydantic Models for Configuration
# =============================================================================


class BacktestConfig(BaseModel):
    """Backtesting configuration with validation."""

    start_date: str
    end_date: str
    initial_capital: float = Field(gt=0)
    rebalance_freq: RebalanceFrequency = RebalanceFrequency.WEEKLY
    transaction_cost_bps: float = Field(ge=0, default=15.0)
    slippage_bps: float = Field(ge=0, default=5.0)
    max_weight_per_asset: float = Field(gt=0, le=1, default=0.1)
    max_leverage: float = Field(gt=0, default=1.0)

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Validate date format."""
        try:
            pd.Timestamp(v)
        except Exception:
            raise ValueError(f"Invalid date format: {v}")
        return v


class PortfolioConfig(BaseModel):
    """Portfolio construction configuration."""

    allocation_method: AllocationMethod = AllocationMethod.TOPK
    topk: int = Field(gt=0, default=20)
    softmax_temperature: float = Field(gt=0, default=1.0)
    zscore_cap: float = Field(gt=0, default=2.0)
    max_weight: float = Field(gt=0, le=1, default=0.1)
    min_weight: float = Field(ge=0, le=1, default=0.01)
    long_only: bool = True


class RiskConfig(BaseModel):
    """Risk management configuration."""

    beta_hedge_enabled: bool = False
    target_beta: float = 0.0
    hedge_instruments: list[str] = Field(default_factory=list)
    volatility_target: float | None = None
    max_drawdown: float = Field(gt=0, le=1, default=0.15)


# =============================================================================
# DataFrame Schema Validators
# =============================================================================


def validate_features_df(df: pd.DataFrame) -> None:
    """
    Validate features DataFrame schema.

    Args:
        df: Features DataFrame to validate

    Raises:
        AssertionError: If schema validation fails
    """
    required_cols = ["date", "asset_id"]
    assert all(c in df.columns for c in required_cols), (
        f"Missing required columns. Required: {required_cols}, Found: {df.columns.tolist()}"
    )
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), "date column must be datetime type"
    assert df["asset_id"].dtype == object, "asset_id column must be string type"
    assert not df["date"].isna().any(), "date column contains NaN values"
    assert not df["asset_id"].isna().any(), "asset_id column contains NaN values"


def validate_labels_df(df: pd.DataFrame) -> None:
    """
    Validate labels DataFrame schema.

    Args:
        df: Labels DataFrame to validate

    Raises:
        AssertionError: If schema validation fails
    """
    required_cols = ["date", "asset_id", "y_reg"]
    assert all(c in df.columns for c in required_cols), (
        f"Missing required columns. Required: {required_cols}, Found: {df.columns.tolist()}"
    )
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), "date column must be datetime type"
    assert pd.api.types.is_numeric_dtype(df["y_reg"]), "y_reg column must be numeric type"


def validate_signals_df(df: pd.DataFrame) -> None:
    """
    Validate signals DataFrame schema.

    Args:
        df: Signals DataFrame to validate

    Raises:
        AssertionError: If schema validation fails
    """
    required_cols = ["date", "asset_id", "score"]
    assert all(c in df.columns for c in required_cols), (
        f"Missing required columns. Required: {required_cols}, Found: {df.columns.tolist()}"
    )
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), "date column must be datetime type"
    assert pd.api.types.is_numeric_dtype(df["score"]), "score column must be numeric type"


def validate_weights_df(df: pd.DataFrame) -> None:
    """
    Validate weights DataFrame schema.

    Args:
        df: Weights DataFrame to validate

    Raises:
        AssertionError: If schema validation fails
    """
    required_cols = ["date", "asset_id", "weight"]
    assert all(c in df.columns for c in required_cols), (
        f"Missing required columns. Required: {required_cols}, Found: {df.columns.tolist()}"
    )
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), "date column must be datetime type"
    assert pd.api.types.is_numeric_dtype(df["weight"]), "weight column must be numeric type"


def validate_prices_df(df: pd.DataFrame) -> None:
    """
    Validate prices DataFrame schema.

    Args:
        df: Prices DataFrame to validate

    Raises:
        AssertionError: If schema validation fails
    """
    required_cols = ["date", "asset_id", "close"]
    assert all(c in df.columns for c in required_cols), (
        f"Missing required columns. Required: {required_cols}, Found: {df.columns.tolist()}"
    )
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), "date column must be datetime type"
    assert pd.api.types.is_numeric_dtype(df["close"]), "close column must be numeric type"


# =============================================================================
# Utility Functions
# =============================================================================


def features_df_from_rows(rows: list[FeatureRow]) -> pd.DataFrame:
    """Convert list of FeatureRow to DataFrame."""
    if not rows:
        return pd.DataFrame(columns=["date", "asset_id"])
    return pd.DataFrame([r.to_dict() for r in rows])


def signals_df_from_scores(scores: list[SignalScore]) -> pd.DataFrame:
    """Convert list of SignalScore to DataFrame."""
    if not scores:
        return pd.DataFrame(columns=["date", "asset_id", "score", "confidence", "model_name"])
    return pd.DataFrame([
        {
            "date": s.date,
            "asset_id": s.asset_id,
            "score": s.score,
            "confidence": s.confidence,
            "model_name": s.model_name,
        }
        for s in scores
    ])


def weights_df_from_weights(weights: list[PortfolioWeight]) -> pd.DataFrame:
    """Convert list of PortfolioWeight to DataFrame."""
    if not weights:
        return pd.DataFrame(columns=["date", "asset_id", "weight"])
    return pd.DataFrame([
        {
            "date": w.date,
            "asset_id": w.asset_id,
            "weight": w.weight,
            "target_shares": w.target_shares,
        }
        for w in weights
    ])
