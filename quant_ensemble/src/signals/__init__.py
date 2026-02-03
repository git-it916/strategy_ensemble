"""
Signal Generation Module

Provides signal generators for the ensemble system.
"""

from .alphas import (
    BetaAlpha,
    BookToMarket,
    DrawdownAlpha,
    EarningsYield,
    FlowAlpha,
    LiquidityAlpha,
    MicrostructureAlpha,
    MomentumAlpha,
    MomentumReversal,
    QualityValue,
    RiskAdjustedMomentum,
    SmartMoney,
    TimeSeriesMomentum,
    ValueAlpha,
    VolatilityAlpha,
    VolatilityTiming,
    VolumeAlpha,
)
from .base import MLModel, RuleBasedAlpha, SignalModel
from .models import (
    MomentumNN,
    MomentumNNModel,
    MomentumNNTrainer,
    RegimeNN,
    RegimeNNModel,
    RegimeNNTrainer,
    ValueNN,
    ValueNNModel,
    ValueNNTrainer,
)

__all__ = [
    # Base
    "SignalModel",
    "RuleBasedAlpha",
    "MLModel",
    # Alphas - Momentum
    "MomentumAlpha",
    "TimeSeriesMomentum",
    "MomentumReversal",
    # Alphas - Value
    "ValueAlpha",
    "BookToMarket",
    "EarningsYield",
    "QualityValue",
    # Alphas - Volatility
    "VolatilityAlpha",
    "DrawdownAlpha",
    "RiskAdjustedMomentum",
    "BetaAlpha",
    "VolatilityTiming",
    # Alphas - Microstructure
    "MicrostructureAlpha",
    "VolumeAlpha",
    "LiquidityAlpha",
    "FlowAlpha",
    "SmartMoney",
    # Models
    "MomentumNN",
    "MomentumNNTrainer",
    "MomentumNNModel",
    "ValueNN",
    "ValueNNTrainer",
    "ValueNNModel",
    "RegimeNN",
    "RegimeNNTrainer",
    "RegimeNNModel",
]
