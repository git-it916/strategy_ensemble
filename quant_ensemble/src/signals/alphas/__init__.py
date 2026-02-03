"""
Rule-Based Alpha Factors

Provides classic quantitative alpha factors.
"""

from .microstructure import (
    FlowAlpha,
    LiquidityAlpha,
    MicrostructureAlpha,
    SmartMoney,
    VolumeAlpha,
)
from .momentum import (
    MomentumAlpha,
    MomentumReversal,
    TimeSeriesMomentum,
)
from .value import (
    BookToMarket,
    EarningsYield,
    QualityValue,
    ValueAlpha,
)
from .volatility import (
    BetaAlpha,
    DrawdownAlpha,
    RiskAdjustedMomentum,
    VolatilityAlpha,
    VolatilityTiming,
)

__all__ = [
    # Momentum
    "MomentumAlpha",
    "TimeSeriesMomentum",
    "MomentumReversal",
    # Value
    "ValueAlpha",
    "BookToMarket",
    "EarningsYield",
    "QualityValue",
    # Volatility
    "VolatilityAlpha",
    "DrawdownAlpha",
    "RiskAdjustedMomentum",
    "BetaAlpha",
    "VolatilityTiming",
    # Microstructure
    "MicrostructureAlpha",
    "VolumeAlpha",
    "LiquidityAlpha",
    "FlowAlpha",
    "SmartMoney",
]
