"""V2 Alpha Suite — 10개 알파 auto-import."""

from src.alphas.v2.momentum_multi_scale import MomentumMultiScale
from src.alphas.v2.funding_carry_enhanced import FundingCarryEnhanced
from src.alphas.v2.momentum_composite import MomentumComposite
from src.alphas.v2.intraday_vwap_v2 import IntradayVWAPV2
from src.alphas.v2.intraday_rsi_v2 import IntradayRSIV2
from src.alphas.v2.derivatives_sentiment import DerivativesSentiment
from src.alphas.v2.mean_reversion_multi_horizon import MeanReversionMultiHorizon
from src.alphas.v2.orderbook_imbalance import OrderbookImbalance
from src.alphas.v2.spread_momentum import SpreadMomentum
from src.alphas.v2.volatility_regime import VolatilityRegime

ALL_ALPHAS = [
    MomentumMultiScale,
    FundingCarryEnhanced,
    MomentumComposite,
    IntradayVWAPV2,
    IntradayRSIV2,
    DerivativesSentiment,
    MeanReversionMultiHorizon,
    OrderbookImbalance,
    SpreadMomentum,
    VolatilityRegime,
]
