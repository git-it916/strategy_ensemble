"""
Sequential Pipeline Module

Data → Context → LLM Judge → Human Approval → Execution

Replaces the parallel voting ensemble with a sequential pipeline
where the LLM acts as the final Fund Manager decision maker.
"""

from .context_builder import ContextBuilder, PipelineContext, StockContext
from .risk_manager import RiskManager, PositionOrder
from .approval_agent import ApprovalAgent
from .universe import (
    build_universe_snapshot,
    infer_market_from_ticker,
    normalize_order_ticker,
)

__all__ = [
    "ContextBuilder",
    "PipelineContext",
    "StockContext",
    "RiskManager",
    "PositionOrder",
    "ApprovalAgent",
    "build_universe_snapshot",
    "infer_market_from_ticker",
    "normalize_order_ticker",
]
