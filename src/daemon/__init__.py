"""
Unified Daemon — Sonnet-Driven Trading System

Single-process daemon that handles:
  - Alpha signal generation (7 rule-based + OpenClaw live + ensemble)
  - Sonnet decision making (entry/exit/SL/TP per position)
  - SL/TP monitoring between cycles (5s interval)
  - Binance order execution via Rebalancer
  - Background research sessions (OpenClaw)
"""

from .signal_aggregator import SignalAggregator
from .sonnet_decision_maker import SonnetDecisionMaker
from .position_store import PositionStore
from .sltp_monitor import SLTPMonitor
from .trade_proposal import TradeProposalBuilder
from .unified_daemon import UnifiedDaemon

__all__ = [
    "SignalAggregator",
    "SonnetDecisionMaker",
    "PositionStore",
    "SLTPMonitor",
    "TradeProposalBuilder",
    "UnifiedDaemon",
]
