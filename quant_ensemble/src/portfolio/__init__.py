"""
Portfolio Management Module

Components for portfolio construction, constraints, and risk management.
"""

from .allocator import PortfolioAllocator, scores_to_weights
from .constraints import ConstraintApplier, PortfolioConstraints, apply_constraints
from .costs import CostEstimate, SlippageModel, TransactionCostModel, estimate_annual_cost
from .risk import RiskBudgetAllocator, RiskManager, RiskMetrics

__all__ = [
    # Allocator
    "PortfolioAllocator",
    "scores_to_weights",
    # Constraints
    "PortfolioConstraints",
    "ConstraintApplier",
    "apply_constraints",
    # Risk
    "RiskManager",
    "RiskMetrics",
    "RiskBudgetAllocator",
    # Costs
    "TransactionCostModel",
    "CostEstimate",
    "SlippageModel",
    "estimate_annual_cost",
]
