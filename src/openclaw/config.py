"""
OpenClaw Configuration

System-wide constants, quality gates, execution policies, and research policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
OPENCLAW_DIR = PROJECT_ROOT / "src" / "openclaw"
OPENCLAW_DATA_DIR = PROJECT_ROOT / "data" / "openclaw"
OPENCLAW_LOGS_DIR = PROJECT_ROOT / "logs" / "openclaw"
GENERATED_ALPHAS_DIR = PROJECT_ROOT / "src" / "alphas" / "openclaw_generated"
REGISTRY_PATH = OPENCLAW_DATA_DIR / "alpha_registry.yaml"
EXPERIMENT_TRACKER_PATH = OPENCLAW_DATA_DIR / "experiments.jsonl"
PERFORMANCE_DIR = OPENCLAW_DATA_DIR / "performance"


# =============================================================================
# Assets
# =============================================================================
ASSETS = ["SOL/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT"]
MAX_LOOKBACK_DAYS = 5


# =============================================================================
# Quality Gates
# =============================================================================
@dataclass(frozen=True)
class QualityGates:
    """Validation gates for new alphas."""

    min_sharpe: float = 1.0
    min_ic: float = 0.03
    max_mdd: float = -0.15           # -15%
    min_backtest_years: int = 2
    max_correlation: float = 0.3
    max_daily_turnover: float = 0.30
    require_oos: bool = True
    oos_split_ratio: float = 0.3     # 30% out-of-sample
    max_is_oos_sharpe_divergence: float = 1.5  # IS Sharpe / OOS Sharpe


# =============================================================================
# Execution Policy
# =============================================================================
@dataclass(frozen=True)
class ExecutionPolicy:
    """Live trading execution policy."""

    max_active_alphas: int = 7
    kill_consecutive_loss_days: int = 5
    kill_min_sharpe: float = 0.2
    max_leverage: float = 5.0
    default_leverage_cap: float = 4.0
    target_mdd_for_leverage: float = 0.20    # target MDD = 20%
    paper_trade_days: int = 14
    signal_interval_minutes: int = 15
    data_refresh_seconds: int = 60           # 1분
    rebalance_threshold: float = 0.02        # 2% weight diff triggers rebalance


# =============================================================================
# Research Policy
# =============================================================================
@dataclass(frozen=True)
class ResearchPolicy:
    """Alpha research session policy."""

    max_llm_calls_per_session: int = 50
    code_gen_model: str = "claude-sonnet-4-20250514"
    ensemble_model: str = "claude-opus-4-6"
    brave_results_per_query: int = 10
    max_ideas_per_session: int = 5
    max_code_gen_retries: int = 2


# =============================================================================
# Weight Allocation
# =============================================================================
@dataclass(frozen=True)
class AllocationPolicy:
    """Ensemble weight allocation policy."""

    risk_parity_blend: float = 0.6     # 60% risk-parity
    llm_blend: float = 0.4            # 40% Claude Opus
    min_weight: float = 0.05          # 최소 5%
    max_weight: float = 0.40          # 최대 40%
    lookback_days: int = 63           # ~3 months for vol estimation


# =============================================================================
# Import Whitelist (for code validation)
# =============================================================================
IMPORT_WHITELIST = frozenset({
    "numpy", "np",
    "pandas", "pd",
    "scipy",
    "sklearn",
    "ta", "talib",
    "math", "statistics",
    "collections", "itertools", "functools",
    "datetime", "typing", "dataclasses",
    "logging", "abc",
    "warnings",
    "__future__",
    "src",  # allow internal imports
})

DANGEROUS_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil",
    "socket", "http", "urllib", "requests", "httpx",
    "pickle", "shelve", "marshal",
    "ctypes", "cffi",
    "signal", "threading", "multiprocessing",
    "importlib", "builtins",
})

DANGEROUS_BUILTINS = frozenset({
    "eval", "exec", "compile",
    "__import__", "globals", "locals",
    "getattr", "setattr", "delattr",
    "open",  # file I/O
})


# =============================================================================
# Search Themes (for Brave API)
# =============================================================================
DEFAULT_SEARCH_THEMES = [
    "crypto futures momentum alpha strategy",
    "cryptocurrency mean reversion trading signal",
    "bitcoin carry trade funding rate alpha",
    "crypto microstructure trading strategy",
    "cross-sectional crypto anomaly factor",
    "crypto volatility trading alpha signal",
    "on-chain metrics trading strategy crypto",
    "crypto order flow imbalance alpha",
]


# =============================================================================
# Singleton Instances
# =============================================================================
QUALITY_GATES = QualityGates()
EXECUTION_POLICY = ExecutionPolicy()
RESEARCH_POLICY = ResearchPolicy()
ALLOCATION_POLICY = AllocationPolicy()
