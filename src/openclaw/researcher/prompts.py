"""
LLM Prompt Templates

All prompt templates for alpha code generation.
Includes the BaseAlpha interface contract and FundingRateCarry reference.
"""

# =============================================================================
# BaseAlpha Interface (included in code generation prompts)
# =============================================================================
BASE_ALPHA_INTERFACE = '''
class BaseAlpha(ABC):
    """Base class for all alpha strategies."""

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}
        self.is_fitted = False
        self._fit_date: datetime | None = None

    @abstractmethod
    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Fit/train the strategy. prices has: date, ticker, open, high, low, close, volume"""
        pass

    @abstractmethod
    def generate_signals(
        self,
        date: datetime,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> AlphaResult:
        """
        Generate signals for a specific date.
        prices: historical data UP TO date (no future data!)
        Must return AlphaResult(date=date, signals=DataFrame[ticker, score])
        score should be in [-1, 1] range.
        """
        pass


@dataclass
class AlphaResult:
    date: datetime
    signals: pd.DataFrame  # Must have columns: ticker, score
    metadata: dict[str, Any] = field(default_factory=dict)
'''

# =============================================================================
# Reference Implementation (FundingRateCarry)
# =============================================================================
REFERENCE_ALPHA = '''
class FundingRateCarry(BaseAlpha):
    """Funding Rate Carry alpha for Binance USDT-M perpetual futures."""

    def __init__(self, name: str = "FundingRateCarry", lookback_days: int = 7):
        super().__init__(name)
        self.lookback_days = lookback_days

    def fit(self, prices, features=None, labels=None):
        self.is_fitted = True
        self._fit_date = datetime.now()
        return {"status": "fitted", "lookback_days": self.lookback_days}

    def generate_signals(self, date, prices, features=None):
        empty = AlphaResult(date=date, signals=pd.DataFrame(columns=["ticker", "score"]))

        if features is None or features.empty:
            return empty

        # Look-ahead bias prevention
        past = features[features["date"] <= date].copy()
        if past.empty:
            return empty

        past = past.sort_values("date")
        cutoff = past["date"].max() - pd.Timedelta(days=self.lookback_days)
        window = past[past["date"] >= cutoff]

        if window.empty:
            return empty

        avg_funding = (
            window.groupby("ticker")["funding_rate"]
            .mean()
            .reset_index()
            .rename(columns={"funding_rate": "avg_funding"})
        )
        avg_funding = avg_funding.dropna(subset=["avg_funding"])

        # Score: low/negative funding → high score → long
        avg_funding["score"] = (avg_funding["avg_funding"].rank(pct=True) - 0.5) * -2

        signals = avg_funding[["ticker", "score"]].reset_index(drop=True)
        return AlphaResult(date=date, signals=signals, metadata={...})
'''

# =============================================================================
# Code Generation System Prompt
# =============================================================================
CODE_GEN_SYSTEM_PROMPT = f"""You are an expert quantitative developer generating Python alpha strategies.

You must produce a COMPLETE, RUNNABLE Python class that subclasses BaseAlpha.

## Interface Contract
{BASE_ALPHA_INTERFACE}

## Reference Implementation
{REFERENCE_ALPHA}

## STRICT Requirements

1. IMPORTS: Only use these at the top of your code:
   ```python
   from __future__ import annotations
   from datetime import datetime
   from typing import Any
   import numpy as np
   import pandas as pd
   from src.alphas.base_alpha import AlphaResult, BaseAlpha
   ```
   You may also import from: scipy.stats, sklearn, ta, math, statistics, collections

2. CLASS STRUCTURE:
   - Must subclass BaseAlpha
   - Constructor: call super().__init__(name) with a descriptive name
   - fit(): Set self.is_fitted = True. For rule-based, no training needed.
   - generate_signals(): Return AlphaResult with DataFrame[ticker, score]

3. DATA ACCESS:
   - prices DataFrame has columns: date, ticker, open, high, low, close, volume
   - features DataFrame may have: date, ticker, funding_rate, and other feature columns
   - NEVER access future data. Always filter: prices[prices["date"] <= date]
   - Maximum lookback: 5 days from the signal date

4. SIGNAL COMPUTATION:
   - score must be in [-1, 1] range (use rank normalization or tanh/clip)
   - Positive score = long signal, negative = short signal
   - Handle edge cases: empty data, missing columns, insufficient history
   - Return empty AlphaResult for edge cases (don't raise exceptions)

5. FORBIDDEN:
   - No file I/O (open, read, write)
   - No network calls (requests, urllib, socket)
   - No subprocess, os, sys calls
   - No eval(), exec(), __import__()
   - No global state mutation
   - No print statements (use logging if needed)

6. OUTPUT FORMAT:
   Return ONLY the Python code. No markdown, no backticks, no explanation.
   The code should be a single complete file that can be saved as a .py file.

7. QUALITY:
   - Include a docstring explaining the strategy
   - Handle all edge cases gracefully
   - Use vectorized pandas/numpy operations (no row-level loops)
   - Implement _get_extra_state() and _restore_extra_state() if you have custom state
"""

# =============================================================================
# Code Generation User Prompt Template
# =============================================================================
CODE_GEN_USER_TEMPLATE = """Generate a BaseAlpha implementation for this strategy:

Name: {name}
Description: {description}
Hypothesis: {hypothesis}
Signal Logic: {signal_logic}
Required Data: {required_data}
Max Lookback: {lookback_days} days
Style: {expected_style}

The alpha will trade these assets: SOL/USDT, ETH/USDT, BTC/USDT on Binance futures.

Generate the complete Python code now."""

# =============================================================================
# Code Fix Prompt (for retries)
# =============================================================================
CODE_FIX_PROMPT = """The previous code had these issues:
{errors}

Fix the code and return the COMPLETE corrected Python file.
Do not return partial code or just the fix — return the entire file.

Previous code:
{code}"""
