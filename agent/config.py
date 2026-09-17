"""
Central configuration for Veto.
Deliberately kept as one small, readable file -- every tunable value
in the project should be traceable back to here, not scattered across
modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# Deliberately scoped narrow: 5 tickers, chosen for liquid tokenized
# availability and a real history of earnings-driven volatility.
TRACKED_TICKERS: list[str] = ["TSLA", "NVDA", "AAPL", "MSFT", "AMZN"]

# Number of independent LLM reasoning passes per decision.
NUM_REASONING_PASSES = 3

# Embargo horizon: how long a decision's outcome stays hidden from the
# agent's own memory/retrieval after the decision is made.
MEMORY_EMBARGO_HOURS = 24

# Default paper-trade sizing per approved trade.
DEFAULT_TRADE_SIZE_USD = 300.0

STORAGE_DIR = "./storage/memory"
LOG_DIR = "./logs"


@dataclass
class LLMProviderConfig:
    """One entry per model/provider used for a reasoning pass."""
    name: str
    provider: str      # e.g. "groq", "google", "anthropic"
    model_id: str       # provider-specific model identifier
    temperature: float = 0.3


# Groq's current (post-August 2026) recommended general-purpose models.
# llama-3.3-70b-versatile was deprecated and fully decommissioned by
# Groq on August 16, 2026 -- if you see a "model_not_found" error again
# in the future, check https://console.groq.com/docs/deprecations for
# the current replacement and update the model_id values below.
DEFAULT_LLM_PROVIDERS: list[LLMProviderConfig] = [
    LLMProviderConfig(name="pass_a", provider="groq", model_id="openai/gpt-oss-120b", temperature=0.2),
    LLMProviderConfig(name="pass_b", provider="groq", model_id="openai/gpt-oss-120b", temperature=0.7),
    LLMProviderConfig(name="pass_c", provider="groq", model_id="openai/gpt-oss-20b", temperature=0.4),
]