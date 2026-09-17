"""
Veto -- a risk-gated earnings-trading agent for tokenized U.S. stocks.

Core idea: most LLM trading agents can't tell you when they're unreliable.
Veto refuses to trade the moment it disagrees with itself, or with what
the price data actually shows -- and logs the refusal as visibly as the
trade.

Modules:
  signals.py    -- deterministic, non-LLM price signal computation
  schemas.py    -- shared data structures (ReasoningPass, ConsensusResult, etc.)
  consensus.py  -- multi-pass LLM agreement scoring + deterministic-signal check
  risk_gate.py  -- final deterministic safety checks before execution
  memory.py     -- embargoed decision storage (prevents lookahead leakage)
"""

__version__ = "0.1.0"
__project_name__ = "Veto"
