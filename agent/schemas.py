"""
Reasoning pass data structures.
These define the contract between "however you call the LLM" and the
consensus scorer. Kept separate from the actual LLM-calling code so the
consensus logic can be tested completely offline, with zero API calls,
using fabricated-but-realistic passes.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal

Direction = Literal["buy", "sell", "hold"]


@dataclass
class ReasoningPass:
    """
    One independent LLM reasoning pass over the same evidence.
    `pass_id` distinguishes independent calls (e.g. "pass_1", "pass_2", "pass_3").
    `raw_rationale` is kept in full for the audit log -- never discarded,
    even for rejected trades.
    """
    pass_id: str
    ticker: str
    direction: Direction
    confidence: int  # 0-100, self-reported by the LLM
    raw_rationale: str
    model_name: str  # which model/config produced this pass, for audit

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConsensusResult:
    """
    Output of the consensus scorer. This is what the risk gate reads.
    `approved` is a *recommendation*, not a guarantee of execution --
    the risk gate applies its own independent checks on top of this.
    """
    ticker: str
    passes: list[ReasoningPass]
    majority_direction: Direction
    agreement_ratio: float          # fraction of passes agreeing with majority_direction, 0-1
    avg_confidence_of_majority: float
    deterministic_signal_agrees: bool
    consensus_score: float          # combined score, 0-1
    approved: bool
    rejection_reasons: list[str]    # empty if approved

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passes"] = [p.to_dict() for p in self.passes]
        return d
