"""
Orchestrator
============
This is the one function a caller actually needs: feed it price
history, reasoning passes, and portfolio state, and it runs the full
Veto pipeline -- deterministic signal, consensus scoring, risk gating,
and embargoed memory recording -- in the correct order, with nothing
skipped.

Deliberately does NOT call any LLM or exchange API itself. Those live
in separate modules (llm_client.py, execution.py) that you wire in
once your API keys are ready. This keeps the orchestrator itself
fully testable with zero network calls, same discipline as every
other module in this project.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import uuid

import pandas as pd

from agent.signals import compute_deterministic_signal, DeterministicSignal
from agent.schemas import ReasoningPass, ConsensusResult
from agent.consensus import score_consensus, score_naive_baseline
from agent.risk_gate import evaluate_risk_gate, PortfolioState, RiskGateResult
from agent.memory import EmbargoedMemoryStore, MemoryRecord
from agent.config import DEFAULT_TRADE_SIZE_USD, MEMORY_EMBARGO_HOURS, STORAGE_DIR


@dataclass
class DecisionCycleResult:
    """
    Full audit trail of one decision cycle, start to finish.
    Every intermediate artifact is kept -- nothing is discarded, even
    for rejected trades. This IS the evidence the submission form asks
    for (Part 3/4: validation data and progress).
    """
    cycle_id: str
    ticker: str
    deterministic_signal: dict
    consensus_result: dict
    risk_gate_result: dict
    naive_baseline: dict           # what a naive single-LLM agent would have done, for comparison
    memory_record: dict
    trade_would_execute: bool
    ran_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_decision_cycle(
    ticker: str,
    price_history: pd.DataFrame,
    reasoning_passes: list[ReasoningPass],
    portfolio: PortfolioState,
    memory_store: EmbargoedMemoryStore | None = None,
    proposed_position_size_usd: float = DEFAULT_TRADE_SIZE_USD,
    decided_at: datetime | None = None,
) -> DecisionCycleResult:
    """
    Runs one full decision cycle for a single ticker:
      1. Compute the deterministic price signal (no LLM)
      2. Score consensus across the provided reasoning passes against that signal
      3. Compute the naive-baseline comparison (what a single-LLM agent would do)
      4. Run the risk gate on the consensus result
      5. Record everything to embargoed memory
      6. Return the full audit trail

    This function does NOT execute a real or paper trade -- that is a
    separate, explicit step (see execution.py) so that "the agent
    decided X" and "the agent acted on X" are never conflated in the
    audit log.
    """
    decided_at = decided_at or datetime.now(timezone.utc)
    cycle_id = str(uuid.uuid4())

    # Step 1: deterministic signal, independent of any LLM call
    signal = compute_deterministic_signal(ticker, price_history)

    # Step 2: consensus scoring
    consensus = score_consensus(reasoning_passes, signal)

    # Step 3: naive baseline, for the dashboard's comparison metric
    naive = score_naive_baseline(reasoning_passes)

    # Step 4: risk gate
    risk_result = evaluate_risk_gate(consensus, portfolio, proposed_position_size_usd)

    trade_would_execute = consensus.approved and risk_result.approved

    # Step 5: record to embargoed memory (always -- rejected decisions are
    # recorded too, since the rejection log is a first-class product output)
    memory_store = memory_store or EmbargoedMemoryStore(STORAGE_DIR, embargo_hours=MEMORY_EMBARGO_HOURS)
    memory_record = memory_store.record_decision(
        record_id=cycle_id,
        consensus_result=consensus,
        risk_gate_result=risk_result,
        executed=trade_would_execute,
        decided_at=decided_at,
    )

    return DecisionCycleResult(
        cycle_id=cycle_id,
        ticker=ticker,
        deterministic_signal=signal.to_dict(),
        consensus_result=consensus.to_dict(),
        risk_gate_result=risk_result.to_dict(),
        naive_baseline=naive,
        memory_record=memory_record.to_dict(),
        trade_would_execute=trade_would_execute,
        ran_at=decided_at.isoformat(),
    )
