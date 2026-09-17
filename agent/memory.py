"""
Embargoed Memory
=================
Implements the "Outcome Embargo" concept directly (see field survey,
arXiv 2605.19337, Section 4.2): an episode recorded at time t cannot
expose its outcome to retrieval until current time >= t + k, where k
is the embargo horizon.

Why this matters: without an embargo, an agent's own memory can leak
the future into its present-time reasoning. E.g. "recall a similar
past trade" retrieval could surface a note like "this failed because
of news that broke the next day" -- information the agent could not
have possibly had at the time it's supposedly reasoning from. The
survey calls this the "Oracle Fallacy." This module makes that
leakage structurally impossible rather than relying on prompt-level
instructions not to do it.

Storage backend: plain JSON files on disk, one file per ticker. This
is intentionally simple -- a hackathon judge should be able to open
the file and read exactly what the agent knew and when, with no
database or infra required to verify it.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

from agent.schemas import ConsensusResult
from agent.risk_gate import RiskGateResult


@dataclass
class TradeOutcome:
    """Populated only once the embargo period has elapsed and the outcome is knowable."""
    realized_at: str          # ISO timestamp when this outcome was actually recorded
    price_at_realization: float
    pnl_usd: float
    pnl_pct: float


@dataclass
class MemoryRecord:
    """
    One full record of a decision: what was proposed, what was decided,
    and -- ONLY once the embargo clears -- what happened.
    `outcome` is None until embargo_clears_at has passed AND someone
    explicitly calls backfill_outcome(). It is never silently populated.
    """
    record_id: str
    ticker: str
    decided_at: str                 # ISO timestamp of the decision
    embargo_clears_at: str          # ISO timestamp -- decided_at + embargo horizon
    consensus_result: dict          # serialized ConsensusResult
    risk_gate_result: dict          # serialized RiskGateResult
    executed: bool                  # did this actually place a paper trade
    outcome: dict | None = None     # serialized TradeOutcome, or None if still embargoed / not yet backfilled

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        return cls(**d)


class EmbargoedMemoryStore:
    """
    File-backed store. One JSON file per ticker under `storage_dir`.
    All read operations that could expose an outcome check the embargo
    clock before returning anything -- there is no code path that
    returns an outcome before its embargo has cleared.
    """

    def __init__(self, storage_dir: str | Path, embargo_hours: int = 24):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embargo_hours = embargo_hours

    def _file_for(self, ticker: str) -> Path:
        return self.storage_dir / f"{ticker}.json"

    def _load_all(self, ticker: str) -> list[dict]:
        path = self._file_for(ticker)
        if not path.exists():
            return []
        with open(path, "r") as f:
            return json.load(f)

    def _save_all(self, ticker: str, records: list[dict]) -> None:
        path = self._file_for(ticker)
        with open(path, "w") as f:
            json.dump(records, f, indent=2, default=str)

    def record_decision(
        self,
        record_id: str,
        consensus_result: ConsensusResult,
        risk_gate_result: RiskGateResult,
        executed: bool,
        decided_at: datetime | None = None,
    ) -> MemoryRecord:
        """
        Stores a decision immediately. Outcome is always None at write time --
        there is no parameter to pass an outcome in here, by design. Outcomes
        can only ever enter the store via backfill_outcome(), which enforces
        the embargo check.
        """
        decided_at = decided_at or datetime.now(timezone.utc)
        embargo_clears_at = decided_at + timedelta(hours=self.embargo_hours)

        record = MemoryRecord(
            record_id=record_id,
            ticker=consensus_result.ticker,
            decided_at=decided_at.isoformat(),
            embargo_clears_at=embargo_clears_at.isoformat(),
            consensus_result=consensus_result.to_dict(),
            risk_gate_result=risk_gate_result.to_dict(),
            executed=executed,
            outcome=None,
        )

        records = self._load_all(record.ticker)
        records.append(record.to_dict())
        self._save_all(record.ticker, records)
        return record

    def backfill_outcome(
        self,
        ticker: str,
        record_id: str,
        outcome: TradeOutcome,
        now: datetime | None = None,
    ) -> MemoryRecord:
        """
        Attempts to attach an outcome to a previously-recorded decision.
        Raises if the embargo has not yet cleared -- this is the
        enforcement point. There is no way to bypass this check from
        outside this function.
        """
        now = now or datetime.now(timezone.utc)
        records = self._load_all(ticker)

        target_idx = None
        for i, r in enumerate(records):
            if r["record_id"] == record_id:
                target_idx = i
                break
        if target_idx is None:
            raise ValueError(f"No record found with id '{record_id}' for ticker '{ticker}'")

        record = MemoryRecord.from_dict(records[target_idx])
        embargo_clears_at = datetime.fromisoformat(record.embargo_clears_at)

        if now < embargo_clears_at:
            raise PermissionError(
                f"Embargo has not cleared for record '{record_id}'. "
                f"Embargo clears at {embargo_clears_at.isoformat()}, current time is {now.isoformat()}. "
                f"Refusing to backfill outcome -- this would leak future information into a "
                f"historical decision record."
            )

        record.outcome = outcome.__dict__.copy()
        records[target_idx] = record.to_dict()
        self._save_all(ticker, records)
        return record

    def get_retrievable_history(self, ticker: str, as_of: datetime | None = None) -> list[MemoryRecord]:
        """
        Returns past records for a ticker, but with outcomes STRIPPED
        for any record whose embargo has not cleared as of `as_of`.
        This is what a reasoning pass should query when it wants
        "similar past episodes" -- it can see that a decision was
        made and what was decided, but not the outcome, until the
        embargo genuinely allows it. This directly prevents the
        Oracle Fallacy at the retrieval layer, not just at the
        backfill layer -- even if an outcome WAS backfilled early by
        some other bug, this function still won't surface it if the
        embargo (relative to `as_of`) hasn't cleared.
        """
        as_of = as_of or datetime.now(timezone.utc)
        raw_records = self._load_all(ticker)
        results = []
        for r in raw_records:
            record = MemoryRecord.from_dict(r)
            embargo_clears_at = datetime.fromisoformat(record.embargo_clears_at)
            if as_of < embargo_clears_at:
                # Embargo still active as of the query time -- strip outcome regardless
                # of whether it happens to be populated in storage.
                record.outcome = None
            results.append(record)
        return results

    def get_all_records_unfiltered(self, ticker: str) -> list[MemoryRecord]:
        """
        Admin/audit-only accessor: returns everything including outcomes,
        regardless of embargo status. This is for the dashboard's
        after-the-fact reporting (e.g. "how did we actually do over
        the full run") -- NOT for anything that feeds back into live
        agent reasoning. Naming it distinctly from get_retrievable_history
        is deliberate so it's never accidentally wired into the live
        decision loop.
        """
        raw_records = self._load_all(ticker)
        return [MemoryRecord.from_dict(r) for r in raw_records]
