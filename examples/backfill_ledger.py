#!/usr/bin/env python3
"""
Backfills the paper trading ledger with every APPROVED decision
currently in storage that hasn't been filled yet.

Usage:
    python examples/backfill_ledger.py

Requires: internet access (fetches a real live price per ticker from
Bitget's public ticker endpoint). Does not require GROQ_API_KEY or
Bitget API credentials -- only the public, no-auth ticker read.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.memory import EmbargoedMemoryStore
from agent.orchestrator import DecisionCycleResult
from agent.paper_ledger import PaperLedger, fetch_live_rtoken_price, PaperLedgerError
from agent.config import STORAGE_DIR, TRACKED_TICKERS


def main():
    store = EmbargoedMemoryStore(STORAGE_DIR)
    ledger = PaperLedger("./storage/paper_ledger", starting_cash=10000.0)

    already_filled_cycle_ids = {f.cycle_id for f in ledger.get_all_fills()}

    filled_count = 0
    skipped_count = 0
    failed_count = 0

    for ticker in TRACKED_TICKERS:
        records = store.get_all_records_unfiltered(ticker)
        for record in records:
            approved = record.consensus_result["approved"] and record.risk_gate_result["approved"]
            direction = record.risk_gate_result["direction"]

            if not approved or direction == "hold":
                continue
            if record.record_id in already_filled_cycle_ids:
                skipped_count += 1
                continue

            decision = DecisionCycleResult(
                cycle_id=record.record_id,
                ticker=record.ticker,
                deterministic_signal={},
                consensus_result=record.consensus_result,
                risk_gate_result=record.risk_gate_result,
                naive_baseline={},
                memory_record=record.to_dict(),
                trade_would_execute=True,
                ran_at=record.decided_at,
            )

            try:
                fill = ledger.fill_decision(decision)
                print(f"  FILLED  [{ticker}] {direction.upper()} {fill.quantity} @ ${fill.fill_price:.2f} "
                      f"(record {record.record_id[:8]})")
                filled_count += 1
            except PaperLedgerError as e:
                print(f"  SKIPPED [{ticker}] record {record.record_id[:8]}: {e}")
                failed_count += 1

    print(f"\nDone. {filled_count} new fills, {skipped_count} already filled, {failed_count} failed.")
    state = ledger.get_account_state()
    print(f"Ledger cash balance: ${state.cash_balance:,.2f}")
    print(f"Positions: {state.positions}")


if __name__ == "__main__":
    main()