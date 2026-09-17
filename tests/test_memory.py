"""
Tests for embargoed memory -- the module that prevents lookahead leakage.
This is the highest-stakes module to get right, so tests here are adversarial:
we actively try to break the embargo, not just check the happy path.
Run with: python3 tests/test_memory.py
"""
import sys
import os
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from datetime import datetime, timezone, timedelta
from agent.signals import compute_deterministic_signal
from agent.schemas import ReasoningPass
from agent.consensus import score_consensus
from agent.risk_gate import evaluate_risk_gate, PortfolioState
from agent.memory import EmbargoedMemoryStore, TradeOutcome

_passed = 0
_failed = 0
TEST_STORAGE_DIR = "/tmp/consensus_desk_test_memory"


def make_price_series(prices):
    dates = pd.date_range(end="2026-09-09", periods=len(prices), freq="D")
    return pd.DataFrame({"date": dates, "close": prices})


def check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        print(f"PASS: {name}")
        _passed += 1
    else:
        print(f"FAIL: {name} {detail}")
        _failed += 1


def make_approved_consensus_and_risk(ticker="TSLA"):
    prices = [100 + i * 1.5 for i in range(25)]
    signal = compute_deterministic_signal(ticker, make_price_series(prices))
    passes = [
        ReasoningPass("p1", ticker, "buy", 90, "Strong beat", "model-a"),
        ReasoningPass("p2", ticker, "buy", 88, "Raised guidance", "model-a"),
        ReasoningPass("p3", ticker, "buy", 92, "Momentum", "model-a"),
    ]
    consensus = score_consensus(passes, signal)
    portfolio = PortfolioState(cash_balance=10000, positions={}, total_portfolio_value=10000, open_orders_count=0)
    risk = evaluate_risk_gate(consensus, portfolio, proposed_position_size_usd=500)
    return consensus, risk


def run_all():
    # Clean slate for every test run
    if os.path.exists(TEST_STORAGE_DIR):
        shutil.rmtree(TEST_STORAGE_DIR)

    store = EmbargoedMemoryStore(TEST_STORAGE_DIR, embargo_hours=24)
    consensus, risk = make_approved_consensus_and_risk("TSLA")

    decided_at = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    record = store.record_decision("rec_1", consensus, risk, executed=True, decided_at=decided_at)

    # TEST 1: decision is recorded with no outcome at write time
    check("decision_recorded_with_no_outcome", record.outcome is None)
    check("embargo_clears_24h_later", record.embargo_clears_at == (decided_at + timedelta(hours=24)).isoformat())

    # TEST 2: attempting to backfill BEFORE embargo clears must raise PermissionError
    outcome = TradeOutcome(
        realized_at=(decided_at + timedelta(hours=2)).isoformat(),
        price_at_realization=115.0,
        pnl_usd=50.0,
        pnl_pct=5.0,
    )
    too_early = decided_at + timedelta(hours=2)  # only 2h in, embargo is 24h
    try:
        store.backfill_outcome("TSLA", "rec_1", outcome, now=too_early)
        check("early_backfill_raises_permission_error", False, "did not raise")
    except PermissionError as e:
        check("early_backfill_raises_permission_error", "Embargo has not cleared" in str(e))

    # TEST 3: confirm the outcome was NOT written despite the attempted early backfill
    all_records = store.get_all_records_unfiltered("TSLA")
    check("failed_backfill_did_not_persist", all_records[0].outcome is None)

    # TEST 4: backfill AFTER embargo clears succeeds
    exactly_at_clear = decided_at + timedelta(hours=24)
    result = store.backfill_outcome("TSLA", "rec_1", outcome, now=exactly_at_clear)
    check("on_time_backfill_succeeds", result.outcome is not None)
    check("on_time_backfill_correct_pnl", result.outcome["pnl_usd"] == 50.0)

    # TEST 5: even though outcome is now stored, get_retrievable_history queried
    # from a time BEFORE the embargo cleared must still hide it -- this is the
    # critical defense-in-depth check for the retrieval layer, independent of backfill.
    decided_at2 = datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc)
    consensus2, risk2 = make_approved_consensus_and_risk("TSLA")
    record2 = store.record_decision("rec_2", consensus2, risk2, executed=True, decided_at=decided_at2)
    late_backfill_time = decided_at2 + timedelta(hours=24)
    store.backfill_outcome("TSLA", "rec_2", outcome, now=late_backfill_time)

    # Now query "as of" a time that is AFTER rec_2 was decided but BEFORE its embargo cleared
    query_time_mid_embargo = decided_at2 + timedelta(hours=10)
    history = store.get_retrievable_history("TSLA", as_of=query_time_mid_embargo)
    rec2_in_history = [r for r in history if r.record_id == "rec_2"][0]
    check("retrieval_hides_outcome_during_embargo_even_if_backfilled", rec2_in_history.outcome is None)

    # TEST 6: querying AFTER embargo clears DOES show the outcome
    query_time_after_embargo = decided_at2 + timedelta(hours=25)
    history = store.get_retrievable_history("TSLA", as_of=query_time_after_embargo)
    rec2_in_history = [r for r in history if r.record_id == "rec_2"][0]
    check("retrieval_shows_outcome_after_embargo_clears", rec2_in_history.outcome is not None)
    check("retrieval_outcome_correct_value", rec2_in_history.outcome["pnl_usd"] == 50.0)

    # TEST 7: rec_1 (older, embargo long cleared) is visible from a query "as of" much later
    history_now = store.get_retrievable_history("TSLA", as_of=datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc))
    rec1_in_history = [r for r in history_now if r.record_id == "rec_1"][0]
    check("old_cleared_record_visible", rec1_in_history.outcome is not None)

    # TEST 8: backfilling a nonexistent record_id raises
    try:
        store.backfill_outcome("TSLA", "does_not_exist", outcome, now=datetime(2030, 1, 1, tzinfo=timezone.utc))
        check("backfill_nonexistent_record_raises", False)
    except ValueError as e:
        check("backfill_nonexistent_record_raises", "No record found" in str(e))

    # TEST 9: get_all_records_unfiltered returns everything regardless of embargo,
    # and is a DIFFERENT function than get_retrievable_history -- confirms the
    # separation is real, not just a flag
    unfiltered = store.get_all_records_unfiltered("TSLA")
    check("unfiltered_accessor_returns_both_records", len(unfiltered) == 2)

    # TEST 10: exactly-at-boundary embargo check (>=, not >) -- confirms no off-by-one
    decided_at3 = datetime(2026, 9, 6, 0, 0, 0, tzinfo=timezone.utc)
    consensus3, risk3 = make_approved_consensus_and_risk("TSLA")
    store.record_decision("rec_3", consensus3, risk3, executed=True, decided_at=decided_at3)
    exact_boundary = decided_at3 + timedelta(hours=24)  # exactly at clearance, should succeed (not before)
    try:
        store.backfill_outcome("TSLA", "rec_3", outcome, now=exact_boundary)
        check("exact_boundary_backfill_succeeds", True)
    except PermissionError:
        check("exact_boundary_backfill_succeeds", False, "raised at exact boundary, should have allowed >=")

    # TEST 11: one microsecond before boundary must still fail
    decided_at4 = datetime(2026, 9, 7, 0, 0, 0, tzinfo=timezone.utc)
    consensus4, risk4 = make_approved_consensus_and_risk("TSLA")
    store.record_decision("rec_4", consensus4, risk4, executed=True, decided_at=decided_at4)
    one_us_before = decided_at4 + timedelta(hours=24) - timedelta(microseconds=1)
    try:
        store.backfill_outcome("TSLA", "rec_4", outcome, now=one_us_before)
        check("one_microsecond_before_boundary_fails", False, "should have raised PermissionError")
    except PermissionError:
        check("one_microsecond_before_boundary_fails", True)

    # TEST 12: separate tickers don't cross-contaminate storage
    consensus_nvda, risk_nvda = make_approved_consensus_and_risk("NVDA")
    store.record_decision("nvda_rec_1", consensus_nvda, risk_nvda, executed=True, decided_at=decided_at)
    tsla_records = store.get_all_records_unfiltered("TSLA")
    nvda_records = store.get_all_records_unfiltered("NVDA")
    check("tickers_isolated_tsla_count", len(tsla_records) == 4)
    check("tickers_isolated_nvda_count", len(nvda_records) == 1)

    # Cleanup
    shutil.rmtree(TEST_STORAGE_DIR)

    print(f"\n{_passed} passed, {_failed} failed")
    return _failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
