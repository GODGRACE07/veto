"""
End-to-end integration test for the orchestrator -- proves the full
pipeline (signal -> consensus -> risk gate -> memory) actually works
together, not just in isolated unit tests.
Run with: python3 tests/test_orchestrator.py
"""
import sys
import os
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from datetime import datetime, timezone
from agent.schemas import ReasoningPass
from agent.risk_gate import PortfolioState
from agent.memory import EmbargoedMemoryStore
from agent.orchestrator import run_decision_cycle

_passed = 0
_failed = 0
TEST_STORAGE_DIR = "/tmp/veto_test_orchestrator_memory"


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


def run_all():
    if os.path.exists(TEST_STORAGE_DIR):
        shutil.rmtree(TEST_STORAGE_DIR)
    store = EmbargoedMemoryStore(TEST_STORAGE_DIR, embargo_hours=24)
    portfolio = PortfolioState(cash_balance=10000, positions={}, total_portfolio_value=10000, open_orders_count=0)

    # SCENARIO 1: strong uptrend + unanimous high-confidence buy -> full pipeline approves and would execute
    uptrend_prices = [100 + i * 1.5 for i in range(25)]
    strong_buy_passes = [
        ReasoningPass("p1", "TSLA", "buy", 90, "Earnings beat expectations significantly", "groq-llama"),
        ReasoningPass("p2", "TSLA", "buy", 88, "Guidance raised for next quarter", "groq-llama"),
        ReasoningPass("p3", "TSLA", "buy", 92, "Revenue and margins both improved", "groq-mixtral"),
    ]
    result = run_decision_cycle(
        ticker="TSLA",
        price_history=make_price_series(uptrend_prices),
        reasoning_passes=strong_buy_passes,
        portfolio=portfolio,
        memory_store=store,
        proposed_position_size_usd=300.0,
        decided_at=datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    check("scenario1_would_execute", result.trade_would_execute is True)
    check("scenario1_has_cycle_id", len(result.cycle_id) > 0)
    check("scenario1_signal_present", result.deterministic_signal["trend_label"] in ("uptrend", "strong_uptrend"))
    check("scenario1_consensus_approved", result.consensus_result["approved"] is True)
    check("scenario1_risk_gate_approved", result.risk_gate_result["approved"] is True)
    check("scenario1_memory_recorded", result.memory_record["record_id"] == result.cycle_id)
    check("scenario1_memory_outcome_is_none_at_write_time", result.memory_record["outcome"] is None)

    # SCENARIO 2: THE key demo case -- unanimous, high-confidence "buy" that
    # CONTRADICTS the deterministic signal (downtrend). Naive baseline would
    # execute; Veto must reject.
    downtrend_prices = [200 - i * 1.5 for i in range(25)]
    contradicting_passes = [
        ReasoningPass("p1", "TSLA", "buy", 90, "Social sentiment looks very bullish", "groq-llama"),
        ReasoningPass("p2", "TSLA", "buy", 88, "Analysts raised price targets", "groq-llama"),
        ReasoningPass("p3", "TSLA", "buy", 92, "Should see a rally soon", "groq-mixtral"),
    ]
    result2 = run_decision_cycle(
        ticker="TSLA",
        price_history=make_price_series(downtrend_prices),
        reasoning_passes=contradicting_passes,
        portfolio=portfolio,
        memory_store=store,
        proposed_position_size_usd=300.0,
        decided_at=datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc),
    )
    check("scenario2_veto_rejects_despite_unanimous_llm_agreement", result2.trade_would_execute is False)
    check("scenario2_consensus_rejected", result2.consensus_result["approved"] is False)
    check(
        "scenario2_rejection_reason_is_deterministic_disagreement",
        any("Deterministic signal disagreement" in r for r in result2.consensus_result["rejection_reasons"]),
    )
    # THIS is the headline comparison metric: naive baseline WOULD have executed
    check("scenario2_naive_baseline_would_have_executed", result2.naive_baseline["would_execute"] is True)
    check(
        "scenario2_veto_vs_naive_diverge",
        result2.trade_would_execute != result2.naive_baseline["would_execute"],
        "This divergence IS the product's core value proposition -- must hold."
    )

    # SCENARIO 3: disagreement among the passes themselves (no clean majority)
    split_passes = [
        ReasoningPass("p1", "TSLA", "buy", 80, "Bullish read", "groq-llama"),
        ReasoningPass("p2", "TSLA", "sell", 75, "Bearish read", "groq-mixtral"),
        ReasoningPass("p3", "TSLA", "hold", 60, "Too uncertain to call", "groq-llama"),
    ]
    result3 = run_decision_cycle(
        ticker="TSLA",
        price_history=make_price_series(uptrend_prices),
        reasoning_passes=split_passes,
        portfolio=portfolio,
        memory_store=store,
        proposed_position_size_usd=300.0,
        decided_at=datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc),
    )
    check("scenario3_three_way_split_rejected", result3.trade_would_execute is False)
    check(
        "scenario3_rejection_is_low_agreement",
        any("Low agreement" in r for r in result3.consensus_result["rejection_reasons"]),
    )

    # SCENARIO 4: verify each cycle produces a UNIQUE cycle_id (no collisions)
    ids = {result.cycle_id, result2.cycle_id, result3.cycle_id}
    check("scenario4_unique_cycle_ids", len(ids) == 3)

    # SCENARIO 5: verify memory actually persisted to disk and is retrievable
    # independent of the orchestrator -- opening a FRESH store instance pointed
    # at the same directory should see all 3 records.
    fresh_store = EmbargoedMemoryStore(TEST_STORAGE_DIR, embargo_hours=24)
    all_tsla_records = fresh_store.get_all_records_unfiltered("TSLA")
    check("scenario5_all_three_decisions_persisted", len(all_tsla_records) == 3)

    # SCENARIO 6: rejected trades still have full rationale in the persisted record
    rejected_record = [r for r in all_tsla_records if r.record_id == result2.cycle_id][0]
    passes_in_record = rejected_record.consensus_result["passes"]
    check(
        "scenario6_rejected_trade_rationale_persisted",
        all(len(p["raw_rationale"]) > 0 for p in passes_in_record),
    )

    # SCENARIO 7: different tickers in the same run don't interfere
    nvda_uptrend = [400 + i * 3 for i in range(25)]
    nvda_passes = [
        ReasoningPass("p1", "NVDA", "buy", 85, "Strong AI demand", "groq-llama"),
        ReasoningPass("p2", "NVDA", "buy", 83, "Datacenter growth", "groq-mixtral"),
    ]
    result_nvda = run_decision_cycle(
        ticker="NVDA",
        price_history=make_price_series(nvda_uptrend),
        reasoning_passes=nvda_passes,
        portfolio=portfolio,
        memory_store=store,
        proposed_position_size_usd=300.0,
        decided_at=datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc),
    )
    check("scenario7_nvda_processed_independently", result_nvda.ticker == "NVDA")
    tsla_after = fresh_store.get_all_records_unfiltered("TSLA")
    nvda_after = fresh_store.get_all_records_unfiltered("NVDA")
    check("scenario7_tsla_count_unaffected", len(tsla_after) == 3)
    check("scenario7_nvda_count_correct", len(nvda_after) == 1)

    shutil.rmtree(TEST_STORAGE_DIR)
    print(f"\n{_passed} passed, {_failed} failed")
    return _failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
