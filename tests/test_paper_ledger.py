"""
Tests for the internal paper trading ledger. Tests the bookkeeping
and refusal logic offline (no network calls) by passing an explicit
live_price rather than letting fill_decision() fetch one -- the
fetch_live_rtoken_price() network function itself is exercised
separately, manually, against the real Bitget public endpoint.
Run with: python3 tests/test_paper_ledger.py
"""
import sys
import os
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from agent.orchestrator import DecisionCycleResult
from agent.paper_ledger import PaperLedger, PaperLedgerError

_passed = 0
_failed = 0
TEST_DIR = "/tmp/veto_test_paper_ledger"


def check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        print(f"PASS: {name}")
        _passed += 1
    else:
        print(f"FAIL: {name} {detail}")
        _failed += 1


def make_decision(trade_would_execute=True, direction="buy", size_usd=300.0, ticker="TSLA"):
    return DecisionCycleResult(
        cycle_id=f"test-{direction}-{ticker}",
        ticker=ticker,
        deterministic_signal={"trend_label": "uptrend", "current_price": 250.0},
        consensus_result={"approved": trade_would_execute, "rejection_reasons": []},
        risk_gate_result={
            "approved": trade_would_execute,
            "direction": direction,
            "final_position_size_usd": size_usd,
            "rejection_reasons": [],
        },
        naive_baseline={"direction": direction, "would_execute": True},
        memory_record={"record_id": "test", "outcome": None},
        trade_would_execute=trade_would_execute,
        ran_at=datetime.now(timezone.utc).isoformat(),
    )


def run_all():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    ledger = PaperLedger(TEST_DIR, starting_cash=10000.0)

    state = ledger.get_account_state()
    check("initial_cash_correct", state.cash_balance == 10000.0)
    check("initial_positions_empty", state.positions == {})

    buy_decision = make_decision(trade_would_execute=True, direction="buy", size_usd=300.0, ticker="TSLA")
    fill = ledger.fill_decision(buy_decision, live_price=250.0)
    check("buy_fill_quantity_correct", fill.quantity == 1.2)
    check("buy_fill_notional_correct", fill.notional_usd == 300.0)
    check("buy_fill_cash_before_correct", fill.cash_balance_before == 10000.0)
    check("buy_fill_cash_after_correct", fill.cash_balance_after == 9700.0)
    check("buy_fill_position_after_correct", fill.position_after == 1.2)
    check("buy_fill_price_source_labeled", fill.price_source == "bitget_live_public_ticker")

    state = ledger.get_account_state()
    check("state_cash_persisted", state.cash_balance == 9700.0)
    check("state_position_persisted", state.positions["TSLA"] == 1.2)

    sell_decision = make_decision(trade_would_execute=True, direction="sell", size_usd=150.0, ticker="TSLA")
    fill2 = ledger.fill_decision(sell_decision, live_price=300.0)
    expected_qty = round(150.0 / 300.0, 4)
    check("sell_fill_quantity_correct", fill2.quantity == expected_qty)
    check("sell_fill_position_after_correct", fill2.position_after == round(1.2 - expected_qty, 4))
    check("sell_fill_cash_increases", fill2.cash_balance_after > fill2.cash_balance_before)

    rejected = make_decision(trade_would_execute=False)
    try:
        ledger.fill_decision(rejected, live_price=250.0)
        check("rejected_decision_refused", False)
    except PaperLedgerError as e:
        check("rejected_decision_refused", "Refusing to fill" in str(e))

    hold_decision = make_decision(trade_would_execute=True, direction="hold")
    try:
        ledger.fill_decision(hold_decision, live_price=250.0)
        check("hold_direction_refused", False)
    except PaperLedgerError as e:
        check("hold_direction_refused", "nothing to fill" in str(e))

    poor_ledger = PaperLedger(TEST_DIR + "_poor", starting_cash=10.0)
    expensive_buy = make_decision(trade_would_execute=True, direction="buy", size_usd=5000.0, ticker="NVDA")
    try:
        poor_ledger.fill_decision(expensive_buy, live_price=400.0)
        check("insufficient_cash_refused", False)
    except PaperLedgerError as e:
        check("insufficient_cash_refused", "Insufficient paper cash" in str(e))

    empty_ledger = PaperLedger(TEST_DIR + "_empty", starting_cash=10000.0)
    oversell = make_decision(trade_would_execute=True, direction="sell", size_usd=100.0, ticker="AAPL")
    try:
        empty_ledger.fill_decision(oversell, live_price=190.0)
        check("oversell_refused", False)
    except PaperLedgerError as e:
        check("oversell_refused", "Insufficient paper position" in str(e))

    all_fills = ledger.get_all_fills()
    check("all_fills_count_correct", len(all_fills) == 2)
    check("all_fills_order_preserved", all_fills[0].direction == "buy" and all_fills[1].direction == "sell")

    fresh_ledger = PaperLedger(TEST_DIR, starting_cash=99999.0)
    fresh_state = fresh_ledger.get_account_state()
    check("fresh_instance_sees_persisted_cash", fresh_state.cash_balance == fill2.cash_balance_after)
    check("starting_cash_ignored_when_account_exists", fresh_state.cash_balance != 99999.0)

    zero_price_decision = make_decision(trade_would_execute=True, direction="buy", size_usd=100.0, ticker="MSFT")
    try:
        ledger.fill_decision(zero_price_decision, live_price=0.0)
        check("zero_price_refused", False)
    except PaperLedgerError as e:
        check("zero_price_refused", "invalid live price" in str(e))

    shutil.rmtree(TEST_DIR, ignore_errors=True)
    shutil.rmtree(TEST_DIR + "_poor", ignore_errors=True)
    shutil.rmtree(TEST_DIR + "_empty", ignore_errors=True)

    print(f"\n{_passed} passed, {_failed} failed")
    return _failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)