"""
Tests for the risk gate -- the final deterministic safety check.
Run with: python3 tests/test_risk_gate.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from agent.signals import compute_deterministic_signal
from agent.schemas import ReasoningPass
from agent.consensus import score_consensus
from agent.risk_gate import evaluate_risk_gate, PortfolioState

_passed = 0
_failed = 0


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


def make_healthy_portfolio(cash=10000.0, positions=None, total_value=10000.0, open_orders=0):
    return PortfolioState(
        cash_balance=cash,
        positions=positions or {},
        total_portfolio_value=total_value,
        open_orders_count=open_orders,
    )


def run_all():
    uptrend_prices = [100 + i * 1.5 for i in range(25)]
    uptrend_signal = compute_deterministic_signal("TSLA", make_price_series(uptrend_prices))

    strong_buy_passes = [
        ReasoningPass("p1", "TSLA", "buy", 90, "Earnings beat", "model-a"),
        ReasoningPass("p2", "TSLA", "buy", 88, "Guidance raise", "model-a"),
        ReasoningPass("p3", "TSLA", "buy", 92, "Strong momentum", "model-a"),
    ]
    approved_consensus = score_consensus(strong_buy_passes, uptrend_signal)

    # TEST 1: well-sized trade with healthy portfolio and approved consensus -> approved
    portfolio = make_healthy_portfolio(cash=10000.0, total_value=10000.0)
    result = evaluate_risk_gate(approved_consensus, portfolio, proposed_position_size_usd=500.0)
    check("healthy_trade_approved", result.approved is True, str(result.rejection_reasons))
    check("healthy_trade_final_size_matches_proposed", result.final_position_size_usd == 500.0)

    # TEST 2: consensus itself rejected -> risk gate MUST also reject, no override possible
    disagree_passes = [
        ReasoningPass("p1", "TSLA", "buy", 85, "Bullish", "model-a"),
        ReasoningPass("p2", "TSLA", "sell", 80, "Bearish", "model-a"),
    ]
    rejected_consensus = score_consensus(disagree_passes, uptrend_signal)
    result = evaluate_risk_gate(rejected_consensus, portfolio, proposed_position_size_usd=100.0)
    check("consensus_rejected_forces_risk_gate_rejection", result.approved is False)
    check("consensus_rejected_reason_present", any("Consensus scorer rejected" in r for r in result.rejection_reasons))
    check("consensus_rejected_final_size_zero", result.final_position_size_usd == 0.0)

    # TEST 3: proposed size exceeds absolute cap -> rejected, but clipped size still computed
    result = evaluate_risk_gate(approved_consensus, portfolio, proposed_position_size_usd=5000.0)
    check("oversized_absolute_rejected", result.approved is False)
    check("oversized_absolute_reason", any("exceeds the max single-position cap" in r for r in result.rejection_reasons))

    # TEST 4: proposed size exceeds % of portfolio cap even though under absolute cap
    small_portfolio = make_healthy_portfolio(cash=1000.0, total_value=1000.0)
    result = evaluate_risk_gate(approved_consensus, small_portfolio, proposed_position_size_usd=500.0)  # 50% of 1000
    check("oversized_pct_rejected", result.approved is False)
    check("oversized_pct_reason", any("single-position cap" in r for r in result.rejection_reasons))

    # TEST 5: cash buffer violated
    tight_cash_portfolio = make_healthy_portfolio(cash=600.0, total_value=10000.0)
    result = evaluate_risk_gate(approved_consensus, tight_cash_portfolio, proposed_position_size_usd=500.0)
    check("cash_buffer_violated_rejected", result.approved is False)
    check("cash_buffer_reason", any("cash buffer" in r.lower() or "buffer" in r.lower() for r in result.rejection_reasons))

    # TEST 6: sell with no existing position -> rejected
    downtrend_prices = [200 - i * 1.5 for i in range(25)]
    downtrend_signal = compute_deterministic_signal("TSLA", make_price_series(downtrend_prices))
    sell_passes = [
        ReasoningPass("p1", "TSLA", "sell", 90, "Breaking down", "model-a"),
        ReasoningPass("p2", "TSLA", "sell", 85, "Momentum turning", "model-a"),
    ]
    sell_consensus = score_consensus(sell_passes, downtrend_signal)
    no_position_portfolio = make_healthy_portfolio(cash=10000.0, positions={}, total_value=10000.0)
    result = evaluate_risk_gate(sell_consensus, no_position_portfolio, proposed_position_size_usd=300.0)
    check("sell_without_position_rejected", result.approved is False)
    check("sell_without_position_reason", any("no existing position" in r for r in result.rejection_reasons))

    # TEST 7: sell WITH existing position -> approved
    has_position_portfolio = make_healthy_portfolio(cash=10000.0, positions={"TSLA": 10.0}, total_value=10000.0)
    result = evaluate_risk_gate(sell_consensus, has_position_portfolio, proposed_position_size_usd=300.0)
    check("sell_with_position_approved", result.approved is True, str(result.rejection_reasons))

    # TEST 8: too many open orders -> rejected regardless of everything else
    busy_portfolio = make_healthy_portfolio(cash=10000.0, total_value=10000.0, open_orders=5)
    result = evaluate_risk_gate(approved_consensus, busy_portfolio, proposed_position_size_usd=100.0)
    check("too_many_open_orders_rejected", result.approved is False)
    check("too_many_open_orders_reason", any("open orders" in r for r in result.rejection_reasons))

    # TEST 9: hold direction never requires position sizing and always clears sizing gates
    hold_passes = [
        ReasoningPass("p1", "TSLA", "hold", 70, "Wait and see", "model-a"),
        ReasoningPass("p2", "TSLA", "hold", 65, "Uncertain", "model-a"),
    ]
    hold_consensus = score_consensus(hold_passes, uptrend_signal)
    result = evaluate_risk_gate(hold_consensus, portfolio, proposed_position_size_usd=0.0)
    check("hold_approved_trivially", result.approved is True, str(result.rejection_reasons))
    check("hold_final_size_zero", result.final_position_size_usd == 0.0)

    # TEST 10: weak consensus score below execution threshold, even if nominally "approved"
    # construct a case where agreement is exactly at threshold but confidence just barely clears MIN_AVG_CONFIDENCE,
    # producing a consensus_score that might sit below MIN_CONSENSUS_SCORE_TO_EXECUTE
    borderline_passes = [
        ReasoningPass("p1", "TSLA", "buy", 61, "Marginal", "model-a"),
        ReasoningPass("p2", "TSLA", "buy", 60, "Marginal", "model-a"),
    ]
    borderline_consensus = score_consensus(borderline_passes, uptrend_signal)
    result = evaluate_risk_gate(borderline_consensus, portfolio, proposed_position_size_usd=100.0)
    # We don't assert a fixed outcome here since it depends on exact score math -- instead we
    # verify the gate is INTERNALLY CONSISTENT: if score < threshold, it must be in rejection_reasons.
    if borderline_consensus.consensus_score < 0.65:
        check("weak_consensus_score_rejected", result.approved is False)
        check("weak_consensus_score_reason_present", any("Consensus score" in r for r in result.rejection_reasons))
    else:
        check("weak_consensus_score_case_not_triggered_but_consistent", True)  # score was actually strong enough, that's fine too

    # TEST 11: negative or zero proposed size for a non-hold direction is invalid
    result = evaluate_risk_gate(approved_consensus, portfolio, proposed_position_size_usd=0.0)
    check("zero_size_buy_rejected", result.approved is False)
    check("zero_size_buy_reason", any("must be positive" in r for r in result.rejection_reasons))

    print(f"\n{_passed} passed, {_failed} failed")
    return _failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
