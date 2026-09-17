"""
Tests for the consensus scorer -- the core differentiator of this project.
Run with: python3 tests/test_consensus.py
(kept dependency-free from pytest so it runs in any environment with pandas/numpy)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from agent.signals import compute_deterministic_signal
from agent.schemas import ReasoningPass
from agent.consensus import score_consensus, score_naive_baseline

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


def run_all():
    uptrend_prices = [100 + i * 1.5 for i in range(25)]
    uptrend_signal = compute_deterministic_signal("TSLA", make_price_series(uptrend_prices))

    passes_unanimous_buy = [
        ReasoningPass("pass_1", "TSLA", "buy", 85, "Strong earnings beat, raising guidance", "model-a"),
        ReasoningPass("pass_2", "TSLA", "buy", 80, "Revenue growth accelerating", "model-a"),
        ReasoningPass("pass_3", "TSLA", "buy", 90, "Beat on EPS and revenue both", "model-a"),
    ]
    result = score_consensus(passes_unanimous_buy, uptrend_signal)
    check("unanimous_high_confidence_approved", result.approved is True, str(result.rejection_reasons))
    check("unanimous_agreement_ratio_is_1", result.agreement_ratio == 1.0)
    check("unanimous_majority_is_buy", result.majority_direction == "buy")

    passes_split = [
        ReasoningPass("pass_1", "TSLA", "buy", 85, "Earnings beat", "model-a"),
        ReasoningPass("pass_2", "TSLA", "buy", 82, "Guidance raise", "model-a"),
        ReasoningPass("pass_3", "TSLA", "sell", 70, "Valuation concerns", "model-a"),
    ]
    result = score_consensus(passes_split, uptrend_signal)
    check("split_2of3_meets_agreement_threshold", abs(result.agreement_ratio - 0.6667) < 0.001)
    check("split_2of3_still_approved_if_confidence_high", result.approved is True, str(result.rejection_reasons))

    passes_disagree = [
        ReasoningPass("pass_1", "TSLA", "buy", 85, "Bullish read", "model-a"),
        ReasoningPass("pass_2", "TSLA", "sell", 80, "Bearish read", "model-a"),
    ]
    result = score_consensus(passes_disagree, uptrend_signal)
    check("even_split_rejected", result.approved is False)
    check("even_split_has_low_agreement_reason", any("Low agreement" in r for r in result.rejection_reasons))

    downtrend_prices = [200 - i * 1.5 for i in range(25)]
    downtrend_signal = compute_deterministic_signal("TSLA", make_price_series(downtrend_prices))
    passes_buy_vs_downtrend = [
        ReasoningPass("pass_1", "TSLA", "buy", 90, "News looks great", "model-a"),
        ReasoningPass("pass_2", "TSLA", "buy", 88, "Sentiment very positive", "model-a"),
        ReasoningPass("pass_3", "TSLA", "buy", 92, "Should rally", "model-a"),
    ]
    result = score_consensus(passes_buy_vs_downtrend, downtrend_signal)
    check("unanimous_buy_vs_downtrend_rejected", result.approved is False)
    check("unanimous_buy_vs_downtrend_reason", any("Deterministic signal disagreement" in r for r in result.rejection_reasons))
    check("unanimous_buy_vs_downtrend_det_agrees_false", result.deterministic_signal_agrees is False)

    passes_low_conf = [
        ReasoningPass("pass_1", "TSLA", "buy", 40, "Weak signal, not sure", "model-a"),
        ReasoningPass("pass_2", "TSLA", "buy", 35, "Marginal case", "model-a"),
        ReasoningPass("pass_3", "TSLA", "buy", 45, "Could go either way", "model-a"),
    ]
    result = score_consensus(passes_low_conf, uptrend_signal)
    check("low_confidence_rejected_despite_agreement", result.approved is False)
    check("low_confidence_reason_present", any("Low confidence" in r for r in result.rejection_reasons))
    check("low_confidence_agreement_still_full", result.agreement_ratio == 1.0)

    passes_hold = [
        ReasoningPass("pass_1", "TSLA", "hold", 70, "Wait for more clarity", "model-a"),
        ReasoningPass("pass_2", "TSLA", "hold", 65, "Too much uncertainty", "model-a"),
    ]
    result = score_consensus(passes_hold, downtrend_signal)
    check("hold_can_be_approved_on_downtrend", result.deterministic_signal_agrees is True)

    try:
        score_consensus([passes_unanimous_buy[0]], uptrend_signal)
        check("single_pass_raises", False)
    except ValueError as e:
        check("single_pass_raises", "at least" in str(e))

    mismatched = [
        ReasoningPass("pass_1", "TSLA", "buy", 85, "x", "model-a"),
        ReasoningPass("pass_2", "NVDA", "buy", 85, "y", "model-a"),
    ]
    try:
        score_consensus(mismatched, uptrend_signal)
        check("mismatched_ticker_raises", False)
    except ValueError as e:
        check("mismatched_ticker_raises", "same ticker" in str(e))

    nvda_prices = [100 + i for i in range(25)]
    nvda_signal = compute_deterministic_signal("NVDA", make_price_series(nvda_prices))
    try:
        score_consensus(passes_unanimous_buy, nvda_signal)
        check("signal_ticker_mismatch_raises", False)
    except ValueError as e:
        check("signal_ticker_mismatch_raises", "does not match" in str(e))

    result = score_consensus(passes_disagree, uptrend_signal)
    check("rejected_trade_retains_rationale", all(len(p.raw_rationale) > 0 for p in result.passes))
    check("rejected_trade_serializes_fully", "passes" in result.to_dict())

    baseline = score_naive_baseline(passes_buy_vs_downtrend)
    check("naive_baseline_would_execute_despite_downtrend", baseline["would_execute"] is True)
    check("naive_baseline_direction_is_first_pass", baseline["direction"] == "buy")

    print(f"\n{_passed} passed, {_failed} failed")
    return _failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
