"""
Tests for the deterministic signal module.
Run with: python3 tests/test_signals.py
(dependency-free -- no pytest required, only pandas/numpy)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from agent.signals import compute_deterministic_signal, signal_agrees_with_direction

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
    try:
        df = make_price_series([100, 101, 102])
        compute_deterministic_signal("TEST", df)
        check("insufficient_history_raises", False)
    except ValueError as e:
        check("insufficient_history_raises", "Insufficient history" in str(e))

    prices = [100 + i * 1.5 for i in range(25)]
    sig = compute_deterministic_signal("TEST", make_price_series(prices))
    check("uptrend_detected", sig.trend_label in ("uptrend", "strong_uptrend"))
    check("uptrend_price_above_ma5", sig.price_above_ma5 is True)
    check("uptrend_bullish_gt_bearish", sig.bullish_confidence > sig.bearish_confidence)

    prices = [200 - i * 1.5 for i in range(25)]
    sig = compute_deterministic_signal("TEST", make_price_series(prices))
    check("downtrend_detected", sig.trend_label in ("downtrend", "strong_downtrend"))
    check("downtrend_bearish_gt_bullish", sig.bearish_confidence > sig.bullish_confidence)

    prices = [100.0] * 25
    sig = compute_deterministic_signal("TEST", make_price_series(prices))
    check("flat_is_sideways", sig.trend_label == "sideways")
    check("flat_zero_volatility", sig.volatility_20d == 0.0)

    prices = [100 + np.sin(i / 3) * 5 + i * 0.3 for i in range(30)]
    df = make_price_series(prices)
    sig1 = compute_deterministic_signal("TEST", df)
    sig2 = compute_deterministic_signal("TEST", df)
    d1, d2 = sig1.to_dict(), sig2.to_dict()
    d1.pop("as_of")
    d2.pop("as_of")
    check("determinism_same_input_same_output", d1 == d2)

    prices = [100 + i for i in range(40)]
    full_df = make_price_series(prices)
    truncated_df = full_df.iloc[:25].reset_index(drop=True)
    sig = compute_deterministic_signal("TEST", truncated_df)
    check("no_lookahead_uses_truncated_last_row", sig.current_price == truncated_df["close"].iloc[-1])
    check("no_lookahead_differs_from_full", sig.current_price != full_df["close"].iloc[-1])

    rng = np.random.RandomState(42)
    calm = [100 + i * 0.5 for i in range(25)]
    noisy = [100 + i * 0.5 + rng.normal(0, 5) for i in range(25)]
    sig_calm = compute_deterministic_signal("CALM", make_price_series(calm))
    sig_noisy = compute_deterministic_signal("NOISY", make_price_series(noisy))
    check("volatility_increases_with_noise", sig_noisy.volatility_20d > sig_calm.volatility_20d)

    prices = [100 - i for i in range(25)]
    sig = compute_deterministic_signal("TEST", make_price_series(prices))
    check("buy_disagrees_with_downtrend", signal_agrees_with_direction(sig, "buy") is False)
    check("sell_agrees_with_downtrend", signal_agrees_with_direction(sig, "sell") is True)

    prices = [100 + i for i in range(25)]
    sig = compute_deterministic_signal("TEST", make_price_series(prices))
    check("hold_always_agrees", signal_agrees_with_direction(sig, "hold") is True)

    prices = [100 + i * 2 for i in range(25)]
    sig = compute_deterministic_signal("TEST", make_price_series(prices))
    check("bullish_confidence_bounded", 0 <= sig.bullish_confidence <= 100)
    check("bearish_confidence_bounded", 0 <= sig.bearish_confidence <= 100)

    print(f"\n{_passed} passed, {_failed} failed")
    return _failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
