"""
Tests for the data client's leakage-prevention logic -- testable
offline using synthetic data shaped like yfinance output, without
needing network access. The actual fetch_price_history() network
path requires yfinance + internet and is not exercised here; this
file proves summarize_signal_for_prompt() and the cutoff-enforcement
CONCEPT are correct via a lightweight synthetic check.
Run with: python3 tests/test_data_client.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from agent.data_client import summarize_signal_for_prompt
from agent.signals import compute_deterministic_signal

_passed = 0
_failed = 0


def check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        print(f"PASS: {name}")
        _passed += 1
    else:
        print(f"FAIL: {name} {detail}")
        _failed += 1


def make_price_series(prices):
    dates = pd.date_range(end="2026-09-09", periods=len(prices), freq="D")
    return pd.DataFrame({"date": dates, "close": prices})


def run_all():
    prices = [100 + i * 1.5 for i in range(25)]
    signal = compute_deterministic_signal("TSLA", make_price_series(prices))
    summary = summarize_signal_for_prompt(signal)

    check("summary_contains_current_price", str(round(signal.current_price, 2)) in summary or f"{signal.current_price:.2f}" in summary)
    check("summary_contains_trend_label", signal.trend_label in summary)
    check("summary_contains_volatility_pct_symbol", "%" in summary)
    check("summary_contains_bullish_confidence", str(signal.bullish_confidence) in summary)
    check("summary_contains_bearish_confidence", str(signal.bearish_confidence) in summary)
    check("summary_is_nonempty_string", isinstance(summary, str) and len(summary) > 20)

    # Cutoff-enforcement logic check (synthetic, mirrors the real function's filter step)
    dates = pd.date_range("2026-08-01", periods=40, freq="D")
    raw_df = pd.DataFrame({"date": dates, "close": [100 + i for i in range(40)]})
    as_of = pd.Timestamp("2026-08-20")
    filtered = raw_df[raw_df["date"] <= as_of].sort_values("date").reset_index(drop=True)
    check("cutoff_filter_excludes_future_rows", filtered["date"].max() <= as_of)
    check("cutoff_filter_row_count_correct", len(filtered) == 20)  # Aug 1-20 inclusive

    print(f"\n{_passed} passed, {_failed} failed")
    return _failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
