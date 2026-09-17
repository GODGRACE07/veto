#!/usr/bin/env python3
"""
Seeds the memory store with demo decision cycles so you can see the
dashboard working WITHOUT needing GROQ_API_KEY or Bitget credentials yet.

IMPORTANT: this does NOT fake the dashboard's numbers. It runs the real
orchestrator (real consensus scoring, real risk gate, real memory
embargo logic) -- the only thing that's fabricated is the INPUT
(the reasoning passes' text and confidence values, since we don't have
a live LLM call yet). Every chart/metric in the dashboard is computed
for real from these inputs, same as it would be from live data.

Clearly label this as demo data in your submission -- do not present
this as real paper-trading evidence. Once you have GROQ_API_KEY,
delete the storage/ folder and run examples/run_live_example.py
repeatedly instead to build up real evidence.

Usage: python3 dashboard/seed_demo_data.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import shutil
import pandas as pd
from datetime import datetime, timezone, timedelta

from agent.schemas import ReasoningPass
from agent.risk_gate import PortfolioState
from agent.memory import EmbargoedMemoryStore
from agent.orchestrator import run_decision_cycle
from agent.config import STORAGE_DIR, MEMORY_EMBARGO_HOURS


def make_price_series(prices: list[float], end_date: str) -> pd.DataFrame:
    dates = pd.date_range(end=end_date, periods=len(prices), freq="D")
    return pd.DataFrame({"date": dates, "close": prices})


def main():
    print("Seeding demo data into the memory store...")
    print(f"Storage directory: {STORAGE_DIR}")

    if os.path.exists(STORAGE_DIR):
        confirm = input(f"'{STORAGE_DIR}' already exists. Delete and reseed? [y/N]: ")
        if confirm.lower() != "y":
            print("Aborted.")
            return
        shutil.rmtree(STORAGE_DIR)

    store = EmbargoedMemoryStore(STORAGE_DIR, embargo_hours=MEMORY_EMBARGO_HOURS)
    # MSFT position pre-seeded so the "unanimous sell, confirmed by downtrend" demo
    # scenario has real shares to sell against -- otherwise the risk gate correctly
    # (and this is a FEATURE, not a bug) refuses to sell a position that doesn't exist.
    portfolio = PortfolioState(
        cash_balance=10000.0,
        positions={"MSFT": 5.0},
        total_portfolio_value=10000.0,
        open_orders_count=0,
    )

    base_date = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    scenarios = []

    # Scenario A: TSLA, unanimous buy, uptrend confirms -> approved, executes
    scenarios.append(dict(
        ticker="TSLA",
        prices=[240 + i * 1.8 for i in range(25)],
        passes=[
            ReasoningPass("pass_a", "TSLA", "buy", 88, "Q3 earnings beat estimates on both revenue and EPS.", "groq:llama-3.3-70b@T0.2"),
            ReasoningPass("pass_b", "TSLA", "buy", 85, "Guidance raised for next quarter, margin expansion noted.", "groq:llama-3.3-70b@T0.7"),
            ReasoningPass("pass_c", "TSLA", "buy", 90, "Delivery numbers strong, demand indicators positive.", "groq:mixtral-8x7b@T0.4"),
        ],
        day_offset=0,
    ))

    # Scenario B: TSLA, unanimous buy, but price actually in DOWNTREND -> the headline demo case
    scenarios.append(dict(
        ticker="TSLA",
        prices=[300 - i * 1.8 for i in range(25)],
        passes=[
            ReasoningPass("pass_a", "TSLA", "buy", 91, "Social sentiment extremely bullish following the announcement.", "groq:llama-3.3-70b@T0.2"),
            ReasoningPass("pass_b", "TSLA", "buy", 87, "Multiple analysts raised price targets after the news.", "groq:llama-3.3-70b@T0.7"),
            ReasoningPass("pass_c", "TSLA", "buy", 93, "Headline numbers look very strong at first read.", "groq:mixtral-8x7b@T0.4"),
        ],
        day_offset=1,
    ))

    # Scenario C: NVDA, split decision -> rejected on low agreement
    scenarios.append(dict(
        ticker="NVDA",
        prices=[480 + i * 1.2 for i in range(25)],
        passes=[
            ReasoningPass("pass_a", "NVDA", "buy", 82, "Datacenter revenue segment beat expectations.", "groq:llama-3.3-70b@T0.2"),
            ReasoningPass("pass_b", "NVDA", "sell", 70, "Gross margin guidance came in below whisper numbers.", "groq:llama-3.3-70b@T0.7"),
            ReasoningPass("pass_c", "NVDA", "hold", 65, "Mixed signals, would wait for more clarity.", "groq:mixtral-8x7b@T0.4"),
        ],
        day_offset=2,
    ))

    # Scenario D: AAPL, unanimous hold -> approved trivially, no trade needed
    scenarios.append(dict(
        ticker="AAPL",
        prices=[190 + (i % 3) * 0.5 for i in range(25)],
        passes=[
            ReasoningPass("pass_a", "AAPL", "hold", 72, "Earnings roughly in line with estimates, no major surprises.", "groq:llama-3.3-70b@T0.2"),
            ReasoningPass("pass_b", "AAPL", "hold", 68, "Services growth steady but not accelerating meaningfully.", "groq:llama-3.3-70b@T0.7"),
        ],
        day_offset=3,
    ))

    # Scenario E: MSFT, unanimous sell, downtrend confirms -> approved, executes
    scenarios.append(dict(
        ticker="MSFT",
        prices=[420 - i * 1.5 for i in range(25)],
        passes=[
            ReasoningPass("pass_a", "MSFT", "sell", 84, "Azure growth decelerated more than expected.", "groq:llama-3.3-70b@T0.2"),
            ReasoningPass("pass_b", "MSFT", "sell", 80, "Capex guidance raised, pressuring near-term margins.", "groq:llama-3.3-70b@T0.7"),
            ReasoningPass("pass_c", "MSFT", "sell", 86, "Cloud segment commentary was more cautious than prior quarter.", "groq:mixtral-8x7b@T0.4"),
        ],
        day_offset=4,
    ))

    # Scenario F: AMZN, low confidence unanimous buy -> rejected on confidence threshold
    scenarios.append(dict(
        ticker="AMZN",
        prices=[185 + i * 1.0 for i in range(25)],
        passes=[
            ReasoningPass("pass_a", "AMZN", "buy", 45, "Results were okay but nothing stood out clearly.", "groq:llama-3.3-70b@T0.2"),
            ReasoningPass("pass_b", "AMZN", "buy", 40, "Slight beat but guidance was vague.", "groq:llama-3.3-70b@T0.7"),
        ],
        day_offset=5,
    ))

    results = []
    for s in scenarios:
        decided_at = base_date + timedelta(days=s["day_offset"])
        price_history = make_price_series(s["prices"], end_date=decided_at.strftime("%Y-%m-%d"))
        result = run_decision_cycle(
            ticker=s["ticker"],
            price_history=price_history,
            reasoning_passes=s["passes"],
            portfolio=portfolio,
            memory_store=store,
            proposed_position_size_usd=300.0,
            decided_at=decided_at,
        )
        results.append(result)
        status = "EXECUTED" if result.trade_would_execute else "VETOED"
        print(f"  [{s['ticker']}] {status} -- {result.consensus_result['majority_direction'].upper()} "
              f"(consensus score: {result.consensus_result['consensus_score']:.2f})")

    print(f"\nSeeded {len(results)} demo decision cycles into {STORAGE_DIR}")
    print("Run `streamlit run dashboard/app.py` to view them.")
    print("\nNOTE: this is clearly-labeled DEMO data using fabricated reasoning-pass text,")
    print("run through the REAL consensus/risk-gate/memory pipeline. Once you have")
    print("GROQ_API_KEY, delete storage/ and use examples/run_live_example.py for real evidence.")


if __name__ == "__main__":
    main()
