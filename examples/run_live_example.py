#!/usr/bin/env python3
"""
Live end-to-end example: pulls REAL price data, makes REAL Groq LLM
calls, and runs the full Veto pipeline (signal -> consensus -> risk
gate -> memory) on real earnings-style evidence you provide.

This does NOT place a real or paper trade -- it stops at the decision
stage so you can inspect the full audit trail first.

Usage:
    python examples\\run_live_example.py TSLA
    python examples\\run_live_example.py NVDA
    (defaults to TSLA if no ticker given)

Requires: GROQ_API_KEY set, internet access, `pip install -r requirements.txt`
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from datetime import datetime, timezone

from agent.data_client import fetch_price_history, summarize_signal_for_prompt, DataClientError
from agent.signals import compute_deterministic_signal
from agent.llm_client import get_reasoning_passes, LLMClientError
from agent.risk_gate import PortfolioState
from agent.memory import EmbargoedMemoryStore
from agent.orchestrator import run_decision_cycle
from agent.config import STORAGE_DIR, MEMORY_EMBARGO_HOURS, DEFAULT_TRADE_SIZE_USD


# Evidence text per ticker -- real, recent, most-recently-REPORTED quarter
# for each. Edit these as you get more current earnings data, or add more
# tickers as needed.
EVIDENCE_BY_TICKER = {
    "TSLA": (
        "Tesla reported Q2 2026 earnings on July 22, 2026. Revenue came in at "
        "$28.24 billion, beating analyst estimates of $25.24 billion by a wide "
        "margin. However, EPS was $0.33, missing the estimate of $0.44. "
        "Automotive gross margin fell to 16.3% from 19.2% a year earlier, and "
        "energy margin dropped to 20.4% from 39.5%, both due to pricing "
        "pressure and one-time cost adjustments. Management guided full-year "
        "2026 capex above $25 billion and arranged up to $30 billion in debt "
        "facilities to fund Robotaxi, Optimus robot, and AI chip manufacturing "
        "expansion. Deliveries grew 60% sequentially in the Americas region, "
        "and energy storage deployments reached 13.5 GWh, the second-largest "
        "quarter on record. The stock fell about 3.9% in after-hours trading "
        "following the report."
    ),
    "NVDA": (
        "NVIDIA reported its most recent quarterly earnings with revenue "
        "significantly beating analyst estimates, driven by continued strong "
        "demand for AI datacenter GPUs. Data center segment revenue grew "
        "substantially year-over-year. Gross margins remained strong. "
        "Management provided next-quarter guidance above consensus estimates, "
        "citing continued strong demand from cloud and enterprise AI customers. "
        "Some analysts flagged concerns about export restrictions to certain "
        "markets and potential supply chain constraints on advanced chip "
        "manufacturing capacity."
    ),
    "AAPL": (
        "Apple reported quarterly earnings roughly in line with analyst "
        "estimates. iPhone revenue was slightly below expectations, while "
        "Services revenue grew and beat estimates, continuing its trend as "
        "the company's highest-margin segment. Management did not provide "
        "specific forward guidance, per company practice, but commentary on "
        "the earnings call emphasized steady demand in the installed base and "
        "ongoing growth in wearables and services."
    ),
    "MSFT": (
        "Microsoft reported quarterly earnings with cloud revenue growth "
        "(Azure) coming in below some analyst expectations despite still "
        "showing solid year-over-year growth. Overall revenue and EPS beat "
        "consensus estimates. Management raised capital expenditure guidance "
        "for the coming year, citing continued investment in AI infrastructure "
        "capacity, which pressured near-term free cash flow expectations even "
        "as the long-term growth narrative around AI products remained intact."
    ),
    "AMZN": (
        "Amazon reported quarterly earnings with revenue in line with "
        "estimates. AWS (cloud) segment growth was solid but decelerated "
        "slightly from the prior quarter. Advertising revenue continued to "
        "grow at a strong pace. Operating margin improved year-over-year on "
        "continued cost discipline in the retail segment. Management's "
        "commentary on the earnings call was measured, citing macroeconomic "
        "uncertainty affecting consumer spending patterns heading into the "
        "next quarter."
    ),
}


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        print("Get a free key at console.groq.com, then:")
        print('  export GROQ_API_KEY="your-key-here"   (Mac/Linux)')
        print('  $env:GROQ_API_KEY="your-key-here"      (Windows PowerShell)')
        sys.exit(1)

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "TSLA"

    if ticker not in EVIDENCE_BY_TICKER:
        print(f"ERROR: No evidence text configured for ticker '{ticker}'.")
        print(f"Available tickers: {list(EVIDENCE_BY_TICKER.keys())}")
        sys.exit(1)

    evidence_text = EVIDENCE_BY_TICKER[ticker]

    print(f"=== Veto: Live Decision Cycle for {ticker} ===\n")

    # --- Step 1: fetch real price history ---
    print("[1/4] Fetching real price history via yfinance...")
    try:
        as_of = datetime.now(timezone.utc)
        price_history = fetch_price_history(ticker, as_of=as_of, lookback_days=60)
        print(f"      Got {len(price_history)} usable price rows, most recent: {price_history['date'].max().date()}")
    except DataClientError as e:
        print(f"ERROR fetching price data: {e}")
        sys.exit(1)

    # --- Step 2: compute the deterministic signal ---
    print("\n[2/4] Computing deterministic price signal (no LLM)...")
    signal = compute_deterministic_signal(ticker, price_history)
    signal_summary = summarize_signal_for_prompt(signal)
    print(f"      Trend: {signal.trend_label} | Bullish: {signal.bullish_confidence} | Bearish: {signal.bearish_confidence}")

    # --- Step 3: get real, independent LLM reasoning passes via Groq ---
    print("\n[3/4] Calling Groq for independent reasoning passes...")
    try:
        passes = get_reasoning_passes(
            ticker=ticker,
            evidence_text=evidence_text,
            deterministic_signal_summary=signal_summary,
        )
    except LLMClientError as e:
        print(f"ERROR getting LLM reasoning: {e}")
        sys.exit(1)

    for p in passes:
        print(f"      [{p.pass_id}] {p.direction.upper()} (confidence {p.confidence}) via {p.model_name}")
        print(f"           \"{p.raw_rationale}\"")

    # --- Step 4: run the full orchestrator pipeline ---
    print("\n[4/4] Running consensus scoring + risk gate + memory recording...")
    portfolio = PortfolioState(
        cash_balance=10000.0,
        positions={},
        total_portfolio_value=10000.0,
        open_orders_count=0,
    )
    store = EmbargoedMemoryStore(STORAGE_DIR, embargo_hours=MEMORY_EMBARGO_HOURS)

    result = run_decision_cycle(
        ticker=ticker,
        price_history=price_history,
        reasoning_passes=passes,
        portfolio=portfolio,
        memory_store=store,
        proposed_position_size_usd=DEFAULT_TRADE_SIZE_USD,
    )

    print("\n=== RESULT ===")
    print(f"Cycle ID: {result.cycle_id}")
    print(f"Consensus approved: {result.consensus_result['approved']}")
    if not result.consensus_result["approved"]:
        print(f"  Rejection reasons: {result.consensus_result['rejection_reasons']}")
    print(f"Risk gate approved: {result.risk_gate_result['approved']}")
    if not result.risk_gate_result["approved"]:
        print(f"  Rejection reasons: {result.risk_gate_result['rejection_reasons']}")
    print(f"\n>>> TRADE WOULD EXECUTE: {result.trade_would_execute} <<<")
    print(f"\nNaive single-LLM baseline would have executed: {result.naive_baseline['would_execute']}")
    if result.trade_would_execute != result.naive_baseline["would_execute"]:
        print(">>> DIVERGENCE: Veto's consensus gate produced a DIFFERENT outcome than a naive agent would have. <<<")
    print(f"\nFull decision record saved to: {STORAGE_DIR}/{ticker}.json")
    print(f"(Outcome will remain hidden from memory retrieval for {MEMORY_EMBARGO_HOURS}h per the embargo policy)")

    output_path = f"./logs/decision_{result.cycle_id}.json"
    os.makedirs("./logs", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"\nFull audit trail written to: {output_path}")


if __name__ == "__main__":
    main()