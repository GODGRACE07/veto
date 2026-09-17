"""
Veto -- Live Interactive Demo
==============================
This page lets a visitor actually RUN the pipeline, not just view past
results. It calls a real LLM (Groq) and real price data (yfinance) live,
using a server-side API key (via Streamlit secrets) so the visitor never
needs their own key.

Run alongside or instead of app.py:
    streamlit run dashboard/live_demo.py

SETUP REQUIRED BEFORE DEPLOYING PUBLICLY:
1. Add your Groq key to Streamlit secrets (NOT hardcoded, NOT committed
   to git). Locally: create .streamlit/secrets.toml with:
       GROQ_API_KEY = "your-key-here"
   On Streamlit Community Cloud: Settings -> Secrets, paste the same.
2. This page enforces a simple per-session rate limit (see MAX_RUNS_PER_SESSION
   below) to protect your Groq free-tier quota from being exhausted by
   public traffic. Adjust the limit based on your actual quota.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from datetime import datetime, timezone

from agent.data_client import fetch_price_history, summarize_signal_for_prompt, DataClientError
from agent.signals import compute_deterministic_signal
from agent.llm_client import get_reasoning_passes, LLMClientError
from agent.risk_gate import PortfolioState
from agent.memory import EmbargoedMemoryStore
from agent.orchestrator import run_decision_cycle
from agent.config import STORAGE_DIR, MEMORY_EMBARGO_HOURS, DEFAULT_TRADE_SIZE_USD, TRACKED_TICKERS


st.set_page_config(page_title="Veto -- Live Demo", page_icon="🛑", layout="wide")

MAX_RUNS_PER_SESSION = 5  # protects your Groq free-tier quota from public traffic; raise if your quota allows

st.title("🛑 Veto -- Live Demo")
st.caption(
    "This page runs the REAL pipeline live: real price data, real independent LLM calls, "
    "real consensus scoring. Nothing here is pre-recorded."
)

# --- Server-side API key setup ---
groq_key_available = False
try:
    groq_key = st.secrets.get("GROQ_API_KEY", None)
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        groq_key_available = True
except Exception:
    # st.secrets raises if no secrets.toml exists at all (e.g. fresh local clone) --
    # fall back to checking a plain environment variable instead, so local dev
    # (export GROQ_API_KEY=...) still works without requiring a secrets file.
    groq_key_available = bool(os.environ.get("GROQ_API_KEY"))

if not groq_key_available:
    st.error(
        "GROQ_API_KEY is not configured on this deployment. If you are the operator: "
        "add it to .streamlit/secrets.toml locally, or to your Streamlit Cloud app's "
        "Settings -> Secrets before deploying publicly."
    )
    st.stop()

# --- Session-level rate limit ---
if "run_count" not in st.session_state:
    st.session_state.run_count = 0

runs_left = MAX_RUNS_PER_SESSION - st.session_state.run_count
st.info(f"Live runs remaining this session: {runs_left} / {MAX_RUNS_PER_SESSION}")

st.divider()

# --- Input form ---
col1, col2 = st.columns([1, 2])

with col1:
    ticker = st.selectbox("Ticker", options=TRACKED_TICKERS)

with col2:
    evidence_text = st.text_area(
        "Earnings / news evidence",
        placeholder=(
            "Example: Company X reported Q3 revenue of $Y billion, beating estimates of $Z billion. "
            "EPS came in at $A vs expected $B. Management raised full-year guidance citing strong "
            "demand in the core segment."
        ),
        height=120,
    )

run_button = st.button("Run Decision Cycle", type="primary", disabled=(runs_left <= 0))

if runs_left <= 0:
    st.warning(
        "Session run limit reached. This protects the demo's API quota for other visitors. "
        "Refresh the page to reset (a new session), or come back later."
    )

st.divider()

# --- Run the live pipeline ---
if run_button:
    if not evidence_text.strip():
        st.error("Please enter some earnings/news evidence before running -- the LLM needs something to reason over.")
        st.stop()

    st.session_state.run_count += 1

    with st.status("Running live decision cycle...", expanded=True) as status:
        # Step 1: fetch real price data
        st.write("Fetching real price history...")
        try:
            as_of = datetime.now(timezone.utc)
            price_history = fetch_price_history(ticker, as_of=as_of, lookback_days=60)
            st.write(f"Got {len(price_history)} price rows, most recent: {price_history['date'].max().date()}")
        except DataClientError as e:
            status.update(label="Failed: could not fetch price data", state="error")
            st.error(
                f"Could not fetch live price data for {ticker}: {e}. "
                f"This can happen if the market data provider is temporarily rate-limiting requests -- try again shortly."
            )
            st.stop()

        # Step 2: deterministic signal
        st.write("Computing deterministic price signal (no LLM)...")
        try:
            signal = compute_deterministic_signal(ticker, price_history)
        except ValueError as e:
            status.update(label="Failed: insufficient price history", state="error")
            st.error(f"Could not compute signal: {e}")
            st.stop()
        signal_summary = summarize_signal_for_prompt(signal)
        st.write(f"Trend: **{signal.trend_label}** | Bullish confidence: {signal.bullish_confidence} | Bearish confidence: {signal.bearish_confidence}")

        # Step 3: real LLM reasoning passes
        st.write("Calling Groq for 3 independent reasoning passes...")
        try:
            passes = get_reasoning_passes(
                ticker=ticker,
                evidence_text=evidence_text,
                deterministic_signal_summary=signal_summary,
            )
        except LLMClientError as e:
            status.update(label="Failed: LLM call error", state="error")
            st.error(
                f"The live LLM call failed: {e}. This can happen due to a temporary rate limit "
                f"or an unexpected response format -- try again in a moment."
            )
            st.stop()

        for p in passes:
            st.write(f"`{p.pass_id}` ({p.model_name}): **{p.direction.upper()}** (confidence {p.confidence}) -- \"{p.raw_rationale}\"")

        # Step 4: full pipeline
        st.write("Running consensus scoring + risk gate + memory recording...")
        portfolio = PortfolioState(cash_balance=10000.0, positions={}, total_portfolio_value=10000.0, open_orders_count=0)
        store = EmbargoedMemoryStore(STORAGE_DIR, embargo_hours=MEMORY_EMBARGO_HOURS)

        try:
            result = run_decision_cycle(
                ticker=ticker,
                price_history=price_history,
                reasoning_passes=passes,
                portfolio=portfolio,
                memory_store=store,
                proposed_position_size_usd=DEFAULT_TRADE_SIZE_USD,
            )
        except Exception as e:
            status.update(label="Failed: pipeline error", state="error")
            st.error(f"Unexpected error running the decision pipeline: {e}")
            st.stop()

        status.update(label="Done", state="complete")

    # --- Result display ---
    st.divider()
    st.subheader("Result")

    if result.trade_would_execute:
        st.success(f"**APPROVED** -- Veto would execute a **{result.consensus_result['majority_direction'].upper()}** paper trade on {ticker}.")
    else:
        st.warning(f"**VETOED** -- Veto refused to act on this evidence.")
        all_reasons = result.consensus_result["rejection_reasons"] + result.risk_gate_result["rejection_reasons"]
        for reason in all_reasons:
            st.write(f"- {reason}")

    naive_would_execute = result.naive_baseline["would_execute"]
    if naive_would_execute != result.trade_would_execute:
        st.info(
            f"**Divergence detected:** a naive single-LLM agent (no consensus check, no price-data check) "
            f"would have {'executed' if naive_would_execute else 'held off on'} this trade. "
            f"Veto's additional checks changed the outcome."
        )

    with st.expander("Full decision detail (JSON)"):
        st.json(result.to_dict())

st.divider()
st.caption(
    "Built for the Bitget AI Hackathon S2. Paper trading only -- no real funds involved. "
    f"Outcomes are hidden from the agent's own memory for {MEMORY_EMBARGO_HOURS}h after each decision."
)
