"""
Veto Dashboard
==============
Reads directly from the embargoed memory store's JSON files on disk.
Requires NO API keys to run -- it only visualizes decisions that were
already recorded by the orchestrator.

Run with:
    streamlit run dashboard/app.py

Theme: light, white/silver background with a bright cyan-blue accent.
Tables are rendered as custom HTML, NOT st.dataframe -- Streamlit's
built-in dataframe widget is a canvas-based grid that only picks up
theme colors from .streamlit/config.toml at process startup, and
CSS cannot override it. Rendering tables as plain HTML instead gives
full, guaranteed control over colors regardless of Streamlit's theme
state, and lets real per-row logos render inline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import html as html_lib
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from agent.memory import EmbargoedMemoryStore
from agent.config import STORAGE_DIR, TRACKED_TICKERS


# ============================================================
# THEME
# ============================================================

BLUE = "#0EA5E9"
BLUE_DEEP = "#0284C7"
BLUE_GLOW = "rgba(14, 165, 233, 0.22)"
BG_PAGE = "#EAF4FC"
BG_CARD = "#FFFFFF"
SILVER = "#E2E6EB"
SILVER_DEEP = "#C7CDD6"
TEXT_PRIMARY = "#0F172A"
TEXT_MUTED = "#64748B"
GREEN = "#059669"
GREEN_BG = "rgba(5, 150, 105, 0.10)"
RED = "#DC2626"
RED_BG = "rgba(220, 38, 38, 0.08)"
BORDER = "#E2E6EB"
SHADOW = "0 2px 12px rgba(15, 23, 42, 0.06)"
SHADOW_HOVER = "0 8px 28px rgba(14, 165, 233, 0.18)"

TICKER_BRAND = {
    "TSLA": {"slug": "tesla", "color": "#E31937", "mono": "TS", "name": "Tesla"},
    "NVDA": {"slug": "nvidia", "color": "#76B900", "mono": "NV", "name": "NVIDIA"},
    "AAPL": {"slug": "apple", "color": "#555555", "mono": "AP", "name": "Apple"},
    "MSFT": {"slug": "microsoft", "color": "#00A4EF", "mono": "MS", "name": "Microsoft"},
    "AMZN": {"slug": "amazon", "color": "#FF9900", "mono": "AZ", "name": "Amazon"},
}

st.set_page_config(page_title="Veto — Bitget AI Hackathon S2", page_icon="🛑", layout="wide")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    .stApp {{ background: {BG_PAGE}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .main .block-container {{ padding-top: 1.5rem; max-width: 1200px; }}

    /* ---- Hero ---- */
    .veto-hero {{
        position: relative;
        padding: 2.4rem 2.2rem;
        border-radius: 20px;
        margin-bottom: 1.4rem;
        background: linear-gradient(120deg, #FFFFFF 0%, #EEF3F8 45%, #FFFFFF 100%);
        background-size: 220% 220%;
        animation: heroShift 7s ease-in-out infinite;
        border: 1px solid {SILVER};
        box-shadow: {SHADOW};
        overflow: hidden;
    }}
    .veto-hero::before {{
        content: ""; position: absolute; top: -70%; left: -25%;
        width: 60%; height: 240%;
        background: radial-gradient(circle, {BLUE_GLOW} 0%, transparent 70%);
        animation: sweep 5.5s ease-in-out infinite; pointer-events: none;
    }}
    .veto-hero::after {{
        content: ""; position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, transparent, {BLUE}, transparent);
        background-size: 200% 100%; animation: scanline 3s linear infinite;
    }}
    @keyframes heroShift {{ 0%,100% {{background-position:0% 50%;}} 50% {{background-position:100% 50%;}} }}
    @keyframes sweep {{ 0%,100% {{transform:translateX(0) translateY(0);}} 50% {{transform:translateX(90%) translateY(30%);}} }}
    @keyframes scanline {{ 0% {{background-position:200% 0;}} 100% {{background-position:-200% 0;}} }}
    .veto-hero-top {{ display: flex; align-items: center; gap: 0.9rem; position: relative; z-index: 1; }}
    .veto-icon {{
        width: 50px; height: 50px; border-radius: 14px;
        background: linear-gradient(135deg, {BLUE_DEEP}, {BLUE});
        display: flex; align-items: center; justify-content: center;
        font-size: 1.7rem; box-shadow: 0 4px 16px {BLUE_GLOW};
        animation: iconPulse 2.4s ease-in-out infinite;
    }}
    @keyframes iconPulse {{ 0%,100% {{box-shadow:0 4px 16px {BLUE_GLOW};}} 50% {{box-shadow:0 6px 28px rgba(14,165,233,0.4);}} }}
    .veto-hero h1 {{ font-size: 2.4rem; font-weight: 900; margin: 0; color: {TEXT_PRIMARY}; letter-spacing: -0.7px; }}
    .veto-live {{
        display: inline-flex; align-items: center; gap: 0.4rem; margin-left: 0.6rem;
        padding: 0.2rem 0.7rem; border-radius: 20px; background: {GREEN_BG};
        border: 1px solid {GREEN}; font-size: 0.7rem; font-weight: 800; color: {GREEN};
        letter-spacing: 0.5px; vertical-align: middle;
    }}
    .veto-live-dot {{ width: 7px; height: 7px; border-radius: 50%; background: {GREEN}; animation: livePulse 1.4s ease-in-out infinite; }}
    @keyframes livePulse {{ 0%,100% {{opacity:1; transform:scale(1);}} 50% {{opacity:0.4; transform:scale(0.8);}} }}
    .veto-hero p {{ font-size: 1.08rem; color: {TEXT_MUTED}; max-width: 850px; margin: 0.9rem 0 0 0; position: relative; z-index: 1; line-height: 1.6; font-weight: 500; }}

    /* ---- Entrance animation ---- */
    [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stVerticalBlock"] > div {{ animation: fadeInUp 0.45s ease-out backwards; }}
    @keyframes fadeInUp {{ from {{opacity:0; transform:translateY(10px);}} to {{opacity:1; transform:translateY(0);}} }}

    /* ---- Problem / Solution ---- */
    .veto-explain-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }}
    .veto-explain-card {{
        background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 16px;
        padding: 1.4rem 1.6rem; box-shadow: {SHADOW};
    }}
    .veto-explain-card h3 {{ margin: 0 0 0.6rem 0 !important; font-size: 1.1rem !important; }}
    .veto-explain-card p {{ margin: 0; line-height: 1.6; font-size: 0.95rem; }}
    .veto-explain-card.problem {{ border-left: 4px solid {RED}; }}
    .veto-explain-card.solution {{ border-left: 4px solid {GREEN}; }}
    .veto-steps {{ display: flex; gap: 1rem; margin-bottom: 1.6rem; flex-wrap: wrap; }}
    .veto-step {{
        flex: 1; min-width: 220px; background: {BG_CARD}; border: 1px solid {BORDER};
        border-radius: 14px; padding: 1.1rem 1.2rem; box-shadow: {SHADOW};
        display: flex; gap: 0.8rem; align-items: flex-start;
    }}
    .veto-step-num {{
        width: 30px; height: 30px; border-radius: 50%; background: {BLUE}; color: white;
        font-weight: 800; font-size: 0.9rem; display: flex; align-items: center;
        justify-content: center; flex-shrink: 0;
    }}
    .veto-step h4 {{ margin: 0 0 0.2rem 0; font-size: 0.92rem; color: {TEXT_PRIMARY}; font-weight: 800; }}
    .veto-step p {{ margin: 0; font-size: 0.82rem; color: {TEXT_MUTED}; line-height: 1.4; }}

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {{
        background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 1.1rem 1.3rem; box-shadow: {SHADOW}; transition: all 0.25s ease;
    }}
    [data-testid="stMetric"]:hover {{ border-color: {BLUE}; box-shadow: {SHADOW_HOVER}; transform: translateY(-3px); }}
    [data-testid="stMetricValue"] {{ color: {BLUE_DEEP} !important; font-weight: 900 !important; }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; font-weight: 600 !important; }}

    h2, h3 {{ color: {TEXT_PRIMARY} !important; font-weight: 800 !important; }}
    h2 {{ border-left: 4px solid {BLUE}; padding-left: 0.8rem; }}
    p, .stMarkdown, [data-testid="stCaptionContainer"] {{ color: {TEXT_MUTED}; }}

    [data-testid="stExpander"] {{
        background: {BG_CARD}; border: 1px solid {BORDER} !important; border-radius: 12px;
        box-shadow: {SHADOW}; transition: all 0.2s ease;
    }}
    [data-testid="stExpander"]:hover {{ border-color: {BLUE} !important; box-shadow: {SHADOW_HOVER}; }}
    [data-testid="stExpander"] summary {{ color: {TEXT_PRIMARY} !important; font-weight: 700 !important; }}

    .stButton > button {{
        background: linear-gradient(90deg, {BLUE_DEEP}, {BLUE}); color: #FFFFFF; font-weight: 700;
        border: none; border-radius: 10px; box-shadow: {SHADOW}; transition: all 0.2s ease;
    }}
    .stButton > button:hover {{ box-shadow: {SHADOW_HOVER}; transform: translateY(-1px); }}
    hr {{ border-color: {BORDER} !important; }}

    .veto-badge {{
        display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 800; letter-spacing: 0.5px;
    }}
    .veto-badge-buy {{ background: {GREEN_BG}; color: {GREEN}; border: 1px solid {GREEN}; }}
    .veto-badge-sell {{ background: {RED_BG}; color: {RED}; border: 1px solid {RED}; }}
    .veto-badge-hold {{ background: {SILVER}; color: {TEXT_MUTED}; border: 1px solid {SILVER_DEEP}; }}

    .veto-footer {{ text-align: center; color: {TEXT_MUTED}; font-size: 0.85rem; padding: 1.5rem 0 0.5rem 0; font-weight: 500; }}
    [data-baseweb="select"] {{ border-color: {BORDER} !important; }}

    /* ---- Custom HTML tables (replaces st.dataframe for guaranteed light theme) ---- */
    .veto-table-wrap {{
        overflow-x: auto; border: 1px solid {BORDER}; border-radius: 12px;
        box-shadow: {SHADOW}; background: {BG_CARD}; margin-bottom: 0.5rem;
    }}
    table.veto-table {{ border-collapse: collapse; width: 100%; font-size: 0.84rem; }}
    table.veto-table th {{
        background: {BG_PAGE}; color: {TEXT_PRIMARY}; text-align: left;
        padding: 0.6rem 0.8rem; font-weight: 700; border-bottom: 2px solid {BORDER};
        white-space: nowrap; position: sticky; top: 0;
    }}
    table.veto-table td {{
        padding: 0.55rem 0.8rem; border-bottom: 1px solid {BORDER};
        color: {TEXT_PRIMARY}; white-space: nowrap; max-width: 260px;
        overflow: hidden; text-overflow: ellipsis;
    }}
    table.veto-table td.veto-wrap-cell {{
        white-space: normal; max-width: 420px; min-width: 280px;
        overflow: visible; text-overflow: clip; line-height: 1.4;
    }}
    table.veto-table tr:last-child td {{ border-bottom: none; }}
    table.veto-table tr:hover td {{ background: {BG_PAGE}; }}
    .veto-row-logo {{
        width: 24px; height: 24px; border-radius: 6px; object-fit: contain; display: block;
    }}
    .veto-row-logo-wrap {{ width: 24px; height: 24px; position: relative; }}
    .veto-row-logo-fallback {{
        width: 24px; height: 24px; border-radius: 50%; color: #fff; font-size: 0.6rem;
        font-weight: 800; align-items: center; justify-content: center;
    }}
    .veto-check {{ color: {GREEN}; font-weight: 800; }}
    .veto-cross {{ color: {TEXT_MUTED}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def direction_badge(direction: str) -> str:
    cls = f"veto-badge-{direction.lower()}"
    return f'<span class="veto-badge {cls}">{direction.upper()}</span>'


def combined_reasons(consensus_result: dict, risk_gate_result: dict) -> list[str]:
    """
    Combines consensus + risk-gate rejection reasons, filtering out the
    risk gate's own restatement of the consensus reasons ("Consensus
    scorer rejected this trade (reasons: ...)") -- that message exists
    for programmatic/log clarity (see risk_gate.py's Gate 0), but in
    the dashboard it duplicates text already shown from
    consensus_result.rejection_reasons verbatim. Filtering it here
    keeps every genuinely distinct reason while removing the repeat.
    """
    reasons = list(consensus_result["rejection_reasons"])
    for r in risk_gate_result["rejection_reasons"]:
        if r.startswith("Consensus scorer rejected this trade"):
            continue
        reasons.append(r)
    return reasons


def logo_cell_html(ticker: str) -> str:
    brand = TICKER_BRAND.get(ticker, {"slug": "", "color": BLUE, "mono": ticker[:2]})
    return f"""<div class="veto-row-logo-wrap">
        <img class="veto-row-logo" src="https://s3-symbol-logo.tradingview.com/{brand['slug']}--big.svg"
             onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />
        <div class="veto-row-logo-fallback" style="background:{brand['color']}; display:none;">{brand['mono']}</div>
    </div>"""


def render_table(df: pd.DataFrame, show_logo_for: str = None, bool_cols: list = None, max_rows: int = None):
    """
    Renders a DataFrame as plain HTML -- guaranteed light theme, real
    logos inline, works regardless of Streamlit's internal canvas
    theme state. show_logo_for: column name containing the ticker to
    render a logo for, as the first column. bool_cols: column names
    to render as check/cross instead of True/False text.
    """
    bool_cols = bool_cols or []
    display = df.head(max_rows) if max_rows else df

    header_cells = ""
    if show_logo_for:
        header_cells += "<th></th>"
    for col in display.columns:
        header_cells += f"<th>{html_lib.escape(str(col))}</th>"

    rows_html = ""
    for _, row in display.iterrows():
        cells = ""
        if show_logo_for:
            cells += f"<td>{logo_cell_html(str(row[show_logo_for]))}</td>"
        for col in display.columns:
            val = row[col]
            td_class = ""
            if col in bool_cols:
                cell = '<span class="veto-check">&#10003;</span>' if val else '<span class="veto-cross">&mdash;</span>'
            else:
                text = str(val)
                cell = html_lib.escape(text)
                if col == "rejection_reasons":
                    td_class = ' class="veto-wrap-cell"'
            cells += f"<td{td_class}>{cell}</td>"
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(
        f'<div class="veto-table-wrap"><table class="veto-table"><thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="veto-hero">
        <div class="veto-hero-top">
            <div class="veto-icon">🛑</div>
            <h1>Veto</h1>
            <span class="veto-live"><span class="veto-live-dot"></span>LIVE DATA</span>
        </div>
        <p>Most AI trading agents can't tell you when they're unreliable. Veto refuses to
        trade the moment it disagrees with itself — and shows you the rejection, not just
        the win.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PROBLEM / SOLUTION -- for judges skimming in seconds
# ============================================================

st.markdown(
    f"""
    <div class="veto-explain-grid">
        <div class="veto-explain-card problem">
            <h3>🔴 The problem</h3>
            <p>Tokenized U.S. stocks trade 24/7 on Bitget, but earnings news drops overnight
            when no human is watching. The obvious fix — an AI agent that trades for you —
            has its own unsolved problem: left alone, it can confidently act on a reading
            that contradicts the actual market, with no warning.</p>
        </div>
        <div class="veto-explain-card solution">
            <h3>🟢 The solution</h3>
            <p>Veto asks the same question three times, independently, and checks the answer
            against real, non-AI price data before it ever acts. If the AI disagrees with
            itself, or with the market, Veto refuses the trade — and shows you exactly why,
            instead of hiding the disagreement.</p>
        </div>
    </div>

    <div class="veto-steps">
        <div class="veto-step">
            <div class="veto-step-num">1</div>
            <div><h4>Ask 3 times</h4><p>Independent LLM reasoning passes, different framings, on the same real evidence.</p></div>
        </div>
        <div class="veto-step">
            <div class="veto-step-num">2</div>
            <div><h4>Check the market</h4><p>A separate, deterministic module checks the answer against real price data.</p></div>
        </div>
        <div class="veto-step">
            <div class="veto-step-num">3</div>
            <div><h4>Approve or Veto</h4><p>Only trades that survive both checks execute. Every rejection is logged, not hidden.</p></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

store = EmbargoedMemoryStore(STORAGE_DIR)

# --- Load all records across all tracked tickers ---
all_records = []
for ticker in TRACKED_TICKERS:
    records = store.get_all_records_unfiltered(ticker)
    all_records.extend(records)

if not all_records:
    st.warning(
        "No decision records found yet. Run `python examples/run_live_example.py TICKER` "
        "(needs GROQ_API_KEY) to generate real records, or run "
        "`python dashboard/seed_demo_data.py` for demo data."
    )
    st.stop()

# --- Build a flat DataFrame for display ---
rows = []
for r in all_records:
    consensus = r.consensus_result
    risk = r.risk_gate_result
    would_execute = consensus["approved"] and risk["approved"]
    rows.append({
        "record_id": r.record_id[:8],
        "ticker": r.ticker,
        "decided_at": r.decided_at,
        "majority_direction": consensus["majority_direction"],
        "agreement_ratio": consensus["agreement_ratio"],
        "consensus_score": consensus["consensus_score"],
        "det_signal_agrees": consensus["deterministic_signal_agrees"],
        "would_execute": would_execute,
        "rejection_reasons": "; ".join(combined_reasons(consensus, risk)) or "—",
    })

df = pd.DataFrame(rows).sort_values("decided_at", ascending=False)

# --- Top-line metrics ---
col1, col2, col3, col4 = st.columns(4)
total = len(df)
executed = df["would_execute"].sum()
rejected = total - executed
reject_rate = rejected / total if total > 0 else 0

col1.metric("Total decisions", total)
col2.metric("Would execute", int(executed))
col3.metric("Vetoed (rejected)", int(rejected))
col4.metric("Veto rate", f"{reject_rate:.0%}")

st.divider()

# --- Veto vs naive baseline ---
st.subheader("Veto vs. naive single-LLM baseline")
st.caption(
    "For each decision, this compares what Veto actually decided against what a naive agent "
    "(single LLM call, no consensus check, no deterministic-signal check) would have done."
)

naive_rows = []
for r in all_records:
    consensus = r.consensus_result
    risk = r.risk_gate_result
    would_execute = consensus["approved"] and risk["approved"]
    passes = consensus["passes"]
    first_pass = passes[0] if passes else None
    naive_would_execute = first_pass["confidence"] >= 50 if first_pass else False
    naive_rows.append({
        "ticker": r.ticker,
        "record_id": r.record_id[:8],
        "veto_would_execute": would_execute,
        "naive_would_execute": naive_would_execute,
        "diverged": would_execute != naive_would_execute,
    })

naive_df = pd.DataFrame(naive_rows)
divergence_count = naive_df["diverged"].sum()
divergence_rate = divergence_count / len(naive_df) if len(naive_df) > 0 else 0

dcol1, dcol2 = st.columns(2)
dcol1.metric("Decisions where Veto diverged from naive baseline", int(divergence_count))
dcol2.metric("Divergence rate", f"{divergence_rate:.0%}")

if divergence_count > 0:
    st.caption(
        "Every divergence row below represents a case where a naive single-LLM agent "
        "would have taken a different action than Veto — most often, a case where the LLM "
        "was confident but Veto's deterministic check caught a disagreement with actual price data."
    )
    diverged_df = naive_df[naive_df["diverged"]][["ticker", "record_id", "veto_would_execute", "naive_would_execute"]]
    render_table(diverged_df, show_logo_for="ticker", bool_cols=["veto_would_execute", "naive_would_execute"])

st.divider()

# --- Full decision log ---
st.subheader("Full decision log")
st.caption("Every decision Veto made — trades executed AND trades rejected, with full reasoning.")

filter_ticker = st.multiselect("Filter by ticker", options=TRACKED_TICKERS, default=[])
display_df = df if not filter_ticker else df[df["ticker"].isin(filter_ticker)]

render_table(
    display_df[[
        "ticker", "record_id", "decided_at", "majority_direction",
        "agreement_ratio", "consensus_score", "det_signal_agrees",
        "would_execute", "rejection_reasons",
    ]],
    show_logo_for="ticker",
    bool_cols=["det_signal_agrees", "would_execute"],
)

st.divider()

# --- Rejected trades detail view ---
st.subheader("Rejected trades — full rationale")
st.caption(
    "The rejection log is a first-class product output, not a hidden failure. "
    "Every rejected trade's full LLM rationale is preserved here for audit."
)

rejected_records = [r for r in all_records if not (r.consensus_result["approved"] and r.risk_gate_result["approved"])]

if not rejected_records:
    st.info("No rejected trades yet.")
else:
    for r in rejected_records[:10]:
        with st.expander(f"{r.ticker} — {r.decided_at} — {r.consensus_result['majority_direction'].upper()}"):
            logo_col, badge_col = st.columns([1, 11])
            with logo_col:
                st.markdown(logo_cell_html(r.ticker), unsafe_allow_html=True)
            with badge_col:
                st.markdown(direction_badge(r.consensus_result["majority_direction"]), unsafe_allow_html=True)
            all_reasons = combined_reasons(r.consensus_result, r.risk_gate_result)
            reasons_md = "\n".join(f"- {reason}" for reason in all_reasons)
            st.markdown("**Rejection reasons:**")
            st.markdown(reasons_md)

            passes_md_lines = []
            for p in r.consensus_result["passes"]:
                line = (
                    f"- `{p['pass_id']}` ({p['model_name']}): **{p['direction'].upper()}** "
                    f"(confidence {p['confidence']}) -- {p['raw_rationale']}"
                )
                passes_md_lines.append(line)
            passes_md = "\n".join(passes_md_lines)
            st.markdown("**Individual reasoning passes:**")
            st.markdown(passes_md)

st.divider()

# --- Paper Trading Ledger ---
st.subheader("Paper trading ledger — real fills, real live prices")
st.caption(
    "Every fill below used a REAL, live rToken market price fetched from Bitget's public "
    "market data at the moment of the fill — not synthetic or estimated. Fills are simulated "
    "locally (Bitget's Demo Trading sandbox does not currently carry Reality/rToken pairs), "
    "but the price input and every number below is genuine. Only trades that got APPROVED "
    "above are eligible to be filled here — run `python examples/backfill_ledger.py` to fill "
    "every approved decision that hasn't been filled yet."
)

try:
    from agent.paper_ledger import PaperLedger

    ledger = PaperLedger("./storage/paper_ledger")
    account_state = ledger.get_account_state()
    all_fills = ledger.get_all_fills()

    lcol1, lcol2, lcol3 = st.columns(3)
    lcol1.metric("Paper cash balance", f"${account_state.cash_balance:,.2f}")
    lcol2.metric("Total fills", len(all_fills))
    total_position_value_note = ", ".join(
        f"{qty} {ticker}" for ticker, qty in account_state.positions.items() if qty > 0
    ) or "none"
    lcol3.metric("Open positions", len([q for q in account_state.positions.values() if q > 0]))

    if account_state.positions:
        st.caption(f"Current holdings: {total_position_value_note}")

    if not all_fills:
        st.info(
            "No fills yet. Run `python examples/backfill_ledger.py` to fill every approved "
            "decision currently in storage, or fill one manually via agent/paper_ledger.py's "
            "fill_decision()."
        )
    else:
        fill_rows = []
        for f in all_fills:
            fill_rows.append({
                "ticker": f.ticker,
                "fill_id": f.fill_id[:12],
                "symbol": f.symbol,
                "direction": f.direction,
                "quantity": f.quantity,
                "fill_price": f.fill_price,
                "notional_usd": f.notional_usd,
                "cash_before": f.cash_balance_before,
                "cash_after": f.cash_balance_after,
                "filled_at": f.filled_at,
                "price_source": f.price_source,
            })
        fill_df = pd.DataFrame(fill_rows).sort_values("filled_at", ascending=False)
        render_table(fill_df, show_logo_for="ticker")

except ImportError:
    st.warning("Paper ledger module not found — run this from the project root.")
except Exception as e:
    st.info(f"No paper ledger data yet, or it could not be loaded: {e}")

st.markdown(
    '<div class="veto-footer">Built for the Bitget AI Hackathon S2 — Agentic Trading track. '
    'No real funds involved; paper trading only.</div>',
    unsafe_allow_html=True,
)