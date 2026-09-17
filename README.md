# Veto

**Most AI trading agents can't tell you when they're unreliable. Veto refuses to trade the moment it disagrees with itself — and shows you the rejection, not just the win.**

Built for the Bitget AI Hackathon S2 — Agentic Trading track, "Earnings-driven trading Agent" sub-theme.

## What this is

Veto is a risk-gated earnings-trading agent for tokenized U.S. stocks. When an
earnings event fires, it does NOT just ask an LLM once and act on the answer.
Instead:

1. It asks the same question **multiple times, independently** (different
   prompts/temperatures/framings)
2. A **separate, deterministic, non-LLM module** reads the actual price data
   and computes its own read of market state (trend, moving averages,
   volatility)
3. A **consensus scorer** checks whether the LLM passes agree with each
   other AND whether their conclusion matches what the price data actually
   shows
4. Only trades that pass BOTH checks reach a **deterministic risk gate**
   (position limits, cash buffers, order caps)
5. Approved trades execute as **real paper trades** — no real money
6. Every decision (approved or rejected) is logged with full rationale —
   the rejection log is a first-class product output, not a hidden failure
7. Trade outcomes are hidden from the agent's own memory until enough real
   time has passed (**embargoed memory**) — this prevents the agent from
   ever "cheating" by seeing the future

## Why this exists (the research behind it)

Two real findings from 2026 research motivate this design:

- A field-wide survey of 92 LLM trading agent papers (arXiv:2605.19337)
  found that **zero of the reproducible studies could be independently
  verified** — the field has a severe reproducibility crisis, and almost
  no papers report transaction costs or handle survivorship bias properly.
- A Harvard study (arXiv:2603.22567, "TrustTrade") found that LLM trading
  agents left alone show **weak decision convergence** — their intermediate
  reasoning frequently contradicts their final action — and that a
  multi-agent consensus mechanism anchored to deterministic price signals
  measurably improves stability and reduces erratic risk-taking.

Veto is a direct, concrete implementation of that fix, scoped to one real
problem: tokenized U.S. stocks trade 24/7, but earnings drop overnight when
no human is watching.

## Architecture

```
Earnings event fires
        |
        +---> Deterministic Signal Module (signals.py)
        |     [no LLM -- pure price-data math]
        |
        +---> Multi-Pass LLM Reasoning (llm_client.py)
        |     [2-3 independent calls, different framings]
        |
        v
   Consensus Scorer (consensus.py)
   - Do the passes agree with each other?
   - Does the majority direction match the deterministic signal?
        |
        v
   Risk Gate (risk_gate.py)
   - Position size limits, cash buffer, order caps
   - Cannot override a consensus rejection
        |
        +---> APPROVED --> Paper trade execution (execution.py, via Bitget Agent Hub)
        |
        +---> REJECTED --> Logged with full reason (this is a product output, not a failure)
        |
        v
   Embargoed Memory (memory.py)
   - Decision stored immediately
   - Outcome hidden until embargo_hours have genuinely passed
        |
        v
   Dashboard (dashboard/)
   - Trades taken, trades rejected + why
   - Veto vs. naive-single-LLM-baseline comparison
```

All of `signals.py`, `consensus.py`, `risk_gate.py`, `memory.py`, and
`orchestrator.py` are pure/offline and have zero network dependencies —
they are fully unit-tested (119 passing tests) without any API key.
Only `llm_client.py` (Groq calls) and `data_client.py` (yfinance calls)
touch the network.

## Project structure

```
veto/
├── agent/
│   ├── __init__.py
│   ├── config.py          # all tunable constants in one place
│   ├── schemas.py         # ReasoningPass, ConsensusResult, etc.
│   ├── signals.py         # deterministic price signal (no LLM)
│   ├── consensus.py       # multi-pass agreement + deterministic check
│   ├── risk_gate.py       # final deterministic safety checks
│   ├── memory.py          # embargoed decision storage
│   ├── orchestrator.py    # wires everything into one pipeline
│   ├── llm_client.py      # Groq API calls (needs GROQ_API_KEY)
│   ├── data_client.py     # yfinance price data (needs internet, no key)
│   └── execution.py       # paper-trade execution via Bitget Agent Hub
├── tests/
│   ├── test_signals.py
│   ├── test_consensus.py
│   ├── test_risk_gate.py
│   ├── test_memory.py
│   ├── test_orchestrator.py
│   ├── test_llm_client.py
│   ├── test_data_client.py
│   └── test_execution.py
├── dashboard/              # (build in next phase)
├── requirements.txt
├── run_all_tests.py
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt --break-system-packages
```
(drop `--break-system-packages` if you're using a virtual environment,
which is recommended: `python3 -m venv venv && source venv/bin/activate`)

### 2. Get a free Groq API key
Go to [console.groq.com](https://console.groq.com), sign up, create an API
key. Then:
```bash
export GROQ_API_KEY="your-key-here"
```

### 3. Get a Bitget Agent Hub demo API key
1. Create/log into a Bitget account at bitget.com
2. Account settings → API Management → create a new key
3. Restrict permissions to read + paper trading only (never withdrawal)
4. Clone/install `github.com/Bitget-AI/agent_hub` per its README
5. Set the required environment variables per that repo's instructions
6. Verify it connects using their CLI's `--paper-trading` flag before
   wiring it into this project's `execution.py` (to be added)

### 4. Run the test suite
```bash
python3 run_all_tests.py
```
You should see `TOTAL: 133 passed, 0 failed`. This runs entirely offline —
no API keys needed for this step.

### 5. Run a real decision cycle end-to-end
See `examples/run_live_example.py` (to be added) for a script that pulls
real price data, calls Groq for real reasoning passes, and runs the full
orchestrator pipeline against live data.

## Design choices worth knowing about

- **Scoped to 5 tickers** (TSLA, NVDA, AAPL, MSFT, AMZN), not hundreds.
  Narrow and deep is stronger evidence than wide and shallow — see
  `agent/config.py` for the reasoning.
- **Rejected trades are logged with full rationale**, never discarded.
  The rejection log is as much a product deliverable as the trade log.
- **The embargo is enforced at both the write layer AND the read layer**
  (see `memory.py`) — even if an outcome is correctly backfilled, querying
  from a timestamp still inside the embargo window will not surface it.
  This is deliberate defense-in-depth against lookahead leakage.
- **Every numeric threshold lives in `config.py` or as a named constant**
  at the top of its module — nothing is a magic number buried in logic.

## What this is NOT

- Not a general-purpose trading bot — scoped to earnings events on a
  handful of tickers, by design
- Not trading real funds — paper trading only
- Not a claim that this makes money reliably — the honest, evidenced claim
  is that it is measurably more stable and auditable than a naive
  single-LLM agent, which the test suite directly demonstrates
  (see `tests/test_orchestrator.py`, Scenario 2)

## License

MIT (or your preferred choice — set before public submission)
