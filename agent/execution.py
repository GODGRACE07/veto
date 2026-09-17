"""
Execution Module
================
The ONLY module that actually places trades. Takes an already-approved
DecisionCycleResult (both consensus AND risk gate must show approved)
and submits it as a paper trade via Bitget's Agent Hub CLI (`bgc`).

This module refuses to execute anything that wasn't already approved
by both upstream gates -- it re-checks on entry rather than trusting
the caller, since a trade-execution function is the single most
dangerous place in this codebase to have a silent bug.

Verified real CLI syntax (github.com/Bitget-AI/agent_hub), confirmed
live 2026-09-12:

    bgc --paper-trading spot spot_place_order --orders '[{"symbol":
        "rTSLAUSDT","side":"buy","orderType":"limit","price":"364",
        "size":"0.01"}]'

Symbol format confirmed live: "r" + TICKER + "USDT" (e.g. "rTSLAUSDT")
is Bitget's genuine tokenized-equity "Reality"/rToken product -- 1:1
backed real stock ownership, traded 24/7. Distinct from "<TICKER>USDT"
(leveraged stock-perpetual futures, wrong instrument) and "<TICKER>on"
(Ondo Stock Tokens, different provider).

--paper-trading is a GLOBAL flag placed immediately after `bgc`,
before the module name -- confirmed via `bgc --help`.

CREDENTIALS: paper trading requires a SEPARATE Bitget Demo API key.
A live-account key with --paper-trading is rejected by Bitget with
"exchange environment is incorrect" (confirmed empirically). This
module reads BITGET_DEMO_API_KEY / BITGET_DEMO_SECRET_KEY /
BITGET_DEMO_PASSPHRASE and maps them onto the names `bgc` expects.

IMPORTANT LIMITATION, confirmed empirically 2026-09-14: Bitget's Demo
Trading environment does NOT carry Reality/rToken pairs -- both
spot_get_ticker and spot_place_order for "rTSLAUSDT" return "Parameter
rTSLAUSDT does not exist" under --paper-trading, even though the same
symbol works against the live/production endpoint. This means
execute_decision() below is correct and will work the moment Bitget's
demo sandbox adds tokenized-stock support, but currently cannot
complete against Reality symbols in practice. See agent/paper_ledger.py
for the working alternative used for actual evidence generation: it
fetches the real live rToken price (which DOES work) and simulates
the fill locally, never calling Bitget's demo order endpoint.

IMPORTANT ASSUMPTION, verify before relying on in a real run:
  - Order type is "limit" priced at the signal's current_price at
    decision time (no live order-book fetch). A production system
    would fetch a fresh quote immediately before placing the order.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import subprocess
import platform
import json
import os

from agent.orchestrator import DecisionCycleResult


class ExecutionError(Exception):
    pass


def _build_demo_environment() -> dict:
    """
    Returns an environment dict for the subprocess call that maps the
    project's demo-specific credentials onto the variable names `bgc`
    actually reads (BITGET_API_KEY / BITGET_SECRET_KEY /
    BITGET_PASSPHRASE).

    Why this indirection exists: Bitget requires a SEPARATE API key for
    Demo Trading -- a live-account key passed with --paper-trading is
    rejected with "exchange environment is incorrect" (confirmed
    empirically 2026-09-12). Keeping demo credentials under distinct
    BITGET_DEMO_* names in the user's environment means live and demo
    keys can coexist without any chance of accidentally sending a real
    order with demo intent, or vice versa.

    Raises ExecutionError with a clear message if the demo credentials
    are not configured, rather than silently falling back to live keys
    -- a silent fallback here would be the single most dangerous
    failure mode in this codebase.
    """
    demo_key = os.environ.get("BITGET_DEMO_API_KEY")
    demo_secret = os.environ.get("BITGET_DEMO_SECRET_KEY")
    demo_passphrase = os.environ.get("BITGET_DEMO_PASSPHRASE")

    missing = [
        name for name, value in [
            ("BITGET_DEMO_API_KEY", demo_key),
            ("BITGET_DEMO_SECRET_KEY", demo_secret),
            ("BITGET_DEMO_PASSPHRASE", demo_passphrase),
        ] if not value
    ]
    if missing:
        raise ExecutionError(
            f"Paper trading requires Bitget Demo API credentials, but these environment "
            f"variables are not set: {', '.join(missing)}. "
            f"Create a Demo API key at bitget.com (Futures -> Demo Trading -> API Management, "
            f"with Spot->Trade permission) and set them. "
            f"NOTE: a live-account API key will NOT work for paper trading -- Bitget rejects "
            f"it with 'exchange environment is incorrect'."
        )

    env = os.environ.copy()
    env["BITGET_API_KEY"] = demo_key
    env["BITGET_SECRET_KEY"] = demo_secret
    env["BITGET_PASSPHRASE"] = demo_passphrase
    return env


@dataclass
class ExecutionResult:
    cycle_id: str
    ticker: str
    symbol: str
    direction: str
    size_usd: float
    quantity: float
    limit_price: float
    executed: bool
    paper_trade: bool
    raw_output: str
    error: str | None
    executed_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _build_symbol(ticker: str) -> str:
    """
    Maps a ticker to its Bitget tokenized-stock (Reality/rToken) spot
    symbol. CONFIRMED against the live API on 2026-09-12:

        bgc spot spot_get_ticker --symbol rTSLAUSDT
        -> returns real ticker data (lastPr, bid/ask, volume)

    Format is "r" + TICKER + "USDT", e.g. "rTSLAUSDT" for Tesla.
    This is Bitget's genuine tokenized-equity product (Reality
    Protocol rTokens, 1:1 backed by real shares) -- NOT the same as
    a plain "<TICKER>USDT" symbol (that format is Bitget's leveraged
    stock-perpetual-futures product, a different instrument entirely,
    and would need the `futures` bgc module instead of `spot`).

    Note: Bitget's own API echoes the symbol back uppercased in
    responses (e.g. "RTSLAUSDT"), but the request itself is
    case-insensitive and "rTSLAUSDT" is accepted as sent.
    """
    return f"r{ticker.upper()}USDT"


def _compute_order_quantity(size_usd: float, limit_price: float) -> float:
    """
    Converts a USD notional size into a base-asset quantity for the
    order, given the limit price. Rounded to 4 decimal places as a
    reasonable default -- Bitget's actual per-symbol size precision
    may differ; verify via the symbol's trading rules before a real run.
    """
    if limit_price <= 0:
        raise ValueError(f"limit_price must be positive, got {limit_price}")
    return round(size_usd / limit_price, 4)


def _build_bgc_command(
    symbol: str,
    direction: str,
    quantity: float,
    limit_price: float,
    paper_trading: bool,
) -> list[str]:
    """
    Builds the real, verified `bgc spot spot_place_order` command.
    Orders are passed as a JSON array string per the documented syntax.

    IMPORTANT: --paper-trading is a GLOBAL flag and must appear
    immediately after `bgc`, before the module name -- confirmed via
    `bgc --help`:
        bgc --paper-trading spot spot_get_ticker --symbol BTCUSDT
    Placing it at the end of the command (after the module/tool/args)
    does not work; this was an actual bug in an earlier version of
    this function, caught by testing against the real CLI.
    """
    order = {
        "symbol": symbol,
        "side": direction,  # "buy" or "sell"
        "orderType": "limit",
        "price": str(limit_price),
        "size": str(quantity),
    }
    cmd = ["bgc"]
    if paper_trading:
        cmd.append("--paper-trading")
    cmd += [
        "spot", "spot_place_order",
        "--orders", json.dumps([order]),
    ]
    return cmd


def execute_decision(
    decision: DecisionCycleResult,
    paper_trading: bool = True,
    dry_run: bool = False,
) -> ExecutionResult:
    """
    Executes a trade for an already-approved decision cycle.

    Hard refuses (raises ExecutionError) if:
      - the decision's own trade_would_execute flag is False
      - direction is "hold" (nothing to execute)
      - paper_trading=False is passed -- this project only ever trades
        Bitget's Demo environment, never live/real funds.

    dry_run=True builds and prints the command without actually
    calling subprocess -- useful for verifying the exact command that
    would run before it ever touches Bitget, even the demo environment.

    NOTE: as of 2026-09-14, Bitget's Demo environment does not carry
    Reality/rToken symbols, so a non-dry-run call against a real
    ticker will currently return a BitgetApiError even though this
    function's logic is correct. See agent/paper_ledger.py for the
    working alternative used to generate real evidence in the meantime.
    """
    if not decision.trade_would_execute:
        raise ExecutionError(
            f"Refusing to execute cycle {decision.cycle_id}: trade_would_execute is False. "
            f"Consensus approved={decision.consensus_result['approved']}, "
            f"risk gate approved={decision.risk_gate_result['approved']}. "
            f"Execution only proceeds for decisions both upstream gates approved."
        )

    direction = decision.risk_gate_result["direction"]
    ticker = decision.ticker
    size_usd = decision.risk_gate_result["final_position_size_usd"]

    if direction == "hold":
        raise ExecutionError(f"Cycle {decision.cycle_id} direction is 'hold' -- nothing to execute.")

    if not paper_trading:
        raise ExecutionError(
            "This project's execution module only supports paper_trading=True. "
            "Live real-money execution is intentionally not implemented here."
        )

    limit_price = decision.deterministic_signal["current_price"]
    symbol = _build_symbol(ticker)
    quantity = _compute_order_quantity(size_usd, limit_price)

    cmd = _build_bgc_command(symbol, direction, quantity, limit_price, paper_trading)

    if dry_run:
        return ExecutionResult(
            cycle_id=decision.cycle_id,
            ticker=ticker,
            symbol=symbol,
            direction=direction,
            size_usd=size_usd,
            quantity=quantity,
            limit_price=limit_price,
            executed=False,
            paper_trade=paper_trading,
            raw_output=f"[DRY RUN] Would execute: {' '.join(cmd)}",
            error=None,
            executed_at=datetime.now(timezone.utc).isoformat(),
        )

    env = _build_demo_environment()
    use_shell = platform.system() == "Windows"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            shell=use_shell,
        )
    except FileNotFoundError:
        raise ExecutionError(
            "bgc command not found. Verify Bitget Agent Hub is installed "
            "(npm install -g bitget-client) and on your PATH."
        )
    except subprocess.TimeoutExpired:
        raise ExecutionError(f"bgc command timed out after 30s for cycle {decision.cycle_id}.")

    stderr_indicates_error = bool(result.stderr and '"ok": false' in result.stderr)

    if result.returncode != 0 or stderr_indicates_error:
        return ExecutionResult(
            cycle_id=decision.cycle_id,
            ticker=ticker,
            symbol=symbol,
            direction=direction,
            size_usd=size_usd,
            quantity=quantity,
            limit_price=limit_price,
            executed=False,
            paper_trade=paper_trading,
            raw_output=result.stdout,
            error=result.stderr or f"Command exited with code {result.returncode}",
            executed_at=datetime.now(timezone.utc).isoformat(),
        )

    return ExecutionResult(
        cycle_id=decision.cycle_id,
        ticker=ticker,
        symbol=symbol,
        direction=direction,
        size_usd=size_usd,
        quantity=quantity,
        limit_price=limit_price,
        executed=True,
        paper_trade=paper_trading,
        raw_output=result.stdout,
        error=None,
        executed_at=datetime.now(timezone.utc).isoformat(),
    )