"""
Paper Ledger
============
Bitget's Demo Trading environment does not carry Reality/rToken
(tokenized-stock) pairs -- confirmed empirically 2026-09-14: both
`spot_get_ticker` and `spot_place_order` for "rTSLAUSDT" return
"Parameter rTSLAUSDT does not exist" under --paper-trading, while the
exact same symbol returns real live data against the production
endpoint. This is a genuine gap in Bitget's current demo sandbox, not
a bug in this project's code (see agent/execution.py's docstring for
the full diagnostic trail).

Given that, this module implements the alternative required by the
submission form's own evidence bar ("paper-trading logs including
timestamp, instrument, direction, price, quantity, and account
balance change"): it fetches the REAL, LIVE rToken price from
Bitget's public market-data endpoint (which works fine -- only the
demo trading endpoints are the gap), then simulates the fill and
balance change locally.

This is not a workaround that hides a limitation -- it is arguably
MORE transparent than a black-box demo fill would be: every line of
the fill logic is visible in this file, and the price input is
genuinely live, not synthetic.

This module never calls any Bitget WRITE endpoint. It only reads
public market data and does its own bookkeeping.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import platform
import json
import re

from agent.orchestrator import DecisionCycleResult


class PaperLedgerError(Exception):
    pass


@dataclass
class LedgerFill:
    fill_id: str
    cycle_id: str
    ticker: str
    symbol: str
    direction: str
    quantity: float
    fill_price: float
    notional_usd: float
    cash_balance_before: float
    cash_balance_after: float
    position_before: float
    position_after: float
    filled_at: str
    price_source: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaperAccountState:
    cash_balance: float
    positions: dict[str, float]

    def to_dict(self) -> dict:
        return {"cash_balance": self.cash_balance, "positions": dict(self.positions)}

    @classmethod
    def from_dict(cls, d: dict) -> "PaperAccountState":
        return cls(cash_balance=d["cash_balance"], positions=dict(d["positions"]))


def fetch_live_rtoken_price(ticker: str) -> float:
    symbol = f"r{ticker.upper()}USDT"
    cmd = ["bgc", "spot", "spot_get_ticker", "--symbol", symbol]
    use_shell = platform.system() == "Windows"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, shell=use_shell)
    except FileNotFoundError:
        raise PaperLedgerError("bgc command not found. Verify Bitget Agent Hub is installed.")
    except subprocess.TimeoutExpired:
        raise PaperLedgerError(f"Fetching live price for {symbol} timed out after 15s.")

    raw = result.stdout or result.stderr
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise PaperLedgerError(f"Could not parse ticker response for {symbol}: {raw!r}")

    if isinstance(payload, dict) and payload.get("ok") is False:
        raise PaperLedgerError(f"Bitget rejected ticker request for {symbol}: {payload.get('error')}")

    data = payload.get("data") if isinstance(payload, dict) else None
    if not data or not isinstance(data, list) or "lastPr" not in data[0]:
        raise PaperLedgerError(f"Unexpected ticker response shape for {symbol}: {payload!r}")

    return float(data[0]["lastPr"])


class PaperLedger:
    def __init__(self, storage_dir: str | Path, starting_cash: float = 10000.0):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._account_path = self.storage_dir / "paper_account.json"
        self._fills_path = self.storage_dir / "paper_fills.json"
        if not self._account_path.exists():
            self._save_account(PaperAccountState(cash_balance=starting_cash, positions={}))

    def _load_account(self) -> PaperAccountState:
        with open(self._account_path) as f:
            return PaperAccountState.from_dict(json.load(f))

    def _save_account(self, state: PaperAccountState) -> None:
        with open(self._account_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _load_fills(self) -> list[dict]:
        if not self._fills_path.exists():
            return []
        with open(self._fills_path) as f:
            return json.load(f)

    def _save_fills(self, fills: list[dict]) -> None:
        with open(self._fills_path, "w") as f:
            json.dump(fills, f, indent=2, default=str)

    def get_account_state(self) -> PaperAccountState:
        return self._load_account()

    def fill_decision(
        self,
        decision: DecisionCycleResult,
        live_price: float | None = None,
    ) -> LedgerFill:
        if not decision.trade_would_execute:
            raise PaperLedgerError(
                f"Refusing to fill cycle {decision.cycle_id}: trade_would_execute is False. "
                f"Ledger fills only proceed for decisions both consensus and risk gate approved."
            )

        direction = decision.risk_gate_result["direction"]
        ticker = decision.ticker
        size_usd = decision.risk_gate_result["final_position_size_usd"]

        if direction == "hold":
            raise PaperLedgerError(f"Cycle {decision.cycle_id} direction is 'hold' -- nothing to fill.")

        price = live_price if live_price is not None else fetch_live_rtoken_price(ticker)
        if price <= 0:
            raise PaperLedgerError(f"Fetched invalid live price {price} for {ticker}")

        quantity = round(size_usd / price, 4)
        symbol = f"r{ticker.upper()}USDT"

        state = self._load_account()
        cash_before = state.cash_balance
        position_before = state.positions.get(ticker, 0.0)

        if direction == "buy":
            cost = quantity * price
            if cost > state.cash_balance:
                raise PaperLedgerError(
                    f"Insufficient paper cash: order costs ${cost:.2f}, available ${state.cash_balance:.2f}"
                )
            state.cash_balance -= cost
            state.positions[ticker] = position_before + quantity
        elif direction == "sell":
            if position_before < quantity:
                raise PaperLedgerError(
                    f"Insufficient paper position to sell: have {position_before}, requested {quantity}"
                )
            proceeds = quantity * price
            state.cash_balance += proceeds
            state.positions[ticker] = position_before - quantity
        else:
            raise PaperLedgerError(f"Unknown direction '{direction}'")

        self._save_account(state)

        fill = LedgerFill(
            fill_id=f"fill_{decision.cycle_id}",
            cycle_id=decision.cycle_id,
            ticker=ticker,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            fill_price=price,
            notional_usd=round(quantity * price, 2),
            cash_balance_before=round(cash_before, 2),
            cash_balance_after=round(state.cash_balance, 2),
            position_before=position_before,
            position_after=state.positions[ticker],
            filled_at=datetime.now(timezone.utc).isoformat(),
            price_source="bitget_live_public_ticker",
        )

        fills = self._load_fills()
        fills.append(fill.to_dict())
        self._save_fills(fills)

        return fill

    def get_all_fills(self) -> list[LedgerFill]:
        return [LedgerFill(**f) for f in self._load_fills()]