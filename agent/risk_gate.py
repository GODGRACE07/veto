"""
Risk Gate
=========
This is the last checkpoint before a trade reaches execution. It is
deliberately dumb and deterministic -- no LLM involvement whatsoever.
This mirrors Bitget's own NightDesk design pattern from their S1
showcase: "AI provides analysis only and cannot place orders on its
own." The consensus scorer decides WHAT the agent wants to do; this
module decides whether that's SAFE to actually do, using fixed,
auditable rules.

A trade must pass BOTH the consensus scorer AND this risk gate to
execute. Passing one without the other is not enough -- this is
intentional defense in depth.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from agent.schemas import ConsensusResult, Direction


@dataclass
class PortfolioState:
    """
    Read-only snapshot of account state at decision time.
    In production this would come directly from Agent Hub / the
    exchange API, never from LLM-generated text -- treating account
    state as something an LLM could "remember" incorrectly is a
    known failure mode (see README's Layer A / Layer B distinction).
    """
    cash_balance: float
    positions: dict[str, float]   # ticker -> quantity currently held (0 if none)
    total_portfolio_value: float
    open_orders_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskGateResult:
    ticker: str
    direction: Direction
    proposed_position_size_usd: float
    approved: bool
    rejection_reasons: list[str]
    final_position_size_usd: float  # 0 if rejected, else the (possibly clipped) size actually sent to execution
    checked_at: str

    def to_dict(self) -> dict:
        return asdict(self)


# Fixed, auditable limits. Named constants so they can be cited directly
# in the write-up and are trivial to justify/adjust in one place.
MAX_SINGLE_POSITION_USD = 2000.0          # hard cap on any single trade's dollar size
MAX_POSITION_PCT_OF_PORTFOLIO = 0.20      # no single ticker position exceeds 20% of total portfolio value
MAX_OPEN_ORDERS = 5                       # refuse new trades if too many orders already open (operational safety)
MIN_CASH_BUFFER_USD = 200.0               # always keep at least this much cash uncommitted
MIN_CONSENSUS_SCORE_TO_EXECUTE = 0.65     # even an "approved" consensus result needs a strong enough score


def evaluate_risk_gate(
    consensus_result: ConsensusResult,
    portfolio: PortfolioState,
    proposed_position_size_usd: float,
) -> RiskGateResult:
    """
    Applies fixed risk rules on top of an already-scored consensus result.
    Note: even a consensus_result.approved == True can still be rejected
    here -- consensus approval means "the reasoning is trustworthy", not
    "it is safe to size this trade the way it was proposed."
    """
    reasons: list[str] = []
    ticker = consensus_result.ticker
    direction = consensus_result.majority_direction

    # Gate 0: never execute a trade the consensus scorer itself rejected.
    # This is a hard stop -- there is no risk-sizing rule that can override
    # a failed consensus check. Defense in depth means both gates must pass.
    if not consensus_result.approved:
        reasons.append(
            f"Consensus scorer rejected this trade (reasons: {'; '.join(consensus_result.rejection_reasons)}). "
            f"Risk gate does not override consensus rejections."
        )

    # Gate 1: consensus score strength, even if nominally "approved"
    if consensus_result.consensus_score < MIN_CONSENSUS_SCORE_TO_EXECUTE:
        reasons.append(
            f"Consensus score {consensus_result.consensus_score:.3f} is below the "
            f"execution threshold of {MIN_CONSENSUS_SCORE_TO_EXECUTE}."
        )

    # Gate 2: hold direction never needs sizing/position checks -- there's nothing to execute
    if direction == "hold":
        # Hold always "passes" the remaining gates trivially -- there is no order to place.
        approved = len(reasons) == 0
        return RiskGateResult(
            ticker=ticker,
            direction=direction,
            proposed_position_size_usd=0.0,
            approved=approved,
            rejection_reasons=reasons,
            final_position_size_usd=0.0,
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    if proposed_position_size_usd <= 0:
        reasons.append(f"Proposed position size must be positive for a '{direction}' order, got {proposed_position_size_usd}.")

    # Gate 3: absolute dollar cap
    clipped_size = proposed_position_size_usd
    if proposed_position_size_usd > MAX_SINGLE_POSITION_USD:
        reasons.append(
            f"Proposed size ${proposed_position_size_usd:,.2f} exceeds the max single-position "
            f"cap of ${MAX_SINGLE_POSITION_USD:,.2f}."
        )
        clipped_size = min(clipped_size, MAX_SINGLE_POSITION_USD)

    # Gate 4: percentage-of-portfolio cap
    if portfolio.total_portfolio_value > 0:
        implied_pct = proposed_position_size_usd / portfolio.total_portfolio_value
        if implied_pct > MAX_POSITION_PCT_OF_PORTFOLIO:
            reasons.append(
                f"Proposed size is {implied_pct:.1%} of total portfolio value, exceeding the "
                f"{MAX_POSITION_PCT_OF_PORTFOLIO:.0%} single-position cap."
            )
            clipped_size = min(clipped_size, portfolio.total_portfolio_value * MAX_POSITION_PCT_OF_PORTFOLIO)

    # Gate 5: cash buffer -- must have enough cash left over AFTER this trade
    if direction == "buy":
        remaining_cash_after = portfolio.cash_balance - proposed_position_size_usd
        if remaining_cash_after < MIN_CASH_BUFFER_USD:
            reasons.append(
                f"Executing this buy would leave ${remaining_cash_after:,.2f} cash, "
                f"below the required buffer of ${MIN_CASH_BUFFER_USD:,.2f}."
            )

    # Gate 6: cannot sell more than currently held
    if direction == "sell":
        current_holding = portfolio.positions.get(ticker, 0.0)
        if current_holding <= 0:
            reasons.append(f"Cannot sell {ticker}: no existing position (current holding = {current_holding}).")

    # Gate 7: too many open orders already -- operational safety valve
    if portfolio.open_orders_count >= MAX_OPEN_ORDERS:
        reasons.append(
            f"Already have {portfolio.open_orders_count} open orders, at or above the "
            f"max of {MAX_OPEN_ORDERS}. Refusing new orders until some resolve."
        )

    approved = len(reasons) == 0
    final_size = clipped_size if approved else 0.0

    return RiskGateResult(
        ticker=ticker,
        direction=direction,
        proposed_position_size_usd=proposed_position_size_usd,
        approved=approved,
        rejection_reasons=reasons,
        final_position_size_usd=round(final_size, 2),
        checked_at=datetime.now(timezone.utc).isoformat(),
    )
