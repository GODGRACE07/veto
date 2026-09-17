"""
Consensus Scorer
=================
This is the core mechanism of the whole project. Design rationale
(see README for full citations):

  - LLM trading agents, left alone, show weak "decision convergence":
    their intermediate reasoning frequently contradicts their final
    action, and different independent calls on identical evidence can
    produce different decisions (TrustTrade, Harvard, arXiv 2603.22567).

  - The fix that paper validates is cross-agent/cross-pass consensus:
    run the reasoning multiple times independently, keep only decisions
    that agree, and separately anchor everything to a deterministic,
    non-LLM signal so narrative reasoning can't drift from what price
    data actually shows.

This module implements exactly that, as a strict, auditable function:
    passes (>=2 independent LLM outputs) + deterministic signal
        -> ConsensusResult (approved/rejected + full reasoning trail)

Nothing here calls an LLM. It only scores outputs it's given. This
keeps it fully unit-testable without any API key or network access,
which matters both for hackathon judges re-running your tests AND for
catching bugs in this logic before it ever touches real trades.
"""

from __future__ import annotations
from collections import Counter

from agent.schemas import ReasoningPass, ConsensusResult, Direction
from agent.signals import DeterministicSignal, signal_agrees_with_direction


# Tunable thresholds -- deliberately named constants, not magic numbers,
# so they can be cited directly in the write-up and changed in one place.
MIN_PASSES_REQUIRED = 2
MIN_AGREEMENT_RATIO = 0.66      # at least 2 of 3 passes (or equivalent) must agree
MIN_AVG_CONFIDENCE = 60         # majority passes must average at least this self-reported confidence
REQUIRE_DETERMINISTIC_AGREEMENT = True  # if True, disagreement with price signal is an automatic reject


def score_consensus(
    passes: list[ReasoningPass],
    deterministic_signal: DeterministicSignal,
) -> ConsensusResult:
    if len(passes) < MIN_PASSES_REQUIRED:
        raise ValueError(
            f"Consensus scoring requires at least {MIN_PASSES_REQUIRED} independent passes, "
            f"got {len(passes)}. A single LLM call cannot produce a consensus by definition."
        )

    tickers = {p.ticker for p in passes}
    if len(tickers) != 1:
        raise ValueError(f"All passes must be for the same ticker, got: {tickers}")
    ticker = passes[0].ticker

    if deterministic_signal.ticker != ticker:
        raise ValueError(
            f"Deterministic signal ticker '{deterministic_signal.ticker}' does not match "
            f"passes ticker '{ticker}'. Refusing to score mismatched evidence."
        )

    rejection_reasons: list[str] = []

    # Step 1: majority direction across passes
    direction_counts = Counter(p.direction for p in passes)
    majority_direction, majority_count = direction_counts.most_common(1)[0]
    agreement_ratio = majority_count / len(passes)

    if agreement_ratio < MIN_AGREEMENT_RATIO:
        rejection_reasons.append(
            f"Low agreement: only {majority_count}/{len(passes)} passes "
            f"({agreement_ratio:.0%}) agreed on '{majority_direction}', "
            f"below the {MIN_AGREEMENT_RATIO:.0%} threshold."
        )

    # Step 2: average self-reported confidence among the majority-agreeing passes only
    majority_passes = [p for p in passes if p.direction == majority_direction]
    avg_confidence = sum(p.confidence for p in majority_passes) / len(majority_passes)

    if avg_confidence < MIN_AVG_CONFIDENCE:
        rejection_reasons.append(
            f"Low confidence: majority-agreeing passes averaged {avg_confidence:.1f} "
            f"confidence, below the {MIN_AVG_CONFIDENCE} threshold."
        )

    # Step 3: does the deterministic, non-LLM signal support this direction?
    det_agrees = signal_agrees_with_direction(deterministic_signal, majority_direction)
    if REQUIRE_DETERMINISTIC_AGREEMENT and not det_agrees:
        rejection_reasons.append(
            f"Deterministic signal disagreement: majority direction '{majority_direction}' "
            f"is not supported by price data (trend_label={deterministic_signal.trend_label}, "
            f"bullish_confidence={deterministic_signal.bullish_confidence}, "
            f"bearish_confidence={deterministic_signal.bearish_confidence}). "
            f"The LLM's reasoning does not match what the price data actually shows."
        )

    # Combined consensus score -- simple weighted average, fully transparent.
    # Weighted: agreement matters most, then confidence, then deterministic backing.
    det_component = 1.0 if det_agrees else 0.0
    consensus_score = (
        0.45 * agreement_ratio
        + 0.30 * (avg_confidence / 100)
        + 0.25 * det_component
    )

    approved = len(rejection_reasons) == 0

    return ConsensusResult(
        ticker=ticker,
        passes=passes,
        majority_direction=majority_direction,
        agreement_ratio=round(agreement_ratio, 4),
        avg_confidence_of_majority=round(avg_confidence, 2),
        deterministic_signal_agrees=det_agrees,
        consensus_score=round(consensus_score, 4),
        approved=approved,
        rejection_reasons=rejection_reasons,
    )


def score_naive_baseline(passes: list[ReasoningPass]) -> dict:
    """
    Implements the comparison baseline: what a naive single-LLM agent would do
    -- i.e., just take the FIRST pass at face value, no consensus, no
    deterministic check. This exists specifically to produce the
    "decision convergence: consensus-gated vs naive baseline" comparison
    metric for the dashboard. It is intentionally simplistic -- that's
    the point, it represents what most other hackathon entries will build.
    """
    if not passes:
        raise ValueError("Cannot score baseline on empty passes list")
    first = passes[0]
    return {
        "ticker": first.ticker,
        "direction": first.direction,
        "confidence": first.confidence,
        "would_execute": first.confidence >= 50,  # naive threshold, no other checks
        "note": "Naive baseline: single LLM call, no consensus, no deterministic check.",
    }
