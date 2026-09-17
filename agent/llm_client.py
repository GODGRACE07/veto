"""
LLM Client
==========
This is the ONLY module in the project that makes real network calls
to an LLM provider. Everything else (signals, consensus, risk_gate,
memory, orchestrator) is pure/offline and fully unit-tested without
this module ever running.

Uses Groq's free-tier API. Produces independent ReasoningPass objects
by varying BOTH temperature and prompt framing across calls -- not
just re-sampling the same prompt at different temperatures, since
that alone doesn't give genuine architectural diversity.

Requires: pip install groq
Requires: GROQ_API_KEY environment variable set.

If you want true cross-provider diversity (recommended, not required),
add a second provider client alongside this one and mix passes from
both in orchestrator calls -- the consensus scorer in consensus.py
doesn't care which provider produced a ReasoningPass, only that each
pass's `model_name` is recorded for the audit trail.
"""

from __future__ import annotations
import os
import json
import re

from agent.schemas import ReasoningPass, Direction
from agent.config import DEFAULT_LLM_PROVIDERS, LLMProviderConfig


class LLMClientError(Exception):
    """Raised when a reasoning pass cannot be obtained or parsed reliably."""
    pass


SYSTEM_PROMPT = """You are a financial analyst evaluating a single earnings-related trading decision for a tokenized U.S. stock.

You will be given:
- A ticker symbol
- Recent earnings/news information
- Deterministic price statistics (trend, moving averages, volatility)

Your job: decide whether the evidence supports BUY, SELL, or HOLD, and how confident you are.

Rules:
- Base your decision ONLY on the evidence provided. Do not assume information you were not given.
- If the evidence is mixed or unclear, prefer HOLD over a low-confidence directional call.
- Keep your rationale to ONE short sentence, no more than 25 words.
- Output ONLY valid JSON in this exact schema, nothing else, no markdown fences:
{"direction": "buy" | "sell" | "hold", "confidence": <integer 0-100>, "rationale": "<one short sentence, max 25 words>"}
"""


def _build_user_prompt(ticker: str, evidence_text: str, deterministic_signal_summary: str, framing: str) -> str:
    """
    `framing` varies the analytical lens across passes to produce genuine
    independence rather than just re-sampling noise. Example framings:
    "conservative risk-averse analyst", "momentum-focused analyst", etc.
    """
    return f"""Ticker: {ticker}

Earnings / news evidence:
{evidence_text}

Deterministic price statistics (independently computed, not from you):
{deterministic_signal_summary}

Analytical framing for this pass: {framing}

Provide your decision as JSON per the schema in the system prompt."""


def _parse_llm_response(raw_text: str) -> tuple[Direction, int, str]:
    """
    Strict JSON parsing with one fallback: strip markdown code fences if
    the model added them despite instructions not to. Raises LLMClientError
    on anything that doesn't cleanly parse -- we do NOT guess or default
    silently, since a silently-wrong parse is worse than a loud failure
    here (this feeds directly into consensus scoring).
    """
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMClientError(f"Could not parse LLM response as JSON: {e}. Raw response: {raw_text!r}")

    if "direction" not in data or "confidence" not in data or "rationale" not in data:
        raise LLMClientError(f"LLM response missing required fields. Got keys: {list(data.keys())}")

    direction = data["direction"].strip().lower()
    if direction not in ("buy", "sell", "hold"):
        raise LLMClientError(f"LLM returned invalid direction '{direction}', expected buy/sell/hold")

    confidence = int(data["confidence"])
    if not (0 <= confidence <= 100):
        raise LLMClientError(f"LLM returned confidence {confidence}, out of valid 0-100 range")

    rationale = str(data["rationale"]).strip()
    if len(rationale) == 0:
        raise LLMClientError("LLM returned empty rationale")

    return direction, confidence, rationale  # type: ignore[return-value]


def _call_groq(model_id: str, temperature: float, system_prompt: str, user_prompt: str) -> str:
    """
    Thin wrapper around the Groq SDK. Isolated into its own function so
    it's the only place that needs to change if you swap providers.
    """
    try:
        from groq import Groq
    except ImportError:
        raise LLMClientError(
            "The 'groq' package is not installed. Run: pip install groq --break-system-packages"
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMClientError(
            "GROQ_API_KEY environment variable is not set. "
            "Get a free key at console.groq.com and set it before running."
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        temperature=temperature,
        max_tokens=500,  # response is a short JSON object -- caps runaway output, stays under free-tier OTPM limits
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


# Analytical framings rotated across passes to add genuine independence
# beyond just temperature variation. Feel free to edit/extend these.
FRAMINGS = [
    "You are a conservative, risk-averse analyst who weighs downside risk heavily.",
    "You are a momentum-focused analyst who weighs recent price action and trend strength heavily.",
    "You are a fundamentals-focused analyst who weighs the earnings numbers themselves over market sentiment.",
]


def get_reasoning_passes(
    ticker: str,
    evidence_text: str,
    deterministic_signal_summary: str,
    providers: list[LLMProviderConfig] | None = None,
) -> list[ReasoningPass]:
    """
    Runs one independent LLM call per configured provider entry, using a
    rotating analytical framing for additional independence, and returns
    a list of ReasoningPass objects ready to feed into consensus.score_consensus().

    Raises LLMClientError if ANY pass fails to produce a parseable result --
    by design, we do not silently drop a failed pass and proceed with
    fewer passes than configured, since that would silently weaken the
    consensus check without anyone noticing. Callers who want partial-
    failure tolerance should catch this and decide explicitly.
    """
    providers = providers or DEFAULT_LLM_PROVIDERS
    passes: list[ReasoningPass] = []

    for i, provider_cfg in enumerate(providers):
        framing = FRAMINGS[i % len(FRAMINGS)]
        user_prompt = _build_user_prompt(ticker, evidence_text, deterministic_signal_summary, framing)

        if provider_cfg.provider == "groq":
            raw_response = _call_groq(
                model_id=provider_cfg.model_id,
                temperature=provider_cfg.temperature,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        else:
            raise LLMClientError(
                f"Provider '{provider_cfg.provider}' is not implemented. "
                f"Currently supported: 'groq'. Add a branch here for additional providers."
            )

        direction, confidence, rationale = _parse_llm_response(raw_response)

        passes.append(
            ReasoningPass(
                pass_id=provider_cfg.name,
                ticker=ticker,
                direction=direction,
                confidence=confidence,
                raw_rationale=rationale,
                model_name=f"{provider_cfg.provider}:{provider_cfg.model_id}@T{provider_cfg.temperature}",
            )
        )

    return passes