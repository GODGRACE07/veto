"""
Tests for the LLM client's parsing logic -- the part that's testable
without a live API key. The actual _call_groq() network function is
NOT tested here (it requires GROQ_API_KEY and network access); this
file proves the response-parsing contract is solid, since a bad parse
would silently corrupt everything downstream in consensus scoring.
Run with: python3 tests/test_llm_client.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.llm_client import _parse_llm_response, LLMClientError, _build_user_prompt

_passed = 0
_failed = 0


def check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        print(f"PASS: {name}")
        _passed += 1
    else:
        print(f"FAIL: {name} {detail}")
        _failed += 1


def run_all():
    # TEST 1: clean valid JSON parses correctly
    clean = '{"direction": "buy", "confidence": 85, "rationale": "Strong earnings beat."}'
    direction, confidence, rationale = _parse_llm_response(clean)
    check("clean_json_direction", direction == "buy")
    check("clean_json_confidence", confidence == 85)
    check("clean_json_rationale", rationale == "Strong earnings beat.")

    # TEST 2: JSON wrapped in markdown fences (model ignored instructions) still parses
    fenced = '```json\n{"direction": "sell", "confidence": 70, "rationale": "Miss on revenue."}\n```'
    direction, confidence, rationale = _parse_llm_response(fenced)
    check("fenced_json_parses", direction == "sell" and confidence == 70)

    # TEST 3: uppercase/mixed-case direction is normalized
    mixed_case = '{"direction": "BUY", "confidence": 60, "rationale": "ok"}'
    direction, _, _ = _parse_llm_response(mixed_case)
    check("direction_normalized_lowercase", direction == "buy")

    # TEST 4: invalid direction value raises
    try:
        _parse_llm_response('{"direction": "strong_buy", "confidence": 80, "rationale": "x"}')
        check("invalid_direction_raises", False)
    except LLMClientError as e:
        check("invalid_direction_raises", "invalid direction" in str(e))

    # TEST 5: confidence out of range raises
    try:
        _parse_llm_response('{"direction": "buy", "confidence": 150, "rationale": "x"}')
        check("confidence_out_of_range_raises", False)
    except LLMClientError as e:
        check("confidence_out_of_range_raises", "out of valid" in str(e))

    # TEST 6: negative confidence raises
    try:
        _parse_llm_response('{"direction": "buy", "confidence": -5, "rationale": "x"}')
        check("negative_confidence_raises", False)
    except LLMClientError as e:
        check("negative_confidence_raises", "out of valid" in str(e))

    # TEST 7: missing required field raises
    try:
        _parse_llm_response('{"direction": "buy", "confidence": 80}')  # no rationale
        check("missing_field_raises", False)
    except LLMClientError as e:
        check("missing_field_raises", "missing required fields" in str(e))

    # TEST 8: empty rationale raises
    try:
        _parse_llm_response('{"direction": "buy", "confidence": 80, "rationale": ""}')
        check("empty_rationale_raises", False)
    except LLMClientError as e:
        check("empty_rationale_raises", "empty rationale" in str(e))

    # TEST 9: completely non-JSON garbage raises cleanly, not a crash
    try:
        _parse_llm_response("I think you should buy this stock because it looks good.")
        check("non_json_garbage_raises_cleanly", False)
    except LLMClientError as e:
        check("non_json_garbage_raises_cleanly", "Could not parse" in str(e))

    # TEST 10: confidence as a string that IS a valid int still coerces
    string_confidence = '{"direction": "hold", "confidence": "55", "rationale": "waiting"}'
    direction, confidence, rationale = _parse_llm_response(string_confidence)
    check("string_confidence_coerces_to_int", confidence == 55 and isinstance(confidence, int))

    # TEST 11: prompt builder includes all required context (sanity check, not exhaustive)
    prompt = _build_user_prompt("TSLA", "Earnings beat by 10%", "trend: uptrend, MA5: 250", "conservative analyst")
    check("prompt_includes_ticker", "TSLA" in prompt)
    check("prompt_includes_evidence", "Earnings beat by 10%" in prompt)
    check("prompt_includes_signal_summary", "trend: uptrend" in prompt)
    check("prompt_includes_framing", "conservative analyst" in prompt)

    print(f"\n{_passed} passed, {_failed} failed")
    return _failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
