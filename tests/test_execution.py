"""
Tests for the execution module's safety/refusal logic and real bgc
command construction. Uses dry_run=True so no real Bitget CLI call is
made. Run with: python3 tests/test_execution.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from datetime import datetime, timezone
from agent.orchestrator import DecisionCycleResult
from agent.execution import (
    execute_decision,
    ExecutionError,
    _build_bgc_command,
    _build_symbol,
    _compute_order_quantity,
    _build_demo_environment,
)

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


def make_decision(trade_would_execute=True, direction="buy", size_usd=300.0, ticker="TSLA", current_price=250.0):
    return DecisionCycleResult(
        cycle_id="test-cycle-123",
        ticker=ticker,
        deterministic_signal={"trend_label": "uptrend", "current_price": current_price},
        consensus_result={"approved": trade_would_execute, "rejection_reasons": []},
        risk_gate_result={
            "approved": trade_would_execute,
            "direction": direction,
            "final_position_size_usd": size_usd,
            "rejection_reasons": [],
        },
        naive_baseline={"direction": direction, "would_execute": True},
        memory_record={"record_id": "test-cycle-123", "outcome": None},
        trade_would_execute=trade_would_execute,
        ran_at=datetime.now(timezone.utc).isoformat(),
    )


def run_all():
    check("symbol_maps_ticker_to_rtoken", _build_symbol("TSLA") == "rTSLAUSDT")
    check("symbol_uppercases_ticker_part", _build_symbol("nvda") == "rNVDAUSDT")

    qty = _compute_order_quantity(size_usd=300.0, limit_price=250.0)
    check("quantity_computed_correctly", qty == 1.2)

    try:
        _compute_order_quantity(300.0, 0.0)
        check("zero_price_raises", False)
    except ValueError as e:
        check("zero_price_raises", "must be positive" in str(e))

    decision = make_decision(trade_would_execute=True, direction="buy", size_usd=300.0, current_price=250.0)
    result = execute_decision(decision, paper_trading=True, dry_run=True)
    check("approved_dry_run_not_executed_flag", result.executed is False)
    check("approved_dry_run_has_bgc_in_output", "bgc" in result.raw_output)
    check("approved_dry_run_has_spot_place_order", "spot_place_order" in result.raw_output)
    check("approved_dry_run_symbol_correct", result.symbol == "rTSLAUSDT")
    check("approved_dry_run_quantity_correct", result.quantity == 1.2)
    check("approved_dry_run_limit_price_correct", result.limit_price == 250.0)

    cmd = _build_bgc_command("rTSLAUSDT", "buy", 1.2, 250.0, paper_trading=True)
    check("command_starts_with_bgc", cmd[0] == "bgc")
    check("command_has_paper_trading_flag_right_after_bgc", cmd[1] == "--paper-trading")
    check("command_contains_spot_module_and_tool", "spot" in cmd and "spot_place_order" in cmd)
    check("command_has_paper_trading_flag", "--paper-trading" in cmd)
    orders_json_index = cmd.index("--orders") + 1
    orders = json.loads(cmd[orders_json_index])
    check("order_json_symbol_correct", orders[0]["symbol"] == "rTSLAUSDT")
    check("order_json_side_correct", orders[0]["side"] == "buy")
    check("order_json_price_correct", orders[0]["price"] == "250.0")
    check("order_json_size_correct", orders[0]["size"] == "1.2")
    check("order_json_type_is_limit", orders[0]["orderType"] == "limit")

    rejected_decision = make_decision(trade_would_execute=False)
    try:
        execute_decision(rejected_decision, paper_trading=True, dry_run=True)
        check("rejected_decision_raises", False)
    except ExecutionError as e:
        check("rejected_decision_raises", "Refusing to execute" in str(e))

    hold_decision = make_decision(trade_would_execute=True, direction="hold")
    try:
        execute_decision(hold_decision, paper_trading=True, dry_run=True)
        check("hold_direction_raises", False)
    except ExecutionError as e:
        check("hold_direction_raises", "nothing to execute" in str(e))

    live_attempt = make_decision(trade_would_execute=True, direction="buy")
    try:
        execute_decision(live_attempt, paper_trading=False, dry_run=True)
        check("live_trading_refused", False)
    except ExecutionError as e:
        check("live_trading_refused", "only supports paper_trading=True" in str(e))

    cmd_no_paper = _build_bgc_command("rNVDAUSDT", "sell", 0.5, 400.0, paper_trading=False)
    check("command_omits_paper_trading_flag_when_false", "--paper-trading" not in cmd_no_paper)
    check("command_no_paper_starts_with_bgc_spot", cmd_no_paper[:2] == ["bgc", "spot"])

    sell_decision = make_decision(trade_would_execute=True, direction="sell", size_usd=150.0, ticker="AAPL", current_price=190.0)
    result = execute_decision(sell_decision, paper_trading=True, dry_run=True)
    check("sell_dry_run_symbol_correct", result.symbol == "rAAPLUSDT")
    check("sell_dry_run_direction_correct", result.direction == "sell")
    expected_qty = round(150.0 / 190.0, 4)
    check("sell_dry_run_quantity_correct", result.quantity == expected_qty)

    # --- Demo credential handling tests ---
    saved = {k: os.environ.get(k) for k in
             ("BITGET_DEMO_API_KEY", "BITGET_DEMO_SECRET_KEY", "BITGET_DEMO_PASSPHRASE")}

    for k in saved:
        os.environ.pop(k, None)
    try:
        _build_demo_environment()
        check("missing_demo_creds_raises", False, "should have raised")
    except ExecutionError as e:
        check("missing_demo_creds_raises", "Demo API credentials" in str(e))
        check("missing_demo_creds_names_missing_vars", "BITGET_DEMO_API_KEY" in str(e))
        check("missing_demo_creds_warns_live_key_wont_work", "exchange environment is incorrect" in str(e))

    os.environ["BITGET_DEMO_API_KEY"] = "test-key"
    try:
        _build_demo_environment()
        check("partial_demo_creds_raises", False, "should have raised")
    except ExecutionError as e:
        check("partial_demo_creds_raises", "BITGET_DEMO_SECRET_KEY" in str(e))
        check("partial_demo_creds_omits_present_var", "BITGET_DEMO_API_KEY" not in str(e).split("not set:")[1])

    os.environ["BITGET_DEMO_API_KEY"] = "demo-key-abc"
    os.environ["BITGET_DEMO_SECRET_KEY"] = "demo-secret-abc"
    os.environ["BITGET_DEMO_PASSPHRASE"] = "demo-pass-abc"
    env = _build_demo_environment()
    check("demo_env_maps_api_key", env["BITGET_API_KEY"] == "demo-key-abc")
    check("demo_env_maps_secret_key", env["BITGET_SECRET_KEY"] == "demo-secret-abc")
    check("demo_env_maps_passphrase", env["BITGET_PASSPHRASE"] == "demo-pass-abc")
    check("demo_env_preserves_other_vars", "PATH" in env)

    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    print(f"\n{_passed} passed, {_failed} failed")
    return _failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)