"""Catalog section A: tool-execution faults.

Four tests per fault, as the phase requires: `apply()` in isolation, the trigger,
the mutation log (including `apply(json_patch, before) == after`), and an
end-to-end run against a fake agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine, Limits
from agent_loop_chaos.context import (
    Counters,
    Crossing,
    FaultContext,
    RunContext,
)
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.jsonpatch import apply as apply_patch
from agent_loop_chaos.targeting import Trigger
from agent_loop_chaos.trace import TraceRecorder


def engine(tmp_path: Path, **kw: Any) -> ChaosEngine:
    """Build an engine writing into a temp directory, validating every trace event.

    Args:
        tmp_path: pytest's temp directory.
        **kw: Constructor overrides.

    Returns:
        The engine.
    """
    kw.setdefault("seed", 1337)
    kw.setdefault("out_dir", tmp_path / ".chaos")
    kw.setdefault("strict_trace", True)
    return ChaosEngine(**kw)


def fault_context(seed: int = 1337, *, key: str = "abc123") -> FaultContext:
    """Build a `FaultContext` for calling `apply()` in isolation.

    Args:
        seed: The run seed.
        key: The fault's spec hash, which keys its RNG stream.

    Returns:
        The context.
    """
    ctx = RunContext(
        run_id="run-test",
        seed=seed,
        started_at="1970-01-01T00:00:00Z",
        limits=Limits(),
        trace=TraceRecorder("run-test"),
        counters=Counters(),
    )
    return FaultContext(
        fault_id="f1", fault_key=key, run=ctx, limits=ctx.limits, counters=ctx.counters
    )


def post(result: Any, *, name: str = "get_weather", call_index: int = 1) -> Crossing:
    """Build a `(tool, post)` crossing.

    Args:
        result: What the tool returned.
        name: The tool name.
        call_index: Which call this is.

    Returns:
        The crossing.
    """
    return Crossing(layer="tool", phase="post", name=name, result=result, call_index=call_index)


def events(result: Any) -> list[dict[str, Any]]:
    """Load the trace a run wrote.

    Args:
        result: The run's `ChaosResult`.

    Returns:
        The trace events.
    """
    return TraceRecorder.load(result.artifacts["trace"])


# ============================================================ A1 ToolCorruptionFault


def test_corruption_applies_the_named_mutation() -> None:
    """1/4 unit: `apply()` produces `replace_result` and the mutated value."""
    from agent_loop_chaos.faults.tool import ToolCorruptionFault

    fault = ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"])
    outcome = fault.apply(post({"temp_c": 21, "city": "Paris"}), fault_context())
    assert outcome.action == "replace_result"
    assert outcome.value == {"city": "Paris"}


def test_corruption_records_a_patch_that_reproduces_the_change() -> None:
    """3/4 mutation log: the recorded diff must actually reconstruct the payload."""
    from agent_loop_chaos.faults.tool import ToolCorruptionFault

    before = {"temp_c": 21, "city": "Paris"}
    outcome = ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]).apply(
        post(before), fault_context()
    )
    log = outcome.mutation
    assert log is not None
    assert log.payload_before == before
    assert apply_patch(log.payload_before, log.json_patch) == log.payload_after


def test_corruption_note_names_the_field_and_the_record_count() -> None:
    """The `note` is what the narrator and AGENT_TASK.md quote, so it must be specific."""
    from agent_loop_chaos.faults.tool import ToolCorruptionFault

    records = [{"temp_c": 1}, {"temp_c": 2}, {"temp_c": 3}]
    outcome = ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]).apply(
        post(records), fault_context()
    )
    assert outcome.note
    assert "temp_c" in outcome.note
    assert "3 of 3" in outcome.note, "catalog: 'dropped key \"temp_c\" from 3 of 3 records'"


def test_corruption_rejects_an_unknown_mutation_at_construction() -> None:
    """Eager validation: a typo in a scenario file must fail at load, not mid-run."""
    from agent_loop_chaos.faults.tool import ToolCorruptionFault

    with pytest.raises(ConfigError, match="drop_ky"):
        ToolCorruptionFault(mutation_type="drop_ky")


def test_corruption_random_is_reproducible_under_a_fixed_seed() -> None:
    """`mutation_type="random"` still has to tell the same story twice."""
    from agent_loop_chaos.faults.tool import ToolCorruptionFault

    payload = {"a": 1, "b": "two", "c": [1, 2, 3]}
    first = ToolCorruptionFault(mutation_type="random").apply(post(payload), fault_context(7))
    second = ToolCorruptionFault(mutation_type="random").apply(post(payload), fault_context(7))
    assert first.value == second.value
    assert first.note == second.note


def test_corruption_random_picks_differently_across_seeds() -> None:
    """Otherwise "random" would be a fixed choice wearing a costume."""
    from agent_loop_chaos.faults.tool import ToolCorruptionFault

    payload = {"a": 1, "b": "two", "c": [1, 2, 3]}
    notes = {
        ToolCorruptionFault(mutation_type="random").apply(post(payload), fault_context(s)).note
        for s in range(12)
    }
    assert len(notes) > 1


def test_corruption_records_its_random_choice_as_a_decision() -> None:
    """D-36: an RNG-derived choice belongs in `randomness.decisions`."""
    from agent_loop_chaos.faults.tool import ToolCorruptionFault

    ctx = fault_context()
    ToolCorruptionFault(mutation_type="random").apply(post({"a": 1}), ctx)
    assert any(d["key"] == "mutation_type" for d in ctx.run.decisions)


def test_corruption_only_accepts_tool_post() -> None:
    """It rewrites what the tool returned, so `pre` makes no sense for it."""
    from agent_loop_chaos.faults.tool import ToolCorruptionFault

    assert ToolCorruptionFault.accepts == frozenset({("tool", "post")})


def test_corruption_fires_exactly_on_the_configured_call(tmp_path: Path) -> None:
    """2/4 trigger: on call 2 of 3, and nowhere else."""
    from agent_loop_chaos.faults.tool import ToolCorruptionFault

    eng = engine(tmp_path)
    eng.register_fault(
        ToolCorruptionFault(mutation_type="empty_json"),
        target_tool="get_weather",
        trigger=Trigger(on_call=2),
    )

    @eng.tool
    def get_weather(day: int) -> dict[str, int]:
        return {"temp_c": day}

    result = eng.run(lambda: [get_weather(1), get_weather(2), get_weather(3)])
    assert result.final_output == [{"temp_c": 1}, {}, {"temp_c": 3}]
    assert result.injected_faults[0]["fire_count"] == 1


def test_corruption_end_to_end_records_the_fault(tmp_path: Path) -> None:
    """4/4 end-to-end: the trace carries `fault_fired` and the report the fault."""
    from agent_loop_chaos.faults.tool import ToolCorruptionFault

    eng = engine(tmp_path)
    eng.register_fault(
        ToolCorruptionFault(mutation_type="unit_swap", keys=["temp_c"]),
        target_tool="get_weather",
    )

    @eng.tool
    def get_weather() -> dict[str, Any]:
        return {"temp_c": 21, "city": "Paris"}

    result = eng.run(lambda: get_weather(), scenario_id="corrupt")
    assert result.final_output["temp_c"] == pytest.approx(69.8)
    assert result.final_output["city"] == "Paris", "unit_swap keeps the label"
    assert result.injected_faults[0]["fired"] is True
    assert "fault_fired" in {e["kind"] for e in events(result)}
    assert result.validate() == []


def pre(*args: Any, name: str = "get_weather", call_index: int = 1) -> Crossing:
    """Build a `(tool, pre)` crossing.

    Args:
        *args: The arguments the agent sent.
        name: The tool name.
        call_index: Which call this is.

    Returns:
        The crossing.
    """
    return Crossing(layer="tool", phase="pre", name=name, args=args, call_index=call_index)


# ================================================================ A2 ToolErrorFault


def test_error_raises_the_configured_exception_class() -> None:
    """1/4 unit: a terminal `raise`, built from an allow-list rather than `eval`."""
    from agent_loop_chaos.faults.tool import ToolErrorFault

    outcome = ToolErrorFault(exc_class="ConnectionError", message="upstream down").apply(
        pre(), fault_context()
    )
    assert outcome.action == "raise"
    assert isinstance(outcome.value, ConnectionError)
    assert "upstream down" in str(outcome.value)


@pytest.mark.parametrize(
    "exc_class",
    ["RuntimeError", "TimeoutError", "ConnectionError", "ValueError", "KeyError"],
)
def test_error_supports_every_allow_listed_class(exc_class: str) -> None:
    """The allow-list from the phase prompt, exactly."""
    from agent_loop_chaos.faults.tool import ToolErrorFault

    outcome = ToolErrorFault(exc_class=exc_class).apply(pre(), fault_context())
    assert type(outcome.value).__name__ == exc_class


def test_error_supports_the_library_provided_simulated_error() -> None:
    """So a scenario can raise something unambiguously ours."""
    from agent_loop_chaos.errors import SimulatedToolError
    from agent_loop_chaos.faults.tool import ToolErrorFault

    outcome = ToolErrorFault(exc_class="SimulatedToolError").apply(pre(), fault_context())
    assert isinstance(outcome.value, SimulatedToolError)


def test_error_never_evaluates_an_arbitrary_class_name() -> None:
    """`eval` on a scenario-file string would be a code-execution hole."""
    from agent_loop_chaos.faults.tool import ToolErrorFault

    with pytest.raises(ConfigError, match="exc_class"):
        ToolErrorFault(exc_class="__import__('os').system")


def test_error_payload_returns_instead_of_raising() -> None:
    """The common real-world case: a tool signals failure in-band and is believed.

    An agent that treats `{"error": ...}` as data produces a `silent_wrong_answer`,
    which is a different and more dangerous failure than a crash.
    """
    from agent_loop_chaos.faults.tool import ToolErrorFault

    outcome = ToolErrorFault(error_type="error_payload", message="quota exceeded").apply(
        pre(), fault_context()
    )
    assert outcome.action == "replace_result"
    assert outcome.value["error"] == "quota exceeded"
    assert outcome.params["in_band"] is True


@pytest.mark.parametrize(
    ("error_type", "status"), [("http_500", 500), ("http_429", 429), ("http_401", 401)]
)
def test_http_error_types_carry_their_status(error_type: str, status: int) -> None:
    """`http_401` exists specifically to catch agents retrying non-retryable errors."""
    from agent_loop_chaos.faults.tool import ToolErrorFault

    outcome = ToolErrorFault(error_type=error_type).apply(pre(), fault_context())
    payload = outcome.value if outcome.action == "replace_result" else outcome.value.args[0]
    assert str(status) in str(payload)


def test_error_rejects_an_unknown_error_type() -> None:
    """Eager validation, with the offending value in the message."""
    from agent_loop_chaos.faults.tool import ToolErrorFault

    with pytest.raises(ConfigError, match="http_418"):
        ToolErrorFault(error_type="http_418")


def test_error_fires_instead_of_calling_the_real_tool(tmp_path: Path) -> None:
    """4/4 end-to-end: `(tool, pre)`, so the real tool never runs."""
    from agent_loop_chaos.faults.tool import ToolErrorFault

    eng = engine(tmp_path)
    eng.register_fault(ToolErrorFault(message="boom"), target_tool="get_weather")
    calls: list[int] = []

    @eng.tool
    def get_weather() -> str:
        calls.append(1)
        return "sunny"

    def agent() -> str:
        try:
            return get_weather()
        except RuntimeError:
            return "handled"

    result = eng.run(agent)
    assert result.final_output == "handled"
    assert calls == [], "the real tool must not have run"
    assert result.injected_faults[0]["fired"] is True


# ============================================================== A4 ToolTimeoutFault


def test_timeout_raises_a_timeout_error() -> None:
    """Distinct from A3: this tests the *handling* path, not the budget."""
    from agent_loop_chaos.faults.tool import ToolTimeoutFault

    outcome = ToolTimeoutFault().apply(pre(), fault_context())
    assert outcome.action == "raise"
    assert isinstance(outcome.value, TimeoutError)


def test_timeout_end_to_end_is_catchable_by_the_agent(tmp_path: Path) -> None:
    """An agent with a timeout budget should survive this."""
    from agent_loop_chaos.faults.tool import ToolTimeoutFault

    eng = engine(tmp_path)
    eng.register_fault(ToolTimeoutFault(), target_tool="slow")

    @eng.tool
    def slow() -> str:
        return "eventually"

    def agent() -> str:
        try:
            return slow()
        except TimeoutError:
            return "degraded gracefully"

    assert eng.run(agent).final_output == "degraded gracefully"


# ================================================================ A10 RateLimitFault


def test_rate_limit_allows_the_first_calls_then_limits() -> None:
    """`after_calls=2` means two get through and the third is throttled."""
    from agent_loop_chaos.faults.tool import RateLimitFault

    fault = RateLimitFault(after_calls=2)
    ctx = fault_context()
    assert fault.apply(pre(call_index=1), ctx).action == "noop"
    assert fault.apply(pre(call_index=2), ctx).action == "noop"
    assert fault.apply(pre(call_index=3), ctx).action == "replace_result"


def test_rate_limit_payload_carries_the_status_and_retry_after() -> None:
    """Respecting `Retry-After` is exactly what this fault probes for."""
    from agent_loop_chaos.faults.tool import RateLimitFault

    outcome = RateLimitFault(after_calls=0, retry_after_s=30).apply(pre(), fault_context())
    assert outcome.value["status"] == 429
    assert outcome.value["retry_after_s"] == 30


def test_rate_limit_accepts_both_tool_and_llm_layers() -> None:
    """Catalog A10 lists `(tool, pre)` and `(llm, pre)`."""
    from agent_loop_chaos.faults.tool import RateLimitFault

    assert ("tool", "pre") in RateLimitFault.accepts
    assert ("llm", "pre") in RateLimitFault.accepts


def test_rate_limit_can_raise_instead_of_returning() -> None:
    """Some clients surface a 429 as an exception; both shapes must be reachable."""
    from agent_loop_chaos.faults.tool import RateLimitFault

    outcome = RateLimitFault(after_calls=0, error_type="exception").apply(pre(), fault_context())
    assert outcome.action == "raise"


def test_rate_limit_end_to_end_throttles_after_the_budget(tmp_path: Path) -> None:
    """A hand-rolled retry loop with no backoff is the weakness this proves."""
    from agent_loop_chaos.faults.tool import RateLimitFault

    eng = engine(tmp_path)
    eng.register_fault(
        RateLimitFault(after_calls=2),
        target_tool="search",
        trigger=Trigger(max_fires=99),
    )

    @eng.tool
    def search(q: str) -> Any:
        return ["hit"]

    result = eng.run(lambda: [search("a"), search("b"), search("c"), search("d")])
    throttled = [r for r in result.final_output if isinstance(r, dict)]
    assert len(throttled) == 2, "calls 3 and 4 are throttled"
    assert all(r["status"] == 429 for r in throttled)


def test_an_in_band_error_envelope_reaches_the_agent_without_the_tool_running(
    tmp_path: Path,
) -> None:
    """Regression guard for D-57.

    Found by the rate-limit end-to-end test: a `(tool, pre)` fault returning
    `replace_result` had no way to short-circuit the call, so the real tool ran and
    the envelope was discarded. The unit test passed throughout, because it only
    exercised `apply()` in isolation — which is exactly the blind spot an
    end-to-end test exists to cover.
    """
    from agent_loop_chaos.faults.tool import ToolErrorFault

    eng = engine(tmp_path)
    eng.register_fault(
        ToolErrorFault(error_type="error_payload", message="quota exceeded"),
        target_tool="search",
    )
    calls: list[int] = []

    @eng.tool
    def search(q: str) -> list[str]:
        calls.append(1)
        return ["real result"]

    result = eng.run(lambda: search("a"))
    assert result.final_output == {"error": "quota exceeded", "code": "tool_error"}
    assert calls == [], "the tool must not have run"
    assert result.tool_calls[0]["faulted"] is True


# ============================================================== A3 ToolLatencyFault


def test_latency_asks_for_a_delay() -> None:
    """1/4 unit: a terminal `delay`, so it is first-wins against other terminals."""
    from agent_loop_chaos.faults.tool import ToolLatencyFault

    outcome = ToolLatencyFault(delay_ms=250).apply(pre(), fault_context())
    assert outcome.action == "delay"
    assert outcome.delay_ms == 250


def test_latency_is_clamped_to_the_limit() -> None:
    """A scenario must not be able to stall a suite for an hour."""
    from agent_loop_chaos.faults.tool import ToolLatencyFault

    ctx = fault_context()
    outcome = ToolLatencyFault(delay_ms=10_000_000).apply(pre(), ctx)
    assert outcome.delay_ms == ctx.limits.max_injected_delay_ms


def test_latency_records_both_the_requested_and_the_applied_delay() -> None:
    """A clamped run must say so, or the report understates what was asked for."""
    from agent_loop_chaos.faults.tool import ToolLatencyFault

    outcome = ToolLatencyFault(delay_ms=10_000_000).apply(pre(), fault_context())
    assert outcome.params["requested_delay_ms"] == 10_000_000
    assert outcome.params["applied_delay_ms"] == outcome.delay_ms


def test_latency_jitter_is_seeded_and_reproducible() -> None:
    """Jitter that varied run to run would break golden comparison."""
    from agent_loop_chaos.faults.tool import ToolLatencyFault

    first = ToolLatencyFault(delay_ms=100, jitter_ms=50).apply(pre(), fault_context(9))
    second = ToolLatencyFault(delay_ms=100, jitter_ms=50).apply(pre(), fault_context(9))
    assert first.delay_ms == second.delay_ms


def test_latency_accumulates_into_injected_delay_ms(tmp_path: Path) -> None:
    """R3: the harness's own delay is subtracted from every budget comparison.

    There is deliberately no latency *probe* — a timing-derived finding would make
    `success` wall-clock dependent (`docs/07` §3 rule 2). The number is recorded so
    it can be excluded, not so it can be judged.
    """
    from agent_loop_chaos.faults.tool import ToolLatencyFault

    eng = engine(tmp_path)
    eng.register_fault(ToolLatencyFault(delay_ms=5), target_tool="slow")

    @eng.tool
    def slow() -> str:
        return "done"

    result = eng.run(lambda: slow())
    assert result.metrics["injected_delay_ms"] == 5
    assert result.final_output == "done", "a delay must not change the result"


# ================================================================= A6 StaleDataFault


def test_stale_replays_an_earlier_result() -> None:
    """1/4 unit: serve call 1's payload in answer to call 2."""
    from agent_loop_chaos.faults.tool import StaleDataFault

    ctx = fault_context()
    ctx.pre_fault_history["prices"] = [{"price": 100}, {"price": 250}]
    outcome = StaleDataFault(serve_call_index=1).apply(
        post({"price": 250}, name="prices", call_index=2), ctx
    )
    assert outcome.action == "replace_result"
    assert outcome.value == {"price": 100}


def test_stale_backdates_a_timestamp_so_the_staleness_is_detectable() -> None:
    """The whole point: an agent that reads the freshness field can catch this.

    Serving old data with a current timestamp would be undetectable, and a fault
    nobody can detect proves nothing.
    """
    from agent_loop_chaos.faults.tool import StaleDataFault

    ctx = fault_context()
    ctx.pre_fault_history["prices"] = [{"price": 100, "as_of": 1_700_000_000}]
    outcome = StaleDataFault(serve_call_index=1, age_field="as_of", age_delta_s=86_400).apply(
        post({"price": 250, "as_of": 1_700_086_400}, name="prices", call_index=2), ctx
    )
    assert outcome.value["as_of"] == 1_700_000_000 - 86_400


def test_stale_is_a_noop_when_there_is_no_earlier_call() -> None:
    """On the first call there is nothing stale to serve."""
    from agent_loop_chaos.faults.tool import StaleDataFault

    outcome = StaleDataFault().apply(post({"price": 100}, name="prices"), fault_context())
    assert outcome.action == "noop"


def test_stale_end_to_end_serves_the_first_payload_twice(tmp_path: Path) -> None:
    """4/4 end-to-end against a tool whose answer really does change."""
    from agent_loop_chaos.faults.tool import StaleDataFault

    eng = engine(tmp_path)
    eng.register_fault(
        StaleDataFault(serve_call_index=1),
        target_tool="price",
        trigger=Trigger(on_call=2),
    )
    counter = iter([100, 250, 400])

    @eng.tool
    def price() -> dict[str, int]:
        return {"price": next(counter)}

    result = eng.run(lambda: [price(), price(), price()])
    assert result.final_output == [{"price": 100}, {"price": 100}, {"price": 400}]


# =========================================================== A7 NonDeterminismFault


def test_non_determinism_returns_a_different_value_each_call() -> None:
    """Same input, different output — the assumption this fault breaks."""
    from agent_loop_chaos.faults.tool import NonDeterminismFault

    fault = NonDeterminismFault(variants=["a", "b", "c"])
    ctx = fault_context()
    seen = [fault.apply(post("x", call_index=i), ctx).value for i in range(1, 4)]
    assert len(set(seen)) > 1


def test_non_determinism_sequence_is_identical_across_runs_with_one_seed() -> None:
    """Different every call, identical every run: the phase prompt's exact ask."""
    from agent_loop_chaos.faults.tool import NonDeterminismFault

    def sequence(seed: int) -> list[Any]:
        fault = NonDeterminismFault(variants=[1, 2, 3, 4, 5])
        ctx = fault_context(seed)
        return [fault.apply(post("x", call_index=i), ctx).value for i in range(1, 6)]

    assert sequence(11) == sequence(11)
    assert sequence(11) != sequence(12)


def test_non_determinism_falls_back_to_a_mutation_without_variants() -> None:
    """Catalog A7 defaults `mutation_type` to `reorder_list`."""
    from agent_loop_chaos.faults.tool import NonDeterminismFault

    outcome = NonDeterminismFault().apply(post([1, 2, 3, 4, 5]), fault_context())
    assert outcome.action == "replace_result"
    assert sorted(outcome.value) == [1, 2, 3, 4, 5]


# =========================================================== A5 ArgumentTamperFault


def test_argument_tamper_mutates_what_the_agent_sent() -> None:
    """1/4 unit: `replace_args`, so the real tool receives the corrupted call."""
    from agent_loop_chaos.faults.tool import ArgumentTamperFault

    outcome = ArgumentTamperFault(mutation_type="negative_numbers", arg_names=["amount"]).apply(
        Crossing(layer="tool", phase="pre", name="refund", kwargs={"amount": 250}),
        fault_context(),
    )
    assert outcome.action == "replace_args"


def test_argument_tamper_is_declared_as_performing_a_real_action() -> None:
    """It sends a mutated call to the real tool, so the D-23 gate must cover it."""
    from agent_loop_chaos.faults.tool import ArgumentTamperFault

    assert ArgumentTamperFault.performs_real_action is True


def test_argument_tamper_refuses_a_side_effecting_tool_without_opt_in() -> None:
    """The gate, exercised through the real fault rather than a stand-in."""
    from agent_loop_chaos.faults.tool import ArgumentTamperFault

    eng = ChaosEngine(write_bundle=False)

    @eng.tool(side_effecting=True)
    def refund(amount: int) -> str:
        return "refunded"

    with pytest.raises(ConfigError, match="refund"):
        eng.register_fault(ArgumentTamperFault(), target_tool="refund")


def test_argument_tamper_flags_arguments_that_were_already_invalid() -> None:
    """Catalog A5: it doubles as an agent-bug detector.

    If the agent already sent something invalid, that is a finding in its own right
    and must be recorded *before* the fault muddies the evidence.
    """
    from agent_loop_chaos.faults.tool import ArgumentTamperFault

    outcome = ArgumentTamperFault(arg_names=["amount"]).apply(
        Crossing(layer="tool", phase="pre", name="refund", kwargs={"amount": None}),
        fault_context(),
    )
    assert outcome.params.get("pre_existing_invalid_args") == ["amount"]


def test_argument_tamper_end_to_end_reaches_the_real_tool(tmp_path: Path) -> None:
    """4/4 end-to-end: the tool observes the tampered value, which is the point."""
    from agent_loop_chaos.faults.tool import ArgumentTamperFault

    eng = engine(tmp_path)
    eng.register_fault(
        ArgumentTamperFault(mutation_type="negative_numbers", arg_names=["amount"]),
        target_tool="transfer",
    )
    seen: list[int] = []

    @eng.tool(side_effecting=False)
    def transfer(amount: int) -> str:
        seen.append(amount)
        return f"moved {amount}"

    result = eng.run(lambda: transfer(amount=250))
    assert seen == [-250], "the real tool must receive the tampered argument"
    assert result.injected_faults[0]["fired"] is True


# ================================================================== A9 LoopTrapFault


def test_loop_trap_pins_the_first_observed_result() -> None:
    """1/4 unit: nothing the agent does changes the answer."""
    from agent_loop_chaos.faults.tool import LoopTrapFault

    fault = LoopTrapFault(pin_after_call=1)
    ctx = fault_context()
    first = fault.apply(post({"cursor": 1}, call_index=1), ctx)
    second = fault.apply(post({"cursor": 2}, call_index=2), ctx)
    assert second.value == first.value == {"cursor": 1}


def test_loop_trap_pins_to_an_explicit_value_when_given_one() -> None:
    """`pin_value` lets a scenario choose the trap rather than discover it."""
    from agent_loop_chaos.faults.tool import LoopTrapFault

    outcome = LoopTrapFault(pin_value={"stuck": True}).apply(post({"a": 1}), fault_context())
    assert outcome.value == {"stuck": True}


def test_loop_trap_pins_a_copy_so_the_agent_cannot_mutate_the_trap() -> None:
    """An agent that mutates the payload in place would otherwise escape."""
    from agent_loop_chaos.faults.tool import LoopTrapFault

    fault = LoopTrapFault()
    ctx = fault_context()
    first = fault.apply(post({"items": [1]}, call_index=1), ctx)
    first.value["items"].append(999)
    second = fault.apply(post({"items": [2]}, call_index=2), ctx)
    assert second.value == {"items": [1]}


def test_loop_trap_end_to_end_hits_max_steps_without_an_error(tmp_path: Path) -> None:
    """The abort is the signal, not a bug.

    Catalog A9 is explicit that this fault is *expected* to exhaust
    `Limits.max_steps`, so the run must end with `limit_hit` set and `error` empty —
    otherwise the harness's own stop would be reported as an agent crash.
    """
    from agent_loop_chaos.faults.tool import LoopTrapFault

    eng = engine(tmp_path, limits=Limits(max_steps=4, max_tool_calls=99, max_llm_calls=99))
    eng.register_fault(LoopTrapFault(), target_tool="poll", trigger=Trigger(max_fires=99))

    @eng.tool
    def poll() -> dict[str, str]:
        return {"status": "pending"}

    @eng.llm
    def think(prompt: str) -> str:
        return "keep waiting"

    def agent() -> str:
        for _ in range(100):
            think("what now")
            if poll()["status"] == "done":
                return "finished"
        return "gave up"

    result = eng.run(agent)
    assert result.loop["limit_hit"] == "max_steps"
    assert result.error is None
    assert result.validate() == []


# ======================================================= A8 DuplicateSideEffectFault


def test_duplicate_side_effect_asks_the_adapter_to_invoke_again() -> None:
    """1/4 unit: `invoke_target` (D-10).

    `apply()` cannot perform the repeat itself — the adapter is the only party that
    knows whether to `await`, which is exactly why the action exists.
    """
    from agent_loop_chaos.faults.tool import DuplicateSideEffectFault

    outcome = DuplicateSideEffectFault(times=3).apply(pre(), fault_context())
    assert outcome.action == "invoke_target"
    assert outcome.params["times"] == 3
    assert outcome.params["return_from"] == "first"


def test_duplicate_side_effect_is_declared_as_performing_a_real_action() -> None:
    """It calls the real tool twice, so the D-23 gate must cover it."""
    from agent_loop_chaos.faults.tool import DuplicateSideEffectFault

    assert DuplicateSideEffectFault.performs_real_action is True


def test_duplicate_side_effect_really_calls_the_tool_twice(tmp_path: Path) -> None:
    """4/4 end-to-end: the double send has to actually happen to be a finding."""
    from agent_loop_chaos.faults.tool import DuplicateSideEffectFault

    eng = engine(tmp_path)
    eng.register_fault(DuplicateSideEffectFault(times=2), target_tool="send_email")
    sent: list[str] = []

    @eng.tool(side_effecting=False)
    def send_email(to: str) -> str:
        sent.append(to)
        return f"sent-{len(sent)}"

    result = eng.run(lambda: send_email("a@b.c"))
    assert sent == ["a@b.c", "a@b.c"], "the tool must really have run twice"
    assert result.final_output == "sent-1", "return_from='first'"


def test_duplicate_side_effect_can_return_the_last_response(tmp_path: Path) -> None:
    """`return_from="last"` models a client that keeps the newest response."""
    from agent_loop_chaos.faults.tool import DuplicateSideEffectFault

    eng = engine(tmp_path)
    eng.register_fault(
        DuplicateSideEffectFault(times=2, return_from="last"), target_tool="send_email"
    )
    sent: list[str] = []

    @eng.tool(side_effecting=False)
    def send_email(to: str) -> str:
        sent.append(to)
        return f"sent-{len(sent)}"

    assert eng.run(lambda: send_email("a@b.c")).final_output == "sent-2"


def test_the_harness_own_invocations_are_attributed_to_the_harness(tmp_path: Path) -> None:
    """R1: the duplicate-side-effect probe must not count the harness's own calls.

    Without this the probe would fire on every run of this fault, including against
    an agent that is perfectly idempotent — which would make the control unpassable.
    """
    from agent_loop_chaos.faults.tool import DuplicateSideEffectFault

    eng = engine(tmp_path)
    eng.register_fault(DuplicateSideEffectFault(times=2), target_tool="send_email")
    captured: list[Any] = []

    @eng.tool(side_effecting=False)
    def send_email(to: str) -> str:
        return "sent"

    def agent() -> str:
        out = send_email("a@b.c")
        captured.append(eng.harness_facts())
        return out

    eng.run(agent)
    assert captured[0].harness_invocation_seqs, "the repeat must be attributed to the harness"
