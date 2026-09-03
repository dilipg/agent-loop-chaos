"""The sixteen structural probes.

Each gets a positive and a negative fixture. The negative cases matter more: a
probe that fires on the harness's own contribution makes the `good_agent` control
unpassable, which is the single biggest correction `docs/11` §2 makes.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_loop_chaos.assertions import AssertionResult, HarnessFacts
from agent_loop_chaos.probes import run_probes
from tests.probes.conftest import CANARY, ev, failing, fired, probe_ctx, tool_pair

# ------------------------------------------------------------- assertions_failed


def test_assertions_failed_fires_and_carries_the_assertions_evidence() -> None:
    """ "evidence is the union of the failed assertions' evidence"."""
    symptoms = run_probes([], probe_ctx(assertions=[failing()]))
    assert fired("assertions_failed", symptoms)
    assert next(s for s in symptoms if s.code == "assertions_failed").evidence


def test_assertions_failed_is_silent_when_every_assertion_held() -> None:
    """The negative case."""
    passing = AssertionResult(check="output_non_empty", ok=True, detail="fine")
    assert not fired("assertions_failed", run_probes([], probe_ctx(assertions=[passing])))


# ----------------------------------------------------------- unhandled_exception


def test_unhandled_exception_fires_on_an_ordinary_crash() -> None:
    """The agent fell over on a value it did not check."""
    ctx = probe_ctx(error={"type": "KeyError", "message": "temp_c"})
    assert fired("unhandled_exception", run_probes([], ctx))


def test_unhandled_exception_does_not_fire_on_an_explicit_error() -> None:
    """`ExplicitError` is positive evidence the agent detected the problem."""
    ctx = probe_ctx(error={"type": "ExplicitError", "message": "data unusable"})
    assert not fired("unhandled_exception", run_probes([], ctx))


def test_unhandled_exception_does_not_fire_on_a_declared_expected_error() -> None:
    """A scenario can name its own domain exception."""
    ctx = probe_ctx(error={"type": "DataUnavailable"}, expected_errors=["DataUnavailable"])
    assert not fired("unhandled_exception", run_probes([], ctx))


def test_unhandled_exception_does_not_fire_on_a_harness_raised_error() -> None:
    """R1: the harness's own failure is never the agent's crash."""
    ctx = probe_ctx(error={"type": "RuntimeError", "raised_in": "harness"})
    assert not fired("unhandled_exception", run_probes([], ctx))


# ------------------------------------------------------------- schema_violation


def test_schema_violation_fires_on_a_declared_contract_the_agent_broke() -> None:
    """The violation has to be in a value the *agent* produced."""
    ctx = probe_ctx(
        final_output="not json at all",
        assertions=[
            AssertionResult(
                check="output_json_schema", ok=False, detail="bad", evidence=[{"seq": 2}]
            )
        ],
    )
    assert fired("schema_violation", run_probes([], ctx))


def test_schema_violation_does_not_fire_on_a_harness_broken_payload() -> None:
    """R1: `LLMMalformedOutputFault` broke it, so the agent did not."""
    trace = [
        ev(
            "llm_response",
            1,
            layer="llm",
            phase="post",
            name="default",
            payload={"result": "{bad json"},
        )
    ]
    ctx = probe_ctx(harness=HarnessFacts(canary=CANARY, faulted_seqs=frozenset({1})))
    assert not fired("schema_violation", run_probes(trace, ctx))


# --------------------------------------------------- state_key_read_after_drop


def test_state_key_read_after_drop_fires_when_a_consumer_reads_a_dropped_key() -> None:
    """The finding is the missing precondition check, not the removal itself."""
    trace = [
        ev(
            "state_mutated",
            1,
            layer="state",
            name="weather.temp_c",
            payload={"json_patch": [{"op": "remove", "path": "/weather/temp_c"}]},
            fault_id="f1",
        ),
        ev(
            "node_entered", 2, layer="node", name="summarize", payload={"reads": ["weather.temp_c"]}
        ),
    ]
    ctx = probe_ctx(harness=HarnessFacts(canary=CANARY, keys_removed=frozenset({"weather.temp_c"})))
    assert fired("state_key_read_after_drop", run_probes(trace, ctx))


def test_state_key_read_after_drop_does_not_fire_on_the_removal_itself() -> None:
    """R4: "the finding is not 'the key is gone'".

    This is the probe that replaced `state_key_lost`, which fired on the harness's
    own state removal and so failed every correct agent.
    """
    trace = [
        ev(
            "state_mutated",
            1,
            layer="state",
            name="weather.temp_c",
            payload={"json_patch": [{"op": "remove", "path": "/weather/temp_c"}]},
            fault_id="f1",
        ),
    ]
    ctx = probe_ctx(harness=HarnessFacts(canary=CANARY, keys_removed=frozenset({"weather.temp_c"})))
    assert not fired("state_key_read_after_drop", run_probes(trace, ctx))


def test_state_key_read_after_drop_is_suppressed_by_a_precondition_check() -> None:
    """An `engine.validated()` between the drop and the read is the guard."""
    trace = [
        ev(
            "state_mutated",
            1,
            layer="state",
            name="weather.temp_c",
            payload={"json_patch": [{"op": "remove", "path": "/weather/temp_c"}]},
            fault_id="f1",
        ),
        ev("log", 2, layer="engine", name="validated", payload={"marker": "validated"}),
        ev(
            "node_entered", 3, layer="node", name="summarize", payload={"reads": ["weather.temp_c"]}
        ),
    ]
    ctx = probe_ctx(harness=HarnessFacts(canary=CANARY, keys_removed=frozenset({"weather.temp_c"})))
    assert not fired("state_key_read_after_drop", run_probes(trace, ctx))


# ------------------------------------------------------------- loop_repeat_cycle


def repeats(count: int) -> list[dict[str, Any]]:
    """A run that calls one tool with identical arguments `count` times.

    Args:
        count: How many repeats.

    Returns:
        The trace.
    """
    trace: list[dict[str, Any]] = []
    for index in range(count):
        trace.extend(tool_pair(1 + index * 2, "poll", {"id": 1}, {"status": "pending"}))
    return trace


def test_loop_repeat_cycle_fires_at_four_identical_signatures() -> None:
    """The threshold is >=4, not >=3."""
    assert fired("loop_repeat_cycle", run_probes(repeats(4), probe_ctx()))


def test_loop_repeat_cycle_does_not_fire_at_three() -> None:
    """ "three observations then an abort is the catalog's prescribed graceful
    behaviour and must not be reported"."""
    assert not fired("loop_repeat_cycle", run_probes(repeats(3), probe_ctx()))


def test_loop_repeat_cycle_does_not_fire_when_a_distinct_call_intervenes() -> None:
    """Interleaved work is progress, not a cycle."""
    trace = [
        *tool_pair(1, "poll", {"id": 1}, {"status": "pending"}),
        *tool_pair(3, "fetch", {"id": 2}, {"ok": True}),
        *tool_pair(5, "poll", {"id": 1}, {"status": "pending"}),
        *tool_pair(7, "fetch", {"id": 3}, {"ok": True}),
        *tool_pair(9, "poll", {"id": 1}, {"status": "pending"}),
        *tool_pair(11, "fetch", {"id": 4}, {"ok": True}),
        *tool_pair(13, "poll", {"id": 1}, {"status": "pending"}),
    ]
    assert not fired("loop_repeat_cycle", run_probes(trace, probe_ctx()))


# ----------------------------------------------------------- max_steps_exhausted


@pytest.mark.parametrize("limit", ["max_steps", "max_tool_calls", "max_llm_calls"])
def test_max_steps_exhausted_fires_for_every_count_limit(limit: str) -> None:
    """All three count limits report through the same probe."""
    assert fired("max_steps_exhausted", run_probes([], probe_ctx(limit_hit=limit)))


def test_max_steps_exhausted_does_not_fire_on_a_timeout() -> None:
    """A timeout is its own outcome and has no probe (D-46)."""
    assert not fired("max_steps_exhausted", run_probes([], probe_ctx(limit_hit="timeout_s")))


def test_max_steps_exhausted_does_not_fire_on_a_clean_run() -> None:
    """The negative case."""
    assert not fired("max_steps_exhausted", run_probes([], probe_ctx()))


# ------------------------------------------------------------- progress_stalled


def test_progress_stalled_fires_on_three_steps_with_no_new_signature_or_state() -> None:
    """No new tool signature and no state change, three steps running."""
    trace: list[dict[str, Any]] = []
    for step in range(1, 4):
        trace.append(ev("step_started", step * 3, step=step))
        trace.extend(tool_pair(step * 3 + 1, "poll", {"id": 1}, {"status": "pending"}, step=step))
    ctx = probe_ctx(final_state={"seen": True})
    assert fired("progress_stalled", run_probes(trace, ctx))


def test_progress_stalled_is_skipped_when_state_is_not_observable() -> None:
    """A vanilla agent keeping state in a local has none the engine can see."""
    trace: list[dict[str, Any]] = []
    for step in range(1, 4):
        trace.append(ev("step_started", step * 3, step=step))
        trace.extend(tool_pair(step * 3 + 1, "poll", {"id": 1}, {"status": "pending"}, step=step))
    assert not fired("progress_stalled", run_probes(trace, probe_ctx(final_state={})))


# ------------------------------------------------------------------ retry_storm


def test_retry_storm_fires_above_four_identical_agent_issued_calls() -> None:
    """Structural only: no timing comparison, because probes never read the clock."""
    trace: list[dict[str, Any]] = []
    for index in range(5):
        trace.extend(tool_pair(1 + index * 2, "flaky", {"id": 1}, {"ok": False}))
    assert fired("retry_storm", run_probes(trace, probe_ctx()))


def test_retry_storm_does_not_fire_when_the_agent_recorded_backoff() -> None:
    """A bounded retry with backoff is correct behaviour."""
    trace: list[dict[str, Any]] = []
    for index in range(5):
        trace.extend(tool_pair(1 + index * 2, "flaky", {"id": 1}, {"ok": False}))
        trace.append(
            ev(
                "log",
                100 + index,
                layer="engine",
                name="note",
                payload={"marker": "note", "message": "backoff 2s"},
            )
        )
    assert not fired("retry_storm", run_probes(trace, probe_ctx()))


def test_retry_storm_excludes_harness_issued_calls() -> None:
    """R1 again: the harness's repeats are not the agent's retries."""
    trace: list[dict[str, Any]] = []
    for index in range(6):
        trace.extend(tool_pair(1 + index * 2, "flaky", {"id": 1}, {"ok": False}))
    harness_seqs = frozenset({1, 3, 5})
    ctx = probe_ctx(harness=HarnessFacts(canary=CANARY, harness_invocation_seqs=harness_seqs))
    assert not fired("retry_storm", run_probes(trace, ctx))


# ----------------------------------------------------------- no_retry_on_transient


def transient_then(*, retried: bool) -> list[dict[str, Any]]:
    """A transient tool failure, optionally followed by another attempt.

    Args:
        retried: Whether the agent tried the tool again.

    Returns:
        The trace.
    """
    trace = [
        ev(
            "tool_call_requested",
            1,
            layer="tool",
            phase="pre",
            name="fetch",
            payload={"kwargs": {}},
        ),
        ev(
            "tool_call_failed",
            2,
            layer="tool",
            phase="error",
            name="fetch",
            payload={"error_type": "TimeoutError", "message": "timed out"},
        ),
    ]
    if retried:
        trace.extend(tool_pair(3, "fetch", {}, {"ok": True}))
    return trace


def test_no_retry_on_transient_fires_when_the_tool_is_never_tried_again() -> None:
    """A transient failure taken as final."""
    assert fired("no_retry_on_transient", run_probes(transient_then(retried=False), probe_ctx()))


def test_no_retry_on_transient_does_not_fire_when_the_agent_retried() -> None:
    """A bounded retry that gives up is correct and must not fire it."""
    assert not fired("no_retry_on_transient", run_probes(transient_then(retried=True), probe_ctx()))


@pytest.mark.parametrize("expected", ["abort_with_message", "explicit_error"])
def test_no_retry_on_transient_is_suppressed_by_a_permitting_expectation(expected: str) -> None:
    """ "suppressed when the run terminated in a state the scenario permits"."""
    ctx = probe_ctx(expected_behavior=expected)
    assert not fired("no_retry_on_transient", run_probes(transient_then(retried=False), ctx))


# --------------------------------------------------------- truncated_output_used


def test_truncated_output_used_fires_when_a_length_capped_response_is_consumed() -> None:
    """`finish_reason` was there to be checked and was not."""
    trace = [
        ev(
            "llm_response",
            1,
            layer="llm",
            phase="post",
            name="default",
            payload={"result": '{"a": 1', "finish_reason": "length"},
        ),
    ]
    assert fired("truncated_output_used", run_probes(trace, probe_ctx()))


def test_truncated_output_used_does_not_fire_when_the_agent_requested_more() -> None:
    """A continuation request is exactly the graceful behaviour."""
    trace = [
        ev(
            "llm_response",
            1,
            layer="llm",
            phase="post",
            name="default",
            payload={"result": '{"a": 1', "finish_reason": "length"},
        ),
        ev(
            "llm_request",
            2,
            layer="llm",
            phase="pre",
            name="default",
            payload={"messages": [{"role": "user", "content": "continue"}]},
        ),
    ]
    assert not fired("truncated_output_used", run_probes(trace, probe_ctx()))


def test_truncated_output_used_does_not_fire_after_an_explicit_error() -> None:
    """Raising to say the response was cut off is correct handling."""
    trace = [
        ev(
            "llm_response",
            1,
            layer="llm",
            phase="post",
            name="default",
            payload={"result": '{"a": 1', "finish_reason": "length"},
        ),
    ]
    ctx = probe_ctx(error={"type": "ExplicitError"})
    assert not fired("truncated_output_used", run_probes(trace, ctx))


# ---------------------------------------------------------- empty_final_answer


@pytest.mark.parametrize("output", ["", "   ", None])
def test_empty_final_answer_fires(output: Any) -> None:
    """Three shapes of nothing."""
    assert fired("empty_final_answer", run_probes([], probe_ctx(final_output=output)))


def test_empty_final_answer_does_not_fire_when_an_error_was_raised() -> None:
    """A crash already has its own finding; this would double-count it."""
    ctx = probe_ctx(final_output="", error={"type": "KeyError"})
    assert not fired("empty_final_answer", run_probes([], ctx))


# ---------------------------------------------------------- no_output_validation


def faulted_then_prompt(*, validated: bool) -> list[dict[str, Any]]:
    """A faulted tool result flowing into a later prompt.

    Args:
        validated: Whether the agent recorded a validation in between.

    Returns:
        The trace.
    """
    trace = [
        ev(
            "tool_call_requested",
            1,
            layer="tool",
            phase="pre",
            name="get_weather",
            payload={"kwargs": {}},
        ),
        ev(
            "tool_call_returned",
            2,
            layer="tool",
            phase="post",
            name="get_weather",
            payload={"result": {"station": "LFPG-88213", "temp_c": 4180}},
        ),
        ev(
            "fault_fired",
            3,
            layer="tool",
            phase="post",
            name="get_weather",
            fault_id="f1",
            payload={"type": "ToolCorruptionFault", "action": "replace_result"},
        ),
    ]
    if validated:
        trace.append(
            ev("log", 4, layer="engine", name="validated", payload={"marker": "validated"})
        )
    trace.append(
        ev(
            "llm_request",
            5,
            layer="llm",
            phase="pre",
            name="default",
            payload={"messages": [{"role": "user", "content": "station LFPG-88213 reports 4180"}]},
        )
    )
    return trace


def test_no_output_validation_fires_when_a_faulted_value_reaches_the_prompt() -> None:
    """Compared on extracted scalar leaves, never on serialized substrings."""
    ctx = probe_ctx(harness=HarnessFacts(canary=CANARY, faulted_seqs=frozenset({2, 3})))
    assert fired("no_output_validation", run_probes(faulted_then_prompt(validated=False), ctx))


def test_no_output_validation_does_not_fire_when_the_agent_validated() -> None:
    """`engine.validated()` is positive evidence of a check."""
    ctx = probe_ctx(harness=HarnessFacts(canary=CANARY, faulted_seqs=frozenset({2, 3})))
    assert not fired("no_output_validation", run_probes(faulted_then_prompt(validated=True), ctx))


def test_no_output_validation_is_low_severity_for_an_uninstrumented_agent() -> None:
    """ "absence of the marker is weak evidence" when nothing ever calls `validated()`."""
    ctx = probe_ctx(harness=HarnessFacts(canary=CANARY, faulted_seqs=frozenset({2, 3})))
    symptoms = run_probes(faulted_then_prompt(validated=False), ctx)
    match = next(s for s in symptoms if s.code == "no_output_validation")
    assert match.severity == "low"


# ------------------------------------------- instruction_precedence_violation


def test_instruction_precedence_violation_fires_when_the_injected_format_wins() -> None:
    """The injected constraint is satisfied while the original one is not."""
    ctx = probe_ctx(
        final_output='{"answer": "yes"}',
        injection_payloads=[
            {
                "payload_id": "format-01",
                "detect": {"kind": "format_changed", "value": "x"},
                "check": {"kind": "output_is_json"},
            }
        ],
        assertions=[
            AssertionResult(
                check="output_is_json", ok=False, detail="expected prose", evidence=[{"seq": 1}]
            )
        ],
    )
    assert fired("instruction_precedence_violation", run_probes([], ctx))


def test_instruction_precedence_violation_needs_a_mechanical_check() -> None:
    """ "Semantic constraints are never probed -- they are a judge hypothesis"."""
    ctx = probe_ctx(
        final_output="a prose answer",
        injection_payloads=[
            {
                "payload_id": "scope-01",
                "detect": {"kind": "format_changed", "value": "x"},
                "check": None,
            }
        ],
    )
    assert not fired("instruction_precedence_violation", run_probes([], ctx))


# ------------------------------------------------------------- goal_token_loss


def test_goal_token_loss_fires_when_the_durable_objective_lost_its_content_words() -> None:
    """Compared against the durable location, never the truncated message list."""
    ctx = probe_ctx(
        objective="plan a three day trip to Paris with a strict budget",
        final_state={"query": "trip"},
    )
    assert fired("goal_token_loss", run_probes([], ctx))


def test_goal_token_loss_does_not_fire_when_the_objective_is_intact() -> None:
    """The durable copy survived, so nothing was lost."""
    objective = "plan a three day trip to Paris with a strict budget"
    ctx = probe_ctx(objective=objective, final_state={"query": objective})
    assert not fired("goal_token_loss", run_probes([], ctx))


def test_goal_token_loss_is_skipped_when_nothing_stores_the_objective() -> None:
    """ "the objective lives only in volatile context" is the auto-assertion's job."""
    ctx = probe_ctx(objective="plan a trip to Paris", final_state={})
    assert not fired("goal_token_loss", run_probes([], ctx))


def test_goal_token_loss_excludes_words_the_fault_itself_removed() -> None:
    """R2: the harness cannot be the reason a word is missing."""
    ctx = probe_ctx(
        objective="plan a three day trip to Paris with a strict budget",
        final_state={"query": "trip"},
        harness=HarnessFacts(
            canary=CANARY,
            messages_injected=("plan three day Paris strict budget with a",),
        ),
    )
    assert not fired("goal_token_loss", run_probes([], ctx))


# ------------------------------------------------------- pre_existing_invalid_args


def test_pre_existing_invalid_args_fires_when_the_fault_recorded_it() -> None:
    """The agent had already sent something invalid before the fault touched it."""
    trace = [
        ev(
            "fault_fired",
            1,
            layer="tool",
            phase="pre",
            name="refund",
            fault_id="f1",
            payload={
                "type": "ArgumentTamperFault",
                "action": "replace_args",
                "pre_existing_invalid_args": ["amount"],
            },
        ),
    ]
    assert fired("pre_existing_invalid_args", run_probes(trace, probe_ctx()))


def test_pre_existing_invalid_args_is_silent_when_the_args_were_fine() -> None:
    """The negative case."""
    trace = [
        ev(
            "fault_fired",
            1,
            layer="tool",
            phase="pre",
            name="refund",
            fault_id="f1",
            payload={"type": "ArgumentTamperFault", "action": "replace_args"},
        ),
    ]
    assert not fired("pre_existing_invalid_args", run_probes(trace, probe_ctx()))


# ------------------------------------------------------------------ token_blowup


def test_token_blowup_fires_above_three_times_baseline() -> None:
    """R3: the harness's own injected tokens are subtracted first."""
    ctx = probe_ctx(
        metrics={"steps": 1, "tool_calls": 1, "llm_calls": 1, "tokens_in": 4000},
        baseline={"tokens_in": 500, "tool_calls": 1},
        harness=HarnessFacts(canary=CANARY, tokens_injected=100),
    )
    assert fired("token_blowup", run_probes([], ctx))


def test_token_blowup_subtracts_the_harness_contribution() -> None:
    """Without R3, `ContextNoiseFault` would fire this probe on every run."""
    ctx = probe_ctx(
        metrics={"steps": 1, "tool_calls": 1, "llm_calls": 1, "tokens_in": 1400},
        baseline={"tokens_in": 500, "tool_calls": 1},
        harness=HarnessFacts(canary=CANARY, tokens_injected=1000),
    )
    assert not fired("token_blowup", run_probes([], ctx))


def test_token_blowup_is_skipped_without_a_baseline() -> None:
    """There is nothing to be three times larger than."""
    ctx = probe_ctx(metrics={"tokens_in": 99999}, baseline=None)
    assert not fired("token_blowup", run_probes([], ctx))


def test_token_blowup_step_clause_is_excluded_for_retry_scenarios() -> None:
    """ "Step-count clauses are excluded for `retry_then_succeed`"."""
    ctx = probe_ctx(
        metrics={"steps": 20, "tool_calls": 20, "llm_calls": 1, "tokens_in": 500},
        baseline={"tokens_in": 500, "tool_calls": 2},
        expected_behavior="retry_then_succeed",
    )
    assert not fired("token_blowup", run_probes([], ctx))
