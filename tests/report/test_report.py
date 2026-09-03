"""The report object: field parity, code pointers, and assembly from a trace.

The parity test is the guard that keeps the dataclass and the schema from drifting.
M0 shipped `Expect` with the schema's field *names* and hand-written types that did
not match; this is the test that catches that class of mistake for the report.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from agent_loop_chaos.report import (
    ChaosResult,
    assemble,
    extract_code_context,
    extract_code_pointers,
)
from agent_loop_chaos.schema import load_schema

# The schema pins `plan_hash` to sha256: plus 64 hex characters.
PLAN_HASH = "sha256:" + "ab" * 32


# ------------------------------------------------------------------- parity


def test_every_dataclass_field_appears_in_the_schema() -> None:
    """Direction one: the code may not write a field the schema does not describe."""
    fields = {f.name for f in dataclasses.fields(ChaosResult)}
    properties = set(load_schema("report")["properties"])
    extra = sorted(fields - properties)
    assert extra == [], f"ChaosResult writes fields absent from the schema: {extra}"


def test_every_schema_property_appears_in_the_dataclass() -> None:
    """Direction two: a schema field nothing ever writes is dead weight."""
    fields = {f.name for f in dataclasses.fields(ChaosResult)}
    properties = set(load_schema("report")["properties"])
    missing = sorted(properties - fields)
    assert missing == [], f"the schema declares fields ChaosResult never writes: {missing}"


def test_every_required_property_is_emitted_on_every_run() -> None:
    """A consumer must never have to guess whether a field is missing or absent."""
    emitted = set(_minimal().to_dict())
    required = set(load_schema("report")["required"])
    assert required <= emitted, f"missing from to_dict(): {sorted(required - emitted)}"


def _minimal() -> ChaosResult:
    """A minimally-populated result that still validates.

    Returns:
        The result.
    """
    from agent_loop_chaos.report import empty_verdict

    return ChaosResult(
        run_id="run-3f9a12c4",
        seed=1337,
        plan_hash=PLAN_HASH,
        started_at="1970-01-01T00:00:00Z",
        finished_at="1970-01-01T00:00:01Z",
        duration_ms=1,
        target={"framework": "vanilla", "adapter_version": "python-3.12"},
        metrics={"steps": 1, "tool_calls": 1, "llm_calls": 1},
        loop={"steps": 1, "max_steps": 25, "limit_hit": None},
        randomness={"seed": 1337, "streams": {}},
        reproduce={"cmd": "alc replay .", "seed": 1337},
        artifacts={"report": "r.json", "trace": "t.jsonl"},
        verdict=empty_verdict(),
    )


def test_a_minimal_result_validates() -> None:
    """The schema is the product; a result that does not satisfy it is not a report."""
    assert _minimal().validate() == []


def test_to_dict_emits_nothing_the_schema_forbids() -> None:
    """`additionalProperties: false` means an extra key fails the whole document."""
    allowed = set(load_schema("report")["properties"])
    assert set(_minimal().to_dict()) <= allowed


# ------------------------------------------------------------- code pointers


def test_code_pointers_keep_only_user_frames() -> None:
    """A pointer into the library's own source sends a coding agent to fix nothing."""
    traceback_text = (
        "Traceback (most recent call last):\n"
        '  File "/proj/src/agent_loop_chaos/engine.py", line 10, in run\n'
        "    output = agent()\n"
        '  File "/venv/lib/python3.12/site-packages/thing/mod.py", line 3, in helper\n'
        "    return inner()\n"
        '  File "/proj/myapp/agent.py", line 42, in summarize\n'
        '    return data[0]["temp_c"]\n'
        "KeyError: temp_c\n"
    )
    pointers = extract_code_pointers({"traceback": traceback_text, "type": "KeyError"}, [])
    files = [p["file"] for p in pointers]
    assert "/proj/myapp/agent.py" in files
    assert not any("agent_loop_chaos" in f for f in files)
    assert not any("site-packages" in f for f in files)


def test_code_pointers_are_innermost_first() -> None:
    """The frame that actually failed is the one to read first."""
    traceback_text = (
        "Traceback (most recent call last):\n"
        '  File "/proj/myapp/outer.py", line 5, in outer\n'
        "    inner()\n"
        '  File "/proj/myapp/inner.py", line 9, in inner\n'
        "    boom()\n"
        "KeyError: x\n"
    )
    pointers = extract_code_pointers({"traceback": traceback_text, "type": "KeyError"}, [])
    assert pointers[0]["file"].endswith("inner.py")


def test_code_pointers_carry_the_line_symbol_and_reason() -> None:
    """`why` is what makes a pointer actionable rather than merely locating."""
    traceback_text = (
        "Traceback (most recent call last):\n"
        '  File "/proj/myapp/agent.py", line 42, in summarize\n'
        '    return data[0]["temp_c"]\n'
        "KeyError: temp_c\n"
    )
    pointer = extract_code_pointers({"traceback": traceback_text, "type": "KeyError"}, [])[0]
    assert pointer["line"] == 42
    assert pointer["symbol"] == "summarize"
    assert pointer["why"]


def test_a_pointer_is_derived_from_the_faulted_crossing_with_no_exception() -> None:
    """The reference example has no exception, and still needs somewhere to look.

    Without this, a silent wrong answer produces a work order with no file to open.
    """
    trace = [
        {
            "seq": 3,
            "kind": "fault_fired",
            "layer": "tool",
            "phase": "post",
            "name": "get_weather_data",
            "fault_id": "f1",
            "payload": {"type": "ToolCorruptionFault", "action": "replace_result"},
        },
    ]
    pointers = extract_code_pointers(None, trace)
    assert pointers
    assert "get_weather_data" in pointers[0]["why"]
    assert pointers[0]["confidence"] < 1.0, "an advisory pointer is not a certainty (D-30)"


def test_code_context_reads_the_surrounding_source(tmp_path: Path) -> None:
    """The judge quotes this in phase 06, so it has to be real source."""
    module = tmp_path / "agent.py"
    module.write_text("\n".join(f"line {n}" for n in range(1, 21)), encoding="utf-8")
    context = extract_code_context([{"file": str(module), "line": 10, "why": "x"}])
    assert context
    assert "line 10" in context[0]["source"]
    assert "line 6" in context[0]["source"], "+-4 lines of context"


def test_code_context_handles_a_missing_file() -> None:
    """A stored run replayed on another machine must not crash the pipeline."""
    assert extract_code_context([{"file": "/nope/missing.py", "line": 3, "why": "x"}]) == []


# ------------------------------------------------------------------- assemble


def test_assemble_is_pure_over_the_trace_and_plan() -> None:
    """ "Keep the post-run pipeline pure over `(trace, plan)`" (`docs/01` §5).

    This is what lets `alc judge` re-judge a stored run with a better model without
    re-running the agent.
    """
    from agent_loop_chaos.assertions import AssertionResult
    from agent_loop_chaos.probes import Symptom

    trace = [
        {"seq": 1, "kind": "run_started", "layer": "engine", "payload": {"seed": 1337}},
        {
            "seq": 2,
            "kind": "tool_call_requested",
            "layer": "tool",
            "phase": "pre",
            "name": "get_weather",
            "payload": {"kwargs": {}},
        },
        {
            "seq": 3,
            "kind": "tool_call_returned",
            "layer": "tool",
            "phase": "post",
            "name": "get_weather",
            "payload": {"result": {}},
        },
        {"seq": 4, "kind": "run_finished", "layer": "engine", "payload": {}},
    ]
    plan = {"seed": 1337, "faults": [], "adapter": "vanilla", "entrypoint": "m:agent"}
    result = assemble(
        trace=trace,
        plan=plan,
        plan_hash=PLAN_HASH,
        run_id="run-3f9a12c4",
        scenario_id="s1",
        started_at="1970-01-01T00:00:00Z",
        finished_at="1970-01-01T00:00:01Z",
        wall_ms=5,
        final_output="The data was unavailable.",
        error=None,
        symptoms=[
            Symptom(code="assertions_failed", severity="high", detail="x", evidence=[{"seq": 3}])
        ],
        assertions=[AssertionResult(check="output_non_empty", ok=True, detail="fine")],
        expected_behavior="graceful_degradation",
    )
    assert result.validate() == []
    assert result.run_id == "run-3f9a12c4"
    assert result.metrics["tool_calls"] == 1


def test_assemble_computes_success_without_a_model() -> None:
    """`success` is computed from probes, assertions and expected_behavior only."""
    plan = {"seed": 1, "faults": []}
    result = assemble(
        trace=[{"seq": 1, "kind": "run_started"}],
        plan=plan,
        plan_hash=PLAN_HASH,
        run_id="run-00000001",
        scenario_id=None,
        started_at="1970-01-01T00:00:00Z",
        finished_at="1970-01-01T00:00:01Z",
        wall_ms=1,
        final_output="fine",
        error=None,
        symptoms=[],
        assertions=[],
        expected_behavior="ignore_and_continue",
    )
    assert result.success is True
    assert result.failure_mode == "none"
    assert result.severity == "info"


def test_assemble_marks_a_failing_run() -> None:
    """The motivating example: a corrupted payload and an invented number."""
    from agent_loop_chaos.assertions import AssertionResult
    from agent_loop_chaos.probes import Symptom

    result = assemble(
        trace=[{"seq": 1, "kind": "run_started"}],
        plan={"seed": 1, "faults": []},
        plan_hash=PLAN_HASH,
        run_id="run-00000001",
        scenario_id=None,
        started_at="1970-01-01T00:00:00Z",
        finished_at="1970-01-01T00:00:01Z",
        wall_ms=1,
        final_output="It is 24C.",
        error=None,
        symptoms=[
            Symptom(code="assertions_failed", severity="high", detail="x", evidence=[{"seq": 1}])
        ],
        assertions=[
            AssertionResult(check="no_unsourced_numbers", ok=False, detail="24C is unsourced")
        ],
        expected_behavior="graceful_degradation",
        destructive_mutation=True,
    )
    assert result.success is False
    assert result.verdict["observed_behavior"] == "hallucinated"
    assert result.failure_mode == "hallucination_on_corrupt_data"
    assert result.severity in {"high", "critical"}


def test_summary_line_reads_as_one_row_of_the_cli_table() -> None:
    """`alc run` prints one of these per scenario."""
    line = _minimal().summary_line()
    assert "pass" in line or "FAIL" in line
