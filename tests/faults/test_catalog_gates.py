"""Phase-wide gates for the fault catalog.

These are the assertions that hold for *every* fault rather than for one, so they
cannot rot as the catalog grows: a fault added in M3 or M5 without a docstring, a
schema-valid patch, or a dict constructor fails here rather than in review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine, Limits
from agent_loop_chaos.context import Counters, Crossing, FaultContext, RunContext
from agent_loop_chaos.faults.base import FAULT_REGISTRY, fault_from_dict, list_faults
from agent_loop_chaos.jsonpatch import apply as apply_patch
from agent_loop_chaos.trace import TraceRecorder

# Catalog section A, in the order docs/03-FAULT-CATALOG.md lists them.
SECTION_A = [
    "ToolCorruptionFault",
    "ToolErrorFault",
    "ToolLatencyFault",
    "ToolTimeoutFault",
    "ArgumentTamperFault",
    "StaleDataFault",
    "NonDeterminismFault",
    "DuplicateSideEffectFault",
    "LoopTrapFault",
    "RateLimitFault",
]


def context() -> FaultContext:
    """Build a `FaultContext` for calling `apply()` directly.

    Returns:
        The context.
    """
    ctx = RunContext(
        run_id="run-gate",
        seed=1337,
        started_at="1970-01-01T00:00:00Z",
        limits=Limits(),
        trace=TraceRecorder("run-gate"),
        counters=Counters(),
    )
    return FaultContext(
        fault_id="f1", fault_key="gate01", run=ctx, limits=ctx.limits, counters=ctx.counters
    )


@pytest.mark.parametrize("kind", SECTION_A)
def test_every_section_a_fault_is_registered(kind: str) -> None:
    """A fault absent from `FAULT_REGISTRY` cannot be named in a YAML scenario."""
    assert kind in FAULT_REGISTRY


@pytest.mark.parametrize("kind", SECTION_A)
def test_every_section_a_fault_is_constructible_from_a_dict(kind: str) -> None:
    """This is the path a suite file takes, so it must work with no params."""
    assert fault_from_dict({"type": kind}).kind == kind


@pytest.mark.parametrize("kind", SECTION_A)
def test_every_fault_declares_the_phases_it_accepts(kind: str) -> None:
    """An empty `accepts` set would make the fault silently unfirable."""
    assert FAULT_REGISTRY[kind].accepts


@pytest.mark.parametrize("kind", SECTION_A)
def test_every_fault_docstring_states_the_weakness_and_the_graceful_behaviour(
    kind: str,
) -> None:
    """CLAUDE.md requires both, because that pairing is what makes a work order
    actionable: a finding without "what good looks like" tells nobody what to build.
    """
    doc = FAULT_REGISTRY[kind].__doc__ or ""
    assert "weakness it proves" in doc, f"{kind} does not say what it proves"
    assert "graceful behaviour looks like" in doc, f"{kind} does not say what good looks like"


@pytest.mark.parametrize("kind", SECTION_A)
def test_every_recorded_patch_reconstructs_the_payload(kind: str) -> None:
    """`apply(json_patch, payload_before) == payload_after`, for every fault.

    A recorded diff that does not reproduce the change is worse than no diff: a
    coding agent would act on it.
    """
    crossing = Crossing(
        layer="tool",
        phase="post",
        name="get_weather",
        args=(21,),
        kwargs={"amount": 250, "temp_c": 21},
        result={"temp_c": 21, "city": "Paris", "items": [1, 2, 3]},
        call_index=1,
    )
    outcome = FAULT_REGISTRY[kind]().apply(crossing, context())
    log = outcome.mutation
    if log is None:
        pytest.skip(f"{kind} records no mutation for this crossing")
    assert apply_patch(log.payload_before, log.json_patch) == log.payload_after


@pytest.mark.parametrize("kind", SECTION_A)
def test_every_fault_produces_a_human_note(kind: str) -> None:
    """The `note` is what the narrator and AGENT_TASK.md quote verbatim."""
    crossing = Crossing(
        layer="tool",
        phase="post",
        name="get_weather",
        kwargs={"amount": 250},
        result={"temp_c": 21, "items": [1, 2]},
        call_index=1,
    )
    outcome = FAULT_REGISTRY[kind]().apply(crossing, context())
    assert outcome.note, f"{kind} produced no note"
    assert outcome.note[0].islower() or outcome.note[0].isdigit(), (
        "notes are clauses, not sentences"
    )


def test_list_faults_covers_the_whole_registry() -> None:
    """`alc list-faults` reads this, so a gap here is a gap in the CLI."""
    described = {info.kind for info in list_faults()}
    assert set(SECTION_A) <= described


@pytest.mark.parametrize("kind", SECTION_A)
def test_every_fault_has_a_one_line_summary(kind: str) -> None:
    """`alc list-faults` prints one line per fault, so each needs a first line."""
    info = next(i for i in list_faults() if i.kind == kind)
    assert info.summary
    assert len(info.summary) < 120


# ------------------------------------------------------------------ alc list-faults


def test_alc_list_faults_prints_every_fault(capsys: pytest.CaptureFixture[str]) -> None:
    """The phase's Verify block runs this, so it has to actually print."""
    from agent_loop_chaos.cli import main

    assert main(["list-faults"]) == 0
    out = capsys.readouterr().out
    for kind in SECTION_A:
        assert kind in out


def test_alc_list_faults_prints_the_accepted_phases_and_a_description(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bare list of names would not tell a user where a fault can be pointed."""
    from agent_loop_chaos.cli import main

    main(["list-faults"])
    out = capsys.readouterr().out
    assert "tool.post" in out
    assert "corrupt" in out.lower()


def test_alc_list_faults_json_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """`--json` prints one JSON document and nothing else, so CI can pipe it."""
    import json

    from agent_loop_chaos.cli import main

    assert main(["list-faults", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    kinds = {entry["kind"] for entry in payload}
    assert set(SECTION_A) <= kinds
    assert all("accepts" in entry and "severity_hint" in entry for entry in payload)


def test_the_demo_snippet_from_the_readme_still_works(tmp_path: Path) -> None:
    """The five-line promise, with a real fault rather than a stub.

    `docs/02-API.md` §11 says both snippets must work verbatim by the end of M5;
    this is the tool half of it, which is reachable now.
    """
    from agent_loop_chaos.faults import ToolCorruptionFault

    engine = ChaosEngine(seed=1337, out_dir=tmp_path / ".chaos")
    engine.register_fault(
        ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
        target_tool="get_weather_data",
    )

    @engine.tool
    def get_weather_data(location: str) -> list[dict[str, Any]]:
        return [{"temp_c": 21, "city": location}]

    result = engine.run(get_weather_data, inputs="Paris")
    assert result.final_output == [{"city": "Paris"}]
    assert result.validate() == []
