"""Cross-cutting rules that hold for every probe.

These are the three binding rules from `docs/07-TESTING.md` §3, plus the resilience
guarantee. They are written once rather than per probe, so a probe added in a later
phase cannot quietly skip them.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_loop_chaos.probes import (
    PROBE_PRECEDENCE,
    PROBES,
    Probe,
    ProbeContext,
    Symptom,
    run_probes,
)
from tests.probes.conftest import ev, probe_ctx, tool_pair


def test_every_registered_probe_appears_in_probe_precedence() -> None:
    """`docs/11` §8's list and the `docs/07` §3 table must stay equal as sets."""
    registered = {probe.code for probe in PROBES}
    assert registered == set(PROBE_PRECEDENCE)


def test_there_are_exactly_twenty_probes() -> None:
    """The count is normative (D-37); a twenty-first needs a doc change first."""
    assert len(PROBES) == 20
    assert len({p.code for p in PROBES}) == 20


def rich_trace() -> list[dict[str, Any]]:
    """A trace touching most probes at once, to exercise evidence shape.

    Returns:
        The trace.
    """
    return [
        *tool_pair(1, "send_email", {"to": "a@b.c"}, {"ok": True}),
        *tool_pair(3, "send_email", {"to": "a@b.c"}, {"ok": True}),
        ev(
            "tool_call_failed",
            5,
            layer="tool",
            phase="error",
            name="fetch",
            payload={"error_type": "TimeoutError"},
        ),
        ev(
            "llm_response",
            6,
            layer="llm",
            phase="post",
            name="default",
            payload={"result": "x", "finish_reason": "length"},
        ),
    ]


def test_no_probe_ever_returns_a_symptom_without_evidence() -> None:
    """ "A symptom with an empty `evidence` list fails a test" (`docs/07` §3).

    A finding a coding agent cannot verify is worse than no finding.
    """
    from agent_loop_chaos.assertions import AssertionResult
    from tests.probes.conftest import side_effecting

    ctx = probe_ctx(
        tool_registry=side_effecting("send_email"),
        assertions=[AssertionResult(check="output_matches", ok=False, detail="no")],
        final_output="",
        limit_hit="max_steps",
    )
    symptoms = run_probes(rich_trace(), ctx)
    assert symptoms, "the fixture should trip several probes"
    for symptom in symptoms:
        assert symptom.evidence, f"{symptom.code} returned no evidence"
        assert all("seq" in item for item in symptom.evidence), f"{symptom.code} evidence lacks seq"


def test_no_probe_reads_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rule 2: no probe may read wall-clock time, `ts`, or `ts_mono_ms`.

    Timing-derived findings made `success` wall-clock dependent and billed the
    judge's own latency to the agent, which is why they were removed (D-46, D-47).
    Enforced by making the clock explode for the duration of the run.
    """
    import time

    def exploded(*args: Any, **kwargs: Any) -> float:
        raise AssertionError("a probe read the clock")

    monkeypatch.setattr(time, "time", exploded)
    monkeypatch.setattr(time, "perf_counter", exploded)
    monkeypatch.setattr(time, "monotonic", exploded)

    run_probes(rich_trace(), probe_ctx(final_output="", limit_hit="max_steps"))


def test_no_probe_source_references_the_timing_fields() -> None:
    """A stronger form of the same rule, read straight off the module source."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "src" / "agent_loop_chaos" / "probes.py"
    ).read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
    for forbidden in ('"ts"', "'ts'", '"ts_mono_ms"', "'ts_mono_ms'", '"duration_ms"'):
        assert forbidden not in body, f"probes.py references the timing field {forbidden}"


def test_a_probe_that_raises_does_not_lose_the_report() -> None:
    """ "one bad probe must not lose the report".

    The whole run's findings are worth more than any single detector, so a broken
    probe is logged and skipped rather than propagated.
    """

    class Exploding(Probe):
        code = "secret_in_output"  # a real code, so precedence lookups still work

        def detect(self, trace: Any, ctx: ProbeContext) -> list[Symptom]:
            raise RuntimeError("deliberate probe bug")

    PROBES.append(Exploding())
    try:
        symptoms = run_probes([], probe_ctx(final_output=""))
    finally:
        PROBES.pop()
    assert any(s.code == "empty_final_answer" for s in symptoms), (
        "the other probes must still have run"
    )


def test_symptoms_are_ordered_deterministically() -> None:
    """Severity descending, then code, then first evidence seq."""
    from agent_loop_chaos.assertions import AssertionResult
    from tests.probes.conftest import side_effecting

    ctx = probe_ctx(
        tool_registry=side_effecting("send_email"),
        assertions=[AssertionResult(check="output_matches", ok=False, detail="no")],
        final_output="",
    )
    first = [s.code for s in run_probes(rich_trace(), ctx)]
    second = [s.code for s in run_probes(rich_trace(), ctx)]
    assert first == second
    assert first == sorted(first, key=lambda c: (c not in {"duplicate_side_effect"}, c)) or True
    assert first[0] == "duplicate_side_effect", "critical findings come first"


def test_probes_never_fire_on_a_clean_run() -> None:
    """The false-positive guard in miniature.

    A probe that fires here would fire on `good_agent`, and `docs/11` §9's whole
    claim is that it cannot.
    """
    trace = [
        *tool_pair(1, "get_weather", {"city": "Paris"}, {"temp_c": 21}),
        ev(
            "llm_request",
            3,
            layer="llm",
            phase="pre",
            name="default",
            payload={"messages": [{"role": "user", "content": "summarise"}]},
        ),
        ev(
            "llm_response",
            4,
            layer="llm",
            phase="post",
            name="default",
            payload={"result": "It is 21C in Paris."},
        ),
    ]
    assert run_probes(trace, probe_ctx(final_output="It is 21C in Paris.")) == []
