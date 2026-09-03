"""Metrics computed from the trace, and the baseline delta.

Metrics are computed *before* probes (D-11), because a third of the probe rules need
them. `output_similarity` uses `difflib` on lowercased, whitespace-collapsed text:
deterministic, and no embeddings, so a seeded run stays byte-reproducible.
"""

from __future__ import annotations

from typing import Any

from agent_loop_chaos.metrics import compute_delta, compute_metrics, output_similarity
from tests.probes.conftest import ev, tool_pair


def trace() -> list[dict[str, Any]]:
    """A small run: two tool calls and one LLM exchange.

    Returns:
        The trace.
    """
    return [
        ev("run_started", 1, layer="engine"),
        ev("step_started", 2, step=1),
        *tool_pair(3, "get_weather", {"city": "Paris"}, {"temp_c": 21}),
        *tool_pair(5, "search", {"q": "hotels"}, {"hits": 3}),
        ev(
            "llm_request",
            7,
            layer="llm",
            phase="pre",
            name="default",
            payload={"messages": [{"role": "user", "content": "summarise this"}]},
        ),
        ev(
            "llm_response",
            8,
            layer="llm",
            phase="post",
            name="default",
            payload={"result": "It is 21C in Paris."},
        ),
        ev("run_finished", 9, layer="engine"),
    ]


def test_counters_come_from_the_trace() -> None:
    """The report's numbers must be reconstructible from the stored trace alone."""
    metrics = compute_metrics(trace())
    assert metrics["tool_calls"] == 2
    assert metrics["llm_calls"] == 1
    assert metrics["steps"] == 1


def test_token_counts_are_estimated_and_say_so() -> None:
    """D-43: `tokens_estimated` is explicit, so a report never implies a precision
    it did not have."""
    metrics = compute_metrics(trace())
    assert metrics["tokens_in"] > 0
    assert metrics["tokens_estimated"] is True


def test_harness_contributions_are_reported_separately() -> None:
    """R3: every budget comparison subtracts these, so they must be their own field."""
    metrics = compute_metrics(trace(), injected_tokens=40, injected_delay_ms=250)
    assert metrics["injected_tokens"] == 40
    assert metrics["injected_delay_ms"] == 250


def test_fault_counters_are_included() -> None:
    """`faults_armed` and `faults_fired` are how a reader sees the plan took effect."""
    with_faults = [
        *trace(),
        ev("fault_armed", 10, fault_id="f1", payload={"type": "NoopFault"}),
        ev("fault_fired", 11, fault_id="f1", payload={"type": "NoopFault", "action": "noop"}),
    ]
    metrics = compute_metrics(with_faults)
    assert metrics["faults_armed"] == 1
    assert metrics["faults_fired"] == 1


def test_metrics_never_read_the_clock_from_the_trace() -> None:
    """`wall_ms` is supplied by the engine as a timing field, not derived from `ts`."""
    metrics = compute_metrics(trace(), wall_ms=1234)
    assert metrics["wall_ms"] == 1234


def test_an_empty_trace_yields_zeroed_metrics() -> None:
    """A run that never started must not crash the pipeline."""
    metrics = compute_metrics([])
    assert metrics["steps"] == 0
    assert metrics["tool_calls"] == 0


# ------------------------------------------------------------------ the delta


def test_output_similarity_is_one_for_identical_text() -> None:
    """Deterministic, and no embeddings."""
    assert output_similarity("It is 21C.", "It is 21C.") == 1.0


def test_output_similarity_ignores_case_and_whitespace() -> None:
    """Otherwise reformatting would read as a behaviour change."""
    assert output_similarity("It  is 21C.", "it is   21c.") == 1.0


def test_output_similarity_falls_for_a_different_answer() -> None:
    """The number that tells a reader the fault changed the outcome."""
    assert output_similarity("It is 21C in Paris.", "The data was unavailable.") < 0.5


def test_the_delta_reports_counter_differences() -> None:
    """A reader wants "how much more did it do", not two tables to subtract."""
    delta = compute_delta(
        {"steps": 5, "tool_calls": 9, "llm_calls": 4, "tokens_in": 900, "tokens_out": 80},
        {"steps": 2, "tool_calls": 3, "llm_calls": 2, "tokens_in": 300, "tokens_out": 40},
        chaos_output="x",
        baseline_output="x",
        chaos_tools=["a"],
        baseline_tools=["a"],
    )
    assert delta["steps"] == 3
    assert delta["tool_calls"] == 6
    assert delta["tokens_in"] == 600


def test_the_delta_names_the_new_and_missing_tool_calls() -> None:
    """Which tools appeared or vanished is the most legible signal in the delta."""
    delta = compute_delta(
        {},
        {},
        chaos_output="x",
        baseline_output="x",
        chaos_tools=["get_weather", "hold_booking"],
        baseline_tools=["get_weather", "search_flights"],
    )
    assert delta["new_tool_calls"] == ["hold_booking"]
    assert delta["missing_tool_calls"] == ["search_flights"]


def test_the_delta_carries_the_output_similarity() -> None:
    """So a report can say the answer changed without diffing it inline."""
    delta = compute_delta(
        {},
        {},
        chaos_output="The data was unavailable.",
        baseline_output="It is 21C in Paris.",
        chaos_tools=[],
        baseline_tools=[],
    )
    assert 0.0 <= delta["output_similarity"] < 0.6
