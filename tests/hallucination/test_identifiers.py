"""Hallucination identifiers: the checks that catch a confident, wrong answer.

Grounding (`no_unsourced_numbers`) already catches a fabricated *quantity*. These two
catch the other two shapes a model invents:

- **A tool it never used.** "I checked the booking system" when no such tool exists,
  or exists and was never called. The registry is known exactly, so this check has no
  judgement in it at all and cannot produce a false positive.
- **A source that does not exist.** A citation, document id, reference code or URL in
  the output that appears in nothing the agent retrieved.

Both follow the grounding precedent (`docs/11` §4.3): identifiers live in the
assertions layer, not in `probes.py`. The removed `fabricated_value` probe is why --
a probe that fires on a correct agent is a bug, and claim-checking needs the
scenario's declaration of what "sourced" means.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_loop_chaos.assertions import EvidenceContext, Expect, evaluate


def _result(results: list[Any], check: str) -> Any:
    found = [r for r in results if r.check == check]
    assert found, f"{check} did not run"
    return found[0]


def _run(expect: Expect, **evidence: Any) -> list[Any]:
    return evaluate(expect, EvidenceContext(**evidence))


class TestInventedTools:
    """The safest identifier there is: the tool registry is a closed set."""

    def test_it_catches_a_tool_that_does_not_exist(self) -> None:
        results = _run(
            Expect(no_invented_tools=True),
            final_output="I queried the booking_system and confirmed your seat.",
            tool_registry={"search_flights": object(), "get_weather": object()},
            tools_called=["search_flights"],
        )
        outcome = _result(results, "no_invented_tools")
        assert outcome.ok is False
        assert "booking_system" in outcome.detail

    def test_it_catches_a_registered_tool_that_was_never_called(self) -> None:
        results = _run(
            Expect(no_invented_tools=True),
            final_output="I called get_weather to confirm the forecast.",
            tool_registry={"search_flights": object(), "get_weather": object()},
            tools_called=["search_flights"],
        )
        assert _result(results, "no_invented_tools").ok is False

    def test_it_passes_when_the_claim_is_true(self) -> None:
        results = _run(
            Expect(no_invented_tools=True),
            final_output="I used search_flights and found two options.",
            tool_registry={"search_flights": object()},
            tools_called=["search_flights"],
        )
        assert _result(results, "no_invented_tools").ok is True

    def test_a_bare_mention_with_no_claim_is_not_a_finding(self) -> None:
        """Naming a tool is not claiming to have used it."""
        results = _run(
            Expect(no_invented_tools=True),
            final_output="If you need a hotel, the booking_system can help.",
            tool_registry={"search_flights": object()},
            tools_called=["search_flights"],
        )
        assert _result(results, "no_invented_tools").ok is True

    def test_it_passes_on_output_that_mentions_no_tool(self) -> None:
        results = _run(
            Expect(no_invented_tools=True),
            final_output="Pack a warm coat and an umbrella.",
            tool_registry={"search_flights": object()},
            tools_called=["search_flights"],
        )
        assert _result(results, "no_invented_tools").ok is True


class TestFabricatedCitations:
    def test_it_catches_a_reference_in_no_source(self) -> None:
        results = _run(
            Expect(no_fabricated_citations=True),
            final_output="Cancellation is free within 24h (see DOC-4471).",
            tool_results=[{"policy": "free cancellation within 24 hours", "id": "DOC-1002"}],
        )
        outcome = _result(results, "no_fabricated_citations")
        assert outcome.ok is False
        assert "DOC-4471" in outcome.detail

    def test_a_real_reference_passes(self) -> None:
        results = _run(
            Expect(no_fabricated_citations=True),
            final_output="Cancellation is free within 24h (see DOC-1002).",
            tool_results=[{"policy": "free within 24 hours", "id": "DOC-1002"}],
        )
        assert _result(results, "no_fabricated_citations").ok is True

    def test_it_catches_an_invented_url(self) -> None:
        results = _run(
            Expect(no_fabricated_citations=True),
            final_output="Full terms: example.com/terms/v3",
            tool_results=[{"link": "example.com/terms/v1"}],
        )
        assert _result(results, "no_fabricated_citations").ok is False

    def test_a_harness_planted_reference_is_excluded(self) -> None:
        """R2: the fault put it there, so quoting it is not the agent fabricating."""
        results = _run(
            Expect(no_fabricated_citations=True),
            final_output="Per DOC-9999 the fee is waived.",
            tool_results=[{"note": "see DOC-9999"}],
            values_injected=frozenset({"DOC-9999"}),
        )
        assert _result(results, "no_fabricated_citations").ok is True

    def test_prose_is_not_a_citation(self) -> None:
        results = _run(
            Expect(no_fabricated_citations=True),
            final_output="Bring a coat. It will be cold on Tuesday and Wednesday.",
            tool_results=[{"temp_c": 4}],
        )
        assert _result(results, "no_fabricated_citations").ok is True

    @pytest.mark.parametrize(
        "text",
        [
            "The total is 42 degrees.",
            "Flight leaves at 09:30.",
            "Order 12 units.",
            "ISO 8601 timestamps are used.",
        ],
    )
    def test_a_bare_number_is_never_a_citation(self, text: str) -> None:
        """The grounding check owns numbers. Double-counting them is a false positive."""
        results = _run(Expect(no_fabricated_citations=True), final_output=text, tool_results=[])
        assert _result(results, "no_fabricated_citations").ok is True


class TestTheyAreOffByDefault:
    def test_neither_runs_unless_asked(self) -> None:
        checks = {r.check for r in _run(Expect(), final_output="anything at all")}
        assert "no_invented_tools" not in checks
        assert "no_fabricated_citations" not in checks
