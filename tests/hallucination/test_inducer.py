"""`HallucinationInducerFault` — the known ways to make a model make things up.

`HallucinationSeedFault` *forces* a wrong answer by rewriting the response. This is
the other half, and the more honest test: it plants the **conditions** under which a
real model fabricates, and then lets the agent decide. Nothing about the response is
touched, so a passing run is the agent genuinely refusing the bait.

Every mode is a deterministic string transform. A model call would make a seeded run
unreproducible, and none of these need fluency to work.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.faults import HallucinationInducerFault
from agent_loop_chaos.faults.base import FAULT_REGISTRY

MODES = (
    "false_premise",
    "unanswerable_request",
    "citation_pressure",
    "authority_bias",
    "leading_question",
    "entity_lookalike",
)


def _apply(fault: Any, messages: list[dict[str, str]]) -> Any:
    from tests.faults.test_llm_faults import context, llm_pre

    return fault.apply(llm_pre([dict(m) for m in messages]), context())


class TestTheCatalogEntry:
    def test_it_is_registered(self) -> None:
        assert FAULT_REGISTRY["HallucinationInducerFault"] is HallucinationInducerFault

    def test_it_acts_before_the_model_call(self) -> None:
        """It plants conditions. Touching the response would be seeding, not inducing."""
        assert HallucinationInducerFault.accepts == frozenset({("llm", "pre")})

    def test_it_performs_no_real_action(self) -> None:
        assert HallucinationInducerFault.performs_real_action is False

    def test_an_unknown_mode_is_refused_at_registration(self) -> None:
        with pytest.raises(ConfigError, match="mode"):
            HallucinationInducerFault(mode="make_it_lie").validate()

    def test_every_mode_is_documented_in_the_docstring(self) -> None:
        doc = HallucinationInducerFault.__doc__ or ""
        for mode in MODES:
            assert mode in doc, f"{mode} is undocumented"


class TestEveryModeInduces:
    @pytest.mark.parametrize("mode", MODES)
    def test_it_changes_the_conversation(self, mode: str) -> None:
        before = [{"role": "user", "content": "What should I pack for Paris?"}]
        outcome = _apply(HallucinationInducerFault(mode=mode), before)
        assert outcome.action == "replace_messages"
        assert outcome.value != before
        assert outcome.note

    @pytest.mark.parametrize("mode", MODES)
    def test_it_never_touches_the_response(self, mode: str) -> None:
        outcome = _apply(HallucinationInducerFault(mode=mode), [{"role": "user", "content": "hi"}])
        assert outcome.action == "replace_messages"

    @pytest.mark.parametrize("mode", MODES)
    def test_it_records_what_it_planted(self, mode: str) -> None:
        """The identifiers need this: R2 excludes anything the harness itself wrote."""
        outcome = _apply(HallucinationInducerFault(mode=mode), [{"role": "user", "content": "hi"}])
        assert outcome.params["inducer_text"]
        assert outcome.params["mode"] == mode
        assert outcome.mutation is not None

    @pytest.mark.parametrize("mode", MODES)
    def test_it_is_deterministic(self, mode: str) -> None:
        messages = [{"role": "user", "content": "What should I pack?"}]
        first = _apply(HallucinationInducerFault(mode=mode), messages)
        second = _apply(HallucinationInducerFault(mode=mode), messages)
        assert first.value == second.value

    @pytest.mark.parametrize("mode", MODES)
    def test_it_does_not_mutate_the_caller_s_messages(self, mode: str) -> None:
        messages = [{"role": "user", "content": "What should I pack?"}]
        _apply(HallucinationInducerFault(mode=mode), messages)
        assert messages == [{"role": "user", "content": "What should I pack?"}]


class TestTheModesDoWhatTheyClaim:
    def test_false_premise_asserts_something_untrue(self) -> None:
        outcome = _apply(
            HallucinationInducerFault(mode="false_premise"),
            [{"role": "user", "content": "Is my booking refundable?"}],
        )
        planted = outcome.params["inducer_text"].lower()
        assert "confirmed" in planted or "established" in planted or "earlier" in planted

    def test_unanswerable_request_asks_for_what_no_tool_has(self) -> None:
        outcome = _apply(
            HallucinationInducerFault(mode="unanswerable_request"),
            [{"role": "user", "content": "Plan my trip."}],
        )
        assert outcome.params["inducer_text"]
        assert outcome.params["expected_refusal"] is True

    def test_citation_pressure_demands_a_source(self) -> None:
        outcome = _apply(
            HallucinationInducerFault(mode="citation_pressure"),
            [{"role": "user", "content": "What is the policy?"}],
        )
        assert "cite" in outcome.params["inducer_text"].lower()

    def test_entity_lookalike_names_a_near_identical_id(self) -> None:
        outcome = _apply(
            HallucinationInducerFault(mode="entity_lookalike"),
            [{"role": "user", "content": "Check booking BK-1042."}],
        )
        planted = outcome.params["inducer_text"]
        assert "BK-1042" not in planted
        assert outcome.params.get("lookalike_of") == "BK-1042"

    def test_entity_lookalike_falls_back_when_there_is_no_id(self) -> None:
        outcome = _apply(
            HallucinationInducerFault(mode="entity_lookalike"),
            [{"role": "user", "content": "Plan a trip."}],
        )
        assert outcome.params["inducer_text"]


class TestPlacement:
    def test_it_lands_where_asked(self) -> None:
        before = [
            {"role": "system", "content": "You are careful."},
            {"role": "user", "content": "What should I pack?"},
        ]
        outcome = _apply(
            HallucinationInducerFault(mode="false_premise", position="before_last_user"), before
        )
        assert len(outcome.value) == len(before) + 1

    def test_an_unknown_position_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="position"):
            HallucinationInducerFault(mode="false_premise", position="sideways").validate()
