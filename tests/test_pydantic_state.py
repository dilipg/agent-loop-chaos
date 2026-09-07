"""State faults against a Pydantic `BaseModel` state.

`StateGraph(MyModel)` with a Pydantic model is LangGraph's own recommended pattern and
what real projects use -- `Annotated[list[...], operator.add]` reducer channels on a
`BaseModel`. `StateView` handled `dict` only: `keys()` returned `[]` for anything
else, and `get`/`set` walked mappings.

So every state fault was **silently inert** against such an agent. Not refused, not
skipped with a reason -- no state crossing existed at all, so `StateDropFault` armed
and never fired, and the run was scored as though the fault had been tried (D-141).

Found by running the library against a real LangGraph 1.2.4 codebase: 11 node
crossings, zero state crossings.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

import pytest

pytest.importorskip("pydantic", reason="pydantic ships with the langgraph extra")

from pydantic import BaseModel, ConfigDict, Field

from agent_loop_chaos.context import StateView


class SkillState(BaseModel):
    """Shaped like the real thing: reducer channels on a model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    customer_id: str = "acme"
    location_context: dict[str, Any] = Field(default_factory=dict)
    signals: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)
    answer: str = ""


def _state() -> SkillState:
    return SkillState(
        location_context={"loc-1": {"name": "Riverside"}},
        signals=[{"kind": "ticket", "score": 3}],
    )


class TestItCanSeeAModelState:
    def test_keys_lists_the_fields(self) -> None:
        keys = StateView(_state()).keys()
        assert set(keys) == {"customer_id", "location_context", "signals", "answer"}

    def test_keys_are_sorted(self) -> None:
        keys = StateView(_state()).keys()
        assert keys == sorted(keys)

    def test_a_dict_state_is_unchanged(self) -> None:
        assert StateView({"b": 1, "a": 2}).keys() == ["a", "b"]

    def test_a_scalar_state_still_has_no_keys(self) -> None:
        assert StateView(42).keys() == []


class TestItCanReadAModelState:
    def test_a_top_level_field(self) -> None:
        assert StateView(_state()).get("customer_id") == "acme"

    def test_a_dotted_path_into_a_dict_field(self) -> None:
        assert StateView(_state()).get("location_context.loc-1.name") == "Riverside"

    def test_a_dotted_path_into_a_list_field(self) -> None:
        assert StateView(_state()).get("signals.0.kind") == "ticket"

    def test_a_missing_field_returns_the_default(self) -> None:
        assert StateView(_state()).get("nope", "fallback") == "fallback"


class TestItCanWriteAModelState:
    def test_a_top_level_field(self) -> None:
        state = _state()
        assert StateView(state).set("answer", "written") is True
        assert state.answer == "written"

    def test_a_retype_lands(self) -> None:
        """What `StateTypeFault` does. The model must not re-validate it away."""
        state = _state()
        assert StateView(state).set("signals", "not a list at all") is True
        assert state.signals == "not a list at all"

    def test_a_nested_write_lands(self) -> None:
        state = _state()
        assert StateView(state).set("location_context.loc-1.name", "Northgate") is True
        assert state.location_context["loc-1"]["name"] == "Northgate"

    def test_an_unknown_field_is_refused_rather_than_invented(self) -> None:
        """A write that silently creates a field would corrupt the state shape."""
        state = _state()
        assert StateView(state).set("not_a_field", 1) is False

    def test_a_frozen_model_is_refused_rather_than_raising(self) -> None:
        """The library must never raise into the agent over a payload shape."""

        class Frozen(BaseModel):
            model_config = ConfigDict(frozen=True)
            value: str = "x"

        assert StateView(Frozen()).set("value", "y") is False


class TestEndToEnd:
    def test_a_state_fault_fires_against_a_model_state(self, tmp_path: Any) -> None:
        """The finding this file exists for: it armed and never fired."""
        from agent_loop_chaos import ChaosEngine
        from agent_loop_chaos.faults import StateDropFault
        from agent_loop_chaos.targeting import Target

        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False)

        def node(state: SkillState) -> SkillState:
            # Reads the field the fault removes, without checking it is there --
            # which is the weakness a state fault exists to find.
            state.answer = f"context={state.location_context['loc-1']['name']}"
            return state

        def agent(inputs: Any) -> str:
            # State crossings happen at node boundaries, which is where the whole
            # state is visible and a partial update has not yet merged.
            done = engine.route_node(node, name="summarize", state=_state())
            return str(getattr(done, "answer", done))

        # `state_key` selects *which crossing*; `keys` selects *what to drop*. Both,
        # because a node boundary can carry many keys and the fault drops only what it
        # was told to.
        engine.register_fault(
            StateDropFault(keys=["location_context"]),
            target=Target(layer="state", state_key="location_context"),
        )
        result = engine.run(agent, inputs="q")
        fired = [f["type"] for f in result.injected_faults if f["fired"]]
        assert fired == ["StateDropFault"], (
            f"the fault armed and never fired: {result.injected_faults[0].get('skipped_reason')}"
        )
        # What the fix is about: the fault reached the state and changed it.
        #
        # `keys_removed` is the evidence, not `json_patch`. A Pydantic model has no
        # JSON form, so `MutationLog.of` cannot diff one and the patch is empty by
        # construction -- `keys_removed` is the fault's own record and is what R4
        # reads to tell a later unguarded read from the removal itself.
        fire = result.injected_faults[0]["fires"][0]
        assert fire["params"]["keys_removed"] == ["location_context"], fire["note"]
        # And it reached the node: this one reads the field with no precondition
        # check, so the drop surfaces as a TypeError attributed to the node's own
        # frame -- which is the weakness a state fault exists to find.
        assert (result.error or {}).get("type") == "TypeError", result.error
        assert result.success is False
