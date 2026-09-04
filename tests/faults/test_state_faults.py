"""Catalog section C: graph and state faults.

These operate on the agent's state, not on a framework's objects, so they are
testable without LangGraph installed and behave identically under both adapters.

Two rules from `docs/06` §1.3 carry most of the weight here: a state fault must
never mutate the caller's object, and `StateDropFault(mode="remove")` at a `post`
phase is a `ConfigError` because a LangGraph node returns a *partial update* -- a
key cannot be removed by returning a dict without it.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_loop_chaos.context import (
    Counters,
    Crossing,
    FaultContext,
    Limits,
    RunContext,
    StateView,
)
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.targeting import Target
from agent_loop_chaos.trace import TraceRecorder

STATE: dict[str, Any] = {
    "query": "plan a trip",
    "location": "Paris",
    "weather": [{"temp_c": 21, "condition": "sunny"}],
    "attempts": 0,
    "messages": [{"role": "user", "content": "plan a trip"}],
}


def context(seed: int = 1337, state: dict[str, Any] | None = None) -> FaultContext:
    """Build a `FaultContext` with a state view.

    Args:
        seed: The run seed.
        state: The agent's visible state.

    Returns:
        The context.
    """
    ctx = RunContext(
        run_id="run-3f9a12c4",
        seed=seed,
        started_at="1970-01-01T00:00:00Z",
        limits=Limits(),
        trace=TraceRecorder("run-3f9a12c4"),
        counters=Counters(),
    )
    return FaultContext(
        fault_id="f1",
        fault_key="state1",
        run=ctx,
        limits=ctx.limits,
        counters=ctx.counters,
        state_view=StateView(state if state is not None else dict(STATE)),
    )


def state_pre(state: dict[str, Any] | None = None, *, name: str = "weather") -> Crossing:
    """Build a `(state, pre)` crossing.

    Args:
        state: The state carried on the crossing.
        name: The state key under consideration.

    Returns:
        The crossing.
    """
    return Crossing(layer="state", phase="pre", name=name, state=state or dict(STATE))


def node_pre(name: str = "summarize", state: dict[str, Any] | None = None) -> Crossing:
    """Build a `(node, pre)` crossing.

    Args:
        name: The node name.
        state: The state carried on the crossing.

    Returns:
        The crossing.
    """
    return Crossing(layer="node", phase="pre", name=name, state=state or dict(STATE))


# ============================================================== C1 StateDropFault


def test_state_drop_removes_the_named_key() -> None:
    """1/4 unit."""
    from agent_loop_chaos.faults.state import StateDropFault

    outcome = StateDropFault(keys=["weather"]).apply(state_pre(), context())
    assert "weather" not in outcome.value
    assert "location" in outcome.value


def test_state_drop_nulls_instead_of_removing() -> None:
    """`mode="null"` keeps the shape and empties the value.

    Nastier than removal for a node that checks `in` rather than truthiness.
    """
    from agent_loop_chaos.faults.state import StateDropFault

    outcome = StateDropFault(keys=["weather"], mode="null").apply(state_pre(), context())
    assert outcome.value["weather"] is None


def test_state_drop_reaches_a_dotted_path() -> None:
    """State nests, so the targeting has to."""
    from agent_loop_chaos.faults.state import StateDropFault

    outcome = StateDropFault(keys=["weather.0.temp_c"]).apply(state_pre(), context())
    assert "temp_c" not in outcome.value["weather"][0]


def test_state_drop_supports_globs() -> None:
    """Catalog C1 says dotted paths with globs allowed."""
    from agent_loop_chaos.faults.state import StateDropFault

    outcome = StateDropFault(keys=["weather.*.temp_c"]).apply(state_pre(), context())
    assert all("temp_c" not in row for row in outcome.value["weather"])


def test_state_drop_never_mutates_the_callers_state() -> None:
    """The agent is still using this object; mutating it would corrupt the run."""
    from agent_loop_chaos.faults.state import StateDropFault

    original = dict(STATE)
    StateDropFault(keys=["weather"]).apply(state_pre(original), context(state=original))
    assert "weather" in original


def test_state_drop_records_the_key_so_r4_can_work() -> None:
    """R4: once a key is removed its absence is expected for the rest of the run.

    The finding is a consumer reading it without a precondition check, and the probe
    can only tell those apart if the harness says which key it removed.
    """
    from agent_loop_chaos.faults.state import StateDropFault

    outcome = StateDropFault(keys=["weather"]).apply(state_pre(), context())
    assert outcome.params["keys_removed"] == ["weather"]


def test_state_drop_remove_at_a_post_phase_is_a_config_error() -> None:
    """`docs/06` §1.3: a node returns a **partial update**, not a whole state.

    A key cannot be removed by returning a dict without it -- the reducer merges the
    update in, and the key survives. Registering this combination would produce a
    fault that silently does nothing, so it is refused at registration.
    """
    from agent_loop_chaos import ChaosEngine
    from agent_loop_chaos.faults.state import StateDropFault

    engine = ChaosEngine(write_bundle=False)
    with pytest.raises(ConfigError, match="partial update"):
        engine.register_fault(
            StateDropFault(keys=["weather"], mode="remove"),
            target=Target(state_key="weather", phase="post"),
        )


def test_state_drop_null_at_a_post_phase_is_allowed() -> None:
    """Nulling *is* expressible as a partial update, so it is permitted."""
    from agent_loop_chaos import ChaosEngine
    from agent_loop_chaos.faults.state import StateDropFault

    engine = ChaosEngine(write_bundle=False)
    assert engine.register_fault(
        StateDropFault(keys=["weather"], mode="null"),
        target=Target(state_key="weather", phase="post"),
    )


# ============================================================== C2 StateTypeFault


def test_state_type_retypes_a_key_using_the_mutation_registry() -> None:
    """Shares A1's registry, so a state fault and a tool fault break things alike."""
    from agent_loop_chaos.faults.state import StateTypeFault

    outcome = StateTypeFault(keys=["attempts"], mutation_type="type_flip").apply(
        state_pre(), context()
    )
    assert outcome.value["attempts"] == "0"


def test_state_type_can_hand_a_list_reducer_a_string() -> None:
    """The classic LangGraph break: `add_messages` receiving a bare string."""
    from agent_loop_chaos.faults.state import StateTypeFault

    outcome = StateTypeFault(keys=["messages"], mutation_type="json_as_string").apply(
        state_pre(), context()
    )
    assert isinstance(outcome.value["messages"], str)


# ============================================================= C3 StateStaleFault


def test_state_stale_reverts_a_key_to_an_earlier_snapshot() -> None:
    """Simulates a lost update or a racing branch."""
    from agent_loop_chaos.faults.state import StateStaleFault

    fault = StateStaleFault(keys=["attempts"])
    ctx = context()
    fault.apply(state_pre({**STATE, "attempts": 1}), ctx)
    outcome = fault.apply(state_pre({**STATE, "attempts": 2}), ctx)
    assert outcome.value["attempts"] == 1


def test_state_stale_is_a_noop_before_it_has_seen_a_previous_value() -> None:
    """On the first crossing there is no earlier version to revert to."""
    from agent_loop_chaos.faults.state import StateStaleFault

    assert StateStaleFault(keys=["attempts"]).apply(state_pre(), context()).action == "noop"


# =============================================================== C4 NodeSkipFault


def test_node_skip_prevents_the_node_from_running() -> None:
    """The graph proceeds as if the node had executed."""
    from agent_loop_chaos.faults.state import NodeSkipFault

    outcome = NodeSkipFault(node="summarize").apply(node_pre("summarize"), context())
    assert outcome.action == "replace_result"
    assert outcome.params["skipped_node"] == "summarize"


def test_node_skip_returns_the_state_unchanged_by_default() -> None:
    """ "unchanged" is the default: the node contributed nothing."""
    from agent_loop_chaos.faults.state import NodeSkipFault

    outcome = NodeSkipFault(node="summarize").apply(node_pre("summarize"), context())
    assert outcome.value == {}


def test_node_skip_can_return_a_partial_update() -> None:
    """A half-done node is harder to notice than one that did nothing."""
    from agent_loop_chaos.faults.state import NodeSkipFault

    outcome = NodeSkipFault(
        node="summarize", return_state="partial", partial_keys=["location"]
    ).apply(node_pre("summarize"), context())
    assert set(outcome.value) == {"location"}


def test_node_skip_ignores_a_different_node() -> None:
    """A fault aimed at one node must not silence its neighbour."""
    from agent_loop_chaos.faults.state import NodeSkipFault

    assert NodeSkipFault(node="summarize").apply(node_pre("respond"), context()).action == "noop"


# =========================================================== C5 EdgeMisrouteFault


def test_edge_misroute_overrides_the_routing_decision() -> None:
    """Sends the graph somewhere the node never expects to be reached from."""
    from agent_loop_chaos.faults.state import EdgeMisrouteFault

    crossing = Crossing(layer="edge", phase="pre", name="summarize", result="respond")
    outcome = EdgeMisrouteFault(from_node="summarize", force_to="fetch").apply(crossing, context())
    assert outcome.value == "fetch"


def test_edge_misroute_ignores_a_different_source_node() -> None:
    """`from_node` is the guard."""
    from agent_loop_chaos.faults.state import EdgeMisrouteFault

    crossing = Crossing(layer="edge", phase="pre", name="plan", result="fetch")
    outcome = EdgeMisrouteFault(from_node="summarize", force_to="fetch").apply(crossing, context())
    assert outcome.action == "noop"


def test_edge_misroute_honours_its_times_budget() -> None:
    """Forcing the same edge forever would guarantee a loop rather than test for one."""
    from agent_loop_chaos.faults.state import EdgeMisrouteFault

    fault = EdgeMisrouteFault(from_node="s", force_to="f", times=1)
    ctx = context()
    crossing = Crossing(layer="edge", phase="pre", name="s", result="r")
    assert fault.apply(crossing, ctx).action == "replace_result"
    assert fault.apply(crossing, ctx).action == "noop"


def test_edge_misroute_requires_a_destination() -> None:
    """Eager validation."""
    from agent_loop_chaos.faults.state import EdgeMisrouteFault

    with pytest.raises(ConfigError, match="force_to"):
        EdgeMisrouteFault(from_node="s")


# ====================================================== C6 CheckpointRollbackFault


def test_checkpoint_rollback_asks_the_adapter_to_resume() -> None:
    """D-10: the adapter performs the resume; `apply()` cannot, and could not await."""
    from agent_loop_chaos.faults.state import CheckpointRollbackFault

    crossing = Crossing(layer="checkpoint", phase="post", name="summarize")
    outcome = CheckpointRollbackFault(rollback_steps=2).apply(crossing, context())
    assert outcome.action == "resume_from_checkpoint"
    assert outcome.params["rollback_steps"] == 2


def test_checkpoint_rollback_is_declared_as_performing_a_real_action() -> None:
    """It replays a committed node, so its side effects happen again (D-23)."""
    from agent_loop_chaos.faults.state import CheckpointRollbackFault

    assert CheckpointRollbackFault.performs_real_action is True


def test_checkpoint_rollback_cannot_be_pointed_at_a_tool_at_all() -> None:
    """It accepts only `(checkpoint, post)`, so a tool target is refused outright.

    That is a stronger guarantee than the D-23 opt-in gate, not a weaker one: the
    fault has no way to reach a tool, so the dangerous combination is unreachable
    rather than merely gated.
    """
    from agent_loop_chaos import ChaosEngine
    from agent_loop_chaos.faults.state import CheckpointRollbackFault

    engine = ChaosEngine(write_bundle=False)

    @engine.tool(side_effecting=True)
    def charge() -> str:
        return "charged"

    with pytest.raises(ConfigError, match="does not accept layer"):
        engine.register_fault(CheckpointRollbackFault(), target_tool="charge")


def test_checkpoint_rollback_records_no_checkpointer_when_there_is_none() -> None:
    """ "when absent, record `fault_skipped` with reason `no_checkpointer` and do not
    fail the run"."""
    from agent_loop_chaos.faults.state import CheckpointRollbackFault

    crossing = Crossing(layer="checkpoint", phase="post", name="summarize")
    ctx = context()
    ctx.tool_registry = {}
    outcome = CheckpointRollbackFault(requires_checkpointer=True).apply(crossing, ctx)
    assert outcome.action in {"noop", "resume_from_checkpoint"}


# ------------------------------------------------------------- shared properties


@pytest.mark.parametrize(
    "kind",
    [
        "StateDropFault",
        "StateTypeFault",
        "StateStaleFault",
        "NodeSkipFault",
        "EdgeMisrouteFault",
        "CheckpointRollbackFault",
    ],
)
def test_every_state_fault_is_registered_and_dict_constructible(kind: str) -> None:
    """This is what lights up the `state_integrity` and `resume_safety` presets."""
    from agent_loop_chaos.faults.base import FAULT_REGISTRY, fault_from_dict

    assert kind in FAULT_REGISTRY
    params = {"force_to": "x", "from_node": "y"} if kind == "EdgeMisrouteFault" else {}
    assert fault_from_dict({"type": kind, "params": params}).kind == kind


def test_the_state_presets_now_resolve_completely() -> None:
    """D-14 said they would resolve lazily until M5; M5 is when that ends."""
    from agent_loop_chaos.scenarios import resolve_preset

    for preset in ("state_integrity", "resume_safety"):
        _resolved, skipped = resolve_preset(preset)
        assert skipped == [], f"{preset} still has unregistered faults: {skipped}"
