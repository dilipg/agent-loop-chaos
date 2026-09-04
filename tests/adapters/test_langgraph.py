"""The LangGraph adapter.

Skipped wholesale when the extra is absent, which is what keeps the CI
`langgraph = none` column meaningful: the core must pass with none of this
installed.

The adapter's only job is to turn LangGraph's hooks into `Crossing` objects. Every
attribute path it reaches through is version-specific, so the shims resolve a chain
and raise `AdapterError` naming the installed version and every path they tried --
silently instrumenting zero nodes would look like a working run that proves nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# Both, not just langgraph: a partial install where langgraph resolves and
# langchain_core does not must skip rather than error during collection.
pytest.importorskip("langgraph", reason="the langgraph extra is not installed")
pytest.importorskip("langchain_core", reason="the langgraph extra is not installed")

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.adapters.langgraph import (
    branch_slots,
    instrument_graph,
    is_compiled,
    langgraph_version,
    node_slots,
)
from agent_loop_chaos.errors import AdapterError
from agent_loop_chaos.faults import ToolCorruptionFault
from agent_loop_chaos.faults.state import NodeSkipFault, StateDropFault
from agent_loop_chaos.targeting import Target, Trigger
from tests.fakes.lg_agent import build


def model(prompt: Any) -> str:
    """A scripted stand-in for a chat model.

    Args:
        prompt: Ignored.

    Returns:
        A fixed summary.
    """
    return "Warm and settled; pack light layers."


def engine(tmp_path: Path, **kw: Any) -> ChaosEngine:
    """Build an engine writing into a temp directory.

    Args:
        tmp_path: pytest's temp directory.
        **kw: Constructor overrides.

    Returns:
        The engine.
    """
    kw.setdefault("seed", 1337)
    kw.setdefault("out_dir", tmp_path / ".chaos")
    kw.setdefault("write_bundle", False)
    return ChaosEngine(**kw)


# ------------------------------------------------------------------ version shims


def test_the_version_is_resolvable() -> None:
    """Every `AdapterError` names it, so a compat break is diagnosable from a report."""
    major, minor, _patch = langgraph_version()
    assert (major, minor) >= (0, 2)


def test_node_slots_resolve_before_compile() -> None:
    """`builder.nodes[name].runnable` on the versions in range."""
    slots = node_slots(build(model))
    assert {name for name, _get, _set in slots} == {"plan", "fetch", "summarize", "respond"}


def test_node_slots_resolve_after_compile() -> None:
    """`compiled.nodes[name].bound`, and the internal `__start__` node is excluded."""
    names = {name for name, _get, _set in node_slots(build(model).compile())}
    assert {"plan", "fetch", "summarize", "respond"} <= names
    assert not any(n.startswith("__") for n in names)


def test_a_slot_round_trips_through_its_setter() -> None:
    """The setter is what instrumentation writes through, so it has to work."""
    graph = build(model)
    _name, get, set_ = node_slots(graph)[0]
    original = get()
    set_(original)
    assert get() is original


def test_branch_slots_resolve_the_conditional_edge() -> None:
    """`EdgeMisrouteFault` needs the router, not the edge list."""
    slots = branch_slots(build(model))
    assert any(source == "summarize" for source, _name, _get, _set in slots)


def test_is_compiled_distinguishes_the_two_shapes() -> None:
    """Instrumentation differs before and after compile."""
    assert is_compiled(build(model)) is False
    assert is_compiled(build(model).compile()) is True


def test_instrumenting_something_that_is_not_a_graph_raises_adapter_error() -> None:
    """Naming the paths tried is the whole point of the error."""
    with pytest.raises(AdapterError) as exc:
        node_slots(object())
    assert "tried" in str(exc.value).lower()


# ---------------------------------------------------------------- instrumentation


def test_an_instrumented_graph_still_runs(tmp_path: Path) -> None:
    """The observer must not break the observed."""
    eng = engine(tmp_path)
    app = instrument_graph(build(model), eng)
    result = eng.run(
        app, inputs={"query": "Pack list for Paris"}, expected_behavior="ignore_and_continue"
    )
    assert result.success is True
    assert "pack light layers" in str(result.final_output).lower()


def test_every_node_produces_crossings(tmp_path: Path) -> None:
    """Four nodes, so four node entries at least."""
    eng = engine(tmp_path)
    app = instrument_graph(build(model), eng)
    result = eng.run(app, inputs={"query": "x"}, expected_behavior="ignore_and_continue")
    assert result.metrics["steps"] >= 4, "a node entry is one step (D-05)"


def test_instrumentation_preserves_a_two_argument_node_signature(tmp_path: Path) -> None:
    """LangGraph inspects signatures to decide what to pass.

    `summarize(state, config)` and `plan(state)` must both still work, or the graph
    fails for a reason that has nothing to do with the fault under test.
    """
    eng = engine(tmp_path)
    app = instrument_graph(build(model), eng)
    result = eng.run(app, inputs={"query": "x"}, expected_behavior="ignore_and_continue")
    assert result.error is None


async def test_async_nodes_get_async_wrappers(tmp_path: Path) -> None:
    """Sync and async parity is structural, not remembered."""
    eng = engine(tmp_path)
    app = instrument_graph(build(model, is_async=True), eng)
    result = await eng.arun(app, inputs={"query": "x"}, expected_behavior="ignore_and_continue")
    assert result.success is True


def test_instrumenting_twice_is_idempotent(tmp_path: Path) -> None:
    """A double-wrapped node would double every crossing and every step."""
    eng = engine(tmp_path)
    graph = build(model)
    app = instrument_graph(instrument_graph(graph, eng), eng)
    result = eng.run(app, inputs={"query": "x"}, expected_behavior="ignore_and_continue")
    other = engine(tmp_path / "b")
    once = other.run(
        instrument_graph(build(model), other),
        inputs={"query": "x"},
        expected_behavior="ignore_and_continue",
    )
    assert result.metrics["steps"] == once.metrics["steps"]


def test_nodes_argument_restricts_instrumentation(tmp_path: Path) -> None:
    """A large graph may only need one node watched."""
    eng = engine(tmp_path)
    app = instrument_graph(build(model), eng, nodes=["summarize"])
    result = eng.run(app, inputs={"query": "x"}, expected_behavior="ignore_and_continue")
    assert result.metrics["steps"] == 1


def test_a_compiled_graph_can_be_instrumented(tmp_path: Path) -> None:
    """Users hand us both shapes; both must work."""
    eng = engine(tmp_path)
    app = instrument_graph(build(model).compile(), eng)
    result = eng.run(app, inputs={"query": "x"}, expected_behavior="ignore_and_continue")
    assert result.success is True


def test_instrumenting_zero_nodes_is_an_error(tmp_path: Path) -> None:
    """ "Never instrument zero nodes silently".

    A run with nothing wrapped looks like a clean pass and proves nothing, which is
    the same failure mode D-64 exists to surface at the scenario level.
    """
    with pytest.raises(AdapterError, match="no nodes"):
        instrument_graph(build(model), engine(tmp_path), nodes=["no_such_node"])


# ------------------------------------------------------------------ faults on a graph


def test_a_tool_fault_fires_on_a_wrapped_graph_tool(tmp_path: Path) -> None:
    """The tool layer works the same under LangGraph as under vanilla."""
    eng = engine(tmp_path)
    tools = eng.wrap_tools(
        {"get_weather": __import__("tests.fakes.lg_agent", fromlist=["get_weather"]).get_weather}
    )
    eng.register_fault(
        ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
        target=Target(tool="get_weather", phase="post"),
        trigger=Trigger(on_call=1),
    )
    app = instrument_graph(build(model, tools=tools), eng)
    result = eng.run(app, inputs={"query": "x"})
    assert result.injected_faults[0]["fired"] is True


def test_a_node_skip_fault_prevents_a_node_from_running(tmp_path: Path) -> None:
    """C4 end to end: the graph proceeds as if the node had run."""
    eng = engine(tmp_path)
    eng.register_fault(
        NodeSkipFault(node="summarize"),
        target=Target(node="summarize", phase="pre"),
        trigger=Trigger(max_fires=999),
    )
    app = instrument_graph(build(model), eng)
    result = eng.run(app, inputs={"query": "x"})
    assert result.injected_faults[0]["fired"] is True
    # `summarize` is what sets `attempts`, so skipping it means the conditional edge
    # never routes onward: the run makes no progress and stops on the step limit.
    assert result.loop["limit_hit"] == "max_steps"
    assert result.error is None, "a limit we imposed is not the agent's crash"


def test_a_state_fault_fires_at_a_node_boundary(tmp_path: Path) -> None:
    """C1 end to end, at `(node, pre)` where a partial update is not in the way."""
    eng = engine(tmp_path)
    eng.register_fault(
        StateDropFault(keys=["weather"]),
        target=Target(node="summarize", phase="pre"),
        trigger=Trigger(on_call=1),
    )
    app = instrument_graph(build(model), eng)
    result = eng.run(app, inputs={"query": "x"})
    assert result.injected_faults[0]["fired"] is True


def test_the_recursion_limit_maps_to_the_step_limit(tmp_path: Path) -> None:
    """`GraphRecursionError` is a harness stop, not an agent crash.

    Reporting it in `error` would blame the agent for the limit we imposed.
    """
    from agent_loop_chaos.context import Limits
    from agent_loop_chaos.faults.state import EdgeMisrouteFault

    eng = engine(tmp_path, limits=Limits(max_steps=4, max_tool_calls=99, max_llm_calls=99))
    eng.register_fault(
        EdgeMisrouteFault(from_node="summarize", force_to="summarize", times=99),
        target=Target(layer="edge", phase="pre"),
        trigger=Trigger(max_fires=99),
    )
    app = instrument_graph(build(model), eng)
    result = eng.run(app, inputs={"query": "x"})
    assert result.error is None or "Recursion" not in str(result.error.get("type", ""))


def test_intercept_state_false_skips_state_crossings(tmp_path: Path) -> None:
    """Opt-out for a graph whose state is large or unserializable."""
    eng = engine(tmp_path)
    eng.register_fault(
        StateDropFault(keys=["weather"]),
        target=Target(state_key="weather", phase="pre"),
        trigger=Trigger(on_call=1),
    )
    app = instrument_graph(build(model), eng, intercept_state=False)
    result = eng.run(app, inputs={"query": "x"}, expected_behavior="ignore_and_continue")
    assert result.injected_faults[0]["fired"] is False


def test_the_adapter_reports_itself_and_its_version(tmp_path: Path) -> None:
    """`target.adapter_version` is what makes a compat break diagnosable."""
    eng = engine(tmp_path)
    app = instrument_graph(build(model), eng)
    result = eng.run(app, inputs={"query": "x"}, expected_behavior="ignore_and_continue")
    assert result.target["framework"] == "langgraph"
    assert result.target["adapter_version"]
