"""Instrument a graph the caller never hands over.

`instrument_graph(app, engine)` needs the graph object. Most services do not have one to
give: `run_user_pipeline(...)` builds it, compiles it, invokes it and returns a result,
so node, edge, state and checkpoint faults had nothing to attach to unless someone wrote
a harness to expose the graph. That harness is what this removes -- `StateGraph.compile`
is patched for the duration of the run, exactly as `BaseChatModel.generate` is, so any
graph compiled while the run is active comes back instrumented.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine, Target
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.faults import NodeSkipFault, StateDropFault
from agent_loop_chaos.interceptors import Registry
from agent_loop_chaos.interceptors.langgraph import LangGraphStrategy
from agent_loop_chaos.interceptors.seams import SeamsStrategy

pytest.importorskip("langgraph", reason="the langgraph extra is not installed")

from tests.fakes import internal_graph


def _engine(tmp_path: Any, **kw: Any) -> ChaosEngine:
    return ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False, **kw)


def _under(engine: ChaosEngine, **kw: Any) -> Any:
    registry = Registry([LangGraphStrategy()])
    registry.attach(engine)
    try:
        return engine.run(internal_graph.run, **kw)
    finally:
        registry.detach()


class TestItInstrumentsAGraphItWasNeverGiven:
    def test_nodes_become_steps(self, tmp_path: Any) -> None:
        result = _under(_engine(tmp_path), inputs="what should I pack?")
        assert result.metrics["steps"] > 0, "no node was instrumented"

    def test_a_node_fault_fires(self, tmp_path: Any) -> None:
        engine = _engine(tmp_path)
        engine.register_fault(NodeSkipFault(), target_node="gather")
        result = _under(engine, inputs="q")
        assert result.injected_faults[0]["fired"] is True
        assert "could not read" in str(result.final_output), (
            "the skip did not reach the agent's answer"
        )

    def test_a_state_fault_fires(self, tmp_path: Any) -> None:
        engine = _engine(tmp_path)
        # `keys=` is what the fault drops; the target only selects the crossing. Left
        # off, the fault drops nothing and D-132 correctly refuses to call it a fire.
        engine.register_fault(
            StateDropFault(keys=["facts"]), target=Target(layer="state", state_key="facts")
        )
        result = _under(engine, inputs="q")
        assert result.injected_faults[0]["fired"] is True

    def test_without_it_nothing_is_instrumented(self, tmp_path: Any) -> None:
        """Otherwise the tests above would pass for the wrong reason."""
        engine = _engine(tmp_path)
        engine.register_fault(NodeSkipFault(), target_node="gather")
        result = engine.run(internal_graph.run, inputs="q")
        assert result.metrics["steps"] == 0
        assert result.injected_faults[0]["fired"] is False


class TestItRestoresCompile:
    def test_compile_is_the_original_afterwards(self, tmp_path: Any) -> None:
        from langgraph.graph import StateGraph

        original = StateGraph.compile
        registry = Registry([LangGraphStrategy()])
        registry.attach(_engine(tmp_path))
        assert StateGraph.compile is not original
        registry.detach()
        assert StateGraph.compile is original

    def test_it_does_nothing_outside_a_run(self, tmp_path: Any) -> None:
        registry = Registry([LangGraphStrategy()])
        registry.attach(_engine(tmp_path))
        try:
            assert "21C" in internal_graph.run("q")
        finally:
            registry.detach()

    def test_a_graph_already_instrumented_is_not_wrapped_twice(self, tmp_path: Any) -> None:
        """A repo whose builder calls `instrument_graph` itself must not end up with
        two layers of wrapping, which would double every crossing."""
        engine = _engine(tmp_path)
        registry = Registry([LangGraphStrategy()])
        registry.attach(engine)
        try:
            result = engine.run(internal_graph.run, inputs="q")
            once = result.metrics["steps"]
        finally:
            registry.detach()

        registry2 = Registry([LangGraphStrategy(), LangGraphStrategy()])
        registry2.attach(engine)
        try:
            twice = engine.run(internal_graph.run, inputs="q").metrics["steps"]
        finally:
            registry2.detach()
        assert twice == once, f"steps doubled: {once} -> {twice}"


class TestTheDefaultSetIncludesIt:
    def test_intercept_covers_the_graph_layers(self, tmp_path: Any) -> None:
        """`intercept=True` must reach nodes too, not only models and tools."""
        engine = _engine(tmp_path, intercept=True)
        engine.register_fault(NodeSkipFault(), target_node="gather")
        result = engine.run(internal_graph.run, inputs="q")
        assert result.injected_faults[0]["fired"] is True, (
            "`intercept=True` did not instrument a graph compiled inside the agent"
        )


class TestAGraphCompiledAtImport:
    """The other common shape, and the one patching `compile` cannot reach.

    A service that reuses one graph across invocations compiles it at import --
    `_workflow = build_workflow()` at module level -- so by the time a run starts there
    is no compile left to intercept. Naming it as a seam is the answer, because the
    graph is exactly the kind of thing a dotted path can point at.
    """

    def test_naming_the_graph_instruments_it(self, tmp_path: Any) -> None:
        engine = _engine(tmp_path)
        engine.register_fault(NodeSkipFault(), target_node="gather")
        registry = Registry([SeamsStrategy({"graph": ["tests.fakes.internal_graph:COMPILED"]})])
        registry.attach(engine)
        try:
            result = engine.run(internal_graph.run_singleton, inputs="q")
        finally:
            registry.detach()

        assert result.metrics["steps"] > 0, "the named graph was not instrumented"
        assert result.injected_faults[0]["fired"] is True
        assert "could not read" in str(result.final_output)

    def test_the_graph_is_left_uninstrumented(self, tmp_path: Any) -> None:
        """`instrument_graph` wraps a graph's slots in place and returns the same
        object, so restoring the attribute undoes nothing. A module-level graph outlives
        the run, and a second scenario would route its crossings to the first engine."""
        from agent_loop_chaos.adapters.langgraph import node_slots

        def inner() -> dict[str, Any]:
            # The container is mutated in place, so its identity says nothing. What
            # changes is the `func` inside it, and the marker stamped on it.
            return {
                name: (getattr(get_(), "func", None), getattr(get_(), "__alc_instrumented__", None))
                for name, get_, _set in node_slots(internal_graph.COMPILED)
            }

        before = inner()
        registry = Registry([SeamsStrategy({"graph": ["tests.fakes.internal_graph:COMPILED"]})])
        registry.attach(_engine(tmp_path))
        during = inner()
        assert during != before, "the graph was never instrumented"
        registry.detach()
        assert inner() == before, "the graph is still instrumented after the run"

    def test_a_path_that_is_not_a_graph_is_refused(self, tmp_path: Any) -> None:
        """Naming the wrong thing must fail at attach time, not mid-run."""
        strategy = SeamsStrategy({"graph": ["tests.fakes.internal_graph:run"]})
        with pytest.raises(ConfigError, match="graph"):
            strategy.attach(_engine(tmp_path), {})
        strategy.detach()

    def test_two_scenarios_in_a_row_both_fire(self, tmp_path: Any) -> None:
        """The property the snapshot exists for. A suite is many scenarios, each with
        its own engine; if the first run left the singleton instrumented, the second
        would keep routing crossings to the first engine and fire nothing."""
        fired = []
        for _ in range(2):
            engine = _engine(tmp_path)
            engine.register_fault(NodeSkipFault(), target_node="gather")
            registry = Registry([SeamsStrategy({"graph": ["tests.fakes.internal_graph:COMPILED"]})])
            registry.attach(engine)
            try:
                result = engine.run(internal_graph.run_singleton, inputs="q")
            finally:
                registry.detach()
            fired.append(result.injected_faults[0]["fired"])
        assert fired == [True, True], f"the second run did not fire: {fired}"
