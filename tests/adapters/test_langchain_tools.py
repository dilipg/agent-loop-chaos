"""A LangChain `@tool` object is a tool, and `engine.tool` has to take one.

`@tool` produces a `StructuredTool`, not a function. It is not callable — it is
invoked through `.invoke()` — so `inspect.signature` on it raises
`TypeError: … is not a callable object` and `engine.tool(cite_source)` died before it
started.

That matters because `@tool` is *the* way tools are declared in a LangChain or
LangGraph codebase. An agent whose entire shared tool layer is `@tool` primitives
could not be instrumented at all, which is most of them (D-138).

The fix wraps the tool's underlying `func`/`coroutine` **in place** and returns the
same object, so anything already holding the tool -- a registry, a bound model, a
`ToolNode` -- goes through the wrapper without being rebound.
"""

from __future__ import annotations

import tempfile
from typing import Any

import pytest

pytest.importorskip("langchain_core", reason="the langgraph extra is not installed")

from langchain_core.tools import tool

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import ToolCorruptionFault


@tool
def cite_source(source_id: str, note: str = "") -> dict[str, Any]:
    """Record a citation, as a shared platform layer would."""
    return {"cited": source_id, "note": note, "score": 42}


@tool
async def fetch_async(location_id: str) -> dict[str, Any]:
    """An async tool, which `@tool` also produces."""
    return {"location": location_id, "score": 7}


def _engine(**kw: Any) -> ChaosEngine:
    return ChaosEngine(seed=1337, out_dir=tempfile.mkdtemp(), judge="rules", **kw)


class TestItAcceptsAToolObject:
    def test_engine_tool_takes_a_structured_tool(self) -> None:
        engine = _engine()
        wrapped = engine.tool(cite_source, name="cite_source")
        assert wrapped is not None

    def test_the_tool_still_works_as_a_tool(self) -> None:
        """It has to keep its LangChain identity, or the graph cannot use it."""
        engine = _engine()
        wrapped = engine.tool(cite_source, name="cite_source")
        assert hasattr(wrapped, "invoke"), "the tool lost its LangChain interface"
        assert wrapped.name == "cite_source"

    def test_it_is_the_same_object(self) -> None:
        """A registry or a bound model already holds it; rebinding is not an option."""
        engine = _engine()
        assert engine.tool(cite_source, name="cite_source") is cite_source

    def test_invoking_it_reaches_the_engine(self) -> None:
        engine = _engine()
        engine.tool(cite_source, name="cite_source")

        @engine.intercept_tools()
        def agent(question: str) -> str:
            return str(cite_source.invoke({"source_id": "src-1"}))

        result = engine.run(agent, inputs="q")
        assert [c["tool"] for c in result.tool_calls] == ["cite_source"]

    def test_a_fault_reaches_it(self) -> None:
        engine = _engine()
        engine.tool(cite_source, name="cite_source")
        engine.register_fault(
            ToolCorruptionFault(mutation_type="drop_key", keys=["score"]),
            target_tool="cite_source",
        )

        @engine.intercept_tools()
        def agent(question: str) -> str:
            return str(cite_source.invoke({"source_id": "src-1"})["score"])

        result = engine.run(agent, inputs="q")
        assert [f["type"] for f in result.injected_faults if f["fired"]] == ["ToolCorruptionFault"]
        assert result.success is False

    def test_the_name_defaults_to_the_tools_own(self) -> None:
        engine = _engine()
        engine.tool(cite_source)

        @engine.intercept_tools()
        def agent(question: str) -> str:
            return str(cite_source.invoke({"source_id": "s"}))

        assert [c["tool"] for c in engine.run(agent, inputs="q").tool_calls] == ["cite_source"]


class TestAsync:
    def test_an_async_tool_is_accepted(self) -> None:
        engine = _engine()
        assert engine.tool(fetch_async, name="fetch_async") is fetch_async

    def test_awaiting_it_reaches_the_engine(self) -> None:
        import asyncio

        engine = _engine()
        engine.tool(fetch_async, name="fetch_async")

        @engine.intercept_tools()
        async def agent(question: str) -> str:
            return str(await fetch_async.ainvoke({"location_id": "loc-1"}))

        result = asyncio.run(engine.arun(agent, inputs="q"))
        assert [c["tool"] for c in result.tool_calls] == ["fetch_async"]


class TestInstrumentObjectTakesARegistry:
    def test_a_dict_of_tools_can_be_wrapped(self) -> None:
        """The shape a shared platform layer exposes: a name-to-tool catalog."""
        engine = _engine()
        registry = {"cite_source": cite_source}
        for name, obj in registry.items():
            engine.tool(obj, name=name)

        @engine.intercept_tools()
        def agent(question: str) -> str:
            return str(registry["cite_source"].invoke({"source_id": "s"}))

        assert [c["tool"] for c in engine.run(agent, inputs="q").tool_calls] == ["cite_source"]
