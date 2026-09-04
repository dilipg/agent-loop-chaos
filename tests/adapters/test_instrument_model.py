"""The model proxy and LangChain tool wrapping.

The proxy forwards everything it does not intercept, so a model keeps behaving like
a model: an adapter that quietly dropped `bind_tools` or a provider-specific
attribute would break the agent for a reason unrelated to any fault.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langchain_core", reason="the langgraph extra is not installed")

from langchain_core.messages import AIMessage, HumanMessage

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.adapters.langgraph import (
    instrument_model,
    wrap_langchain_tools,
)
from agent_loop_chaos.faults import LLMEmptyFault, LLMMalformedOutputFault
from agent_loop_chaos.targeting import Target, Trigger


class FakeChatModel:
    """Enough of a chat model to be bound, invoked and streamed."""

    def __init__(self, reply: str = "it is sunny") -> None:
        """Initialise.

        Args:
            reply: What every invocation returns.
        """
        self.reply = reply
        self.received: list[Any] = []
        self.bound_tools: list[Any] = []
        self.provider = "fake"

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        """Record the call and answer.

        Args:
            messages: The conversation.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            The scripted reply.
        """
        self.received.append(messages)
        return AIMessage(content=self.reply)

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        """Async twin of `invoke`.

        Args:
            messages: The conversation.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            The scripted reply.
        """
        return self.invoke(messages)

    def stream(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Yield the reply in two chunks.

        Args:
            messages: The conversation.
            *args: Ignored.
            **kwargs: Ignored.

        Yields:
            Chunks of the reply.
        """
        self.received.append(messages)
        yield AIMessage(content=self.reply[:5])
        yield AIMessage(content=self.reply[5:])

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChatModel:
        """Return a bound copy, as a real chat model does.

        Args:
            tools: The tools to bind.
            **kwargs: Ignored.

        Returns:
            A new model carrying the tools.
        """
        bound = FakeChatModel(self.reply)
        bound.bound_tools = list(tools)
        return bound


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


def test_the_proxy_forwards_unknown_attributes(tmp_path: Path) -> None:
    """A model must keep behaving like a model."""
    proxy = instrument_model(FakeChatModel(), engine(tmp_path))
    assert proxy.provider == "fake"


def test_invoke_is_intercepted_and_still_answers(tmp_path: Path) -> None:
    """The observer must not break the observed."""
    eng = engine(tmp_path)
    proxy = instrument_model(FakeChatModel(), eng)
    result = eng.run(
        lambda: proxy.invoke([HumanMessage(content="hi")]).content,
        expected_behavior="ignore_and_continue",
    )
    assert result.final_output == "it is sunny"
    assert result.metrics["llm_calls"] == 1


async def test_ainvoke_is_intercepted(tmp_path: Path) -> None:
    """Sync and async parity."""
    eng = engine(tmp_path)
    proxy = instrument_model(FakeChatModel(), eng)

    async def agent() -> str:
        return (await proxy.ainvoke([HumanMessage(content="hi")])).content

    result = await eng.arun(agent, expected_behavior="ignore_and_continue")
    assert result.final_output == "it is sunny"


def test_bind_tools_returns_a_still_instrumented_model(tmp_path: Path) -> None:
    """A bound copy that lost its instrumentation would silently stop being watched.

    Agents bind tools as a matter of course, so this is the common path, not an edge
    case.
    """
    eng = engine(tmp_path)
    proxy = instrument_model(FakeChatModel(), eng)
    bound = proxy.bind_tools([{"name": "get_weather"}])
    result = eng.run(
        lambda: bound.invoke([HumanMessage(content="hi")]).content,
        expected_behavior="ignore_and_continue",
    )
    assert result.metrics["llm_calls"] == 1, "the bound copy is still instrumented"
    assert bound.bound_tools


def test_a_post_fault_reaches_the_model_response(tmp_path: Path) -> None:
    """LLM faults work through the proxy exactly as under vanilla."""
    eng = engine(tmp_path)
    eng.register_fault(
        LLMEmptyFault(mode="empty_string"),
        target=Target(llm="default", phase="post"),
        trigger=Trigger(on_call=1),
    )
    proxy = instrument_model(FakeChatModel(), eng)
    result = eng.run(
        lambda: proxy.invoke([HumanMessage(content="hi")]), expected_behavior="ignore_and_continue"
    )
    assert result.injected_faults[0]["fired"] is True


def test_a_pre_fault_sees_normalized_messages(tmp_path: Path) -> None:
    """The whole point of `_lc_messages`: one fault, both adapters."""
    from agent_loop_chaos.faults import ContextNoiseFault

    eng = engine(tmp_path)
    eng.register_fault(
        ContextNoiseFault(noise_type="conflicting_instruction"),
        target=Target(llm="default", phase="pre"),
        trigger=Trigger(on_call=1),
    )
    model = FakeChatModel()
    proxy = instrument_model(model, eng)
    eng.run(
        lambda: proxy.invoke([HumanMessage(content="hi")]), expected_behavior="ignore_and_continue"
    )
    assert len(model.received[0]) == 2, "the model received the injected message"


def test_streaming_is_materialized_and_faulted_once(tmp_path: Path) -> None:
    """v0.1 limitation, recorded rather than pretended away.

    Per-chunk faults would need a streaming-aware fault protocol that does not exist;
    materializing and faulting the whole response once is honest and is written into
    the trace as a `log` event.
    """
    eng = engine(tmp_path)
    eng.register_fault(
        LLMMalformedOutputFault(mode="prose_instead_of_json"),
        target=Target(llm="default", phase="post"),
        trigger=Trigger(on_call=1),
    )
    proxy = instrument_model(FakeChatModel(), eng)

    def agent() -> list[Any]:
        return list(proxy.stream([HumanMessage(content="hi")]))

    result = eng.run(agent, expected_behavior="ignore_and_continue")
    assert len(result.final_output) == 1, "materialized into a single chunk"
    assert result.injected_faults[0]["fired"] is True


def test_the_streaming_limitation_is_recorded_in_the_trace(tmp_path: Path) -> None:
    """A limitation nobody can see is indistinguishable from a bug."""
    from agent_loop_chaos.trace import TraceRecorder

    eng = ChaosEngine(seed=1, out_dir=tmp_path / ".chaos", strict_trace=True)
    proxy = instrument_model(FakeChatModel(), eng)
    result = eng.run(
        lambda: list(proxy.stream([HumanMessage(content="hi")])),
        expected_behavior="ignore_and_continue",
        scenario_id="stream",
    )
    notes = [
        e
        for e in TraceRecorder.load(result.artifacts["trace"])
        if e["kind"] == "log" and "stream" in str(e.get("payload", {})).lower()
    ]
    assert notes, "the streaming limitation must be written into the trace"


def test_wrap_langchain_tools_preserves_the_tool_objects(tmp_path: Path) -> None:
    """LangChain dispatches on the tool's `name`, so it must survive wrapping."""
    from langchain_core.tools import tool

    @tool
    def get_weather(location: str) -> str:
        """Look up the weather.

        Args:
            location: Where.

        Returns:
            A description.
        """
        return f"sunny in {location}"

    eng = engine(tmp_path)
    wrapped = wrap_langchain_tools([get_weather], eng)
    assert wrapped[0].name == "get_weather"
    result = eng.run(
        lambda: wrapped[0].invoke({"location": "Paris"}), expected_behavior="ignore_and_continue"
    )
    assert "Paris" in str(result.final_output)
    assert result.metrics["tool_calls"] == 1


def test_a_tool_fault_fires_through_a_wrapped_langchain_tool(tmp_path: Path) -> None:
    """One fault catalog, both tool shapes."""
    from langchain_core.tools import tool

    from agent_loop_chaos.faults import ToolCorruptionFault

    @tool
    def get_weather(location: str) -> dict[str, Any]:
        """Look up the weather.

        Args:
            location: Where.

        Returns:
            The reading.
        """
        return {"temp_c": 21, "city": location}

    eng = engine(tmp_path)
    eng.register_fault(
        ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
        target=Target(tool="get_weather", phase="post"),
        trigger=Trigger(on_call=1),
    )
    wrapped = wrap_langchain_tools([get_weather], eng)
    result = eng.run(lambda: wrapped[0].invoke({"location": "Paris"}))
    assert result.injected_faults[0]["fired"] is True
