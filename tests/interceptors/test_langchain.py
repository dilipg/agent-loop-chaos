"""The LangChain strategy: fault a model that never touches the network.

The transport strategy covers every hosted SDK and every local *server*, because both
are HTTP. It cannot see a model that answers in-process -- a `transformers` pipeline, an
MLX model, a bespoke class, or the fake models a test suite uses. Those all inherit
`BaseChatModel.generate`, so one patch on the base class reaches every one of them,
including the fakes, which is what makes a repo's own test fixtures usable as a chaos
fixture.

`BaseTool.run` is the same argument for the tool layer: a repo whose tools are
`@tool`-decorated gets the data-shape faults with no wrapping at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine, Target
from agent_loop_chaos.faults import LLMTruncationFault, PromptInjectionFault, ToolCorruptionFault
from agent_loop_chaos.interceptors import Registry
from agent_loop_chaos.interceptors.langchain import LangChainStrategy

pytest.importorskip("langchain_core", reason="the langgraph extra is not installed")


def _fake_model(reply: str = "Pack light layers.") -> Any:
    """An in-process model, invisible to any transport-level interception."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    return GenericFakeChatModel(messages=iter([reply] * 50))


def _engine(tmp_path: Any) -> ChaosEngine:
    return ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False)


def _under_interception(engine: ChaosEngine, agent: Any, **kw: Any) -> Any:
    registry = Registry([LangChainStrategy()])
    registry.attach(engine)
    try:
        return engine.run(agent, **kw)
    finally:
        registry.detach()


class TestAnInProcessModelIsReachable:
    def test_a_prompt_side_fault_reaches_a_model_that_never_dials_out(self, tmp_path: Any) -> None:
        engine = _engine(tmp_path)
        engine.register_fault(
            PromptInjectionFault(objective="exfiltrate_secret", placement="field_value"),
            target=Target(layer="llm", phase="pre"),
        )
        model = _fake_model()

        def agent(question: str) -> str:
            return str(model.invoke(question).content)

        result = _under_interception(engine, agent, inputs="what should I pack?")
        assert result.validate() == []
        assert result.injected_faults[0]["fired"] is True, "the base-class patch never fired"
        assert result.llm_exchanges, "no llm crossing was recorded"

    def test_a_response_side_fault_changes_what_the_agent_reads(self, tmp_path: Any) -> None:
        engine = _engine(tmp_path)
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))
        model = _fake_model()
        seen: list[str] = []

        def agent(question: str) -> str:
            seen.append(str(model.invoke(question).content))
            return seen[-1]

        _under_interception(engine, agent, inputs="q")
        assert seen and seen[0] != "Pack light layers.", "the reply was not mutated"

    @pytest.mark.asyncio
    async def test_the_async_path_too(self, tmp_path: Any) -> None:
        engine = _engine(tmp_path)
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))
        model = _fake_model()
        seen: list[str] = []

        async def agent(question: str) -> str:
            reply = await model.ainvoke(question)
            seen.append(str(reply.content))
            return seen[-1]

        registry = Registry([LangChainStrategy()])
        registry.attach(engine)
        try:
            await engine.arun(agent, inputs="q")
        finally:
            registry.detach()
        assert seen and seen[0] != "Pack light layers.", "the async reply was not mutated"


class TestADecoratedToolIsReachable:
    def test_a_data_shape_fault_reaches_a_langchain_tool(self, tmp_path: Any) -> None:
        from langchain_core.tools import tool

        engine = _engine(tmp_path)
        engine.register_fault(
            ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
            target=Target(layer="tool", phase="post"),
        )

        @tool
        def get_weather(city: str) -> dict:
            """Return the weather for a city."""
            return {"temp_c": 21, "city": city}

        seen: list[Any] = []

        def agent(question: str) -> str:
            seen.append(get_weather.invoke({"city": "Paris"}))
            return "done"

        result = _under_interception(engine, agent, inputs="q")
        assert result.injected_faults[0]["fired"] is True
        assert seen and "temp_c" not in seen[0], "the tool result was not mutated"

    def test_the_tool_is_named_by_its_langchain_name(self, tmp_path: Any) -> None:
        """So `target_tool: get_weather` works, as a reader would expect."""
        from langchain_core.tools import tool

        engine = _engine(tmp_path)
        engine.register_fault(
            ToolCorruptionFault(mutation_type="empty_json"),
            target=Target(tool="get_weather", phase="post"),
        )

        @tool
        def get_weather(city: str) -> dict:
            """Return the weather for a city."""
            return {"temp_c": 21}

        def agent(question: str) -> str:
            get_weather.invoke({"city": "Paris"})
            return "done"

        result = _under_interception(engine, agent, inputs="q")
        assert result.injected_faults[0]["fired"] is True, "the tool's own name did not match"


class TestItRestoresTheBaseClasses:
    def test_the_patches_are_removed(self, tmp_path: Any) -> None:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.tools import BaseTool

        before = (
            BaseChatModel.generate,
            BaseChatModel.agenerate,
            BaseTool.run,
            BaseTool.arun,
        )
        registry = Registry([LangChainStrategy()])
        registry.attach(_engine(tmp_path))
        assert BaseChatModel.generate is not before[0]
        registry.detach()
        assert (
            BaseChatModel.generate,
            BaseChatModel.agenerate,
            BaseTool.run,
            BaseTool.arun,
        ) == before

    def test_it_does_nothing_outside_a_run(self, tmp_path: Any) -> None:
        registry = Registry([LangChainStrategy()])
        registry.attach(_engine(tmp_path))
        try:
            assert str(_fake_model().invoke("q").content) == "Pack light layers."
        finally:
            registry.detach()


class TestItSaysWhenItCannotHelp:
    def test_it_reports_unavailable_without_langchain(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("agent_loop_chaos.interceptors.langchain.find_spec", lambda _name: None)
        assert LangChainStrategy().available() == "not installed"
