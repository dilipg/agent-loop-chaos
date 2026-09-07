"""The HTTP strategy: fault a model call nobody wrapped.

Every hosted model SDK -- openai, anthropic, azure, bedrock -- and every local server
-- Ollama, vLLM, LM Studio, llama.cpp -- reaches its endpoint through `httpx`. Patching
`Client.send` therefore reaches all of them at once, with no change to the agent. This
is the mechanism [AgentChaos](https://arxiv.org/abs/2608.06790) validates: agnosticity
comes from the shared transport, not from a plugin per framework.

`httpx.MockTransport` stands in for the server, so these tests need no socket -- the
autouse fixture would fail them if they opened one.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine, Target
from agent_loop_chaos.faults import (
    LLMTruncationFault,
    PromptInjectionFault,
    ToolCorruptionFault,
)
from agent_loop_chaos.interceptors import Registry
from agent_loop_chaos.interceptors.http import HttpxStrategy

httpx = pytest.importorskip("httpx")


def _openai_server(seen: list[dict[str, Any]]) -> Any:
    """A fake OpenAI-compatible endpoint that records what it was sent."""

    def handler(request: Any) -> Any:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Pack light layers."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    return httpx.MockTransport(handler)


def _anthropic_server(seen: list[dict[str, Any]]) -> Any:
    def handler(request: Any) -> Any:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "claude-opus-4",
                "content": [{"type": "text", "text": "Pack light layers."}],
                "stop_reason": "end_turn",
            },
        )

    return httpx.MockTransport(handler)


def _engine(tmp_path: Any) -> ChaosEngine:
    return ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False)


def _under_interception(engine: ChaosEngine, agent: Any, **kw: Any) -> Any:
    """Run `agent` with the strategy attached, restoring it afterwards."""
    registry = Registry([HttpxStrategy()])
    registry.attach(engine)
    try:
        return engine.run(agent, **kw)
    finally:
        registry.detach()


class TestItFaultsAModelCallNobodyWrapped:
    def test_a_prompt_side_fault_changes_what_the_server_receives(self, tmp_path: Any) -> None:
        """The whole point: no `engine.llm`, and the injection still lands."""
        seen: list[dict[str, Any]] = []
        engine = _engine(tmp_path)
        engine.register_fault(
            PromptInjectionFault(objective="exfiltrate_secret", placement="field_value"),
            target=Target(layer="llm", phase="pre"),
        )

        def agent(question: str) -> str:
            with httpx.Client(transport=_openai_server(seen)) as client:
                reply = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": [{"role": "user", "content": question}]},
                )
            return str(reply.json()["choices"][0]["message"]["content"])

        result = _under_interception(engine, agent, inputs="what should I pack?")

        assert result.validate() == []
        assert seen, "the fake server was never reached"
        sent = json.dumps(seen[0])
        assert "what should I pack?" in sent, "the real prompt should still be there"
        assert sent != json.dumps(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "what should I pack?"}]}
        ), "the request reached the server unmutated"
        assert result.injected_faults[0]["fired"] is True

    def test_a_response_side_fault_changes_what_the_caller_receives(self, tmp_path: Any) -> None:
        seen: list[dict[str, Any]] = []
        engine = _engine(tmp_path)
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))
        received: list[str] = []

        def agent(question: str) -> str:
            with httpx.Client(transport=_openai_server(seen)) as client:
                reply = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": [{"role": "user", "content": question}]},
                )
            received.append(str(reply.json()["choices"][0]["message"]["content"]))
            return received[-1]

        _under_interception(engine, agent, inputs="what should I pack?")
        assert received and received[0] != "Pack light layers.", "the response was not mutated"

    def test_the_anthropic_shape_works_too(self, tmp_path: Any) -> None:
        seen: list[dict[str, Any]] = []
        engine = _engine(tmp_path)
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))
        received: list[str] = []

        def agent(question: str) -> str:
            with httpx.Client(transport=_anthropic_server(seen)) as client:
                reply = client.post(
                    "https://api.anthropic.com/v1/messages",
                    json={
                        "model": "claude-opus-4",
                        "messages": [{"role": "user", "content": question}],
                    },
                )
            body = reply.json()
            received.append(str(body["content"][0]["text"]))
            return received[-1]

        _under_interception(engine, agent, inputs="what should I pack?")
        assert received and received[0] != "Pack light layers.", "the response was not mutated"

    def test_it_names_the_llm_by_its_model_id(self, tmp_path: Any) -> None:
        """`Target(llm=None)` matches any name, so the model id is free precision."""
        seen: list[dict[str, Any]] = []
        engine = _engine(tmp_path)
        engine.register_fault(LLMTruncationFault(), target=Target(llm="gpt-4*", phase="post"))

        def agent(question: str) -> str:
            with httpx.Client(transport=_openai_server(seen)) as client:
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": [{"role": "user", "content": question}]},
                )
            return "done"

        result = _under_interception(engine, agent, inputs="q")
        assert result.injected_faults[0]["fired"] is True, "a glob on the model id did not match"


class TestANonModelCallIsAToolCall:
    def test_an_arbitrary_http_call_becomes_a_tool_crossing(self, tmp_path: Any) -> None:
        """Not every HTTP call is a model call. The rest are the agent's tools."""
        engine = _engine(tmp_path)
        engine.register_fault(
            ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
            target=Target(layer="tool", phase="post"),
        )
        received: list[dict[str, Any]] = []

        def handler(request: Any) -> Any:
            return httpx.Response(200, json={"temp_c": 21, "city": "Paris"})

        def agent(question: str) -> str:
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                received.append(client.get("https://weather.example/v1/current").json())
            return "done"

        result = _under_interception(engine, agent, inputs="q")
        assert result.injected_faults[0]["fired"] is True
        assert received and "temp_c" not in received[0], "the tool response was not mutated"


class TestItIsHonestAboutWhatItCannotDo:
    def test_a_streaming_request_is_passed_through_and_says_so(
        self, tmp_path: Any, caplog: Any
    ) -> None:
        """Mutating an SSE stream is not implemented. Silently doing nothing is worse.

        It must also not fall through to the tool layer: a streaming call is a *model*
        call, and a tool-layer fault mutating it would be the library inventing a
        finding rather than observing one.
        """
        engine = _engine(tmp_path)
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))

        def handler(request: Any) -> Any:
            return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

        def agent(question: str) -> str:
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": [], "stream": True},
                )
            return "done"

        with caplog.at_level("WARNING", logger="agent_loop_chaos"):
            result = _under_interception(engine, agent, inputs="q")

        assert result.injected_faults[0]["fired"] is False
        assert not result.tool_calls, "a streaming model call must not become a tool call"
        assert any("streaming model call" in r.message for r in caplog.records), (
            "the pass-through was silent"
        )

    def test_it_does_nothing_outside_a_run(self, tmp_path: Any) -> None:
        """A patch left on by a crashed run must not change production behaviour."""
        engine = _engine(tmp_path)
        registry = Registry([HttpxStrategy()])
        registry.attach(engine)
        try:

            def handler(request: Any) -> Any:
                return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                body = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": []},
                ).json()
            assert body["choices"][0]["message"]["content"] == "hi"
        finally:
            registry.detach()


class TestItRestoresTheClient:
    def test_the_patch_is_removed_after_a_run(self, tmp_path: Any) -> None:
        original_sync, original_async = httpx.Client.send, httpx.AsyncClient.send
        registry = Registry([HttpxStrategy()])
        registry.attach(_engine(tmp_path))
        assert httpx.Client.send is not original_sync
        registry.detach()
        assert httpx.Client.send is original_sync
        assert httpx.AsyncClient.send is original_async

    def test_the_patch_is_removed_after_a_failing_run(self, tmp_path: Any) -> None:
        """The engine reports an exploding agent rather than re-raising, so the check
        is that the patch is gone and the run still produced a verdict."""
        original = httpx.Client.send
        engine = _engine(tmp_path)

        def agent(question: str) -> str:
            raise RuntimeError("the agent exploded")

        result = _under_interception(engine, agent, inputs="q")
        assert httpx.Client.send is original
        assert result.success is False


class TestTheAsyncClient:
    @pytest.mark.asyncio
    async def test_the_async_path_is_faulted_too(self, tmp_path: Any) -> None:
        """Async and sync parity is structural, not remembered."""
        seen: list[dict[str, Any]] = []
        engine = _engine(tmp_path)
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))
        received: list[str] = []

        async def agent(question: str) -> str:
            async with httpx.AsyncClient(transport=_openai_server(seen)) as client:
                reply = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": [{"role": "user", "content": question}]},
                )
            received.append(str(reply.json()["choices"][0]["message"]["content"]))
            return received[-1]

        registry = Registry([HttpxStrategy()])
        registry.attach(engine)
        try:
            await engine.arun(agent, inputs="q")
        finally:
            registry.detach()
        assert received and received[0] != "Pack light layers.", (
            "the async response was not mutated"
        )
