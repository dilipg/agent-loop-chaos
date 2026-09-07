"""`ChaosEngine(intercept=True)`: the one switch that removes the harness.

With it on, an agent needs no `engine.tool`, no `engine.llm` and no builder -- the
strategies find the calls. Off is the default, because patching a process-wide HTTP
client is not something to do behind a caller's back.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine, Target
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.faults import LLMTruncationFault

httpx = pytest.importorskip("httpx")


def _server() -> Any:
    def handler(request: Any) -> Any:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Pack light layers."}}]},
        )

    return httpx.MockTransport(handler)


def _agent(seen: list[str]) -> Any:
    def agent(question: str) -> str:
        with httpx.Client(transport=_server()) as client:
            reply = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": question}]},
            )
        seen.append(str(reply.json()["choices"][0]["message"]["content"]))
        return seen[-1]

    return agent


class TestTheSwitch:
    def test_on_it_faults_an_unwrapped_agent(self, tmp_path: Any) -> None:
        """No decorators, no builder, no harness: the whole point of the phase."""
        seen: list[str] = []
        engine = ChaosEngine(
            seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False, intercept=True
        )
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))

        result = engine.run(_agent(seen), inputs="what should I pack?")

        assert result.validate() == []
        assert result.injected_faults[0]["fired"] is True
        assert seen and seen[0] != "Pack light layers.", "the reply reached the agent unmutated"

    def test_off_by_default_nothing_is_intercepted(self, tmp_path: Any) -> None:
        seen: list[str] = []
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False)
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))

        result = engine.run(_agent(seen), inputs="q")

        assert result.injected_faults[0]["fired"] is False
        assert seen == ["Pack light layers."]

    def test_it_restores_the_client_afterwards(self, tmp_path: Any) -> None:
        original = httpx.Client.send
        engine = ChaosEngine(
            seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False, intercept=True
        )
        engine.run(_agent([]), inputs="q")
        assert httpx.Client.send is original

    @pytest.mark.asyncio
    async def test_the_async_run_is_wired_too(self, tmp_path: Any) -> None:
        seen: list[str] = []
        engine = ChaosEngine(
            seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False, intercept=True
        )
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))

        async def agent(question: str) -> str:
            async with httpx.AsyncClient(transport=_server()) as client:
                reply = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": []},
                )
            seen.append(str(reply.json()["choices"][0]["message"]["content"]))
            return seen[-1]

        await engine.arun(agent, inputs="q")
        assert seen and seen[0] != "Pack light layers."


class TestItRefusesToProveNothing:
    def test_intercept_on_with_nothing_to_patch_is_an_error(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """Asking for interception and silently getting none is the failure
        this whole phase exists to remove."""

        class _Nothing:
            name = "httpx"

            def available(self) -> str | None:
                return "not installed"

            def attach(self, engine: Any, seen: dict[str, int]) -> list[Any]:
                raise AssertionError("must not be called")

            def detach(self) -> None:
                pass

        monkeypatch.setattr(
            "agent_loop_chaos.interceptors.default_strategies", lambda: [_Nothing()]
        )
        engine = ChaosEngine(
            seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False, intercept=True
        )
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))

        with pytest.raises(ConfigError) as caught:
            engine.run(_agent([]), inputs="q")
        message = str(caught.value)
        assert "httpx" in message and "not installed" in message
