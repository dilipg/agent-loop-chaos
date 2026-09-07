"""`seams:` -- name a call by dotted path when nothing else can find it.

Two strategies cover the calls that go through a shared library: `httpx` for anything
that speaks HTTP, `langchain-core` for anything that inherits its base classes. Neither
sees a hand-rolled client -- an in-house `LLMClient.chat`, a repository function reading
Mongo through a driver that is not HTTP, an SDK with its own transport.

That is what this is for, and it is the reason `AttachReport.diagnose()` tells a stuck
reader to name the seam explicitly. A hint pointing at a feature that does not exist
would be worse than no hint.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine, Target
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.faults import (
    LLMTruncationFault,
    PromptInjectionFault,
    ToolCorruptionFault,
)
from agent_loop_chaos.interceptors import Registry
from agent_loop_chaos.interceptors.seams import SeamsStrategy
from tests.fakes import homegrown


def _engine(tmp_path: Any, **kw: Any) -> ChaosEngine:
    return ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False, **kw)


class TestItReachesAHandRolledClient:
    def test_a_method_named_by_dotted_path_becomes_an_llm_crossing(self, tmp_path: Any) -> None:
        engine = _engine(tmp_path)
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))
        registry = Registry([SeamsStrategy({"llm": ["tests.fakes.homegrown:LLMClient.chat"]})])
        registry.attach(engine)
        seen: list[str] = []
        try:
            result = engine.run(lambda q: seen.append(homegrown.LLMClient().chat(q)), inputs="q")
        finally:
            registry.detach()

        assert result.injected_faults[0]["fired"] is True
        assert seen and seen[0] != homegrown.CANNED, "the reply was not mutated"

    def test_a_module_function_becomes_a_tool_crossing(self, tmp_path: Any) -> None:
        engine = _engine(tmp_path)
        engine.register_fault(
            ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
            target=Target(layer="tool", phase="post"),
        )
        registry = Registry([SeamsStrategy({"tools": ["tests.fakes.homegrown:fetch_weather"]})])
        registry.attach(engine)
        seen: list[Any] = []
        try:
            engine.run(lambda q: seen.append(homegrown.fetch_weather("Paris")), inputs="q")
        finally:
            registry.detach()

        assert seen and "temp_c" not in seen[0], "the payload was not mutated"

    def test_a_glob_matches_every_function_in_the_module(self, tmp_path: Any) -> None:
        """The shape a repository actually has: a module of repository readers."""
        engine = _engine(tmp_path)
        registry = Registry([SeamsStrategy({"tools": ["tests.fakes.homegrown:fetch_*"]})])
        report = registry.attach(engine)
        try:
            names = {a.target for a in report.attached}
        finally:
            registry.detach()
        assert names == {
            "tests.fakes.homegrown:fetch_weather",
            "tests.fakes.homegrown:fetch_tickets",
        }

    def test_the_seam_is_named_by_its_attribute_path(self, tmp_path: Any) -> None:
        """So `tool: fetch_weather` and `llm: "LLMClient.chat"` work as written."""
        engine = _engine(tmp_path)
        engine.register_fault(
            ToolCorruptionFault(mutation_type="empty_json"),
            target=Target(tool="fetch_weather", phase="post"),
        )
        registry = Registry([SeamsStrategy({"tools": ["tests.fakes.homegrown:fetch_weather"]})])
        registry.attach(engine)
        try:
            result = engine.run(lambda q: homegrown.fetch_weather("Paris"), inputs="q")
        finally:
            registry.detach()
        assert result.injected_faults[0]["fired"] is True


class TestItRestoresWhatItPatched:
    def test_the_module_and_the_class_are_left_as_they_were(self, tmp_path: Any) -> None:
        before = (homegrown.fetch_weather, homegrown.LLMClient.chat)
        registry = Registry(
            [
                SeamsStrategy(
                    {
                        "tools": ["tests.fakes.homegrown:fetch_weather"],
                        "llm": ["tests.fakes.homegrown:LLMClient.chat"],
                    }
                )
            ]
        )
        registry.attach(_engine(tmp_path))
        assert homegrown.fetch_weather is not before[0]
        registry.detach()
        assert (homegrown.fetch_weather, homegrown.LLMClient.chat) == before


class TestItRefusesNonsenseAtAttachTime:
    @pytest.mark.parametrize(
        "spec",
        [
            "no_colon_here",
            "tests.fakes.homegrown:does_not_exist",
            "nope.no.such.module:thing",
            "tests.fakes.homegrown:LLMClient.no_such_method",
        ],
    )
    def test_a_bad_spec_is_a_config_error(self, spec: str, tmp_path: Any) -> None:
        """At attach time, not mid-run: a typo must not read as an agent failure."""
        strategy = SeamsStrategy({"tools": [spec]})
        with pytest.raises(ConfigError) as caught:
            strategy.attach(_engine(tmp_path), {})
        strategy.detach()
        assert spec.split(":")[-1] in str(caught.value) or "module:attr" in str(caught.value)

    def test_an_unknown_layer_is_a_config_error(self, tmp_path: Any) -> None:
        strategy = SeamsStrategy({"node": ["tests.fakes.homegrown:fetch_weather"]})
        with pytest.raises(ConfigError, match=r"unknown layer"):
            strategy.attach(_engine(tmp_path), {})


class TestTheSuiteSurface:
    def test_seams_turn_interception_on_by_themselves(self, tmp_path: Any) -> None:
        """Declaring a seam and having it silently ignored is the failure mode this
        whole area exists to remove, so `seams:` does not also need `intercept: true`."""
        engine = _engine(tmp_path, seams={"tools": ["tests.fakes.homegrown:fetch_weather"]})
        engine.register_fault(
            ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
            target=Target(layer="tool", phase="post"),
        )
        seen: list[Any] = []
        engine.run(lambda q: seen.append(homegrown.fetch_weather("Paris")), inputs="q")
        assert seen and "temp_c" not in seen[0], "a declared seam was ignored"

    def test_a_scenario_can_declare_them(self, tmp_path: Any) -> None:
        import json

        from agent_loop_chaos.scenarios import load_suite

        path = tmp_path / "suite.json"
        path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "scenarios": [
                        {
                            "id": "s.1",
                            "entrypoint": "x:build",
                            "seams": {"llm": ["app.llm:Client.chat"]},
                            "faults": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert load_suite(path).scenarios[0].seams == {"llm": ["app.llm:Client.chat"]}


class TestAMethodSeamSeesTheRealPayload:
    """Found on a real service. Patching a class attribute means the wrapper is called
    as `(self, messages)`, so the engine took `self` for the LLM payload and every
    prompt-side fault was disabled with an unnormalizable payload."""

    def test_the_payload_is_the_argument_not_the_instance(self, tmp_path: Any) -> None:
        engine = _engine(tmp_path)
        engine.register_fault(
            PromptInjectionFault(objective="exfiltrate_secret", placement="field_value"),
            target=Target(layer="llm", phase="pre"),
        )
        registry = Registry([SeamsStrategy({"llm": ["tests.fakes.homegrown:LLMClient.chat"]})])
        registry.attach(engine)
        try:
            result = engine.run(
                lambda q: homegrown.LLMClient().chat("what should I pack?"), inputs="q"
            )
        finally:
            registry.detach()

        assert result.injected_faults[0]["fired"] is True, "a pre-phase fault could not fire"
        exchange = result.llm_exchanges[0]
        assert exchange["messages"], "no prompt was recorded"
        assert "LLMClient" not in str(exchange["messages"]), "`self` was taken for the prompt"

    @pytest.mark.asyncio
    async def test_an_async_method_seam_works_too(self, tmp_path: Any) -> None:
        engine = _engine(tmp_path)
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))
        registry = Registry([SeamsStrategy({"llm": ["tests.fakes.homegrown:LLMClient.achat"]})])
        registry.attach(engine)
        seen: list[str] = []
        try:

            async def agent(q: str) -> str:
                seen.append(await homegrown.LLMClient().achat(q))
                return seen[-1]

            await engine.arun(agent, inputs="q")
        finally:
            registry.detach()
        assert seen and seen[0] != homegrown.CANNED, "the async reply was not mutated"


class TestASeamCanBeGivenAName:
    """Also found on a real service: a suite naturally targets `llm: summarizer`, not
    `llm: "StubModel.ainvoke"`. Without an alias the scenario silently matches nothing,
    which is the vacuous pass this whole area exists to prevent."""

    def test_an_alias_is_what_a_target_matches(self, tmp_path: Any) -> None:
        engine = _engine(tmp_path)
        engine.register_fault(LLMTruncationFault(), target=Target(llm="summarizer", phase="post"))
        registry = Registry(
            [SeamsStrategy({"llm": ["summarizer=tests.fakes.homegrown:LLMClient.chat"]})]
        )
        registry.attach(engine)
        seen: list[str] = []
        try:
            engine.run(lambda q: seen.append(homegrown.LLMClient().chat(q)), inputs="q")
        finally:
            registry.detach()
        assert seen and seen[0] != homegrown.CANNED, "the alias did not match the target"

    def test_the_alias_is_reported_as_the_attachment_target(self, tmp_path: Any) -> None:
        strategy = SeamsStrategy({"tools": ["weather=tests.fakes.homegrown:fetch_weather"]})
        report_seen: dict[str, int] = {}
        points = strategy.attach(_engine(tmp_path), report_seen)
        strategy.detach()
        assert [p.target for p in points] == ["weather=tests.fakes.homegrown:fetch_weather"]

    def test_an_alias_over_a_glob_is_refused(self, tmp_path: Any) -> None:
        """One name cannot stand for several callables, and silently naming only the
        first would be worse than refusing."""
        strategy = SeamsStrategy({"tools": ["all=tests.fakes.homegrown:fetch_*"]})
        with pytest.raises(ConfigError, match="glob"):
            strategy.attach(_engine(tmp_path), {})
        strategy.detach()


class TestABadSeamIsNotSwallowed:
    """The registry caught every exception from `attach` and recorded it as
    "unavailable". A typo in a dotted path then produced a green run with no seam --
    the exact failure mode `seams:` was added to fix."""

    def test_a_config_error_from_a_strategy_propagates(self, tmp_path: Any) -> None:
        registry = Registry([SeamsStrategy({"tools": ["tests.fakes.homegrown:nope"]})])
        with pytest.raises(ConfigError, match="nope"):
            registry.attach(_engine(tmp_path))
        registry.detach()
