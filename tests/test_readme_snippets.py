"""The two integration snippets from `docs/02-API.md` §11, run verbatim.

"Both snippets must work verbatim by the end of phase 05." A quickstart that does
not run is worse than none: it costs a reader their first half hour and their trust.

Kept in sync by construction -- if either snippet changes, this file must change with
it, and the test names say which one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_the_five_line_langgraph_promise(tmp_path: Path, monkeypatch: Any) -> None:
    """`docs/02-API.md` §11, the LangGraph snippet.

    The promise is five lines: construct, register, instrument, run, print. Anything
    more and the "≤5 lines" success criterion in `docs/00-VISION.md` is not met.
    """
    pytest.importorskip("langgraph", reason="the langgraph extra is not installed")
    pytest.importorskip("langchain_core", reason="the langgraph extra is not installed")
    monkeypatch.chdir(tmp_path)

    from tests.fakes.lg_agent import build

    app = build(lambda prompt: "Warm; pack light layers.").compile()

    # --- the snippet ------------------------------------------------------------
    from agent_loop_chaos import ChaosEngine
    from agent_loop_chaos.adapters.langgraph import instrument_graph
    from agent_loop_chaos.faults import ToolCorruptionFault

    engine = ChaosEngine(seed=1337)
    engine.register_fault(
        ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
        target_tool="get_weather_data",
    )
    result = engine.run(instrument_graph(app, engine), inputs={"query": "Pack list for Paris"})
    print(result.to_json())
    # --- end of the snippet -----------------------------------------------------

    assert result.run_id
    assert result.validate() == []


def test_the_plain_python_blueprint_shape(tmp_path: Path, monkeypatch: Any) -> None:
    """`docs/02-API.md` §11, the vanilla snippet.

    This is the original blueprint's call shape, kept working on purpose: the API
    doc froze it, and `run_with_state` exists precisely so it keeps parsing.
    """
    monkeypatch.chdir(tmp_path)

    # --- the snippet ------------------------------------------------------------
    from agent_loop_chaos import ChaosEngine
    from agent_loop_chaos.faults import ToolCorruptionFault

    chaos = ChaosEngine()
    chaos.register_fault(
        ToolCorruptionFault(mutation_type="empty_json"), target_tool="get_weather_data"
    )

    @chaos.tool
    def get_weather_data(location: str) -> list[dict[str, Any]]:
        return [{"temp_c": 21, "city": location}]

    def llm_generate(user_query: str, data: Any) -> str:
        return f"{user_query}: {data}"

    @chaos.intercept_tools()
    def weather_agent(user_query: str, state: dict[str, Any]) -> str:
        data = get_weather_data(state["location"])
        return llm_generate(user_query, data)

    result = chaos.run_with_state(
        weather_agent, query="Pack list for Paris", initial_state={"location": "Paris"}
    )
    print(result.to_json())
    # --- end of the snippet -----------------------------------------------------

    assert result.run_id
    assert result.validate() == []
    assert result.injected_faults[0]["fired"] is True


def test_the_snippets_are_still_the_ones_in_the_doc() -> None:
    """The doc and these tests must not drift apart.

    A snippet that works here and not in the README is the same failure as one that
    works nowhere -- the reader only ever sees the README.
    """
    doc = (Path(__file__).resolve().parents[1] / "docs" / "02-API.md").read_text(encoding="utf-8")
    section = doc.split("## 11. The 5-line integration promise")[1]
    for line in (
        "engine = ChaosEngine(seed=1337)",
        "instrument_graph(app, engine)",
        "chaos = ChaosEngine()",
        "@chaos.intercept_tools()",
        "chaos.run_with_state(",
    ):
        assert line in section, f"the doc no longer contains {line!r}"


# ---------------------------------------------------------------------------------
# The README's integration snippets.
#
# A quickstart that does not run costs a reader their first half hour and their
# trust, and the README is the first thing anyone sees. Each snippet below is the
# README's, verbatim, with only the surrounding fixture changed.
# ---------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _readme() -> str:
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


class TestTheIntegrationSnippets:
    def test_the_plain_python_decorators(self, tmp_path: Path, monkeypatch: Any) -> None:
        """README, "Integrating with your agent → Plain Python"."""
        monkeypatch.chdir(tmp_path)
        from agent_loop_chaos import ChaosEngine

        engine = ChaosEngine(seed=1337, judge="rules", write_bundle=False)

        @engine.tool
        def get_weather(city: str) -> dict:
            return {"temp_c": 24, "condition": "sunny"}

        @engine.llm
        def complete(prompt: str) -> str:
            return "Pack light layers."

        @engine.intercept_tools()
        def agent(question: str) -> str:
            data = get_weather("Paris")
            return complete(f"{question}\n{data}")

        result = engine.run(agent, inputs="what should I pack?")
        assert result.validate() == []
        assert result.tool_calls, "the decorator did not attach the tool"
        assert result.llm_exchanges, "@engine.llm did not attach the model"

    def test_the_side_effecting_declaration(self, tmp_path: Path, monkeypatch: Any) -> None:
        """README: `@engine.tool(side_effecting=True)` is enforced, not decorative."""
        monkeypatch.chdir(tmp_path)
        from agent_loop_chaos import ChaosEngine
        from agent_loop_chaos.errors import ConfigError
        from agent_loop_chaos.faults import DuplicateSideEffectFault

        engine = ChaosEngine(seed=1337, judge="rules", write_bundle=False)

        @engine.tool(side_effecting=True)
        def hold_booking(flight_id: str, passenger: str) -> dict:
            return {"held": flight_id}

        with pytest.raises(ConfigError, match="allow_side_effects"):
            engine.register_fault(DuplicateSideEffectFault(times=2), target_tool="hold_booking")

    def test_instrument_object(self, tmp_path: Path, monkeypatch: Any) -> None:
        """README, "Integrating with your agent → An agent class"."""
        monkeypatch.chdir(tmp_path)
        from agent_loop_chaos import ChaosEngine

        class MyAgent:
            def search(self, q: str) -> dict:
                return {"hits": [q]}

            def book(self, ref: str) -> dict:
                return {"booked": ref}

            def complete(self, prompt: str) -> str:
                return "done"

            def run(self, inputs: str) -> str:
                return self.complete(str(self.search(inputs)))

        engine = ChaosEngine(seed=1337, judge="rules", write_bundle=False)
        agent = MyAgent()
        engine.instrument_object(
            agent,
            tools={"search": False, "book": True},
            llm_methods=["complete"],
        )
        result = engine.run(agent.run, inputs="Find me a flight to Paris")
        assert result.validate() == []
        assert [c["tool"] for c in result.tool_calls] == ["search"]

    def test_the_builder_entrypoint_contract(self, tmp_path: Path, monkeypatch: Any) -> None:
        """README, "The entrypoint contract".

        The rule the README states: a first parameter named `engine` makes the
        callable a builder. The same function has to serve production, where
        `engine` is `None`, or the thing tested is not the thing shipped.
        """
        monkeypatch.chdir(tmp_path)
        from agent_loop_chaos import ChaosEngine
        from agent_loop_chaos.loop import resolve_entrypoint

        def get_weather(city: str) -> dict:
            return {"temp_c": 24}

        def build(engine: Any = None) -> Any:  # the README's exact shape
            tools = {"get_weather": get_weather}
            if engine is not None:
                tools = {name: engine.tool(fn, name=name) for name, fn in tools.items()}

            def agent(question: str) -> str:
                return str(tools["get_weather"]("Paris"))

            return engine.intercept_tools()(agent) if engine else agent

        assert build()("q"), "the builder must work with no engine, for production"

        engine = ChaosEngine(seed=1337, judge="rules", write_bundle=False)
        agent = resolve_entrypoint(build, engine)
        result = engine.run(agent, inputs="what should I pack?")
        assert [c["tool"] for c in result.tool_calls] == ["get_weather"]

    def test_the_pytest_snippet(self, tmp_path: Path) -> None:
        """README, "In pytest" — the per-fault parametrized shape."""
        from agent_loop_chaos import ChaosEngine
        from agent_loop_chaos.faults import ToolCorruptionFault

        def build_agent(engine: Any) -> Any:
            @engine.tool
            def get_weather(city: str) -> dict:
                return {"temp_c": 24}

            @engine.intercept_tools()
            def agent(question: str) -> str:
                weather = get_weather("Paris")
                if "temp_c" not in weather:
                    return "The forecast is unavailable, so I cannot advise."
                return f"About {weather['temp_c']}C."

            return agent

        for mutation in ("drop_key", "type_flip", "empty_json", "unit_swap"):
            engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules")
            engine.register_fault(
                ToolCorruptionFault(mutation_type=mutation), target_tool="get_weather"
            )
            result = engine.run(build_agent(engine), inputs="what should I pack for Paris?")
            assert result.validate() == []

    def test_the_suite_snippet(self, tmp_path: Path) -> None:
        """README, "In pytest" — the whole-suite shape."""
        import json

        from agent_loop_chaos import load_suite
        from agent_loop_chaos.loop import run_suite

        path = tmp_path / "suite.json"
        path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "scenarios": [
                        {
                            "id": "tool.drop_key",
                            "entrypoint": "tests.fakes.apps:build_naive",
                            "faults": [
                                {
                                    "type": "ToolCorruptionFault",
                                    "params": {"mutation_type": "drop_key", "keys": ["temp_c"]},
                                    "trigger": {"on_call": 1},
                                }
                            ],
                        }
                    ],
                }
            )
        )
        suite = load_suite(path)
        results = run_suite(suite.scenarios, out_dir=tmp_path, judge="rules")
        assert [r.scenario_id for r in results] == ["tool.drop_key"]


class TestTheReadmeStaysTrue:
    """The numbers went stale once. They cannot again without failing here."""

    def test_the_fault_count_is_right(self) -> None:
        from agent_loop_chaos.faults import list_faults

        assert f"{len(list_faults())} kinds" in _readme()

    def test_the_probe_count_is_right(self) -> None:
        from agent_loop_chaos.probes import PROBE_PRECEDENCE

        assert f"{len(PROBE_PRECEDENCE)} probes" in _readme()

    def test_the_preset_names_are_right(self) -> None:
        from agent_loop_chaos.scenarios import PRESETS

        readme = _readme()
        for name in PRESETS:
            assert name in readme, f"preset {name} is not in the README"

    def test_the_demo_suite_size_is_right(self, repo_root: Path) -> None:
        from agent_loop_chaos.scenarios import load_suite

        suite = load_suite(repo_root / "examples" / "scenarios" / "demo_suite.yaml")
        assert f"{len(suite.scenarios)}-scenario suite" in _readme()

    def test_every_cli_subcommand_is_documented(self) -> None:
        from agent_loop_chaos.cli import build_parser

        readme = _readme()
        actions = build_parser()._subparsers._group_actions[0]  # type: ignore[union-attr]
        for name in actions.choices:  # type: ignore[attr-defined]
            assert f"alc {name}" in readme, f"`alc {name}` is undocumented in the README"
