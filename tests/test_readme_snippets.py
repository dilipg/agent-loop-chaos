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


class TestTheReadmePathsExist:
    """A reader runs `alc init` and then copy-pastes the next command.

    The README referenced `chaos/suite.yaml` eight times. `alc init` writes
    `chaos/quickstart.yaml`. Every command after the quickstart pointed at a file
    nothing creates.
    """

    def test_every_chaos_path_is_one_init_writes(self, tmp_path: Path, monkeypatch: Any) -> None:
        import re

        from agent_loop_chaos.cli import main

        monkeypatch.chdir(tmp_path)
        assert main(["init"]) == 0
        scaffolded = {str(p.relative_to(tmp_path)) for p in (tmp_path / "chaos").iterdir()}

        # Scope: files the README tells a reader to *read*. `(?<![.\w])` keeps
        # `.chaos/suite.json` -- the output directory -- out of it, and the flat
        # `chaos/*.ext` shape keeps out nested paths like `chaos/cassettes/tape.json`,
        # which the command shown creates rather than expects to exist.
        referenced = set(
            re.findall(r"(?<![.\w])chaos/[A-Za-z0-9_.-]+\.(?:yaml|yml|json)", _readme())
        )
        missing = referenced - scaffolded
        assert not missing, (
            f"the README references {sorted(missing)}, which `alc init` never writes"
        )

    def test_the_referenced_suite_actually_loads(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Not just present -- parseable, so the next command in the README works."""
        pytest.importorskip("yaml", reason="the yaml extra is not installed")
        from agent_loop_chaos.cli import main
        from agent_loop_chaos.scenarios import load_suite

        monkeypatch.chdir(tmp_path)
        main(["init"])
        suite = load_suite(tmp_path / "chaos" / "quickstart.yaml")
        assert suite.scenarios


class TestTheReadmeCoversRunningAgainstARealAgent:
    """The first question anyone integrating asks, and the one with a leak behind it.

    `report.json` is written to be attached to tickets. A reader who never learns
    about `redact_keys` will not know their custom credential header is in it.
    """

    def test_it_says_the_engine_needs_no_credential(self) -> None:
        assert "never needs a credential" in _readme()

    def test_it_names_the_escape_hatch_for_a_custom_header(self) -> None:
        assert "redact_keys" in _readme()

    def test_it_tells_the_reader_to_verify_rather_than_trust(self) -> None:
        readme = _readme()
        assert "not in json.dumps(result.to_dict())" in readme

    def test_it_explains_the_side_effect_gate(self) -> None:
        readme = _readme()
        assert "side_effecting=True" in readme
        assert "allow_side_effects" in readme

    def test_every_expect_check_it_shows_is_real(self) -> None:
        """A scenario field the README invents would fail at load, not at review."""
        import dataclasses
        import re

        from agent_loop_chaos.assertions import Expect

        declared = {f.name for f in dataclasses.fields(Expect)}
        block = _readme().split("    expect:                                   #")[1]
        block = block.split("\n\n")[0]
        shown = set(re.findall(r"^      ([a-z_]+):", block, re.M))
        assert shown, "the annotated scenario no longer shows an expect block"
        assert shown <= declared, f"the README shows checks that do not exist: {shown - declared}"


class TestTheReadmeOnboardsARealRepo:
    """The section that exists so nobody needs a coding agent to write a harness.

    Onboarding one real repository (a 10-node LangGraph pipeline with Mongo behind it)
    took ~200 lines of hand-written harness before a single fault could fire. Every
    blocker hit on the way is documented here with its workaround, because the next
    person hits the same eight and has no transcript to read.
    """

    @staticmethod
    def _section() -> str:
        body = _readme().split("## Onboarding a real repo")
        assert len(body) == 2, "the onboarding section is gone"
        return body[1].split("\n## ")[0]

    def test_the_section_exists(self) -> None:
        assert self._section().strip()

    @pytest.mark.parametrize(
        "blocker",
        [
            "env",  # settings validated at import
            "constructor",  # a DB client built before the graph exists
            "no model",  # nothing to answer offline
            "async",  # a coroutine entrypoint (D-114)
            "no tool or llm layer",  # nothing for a payload fault to attach to
            "never fire",  # armed, inert, proves nothing (D-64/D-132)
            "side-effecting",  # the safety gate
            "virtualenv",  # uv/monorepo/hidden-.pth (D-125)
        ],
    )
    def test_it_covers_each_blocker(self, blocker: str) -> None:
        assert blocker in self._section().lower(), f"{blocker!r} is undocumented"

    def test_it_quotes_the_warning_the_library_actually_prints(self, repo_root: Path) -> None:
        """A reader greps the message they saw. It has to be the real one."""
        printed = "this scenario proves nothing"
        source = (repo_root / "src" / "agent_loop_chaos" / "cli.py").read_text()
        assert printed in source, "the warning moved; update the README quote with it"
        assert printed in self._section(), "the README no longer quotes the real warning"

    def test_it_warns_that_a_stub_model_narrows_the_result(self) -> None:
        """The vacuous pass: the fault fires, but a canned reply cannot respond to it.

        Read as a clean bill of health, this is the most misleading green in the tool.
        """
        section = self._section().lower()
        assert "stub" in section
        assert "does not prove" in section or "not prove" in section

    def test_every_yaml_key_it_shows_is_real(self) -> None:
        """The first draft of this section offered an `env:` block. There is no such key.

        A workaround naming a field that does not exist sends the reader to a
        `ConfigError` at load time, which is worse than sending them nowhere.
        """
        import dataclasses
        import json
        import re

        from agent_loop_chaos import Scenario, Target, Trigger

        schemas = Path(__file__).resolve().parents[1] / "schemas"
        schema = json.loads((schemas / "chaos_report.schema.json").read_text())
        real = (
            {f.name for cls in (Scenario, Target, Trigger) for f in dataclasses.fields(cls)}
            | {"faults", "type", "params", "target", "trigger", "preset", "scenarios", "defaults"}
            # Names the section quotes from output rather than offers as config: a
            # report field, the CLI itself, and a log level.
            | set(schema["properties"])
            | {"alc", "warning"}
        )
        shown = set(re.findall(r"`([a-z_]+):(?: |`|\")", self._section()))
        assert shown, "the section shows no keys at all; did it lose its examples?"
        assert shown <= real, f"the README shows keys that do not exist: {sorted(shown - real)}"

    def test_the_commands_it_shows_exist(self) -> None:
        """A workaround naming a flag we never shipped is worse than no workaround."""
        import re

        from agent_loop_chaos.cli import build_parser

        parser = build_parser()
        subcommands = set(parser._subparsers._group_actions[0].choices)  # type: ignore[union-attr,attr-defined]
        for shown in set(re.findall(r"\balc ([a-z-]+)", self._section())):
            assert shown in subcommands, f"the README shows `alc {shown}`, which does not exist"


class TestTheInstallLineIsUsable:
    """Until it is on PyPI, the install line is the first thing that can be wrong.

    `#egg=` is deprecated and pip 26 will refuse it, so the README must show the
    PEP 508 `name[extras] @ url` form (D-137).
    """

    def test_it_says_it_is_not_on_pypi_yet(self) -> None:
        install = _readme().split("## Install")[1].split("## ")[0]
        assert "not on PyPI yet" in install

    def test_it_uses_the_pep_508_form(self) -> None:
        install = _readme().split("## Install")[1].split("## ")[0]
        assert "@ git+https://" in install
        assert "#egg=" not in install, "pip 26 refuses the egg fragment"

    def test_it_pins_a_tag_that_exists(self, repo_root: Path) -> None:
        """An install line pointing at a tag nobody cut is worse than none."""
        import re
        import subprocess

        install = _readme().split("## Install")[1].split("## ")[0]
        pinned = re.search(r"@(v\d+\.\d+\.\d+)", install)
        assert pinned, "the install line names no tag"
        tags = subprocess.run(
            ["git", "tag"], capture_output=True, text=True, cwd=repo_root
        ).stdout.split()
        assert pinned.group(1) in tags, f"{pinned.group(1)} is not a tag"

    def test_the_pinned_tag_is_the_current_version(self, repo_root: Path) -> None:
        import re

        from agent_loop_chaos import __version__

        install = _readme().split("## Install")[1].split("## ")[0]
        pinned = re.search(r"@v(\d+\.\d+\.\d+)", install)
        assert pinned and pinned.group(1) == __version__, (
            f"the README installs v{pinned.group(1) if pinned else '?'} but the code is "
            f"{__version__}"
        )
