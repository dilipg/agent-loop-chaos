"""The pytest plugin: a repository's own fixtures are the chaos fixtures.

A project with a working test suite has already solved the hard half of onboarding --
constructing the app offline, with fake databases and a stub model. Making that work
reachable is cheaper than any fixture format this library could invent, so the plugin
does one thing: hand over a configured engine and let their fixtures build the agent.

The plugin ships as a `pytest11` entry point, which means it loads in every pytest run
in any environment where the library is installed. So the tests that matter most here
are the ones asserting it stays out of the way: no autouse fixture, no import cost, and
nothing that changes a run that never asked for it.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

# A sub-run gets its own ini. Without the asyncio option set, `pytest_asyncio` emits a
# deprecation warning at configure time, and this repo turns warnings into errors --
# which surfaces as an INTERNALERROR in the sub-run and has nothing to do with the
# plugin under test.
SUBRUN_INI = """
[pytest]
asyncio_default_fixture_loop_scope = function
"""


class TestItStaysOutOfTheWay:
    def test_it_adds_no_autouse_fixture(self) -> None:
        """An autouse fixture here would run in every unrelated project's suite."""
        from agent_loop_chaos import pytest_plugin

        for name, obj in vars(pytest_plugin).items():
            marker = getattr(obj, "_pytestfixturefunction", None)
            if marker is not None:
                assert not marker.autouse, f"{name} is autouse"

    def test_importing_it_costs_no_optional_dependency(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src"
        code = (
            "import sys, agent_loop_chaos.pytest_plugin as p;"
            "print(','.join(sorted(m for m in ('httpx','langchain_core','langgraph')"
            " if m in sys.modules)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        assert out == "", f"importing the plugin pulled in {out}"


class TestTheEngineFixture:
    def test_it_hands_over_a_usable_engine(self, pytester: pytest.Pytester) -> None:
        """The shape a colleague writes: their fixture builds the agent, ours breaks it."""
        pytester.makeini(SUBRUN_INI)
        pytester.makepyfile(
            """
            import pytest
            from agent_loop_chaos import Target
            from agent_loop_chaos.faults import ToolCorruptionFault

            @pytest.fixture
            def agent(chaos_engine):
                # Stands in for a project's own conftest fixture.
                tool = chaos_engine.tool(lambda city: {"temp_c": 21}, name="get_weather")
                def run(question):
                    return f"it is {tool('Paris').get('temp_c', 'unknown')}"
                return run

            def test_it_degrades(chaos_engine, agent):
                chaos_engine.register_fault(
                    ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
                    target=Target(layer="tool", phase="post"),
                )
                result = chaos_engine.run(agent, inputs="what should I pack?")
                assert result.injected_faults[0]["fired"] is True
                assert "unknown" in result.final_output
            """
        )
        pytester.runpytest("-q").assert_outcomes(passed=1)

    def test_it_writes_no_bundle_into_the_project(self, pytester: pytest.Pytester) -> None:
        """A test run must not litter a repository with .chaos directories."""
        pytester.makeini(SUBRUN_INI)
        pytester.makepyfile(
            """
            def test_no_bundle(chaos_engine, tmp_path):
                chaos_engine.run(lambda q: "fine", inputs="q")
            """
        )
        pytester.runpytest("-q").assert_outcomes(passed=1)
        assert not (pytester.path / ".chaos").exists()

    def test_the_marker_configures_it(self, pytester: pytest.Pytester) -> None:
        """`@pytest.mark.chaos(...)` is how a test asks for interception or a seed."""
        pytester.makeini(SUBRUN_INI)
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.chaos(seed=99, intercept=True)
            def test_configured(chaos_engine):
                assert chaos_engine.seed == 99
                assert chaos_engine.intercept is True

            def test_defaults(chaos_engine):
                assert chaos_engine.intercept is False
            """
        )
        pytester.runpytest("-q").assert_outcomes(passed=2)

    def test_the_marker_is_registered(self, pytester: pytest.Pytester) -> None:
        """An unregistered marker warns, and `-W error` turns that into a failure."""
        pytester.makeini(SUBRUN_INI)
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.chaos(seed=1)
            def test_marked(chaos_engine):
                assert chaos_engine.seed == 1
            """
        )
        pytester.runpytest("-q", "-W", "error::pytest.PytestUnknownMarkWarning").assert_outcomes(
            passed=1
        )
