"""A pytest fixture, so a project's own fixtures become its chaos fixtures.

A repository with a working test suite has already solved the expensive half of
onboarding: constructing the app offline, with fake databases and a stub model in its
`conftest.py`. Reaching that work is cheaper than any fixture format this library could
invent, so this plugin does exactly one thing -- hand over a configured `ChaosEngine`
and let the project's own fixtures build the agent.

    @pytest.mark.chaos(intercept=True)
    def test_it_degrades(chaos_engine, graph, seeded_db):   # graph, seeded_db are yours
        chaos_engine.register_fault(NodeSkipFault(), target_node="revalidation")
        result = chaos_engine.run(graph, inputs={...})
        assert result.success, result.failure_mode

The plugin ships as a `pytest11` entry point, so it loads in every pytest run in any
environment where this library is installed. That imposes one hard rule: it must stay
out of the way. No autouse fixture, no import of an optional dependency, and nothing
that changes a run which never asked for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:  # pragma: no cover - types only
    from .engine import ChaosEngine

__all__ = ["chaos_engine", "pytest_configure"]

_MARKER = "chaos"


def pytest_configure(config: pytest.Config) -> None:
    """Register the `chaos` marker.

    An unregistered marker raises `PytestUnknownMarkWarning`, which a project running
    with `-W error` would see as a failure caused by us.

    Args:
        config: The pytest config.
    """
    config.addinivalue_line(
        "markers",
        "chaos(**options): configure the `chaos_engine` fixture -- seed, intercept, "
        "seams, cassette, judge, intensity.",
    )


def _options(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Collect engine options from the closest `chaos` marker.

    Args:
        request: The fixture request.

    Returns:
        The marker's keyword arguments, or an empty mapping when there is no marker.
    """
    marker = request.node.get_closest_marker(_MARKER)
    return dict(marker.kwargs) if marker is not None else {}


@pytest.fixture
def chaos_engine(request: pytest.FixtureRequest, tmp_path: Any) -> ChaosEngine:
    """A `ChaosEngine` configured for a test run.

    Defaults chosen so a test is a test: the rules judge, so nothing dials out; no
    bundle, so a run does not litter the repository with `.chaos` directories; and
    `tmp_path` as the output directory for the little it does write.

    Args:
        request: The fixture request, carrying any `chaos` marker.
        tmp_path: pytest's per-test directory.

    Returns:
        The engine. Register faults on it and hand it your own agent fixture.
    """
    from .engine import ChaosEngine

    options: dict[str, Any] = {
        "seed": 1337,
        "out_dir": tmp_path,
        "judge": "rules",
        "write_bundle": False,
    }
    options.update(_options(request))
    if isinstance(options.get("cassette"), str):
        from .cassettes import Cassette

        options["cassette"] = Cassette(options["cassette"], mode="replay")
    return ChaosEngine(**options)
