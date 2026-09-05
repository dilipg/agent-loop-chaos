"""The conformance suite: every agent shape, through the same fault battery.

The library's claim is that it attaches to an agent loop without caring how that loop
is written. `examples/patterns/` is a pool of shapes people actually ship -- a ReAct
text loop, an OpenAI tool-calling loop, an async agent, a class with state on `self`,
a supervisor delegating to specialists, a fixed pipeline with no loop, a streaming
accumulator, a retrieve-rerank-generate chain -- and this file runs all of them.

A pattern that cannot be instrumented, or whose fault never fires, or whose hardened
twin trips a probe, is a gap in the library rather than in the example. That is what
makes this a conformance suite: the assertions are about the harness, not the agents.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from examples.patterns import PatternSpec, load_all

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults.base import fault_from_dict
from agent_loop_chaos.targeting import Target, Trigger

PATTERNS = load_all()
NAMES = sorted(PATTERNS)


def _run(spec: PatternSpec, builder: Any, tmp_path: Path, *, armed: bool = True) -> Any:
    """Run one pattern under its declared faults."""
    engine = ChaosEngine(
        seed=1337,
        out_dir=tmp_path,
        write_bundle=False,
        strict_schema=True,
        judge="rules",
    )
    if armed:
        for spec_dict in spec.faults:
            target = spec_dict.get("target") or {}
            trigger = spec_dict.get("trigger") or {}
            engine.register_fault(
                fault_from_dict(dict(spec_dict)),
                target=Target(**target) if target else None,
                trigger=Trigger(**trigger) if trigger else None,
            )
    kwargs: dict[str, Any] = {
        "inputs": spec.inputs,
        "initial_state": dict(spec.initial_state or {}) or None,
        "scenario_id": spec.name,
        "expected_behavior": spec.expected_behavior if armed else "ignore_and_continue",
    }
    agent = builder(engine)
    if spec.is_async:
        return asyncio.run(engine.arun(agent, **kwargs))
    return engine.run(agent, **kwargs)


def _call(agent: Any, inputs: Any) -> Any:
    """Invoke an agent the way the engine resolves inputs (D-19).

    A mapping is spread as keyword arguments; anything else is passed positionally.
    Calling a dict positionally is what the engine specifically does not do, and a
    test that got it wrong would blame the example for the test's mistake.

    Args:
        agent: The built agent.
        inputs: Its payload.

    Returns:
        Whatever the agent returned.
    """
    if isinstance(inputs, Mapping):
        return agent(**inputs)
    return agent(inputs)


def test_the_pool_covers_the_shapes_that_matter() -> None:
    """A pool missing async, or missing a no-loop shape, proves less than it looks."""
    assert len(PATTERNS) >= 8, "the point of the pool is breadth"
    assert any(s.is_async for s in PATTERNS.values()), "async parity needs an async example"
    assert "pipeline" in PATTERNS, "a chain with no agent loop is the commonest shipped shape"


@pytest.mark.parametrize("name", NAMES)
class TestEveryShapeIsInstrumentable:
    def test_the_happy_path_runs_uninstrumented(self, name: str) -> None:
        """`engine=None` must still produce a real answer, or the example is a mock."""
        spec = PATTERNS[name]
        agent = spec.build(None)
        out = asyncio.run(_call(agent, spec.inputs)) if spec.is_async else _call(agent, spec.inputs)
        assert out and str(out).strip(), f"{name} produced nothing without an engine"

    def test_the_declared_fault_actually_fires(self, name: str, tmp_path: Path) -> None:
        """A scenario whose fault never fires proves nothing (D-64)."""
        result = _run(PATTERNS[name], PATTERNS[name].build, tmp_path)
        fired = [f["type"] for f in result.injected_faults if f.get("fired")]
        assert fired, f"{name}: nothing fired; the harness did not attach to this shape"

    def test_the_naive_agent_fails(self, name: str, tmp_path: Path) -> None:
        """The planted weakness has to be reachable, or the example is decorative."""
        result = _run(PATTERNS[name], PATTERNS[name].build, tmp_path)
        assert result.success is False, f"{name}: the weakness was not reached"
        assert result.failure_mode != "none"

    def test_the_hardened_agent_passes(self, name: str, tmp_path: Path) -> None:
        """The negative control. A probe that fires here has a false positive."""
        result = _run(PATTERNS[name], PATTERNS[name].build_fixed, tmp_path)
        assert result.success is True, (
            f"{name}: the hardened twin failed with {result.failure_mode} "
            f"({[s['code'] for s in result.symptoms]}). Either the fix is incomplete "
            "or a probe has a false positive -- diagnose before softening anything."
        )

    def test_no_probe_fires_on_a_clean_run(self, name: str, tmp_path: Path) -> None:
        """With nothing injected, both trees must look untouched."""
        for builder in (PATTERNS[name].build, PATTERNS[name].build_fixed):
            result = _run(PATTERNS[name], builder, tmp_path, armed=False)
            assert result.symptoms == [], f"{name}: {[s['code'] for s in result.symptoms]}"

    def test_the_report_is_schema_valid(self, name: str, tmp_path: Path) -> None:
        from agent_loop_chaos.schema import validate_obj

        result = _run(PATTERNS[name], PATTERNS[name].build, tmp_path)
        assert validate_obj(result.to_dict(), "report") == []

    def test_it_is_deterministic(self, name: str, tmp_path: Path) -> None:
        from tests.normalize import normalize

        a = _run(PATTERNS[name], PATTERNS[name].build, tmp_path)
        b = _run(PATTERNS[name], PATTERNS[name].build, tmp_path)
        assert normalize(a.to_dict()) == normalize(b.to_dict()), f"{name} is not reproducible"


@pytest.mark.parametrize("name", NAMES)
def test_the_weakness_is_not_announced_in_the_code(name: str) -> None:
    """A reader should have to think to spot the bug.

    The examples are meant to look like code someone would ship. A comment saying
    "BUG:" turns a credible demo into a strawman, and the explanation belongs in the
    module docstring where a reader looks *after* being surprised.
    """
    source = Path(f"examples/patterns/{name}.py").read_text(encoding="utf-8")
    body = source.split('"""', 2)[-1]  # everything after the module docstring
    for giveaway in ("BUG:", "FIXME", "XXX", "the bug is", "deliberately broken"):
        assert giveaway.lower() not in body.lower(), f"{name} announces its own weakness"


class TestTheRunnableSuite:
    """`examples/scenarios/patterns_suite.yaml` — the pool, from the CLI.

    The registry is the source of truth; the YAML is generated from it. A
    hand-maintained copy drifts, and the drift shows up as a scenario that proves
    nothing (D-64).
    """

    def test_the_committed_file_is_current(self, repo_root: Path) -> None:
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(repo_root / "tools" / "gen_patterns_suite.py"), "--check"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_it_loads_and_covers_every_pattern(self, repo_root: Path) -> None:
        from agent_loop_chaos.scenarios import load_suite

        suite = load_suite(repo_root / "examples" / "scenarios" / "patterns_suite.yaml")
        ids = {s.id for s in suite.scenarios}
        assert ids == {f"pattern.{n}{x}" for n in NAMES for x in ("", ".fixed")}

    def test_every_scenario_declares_a_fault(self, repo_root: Path) -> None:
        from agent_loop_chaos.scenarios import load_suite

        suite = load_suite(repo_root / "examples" / "scenarios" / "patterns_suite.yaml")
        assert all(s.faults for s in suite.scenarios)

    @pytest.mark.slow
    def test_the_whole_suite_splits_clean(self, repo_root: Path, tmp_path: Path) -> None:
        """Eight weaknesses found, eight hardened twins untouched.

        The single number that says the harness attaches to any loop shape *and*
        keeps its false-positive rate at zero.
        """
        from agent_loop_chaos.loop import run_suite
        from agent_loop_chaos.scenarios import load_suite

        suite = load_suite(repo_root / "examples" / "scenarios" / "patterns_suite.yaml")
        results = run_suite(suite.scenarios, out_dir=tmp_path, judge="rules")
        by_id = {r.scenario_id: r for r in results}

        failed = sorted(i for i, r in by_id.items() if not r.success)
        assert failed == sorted(f"pattern.{n}" for n in NAMES), (
            "a naive shape passed or a hardened twin failed; the second is a probe "
            "false positive and the probe is wrong until proven otherwise"
        )
        assert all(
            [f["type"] for f in by_id[f"pattern.{n}"].injected_faults if f.get("fired")]
            for n in NAMES
        ), "a scenario whose fault never fired proves nothing"
