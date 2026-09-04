"""The demo agent, and the proof the library's claim survives contact with it.

`examples/trip_planner` is written to look like code someone would ship, with twelve
planted weaknesses and no comment announcing any of them. `examples/trip_planner_fixed`
is the same agent with all twelve closed, and it is the demo suite's **negative
control**: a scenario that fails on it means a probe has a false positive or a fix is
incomplete, and the answer is to diagnose that, never to soften the scenario.

The suite runs are marked `slow` because each one executes thirty scenarios twice.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "examples" / "scenarios" / "demo_suite.yaml"

pytest.importorskip("langgraph", reason="the demo agent is a LangGraph app")


def _run_suite(tmp_path: Path, entrypoint: str | None = None) -> dict[str, Any]:
    """Run the demo suite against one tree and return its `suite.json`."""
    out = tmp_path / ("fixed" if entrypoint else "buggy")
    argv = [
        sys.executable,
        "-m",
        "agent_loop_chaos.cli",
        "run",
        str(SUITE),
        "--judge",
        "rules",
        "--out",
        str(out),
    ]
    if entrypoint:
        argv += ["--entrypoint", entrypoint]
    env = {"PYTHONPATH": f"{REPO / 'src'}:{REPO}", "PATH": "/usr/bin:/bin"}
    subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=900, env=env)
    return json.loads((out / "suite.json").read_text())


class TestHappyPath:
    """If the happy path is broken the chaos results are meaningless."""

    @pytest.mark.parametrize("tree", ["trip_planner", "trip_planner_fixed"])
    def test_it_runs_and_answers(self, tree: str) -> None:
        import importlib

        app = importlib.import_module(f"examples.{tree}.app")
        answer = app.graph.invoke({"query": app.QUERY, "messages": [], "attempts": 0})
        assert answer.get("answer"), f"{tree} produced no answer"

    def test_both_trees_agree_when_nothing_is_injected(self) -> None:
        """The fix must not change the product, only its behaviour under stress."""
        import importlib

        outs = []
        for tree in ("trip_planner", "trip_planner_fixed"):
            app = importlib.import_module(f"examples.{tree}.app")
            final = app.graph.invoke({"query": app.QUERY, "messages": [], "attempts": 0})
            outs.append(str(final["answer"]))
        # Same flight, same packing list. The wording differs only where the fixed
        # tree names the destination.
        for token in ("612", "Air India", "light shirts"):
            assert all(token in o for o in outs), token

    def test_no_comment_gives_away_a_bug(self) -> None:
        """A reader should have to think to spot the weaknesses."""
        for path in (REPO / "examples" / "trip_planner").rglob("*.py"):
            body = path.read_text(encoding="utf-8").split('"""', 2)[-1]
            for giveaway in ("BUG:", "FIXME", "XXX", "deliberately", "on purpose"):
                assert giveaway.lower() not in body.lower(), f"{path.name} gives it away"


@pytest.mark.slow
class TestTheProof:
    """The two numbers the README is allowed to quote."""

    def test_the_buggy_tree_finds_at_least_six_distinct_failure_modes(self, tmp_path: Path) -> None:
        suite = _run_suite(tmp_path)
        modes = {m for m in suite["failure_modes"] if m != "unknown"}
        assert len(modes) >= 6, f"only {len(modes)} distinct modes: {sorted(modes)}"

    def test_the_fixed_tree_finds_nothing(self, tmp_path: Path) -> None:
        suite = _run_suite(tmp_path, "examples.trip_planner_fixed.app:build_app")
        failures = [r["scenario_id"] for r in suite["results"] if not r["success"]]
        assert failures == [], (
            f"the negative control failed {failures}. Either a fix is incomplete or a "
            "probe has a false positive -- diagnose which; do not soften the scenario."
        )

    def test_the_dry_run_control_passes_on_both_trees(self, tmp_path: Path) -> None:
        for entrypoint in (None, "examples.trip_planner_fixed.app:build_app"):
            suite = _run_suite(tmp_path, entrypoint)
            control = [r for r in suite["results"] if r["scenario_id"] == "control.dry_run"]
            assert control and control[0]["success"], f"control failed ({entrypoint})"

    def test_every_scenario_fires_a_fault_on_the_buggy_tree(self, tmp_path: Path) -> None:
        """A scenario that never fires proves nothing (D-64). This catches dead ones."""
        suite = _run_suite(tmp_path)
        dead = []
        for result in suite["results"]:
            if result["scenario_id"] == "control.dry_run":
                continue  # armed but never applied, by design (D-12)
            report = json.loads(Path(result["report"]).read_text())
            if not any(f.get("fired") for f in report["injected_faults"]):
                dead.append(result["scenario_id"])
        assert dead == [], f"these scenarios armed a fault that never fired: {dead}"


def test_the_vanilla_example_runs_under_the_engine(tmp_path: Path) -> None:
    """`examples/vanilla_agent.py` is the blueprint's plain-Python shape."""
    from examples.vanilla_agent import build

    from agent_loop_chaos import ChaosEngine
    from agent_loop_chaos.faults import ToolCorruptionFault
    from agent_loop_chaos.targeting import Target, Trigger

    engine = ChaosEngine(seed=1337, out_dir=tmp_path, write_bundle=False, judge="rules")
    engine.register_fault(
        ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
        target=Target(tool="get_weather_data"),
        trigger=Trigger(on_call=1),
    )
    result = engine.run(build(engine), inputs={"question": "weather in Paris?"}, scenario_id="v")
    assert result.success is False, "the plain-Python example should find something"
    assert any(f["fired"] for f in result.injected_faults)
