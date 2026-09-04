"""The run directory exists and grows *during* a run, not after it.

Two engine guarantees the dashboard needs, both worth having on their own: a reader
can see a run in progress, and a crashed process leaves a partial trace rather than
nothing at all.

None of this may change the content of a finished `report.json` -- the golden tests
assert that separately, and they must pass untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import ToolCorruptionFault
from agent_loop_chaos.targeting import Target, Trigger


class TestTheDirectoryOpensAtRunStart:
    def test_the_trace_is_readable_mid_run(self, tmp_path: Path) -> None:
        """A second handle, opened while the agent is still working, sees events."""
        seen: dict[str, Any] = {}

        def agent(question: Any = None, engine: ChaosEngine | None = None) -> str:
            # Read from a *separate* handle, exactly as the watcher will.
            traces = list(tmp_path.rglob("trace.jsonl"))
            seen["traces"] = traces
            seen["lines"] = traces[0].read_text(encoding="utf-8").splitlines() if traces else []
            return "done"

        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        engine.run(
            lambda question=None: agent(question),
            inputs={"question": "?"},
            scenario_id="live",
            expected_behavior="ignore_and_continue",
        )

        assert seen["traces"], "no trace.jsonl existed while the agent was running"
        assert seen["lines"], "the trace was empty mid-run; nothing was flushed"
        first = json.loads(seen["lines"][0])
        assert first["kind"] == "run_started"

    def test_the_plan_is_written_before_the_agent_runs(self, tmp_path: Path) -> None:
        seen: dict[str, Any] = {}

        def agent(question: Any = None) -> str:
            seen["plans"] = list(tmp_path.rglob("plan.json"))
            return "done"

        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        engine.register_fault(
            ToolCorruptionFault(mutation_type="drop_key", keys=["x"]),
            target=Target(tool="nothing"),
            trigger=Trigger(on_call=1),
        )
        engine.run(
            agent,
            inputs={"question": "?"},
            scenario_id="live2",
            expected_behavior="ignore_and_continue",
        )
        assert seen["plans"], "plan.json was not written until the run finished"
        plan = json.loads(seen["plans"][0].read_text())
        assert plan["faults"], "the plan was written before the faults were frozen"

    def test_a_crashed_run_still_leaves_its_trace(self, tmp_path: Path) -> None:
        """The partial trace is the whole point: a run that died is still readable."""

        def agent(question: Any = None) -> str:
            raise RuntimeError("died mid-run")

        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        result = engine.run(agent, inputs={"question": "?"}, scenario_id="crash")
        trace = Path(result.artifacts["trace"])
        assert trace.is_file()
        kinds = [json.loads(line)["kind"] for line in trace.read_text().splitlines()]
        assert "run_started" in kinds

    def test_the_finished_report_is_unchanged(self, tmp_path: Path) -> None:
        """Opening early must not alter what a finished run produces."""
        from agent_loop_chaos.schema import validate_obj

        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=True)
        result = engine.run(
            lambda question=None: "done",
            inputs={"question": "?"},
            scenario_id="unchanged",
            expected_behavior="ignore_and_continue",
        )
        assert validate_obj(result.to_dict(), "report") == []
        assert Path(result.artifacts["report"]).is_file()


class TestSuiteJsonIsLive:
    """A viewer needs the denominator before the suite has finished."""

    @staticmethod
    def _suite(tmp_path: Path) -> Any:
        from agent_loop_chaos.scenarios import Scenario

        return [
            Scenario(
                id=f"s{i}",
                entrypoint="tests.fakes.apps:build_good",
                inputs=None,
                faults=[],
                expected_behavior="ignore_and_continue",
            )
            for i in range(3)
        ]

    def test_it_exists_before_the_first_scenario_finishes(self, tmp_path: Path) -> None:
        from agent_loop_chaos.loop import run_suite

        observed: list[dict[str, Any]] = []

        def peek(_result: Any) -> None:
            path = tmp_path / "suite.json"
            if path.is_file():
                observed.append(json.loads(path.read_text()))

        run_suite(self._suite(tmp_path), out_dir=tmp_path, judge="rules", on_result=peek)
        assert observed, "suite.json did not exist while the suite was running"
        assert observed[0]["status"] == "running"
        assert observed[0]["planned"] == ["s0", "s1", "s2"]

    def test_it_reports_progress_as_it_goes(self, tmp_path: Path) -> None:
        from agent_loop_chaos.loop import run_suite

        counts: list[int] = []

        def peek(_result: Any) -> None:
            document = json.loads((tmp_path / "suite.json").read_text())
            counts.append(document["passed"] + document["failed"])

        run_suite(self._suite(tmp_path), out_dir=tmp_path, judge="rules", on_result=peek)
        assert counts == [1, 2, 3], f"progress did not advance per scenario: {counts}"

    def test_it_ends_completed_with_a_finish_time(self, tmp_path: Path) -> None:
        from agent_loop_chaos.loop import run_suite

        run_suite(self._suite(tmp_path), out_dir=tmp_path, judge="rules")
        document = json.loads((tmp_path / "suite.json").read_text())
        assert document["status"] == "completed"
        assert document["finished_at"]
        assert document["schema_version"] == "1.1"

    def test_it_names_the_scenario_in_flight(self, tmp_path: Path) -> None:
        from agent_loop_chaos.loop import run_suite

        seen: list[Any] = []

        def peek(_result: Any) -> None:
            seen.append(json.loads((tmp_path / "suite.json").read_text())["current"])

        run_suite(self._suite(tmp_path), out_dir=tmp_path, judge="rules", on_result=peek)
        assert seen[0] is not None, "`current` should name what is running"

    def test_a_reader_never_sees_half_a_document(self, tmp_path: Path) -> None:
        """Atomic write: temp file plus `os.replace`."""
        from agent_loop_chaos.loop import run_suite

        def peek(_result: Any) -> None:
            for _ in range(20):
                json.loads((tmp_path / "suite.json").read_text())  # raises on a torn read

        run_suite(self._suite(tmp_path), out_dir=tmp_path, judge="rules", on_result=peek)

    def test_it_still_validates(self, tmp_path: Path) -> None:
        from agent_loop_chaos.loop import run_suite
        from agent_loop_chaos.schema import validate_obj

        run_suite(self._suite(tmp_path), out_dir=tmp_path, judge="rules")
        document = json.loads((tmp_path / "suite.json").read_text())
        assert validate_obj(document, "suite") == []
