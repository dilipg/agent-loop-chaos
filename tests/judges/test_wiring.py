"""The judge's seam into the engine, the bundle and the CLI.

The judge runs between `assemble` and the bundle write, so `report.json`,
`judge.json` and `AGENT_TASK.md` all carry the same verdict. The engine must survive
a judge that misbehaves: a judge failure is an `internal_error` event and a rules
verdict, never an exception into the run under test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.judges.base import JudgeEvidence, Verdict
from agent_loop_chaos.judges.rules import RuleJudge
from agent_loop_chaos.scenarios import Scenario


def _run(tmp_path: Path, judge: Any) -> Any:
    """Run the drop-key scenario through the shared suite runner.

    Not `engine.run(_scenario())`: that passes the `Scenario` *object* as the target,
    so no fault is registered and the engine fails with a `ConfigError` about an
    uninspectable callable. Every assertion in this file used to pass against that
    -- including "the agent crashed" -- which `error.raised_in` exposed the moment it
    started recording who actually raised.

    Args:
        tmp_path: Where to write the bundle.
        judge: The judge to use.

    Returns:
        The scenario's result.
    """
    from agent_loop_chaos.loop import run_suite

    return run_suite([_scenario()], out_dir=tmp_path, judge=judge)[0]


def _scenario() -> Scenario:
    """The drop-key scenario against the naive fake: a known, reproducible crash."""
    return Scenario(
        id="tool.drop_required_key",
        entrypoint="tests.fakes.apps:build_naive",
        inputs=None,
        faults=[
            {
                "type": "ToolCorruptionFault",
                "params": {"mutation_type": "drop_key", "keys": ["temp_c"]},
                "trigger": {"on_call": 1},
            }
        ],
        expected_behavior="graceful_degradation",
    )


class TestEngineWiring:
    def test_rules_judge_populates_the_verdict(self, tmp_path: Path) -> None:
        engine = ChaosEngine(out_dir=tmp_path, judge="rules", seed=7)
        result = engine.run(_scenario())
        assert result.verdict["narrative"]
        assert result.verdict["judge_meta"]["kind"] == "rules"
        assert result.verdict["confidence"] == 1.0

    def test_judge_fields_reach_the_report_top_level(self, tmp_path: Path) -> None:
        result = _run(tmp_path, "rules")
        assert result.refinement_hint == result.verdict["refinement_hint"]
        assert result.root_cause_hypothesis == result.verdict["root_cause_hypothesis"]
        assert result.suggested_fixes == result.verdict["suggested_fixes"]
        assert result.chaos_narrative == result.verdict["narrative"]

    def test_judge_cannot_change_success(self, tmp_path: Path) -> None:
        class Liar:
            name = "liar"

            def judge(self, ev: JudgeEvidence) -> Verdict:
                verdict = RuleJudge().judge(ev)
                verdict.passed = not ev.passed  # the thing that must never work
                return verdict

        result = _run(tmp_path, Liar())
        assert result.verdict["passed"] == result.success

    def test_a_raising_judge_is_an_internal_error_not_a_crash(self, tmp_path: Path) -> None:
        class Exploding:
            name = "boom"

            def judge(self, ev: JudgeEvidence) -> Verdict:
                raise RuntimeError("judge exploded")

        result = _run(tmp_path, Exploding())
        assert result.verdict["narrative"]  # rules verdict stood in
        events = [
            json.loads(line) for line in Path(result.artifacts["trace"]).read_text().splitlines()
        ]
        internal = [e for e in events if e["kind"] == "internal_error"]
        assert any("judge" in str(e.get("name")) for e in internal)
        # The agent's own behaviour is unchanged by the judge blowing up.
        assert result.failure_mode == "crash_unhandled_exception"

    def test_judge_json_is_written_for_a_model_judge(self, tmp_path: Path) -> None:
        class Recording:
            name = "recording"
            last_call: ClassVar[dict[str, Any]] = {
                "request": {"model": "x"},
                "response": {"content": "{}"},
            }

            def judge(self, ev: JudgeEvidence) -> Verdict:
                return RuleJudge().judge(ev)

        result = _run(tmp_path, Recording())
        path = Path(result.artifacts["run_dir"]) / "judge.json"
        assert path.is_file()
        stored = json.loads(path.read_text())
        assert stored["request"]["model"] == "x"
        assert stored["verdict"]["narrative"]

    def test_no_judge_json_when_the_judge_never_called_a_model(self, tmp_path: Path) -> None:
        result = _run(tmp_path, "rules")
        assert not (Path(result.artifacts["run_dir"]) / "judge.json").is_file()

    def test_report_is_still_schema_valid(self, tmp_path: Path) -> None:
        from agent_loop_chaos.schema import validate_obj

        result = _run(tmp_path, "rules")
        assert validate_obj(result.to_dict(), "report") == []
        assert validate_obj(result.verdict, "verdict") == []

    def test_agent_task_carries_the_judge_hint(self, tmp_path: Path) -> None:
        result = _run(tmp_path, "rules")
        task = Path(result.artifacts["agent_task"]).read_text()
        assert result.refinement_hint is not None
        assert result.refinement_hint.split(":")[0][:40] in task


class TestSuiteJson:
    def test_disagreement_rate_and_latency_recorded(self, tmp_path: Path) -> None:
        from agent_loop_chaos.bundle import write_suite_json

        class Disagreeing:
            name = "d"

            def judge(self, ev: JudgeEvidence) -> Verdict:
                verdict = RuleJudge().judge(ev)
                verdict.judge_disagreement = "model said x, probes said y"
                verdict.judge_meta = type(verdict.judge_meta)(kind="ensemble", latency_ms=120)
                return verdict

        engine = ChaosEngine(out_dir=tmp_path, judge=Disagreeing(), seed=7)
        results = [engine.run(_scenario())]
        suite = json.loads(Path(write_suite_json(tmp_path, results, seed=7)).read_text())
        assert suite["judge_disagreement_rate"] == 1.0
        assert suite["judge_latency_ms_total"] == 120
        # D-25: `coverage`'s namespace is fault kinds and nothing else.
        assert all(not k.startswith("judge") for k in suite["coverage"])

    def test_rate_is_zero_not_absent_when_nothing_disagreed(self, tmp_path: Path) -> None:
        from agent_loop_chaos.bundle import write_suite_json

        engine = ChaosEngine(out_dir=tmp_path, judge="rules", seed=7)
        results = [engine.run(_scenario())]
        suite = json.loads(Path(write_suite_json(tmp_path, results, seed=7)).read_text())
        assert suite["judge_disagreement_rate"] == 0.0


class TestAlcJudge:
    """`alc judge <run_dir>` re-judges from disk without re-running the agent."""

    @staticmethod
    def _run(tmp_path: Path) -> Path:
        result = _run(tmp_path, "rules")
        return Path(result.artifacts["run_dir"])

    def test_rewrites_the_bundle_with_a_new_judge_meta(self, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import main

        run_dir = self._run(tmp_path)
        before = json.loads((run_dir / "report.json").read_text())
        (run_dir / "report.json").write_text(
            json.dumps({**before, "verdict": {**before["verdict"], "narrative": "STALE"}})
        )

        # Exit 1, not 0: this run failed, and `alc judge` uses `alc run`'s exit
        # codes so CI can gate on a re-judge the same way it gates on a run.
        assert main(["judge", str(run_dir), "--judge", "rules"]) == 1
        after = json.loads((run_dir / "report.json").read_text())
        assert after["verdict"]["narrative"] != "STALE"
        assert after["verdict"]["judge_meta"]["kind"] == "rules"

    def test_does_not_re_run_the_agent(self, tmp_path: Path, monkeypatch: Any) -> None:
        from agent_loop_chaos.cli import main

        run_dir = self._run(tmp_path)
        before = json.loads((run_dir / "report.json").read_text())
        main(["judge", str(run_dir), "--judge", "rules"])
        after = json.loads((run_dir / "report.json").read_text())
        # Identity, metrics and the trace are untouched; only judgement changes.
        for field in ("run_id", "seed", "plan_hash", "metrics", "symptoms", "success"):
            assert after[field] == before[field], field

    def test_success_is_not_recomputed_by_the_judge(self, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import main

        run_dir = self._run(tmp_path)
        before = json.loads((run_dir / "report.json").read_text())["success"]
        main(["judge", str(run_dir), "--judge", "rules"])
        assert json.loads((run_dir / "report.json").read_text())["success"] == before

    def test_updates_agent_task(self, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import main

        run_dir = self._run(tmp_path)
        (run_dir / "AGENT_TASK.md").write_text("STALE")
        main(["judge", str(run_dir), "--judge", "rules"])
        assert (run_dir / "AGENT_TASK.md").read_text() != "STALE"

    def test_report_stays_schema_valid(self, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import main
        from agent_loop_chaos.schema import validate_obj

        run_dir = self._run(tmp_path)
        main(["judge", str(run_dir), "--judge", "rules"])
        assert validate_obj(json.loads((run_dir / "report.json").read_text()), "report") == []

    def test_missing_run_dir_is_a_usage_error(self, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import main

        assert main(["judge", str(tmp_path / "nope")]) == 2


class TestCliFlags:
    @pytest.mark.parametrize(
        "flag", ["--judge", "--model", "--base-url", "--transport", "--allow-remote-judge"]
    )
    def test_run_accepts_the_judge_flags(self, flag: str) -> None:
        from agent_loop_chaos.cli import build_parser

        actions = {a for action in build_parser()._actions for a in action.option_strings}
        subparsers = [a for a in build_parser()._actions if hasattr(a, "choices") and a.choices]
        run_parser = subparsers[0].choices["run"]  # type: ignore[union-attr]
        options = {opt for action in run_parser._actions for opt in action.option_strings}
        assert flag in options or flag in actions

    def test_narrate_all_is_off_by_default(self) -> None:
        from agent_loop_chaos.cli import build_parser

        args = build_parser().parse_args(["run", "x.yaml"])
        assert args.narrate_all is False
