"""`--jobs N` — parallel scenarios, or an honest refusal.

The flag was accepted, documented, and ignored. That is the same failure the library
exists to catch: something that reports it did the thing while doing nothing. A user
sets `--jobs 8` on a 30-scenario suite, sees no speed-up, and has no way to tell
whether their agent is slow or the flag is a lie.

The contract that makes it safe: **the same seeds produce the same reports whatever
`--jobs` is**. Scenarios are independent runs with their own engine, and `run_id`
derives from `(seed, scenario_id, plan_hash, attempt)` rather than from order, so
parallelism must not be able to change a verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos.loop import run_suite
from agent_loop_chaos.scenarios import Scenario

FAULT = {
    "type": "ToolCorruptionFault",
    "params": {"mutation_type": "drop_key", "keys": ["temp_c"]},
    "trigger": {"on_call": 1},
}


def _scenarios(count: int = 6) -> list[Scenario]:
    return [
        Scenario(
            id=f"tool.drop_key_{i}",
            entrypoint="tests.fakes.apps:build_naive",
            inputs=None,
            faults=[FAULT],
            expected_behavior="graceful_degradation",
        )
        for i in range(count)
    ]


def _normalized(results: list[Any]) -> list[dict[str, Any]]:
    from tests.normalize import normalize

    return [normalize(r.to_dict()) for r in sorted(results, key=lambda r: r.scenario_id or "")]


class TestParallelMatchesSerial:
    def test_the_same_verdicts_come_back(self, tmp_path: Path) -> None:
        serial = run_suite(_scenarios(), out_dir=tmp_path / "s", judge="rules")
        parallel = run_suite(_scenarios(), out_dir=tmp_path / "p", judge="rules", jobs=4)
        assert _normalized(serial) == _normalized(parallel), (
            "parallelism changed a report; a verdict must not depend on scheduling"
        )

    def test_results_come_back_in_scenario_order(self, tmp_path: Path) -> None:
        """A suite's output order must not depend on which worker finished first."""
        results = run_suite(_scenarios(), out_dir=tmp_path, judge="rules", jobs=4)
        assert [r.scenario_id for r in results] == [s.id for s in _scenarios()]

    def test_every_scenario_runs(self, tmp_path: Path) -> None:
        results = run_suite(_scenarios(), out_dir=tmp_path, judge="rules", jobs=4)
        assert len(results) == 6

    def test_run_ids_are_unchanged_by_parallelism(self, tmp_path: Path) -> None:
        serial = run_suite(_scenarios(), out_dir=tmp_path / "s", judge="rules")
        parallel = run_suite(_scenarios(), out_dir=tmp_path / "p", judge="rules", jobs=4)
        assert sorted(r.run_id for r in serial) == sorted(r.run_id for r in parallel)


class TestItStaysHonest:
    def test_jobs_one_is_the_serial_path(self, tmp_path: Path) -> None:
        results = run_suite(_scenarios(2), out_dir=tmp_path, judge="rules", jobs=1)
        assert len(results) == 2

    def test_fail_fast_forces_serial(self, tmp_path: Path) -> None:
        """`--fail-fast` means "stop at the first failure", which needs an order."""
        results = run_suite(_scenarios(), out_dir=tmp_path, judge="rules", jobs=4, fail_fast=True)
        assert len(results) == 1, "fail-fast must not keep launching work in parallel"

    def test_on_result_is_called_once_per_scenario(self, tmp_path: Path) -> None:
        seen: list[str] = []
        run_suite(
            _scenarios(),
            out_dir=tmp_path,
            judge="rules",
            jobs=4,
            on_result=lambda r: seen.append(str(r.scenario_id)),
        )
        assert sorted(seen) == sorted(s.id for s in _scenarios())


class TestTheCli:
    def test_jobs_reaches_the_runner(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_loop_chaos import cli

        captured: dict[str, Any] = {}

        def spy(scenarios: Any, **kwargs: Any) -> list[Any]:
            captured.update(kwargs)
            return []

        monkeypatch.setattr("agent_loop_chaos.loop.run_suite", spy)
        monkeypatch.setattr(
            "agent_loop_chaos.cli.write_suite_json", lambda *a, **k: "x", raising=False
        )

        suite = tmp_path / "s.json"
        suite.write_text(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "id": "a",
                            "entrypoint": "tests.fakes.apps:build_good",
                            "expected_behavior": "ignore_and_continue",
                            "faults": [],
                        }
                    ]
                }
            )
        )
        cli.main(["run", str(suite), "--jobs", "3", "--out", str(tmp_path / "o"), "--quiet"])
        assert captured.get("jobs") == 3
