"""Intensity, end to end: YAML -> plan -> report -> CLI.

The dial is only worth having if it reaches a run, and only safe if it is recorded.
Every run says what level produced it and which faults the dial added, because a plan
you cannot explain is a plan you cannot reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos.cli import main
from agent_loop_chaos.intensity import DEFAULT_LEVEL
from agent_loop_chaos.loop import run_suite
from agent_loop_chaos.scenarios import Scenario, load_suite

FAULT: dict[str, Any] = {
    "type": "ToolCorruptionFault",
    "params": {"mutation_type": "drop_key", "keys": ["temp_c"]},
    "trigger": {"on_call": 1, "max_fires": 1},
}


def _scenario(**kw: Any) -> Scenario:
    return Scenario(
        id="tool.drop_key",
        entrypoint="tests.fakes.apps:build_naive",
        inputs=None,
        faults=[dict(FAULT)],
        expected_behavior="graceful_degradation",
        **kw,
    )


class TestTheScenarioCarriesIt:
    def test_it_defaults_to_standard(self) -> None:
        assert _scenario().intensity == DEFAULT_LEVEL

    def test_a_suite_file_can_set_it(self, tmp_path: Path) -> None:
        path = tmp_path / "suite.json"
        path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "defaults": {"intensity": 8},
                    "scenarios": [{"id": "s", "entrypoint": "m:a", "faults": [FAULT]}],
                }
            )
        )
        assert load_suite(path).scenarios[0].intensity == 8

    def test_a_scenario_overrides_the_default(self, tmp_path: Path) -> None:
        path = tmp_path / "suite.json"
        path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "defaults": {"intensity": 8},
                    "scenarios": [
                        {"id": "s", "entrypoint": "m:a", "intensity": 2, "faults": [FAULT]}
                    ],
                }
            )
        )
        assert load_suite(path).scenarios[0].intensity == 2

    def test_a_level_off_the_dial_is_refused_at_load(self, tmp_path: Path) -> None:
        path = tmp_path / "suite.json"
        path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "scenarios": [
                        {"id": "s", "entrypoint": "m:a", "intensity": 42, "faults": [FAULT]}
                    ],
                }
            )
        )
        with pytest.raises(Exception, match="intensity"):
            load_suite(path)


class TestItReachesTheRun:
    def test_the_report_records_the_level(self, tmp_path: Path) -> None:
        [result] = run_suite([_scenario(intensity=7)], out_dir=tmp_path, judge="rules")
        assert result.to_dict()["intensity"] == {
            "level": 7,
            "label": "relentless",
            "summary": "Sustained pressure and two extra faults compounding.",
        }

    def test_the_default_still_records_itself(self, tmp_path: Path) -> None:
        [result] = run_suite([_scenario()], out_dir=tmp_path, judge="rules")
        assert result.to_dict()["intensity"]["level"] == DEFAULT_LEVEL

    def test_high_intensity_adds_faults(self, tmp_path: Path) -> None:
        low = run_suite([_scenario(intensity=3)], out_dir=tmp_path / "a", judge="rules")[0]
        high = run_suite([_scenario(intensity=9)], out_dir=tmp_path / "b", judge="rules")[0]
        assert len(high.injected_faults) > len(low.injected_faults)

    def test_an_added_fault_says_where_it_came_from(self, tmp_path: Path) -> None:
        [result] = run_suite([_scenario(intensity=9)], out_dir=tmp_path, judge="rules")
        origins = {f.get("origin", "") for f in result.injected_faults}
        assert "" in origins, "the declared fault lost its identity"
        assert any(o.startswith("intensity:9:") for o in origins)

    def test_a_real_action_fault_is_never_added_by_the_dial(self, tmp_path: Path) -> None:
        """SAFETY.md section 1 / D-23: turning a dial is not an opt-in."""
        [result] = run_suite([_scenario(intensity=10)], out_dir=tmp_path, judge="rules")
        added = [f for f in result.injected_faults if f.get("origin")]
        assert not [
            f
            for f in added
            if f["type"]
            in {"DuplicateSideEffectFault", "CheckpointRollbackFault", "ArgumentTamperFault"}
        ]

    def test_the_plan_hash_moves_with_the_level(self, tmp_path: Path) -> None:
        """A different level is a different experiment and must not reuse a run id."""
        low = run_suite([_scenario(intensity=2)], out_dir=tmp_path / "a", judge="rules")[0]
        high = run_suite([_scenario(intensity=8)], out_dir=tmp_path / "b", judge="rules")[0]
        assert low.plan_hash != high.plan_hash
        assert low.run_id != high.run_id

    def test_the_plan_file_records_it(self, tmp_path: Path) -> None:
        run_suite([_scenario(intensity=6)], out_dir=tmp_path, judge="rules")
        plan = json.loads(next(tmp_path.rglob("plan.json")).read_text())
        # An integer, not the rendered description: `plan.json` is what `alc replay`
        # rebuilds from and what `plan_hash` covers, so it holds identity, not prose.
        assert plan["intensity"] == 6


class TestTheDefaultIsTheIdentity:
    def test_level_three_matches_no_level_at_all(self, tmp_path: Path) -> None:
        """The load-bearing test: a dial nobody turned changes nothing."""
        from tests.normalize import normalize

        plain = run_suite([_scenario()], out_dir=tmp_path / "a", judge="rules")[0]
        dialled = run_suite(
            [_scenario(intensity=DEFAULT_LEVEL)], out_dir=tmp_path / "b", judge="rules"
        )[0]
        assert normalize(plain.to_dict()) == normalize(dialled.to_dict())


class TestTheCli:
    def test_the_flag_overrides_every_scenario(self, tmp_path: Path) -> None:
        suite = tmp_path / "suite.json"
        suite.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "scenarios": [
                        {
                            "id": "tool.drop_key",
                            "entrypoint": "tests.fakes.apps:build_naive",
                            "intensity": 2,
                            "expected_behavior": "graceful_degradation",
                            "faults": [FAULT],
                        }
                    ],
                }
            )
        )
        out = tmp_path / ".chaos"
        main(
            [
                "run",
                str(suite),
                "--out",
                str(out),
                "--judge",
                "rules",
                "--quiet",
                "--intensity",
                "9",
            ]
        )
        report = json.loads(next(out.rglob("report.json")).read_text())
        assert report["intensity"]["level"] == 9

    def test_a_level_off_the_dial_is_a_usage_error(self, tmp_path: Path) -> None:
        suite = tmp_path / "suite.json"
        suite.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "scenarios": [
                        {"id": "s", "entrypoint": "tests.fakes.apps:build_naive", "faults": [FAULT]}
                    ],
                }
            )
        )
        assert main(["run", str(suite), "--out", str(tmp_path / "o"), "--intensity", "77"]) == 2
