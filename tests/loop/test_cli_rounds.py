"""`alc run --rounds N` — the loop from the command line.

Exit codes matter here: CI gates on them. 1 means a scenario still failed in the
final round; 4 means the harness itself was altered and nothing in the report can be
trusted (D-26).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_loop_chaos.cli import EXIT_FAILED, EXIT_OK, EXIT_TAMPERED, build_parser, main
from tests.fakes import swappable

SUITE = {
    "scenarios": [
        {
            "id": "tool.drop_key",
            "entrypoint": "tests.fakes.swappable:build",
            "expected_behavior": "graceful_degradation",
            "faults": [
                {
                    "type": "ToolCorruptionFault",
                    "params": {"mutation_type": "drop_key", "keys": ["temp_c"]},
                    "trigger": {"on_call": 1},
                }
            ],
        }
    ]
}


@pytest.fixture(autouse=True)
def _reset_fakes() -> Iterator[None]:
    swappable.reset()
    yield
    swappable.reset()


def _suite_file(tmp_path: Path, document: dict[str, object] | None = None) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(document or SUITE))
    return path


class TestFlags:
    def test_rounds_and_stop_when_parse(self) -> None:
        args = build_parser().parse_args(
            ["run", "s.yaml", "--rounds", "3", "--stop-when", "rounds"]
        )
        assert args.rounds == 3
        assert args.stop_when == "rounds"

    def test_rounds_defaults_to_none_meaning_a_single_run(self) -> None:
        assert build_parser().parse_args(["run", "s.yaml"]).rounds is None


class TestExitCodes:
    def test_still_failing_exits_1(self, tmp_path: Path) -> None:
        code = main(
            [
                "run",
                str(_suite_file(tmp_path)),
                "--rounds",
                "2",
                "--judge",
                "rules",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert code == EXIT_FAILED

    def test_everything_passing_exits_0(self, tmp_path: Path) -> None:
        swappable.FIXED = True
        code = main(
            [
                "run",
                str(_suite_file(tmp_path)),
                "--rounds",
                "2",
                "--judge",
                "rules",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert code == EXIT_OK

    def test_a_failing_control_exits_4(self, tmp_path: Path) -> None:
        swappable.BROKEN_SIBLING = True
        document = {
            "scenarios": [
                {
                    "id": "control.dry_run",
                    "entrypoint": "tests.fakes.swappable:build_sibling",
                    "expected_behavior": "ignore_and_continue",
                    "dry_run": True,
                    "faults": [],
                },
                *SUITE["scenarios"],  # type: ignore[misc]
            ]
        }
        code = main(
            [
                "run",
                str(_suite_file(tmp_path, document)),
                "--rounds",
                "2",
                "--judge",
                "rules",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert code == EXIT_TAMPERED

    def test_the_four_exit_codes_are_distinct(self) -> None:
        from agent_loop_chaos.cli import EXIT_INTERNAL, EXIT_USAGE

        assert len({EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_INTERNAL, EXIT_TAMPERED}) == 5
        assert EXIT_TAMPERED == 4


class TestOutput:
    def test_the_markdown_table_is_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "run",
                str(_suite_file(tmp_path)),
                "--rounds",
                "2",
                "--judge",
                "rules",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        printed = capsys.readouterr().out
        assert "| scenario | r1 | r2 | outcome |" in printed
        assert "tool.drop_key" in printed

    def test_suite_json_is_written_at_1_1(self, tmp_path: Path) -> None:
        from agent_loop_chaos.schema import validate_obj

        out = tmp_path / "out"
        main(
            [
                "run",
                str(_suite_file(tmp_path)),
                "--rounds",
                "2",
                "--judge",
                "rules",
                "--out",
                str(out),
            ]
        )
        document = json.loads((out / "suite.json").read_text())
        assert document["schema_version"] == "1.1"
        assert document["rounds_planned"] == 2
        assert validate_obj(document, "suite") == []

    def test_json_mode_emits_exactly_one_object(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "run",
                str(_suite_file(tmp_path)),
                "--rounds",
                "2",
                "--judge",
                "rules",
                "--json",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        printed = capsys.readouterr().out
        document = json.loads(printed)  # raises if anything else was printed
        assert document["rounds"]

    def test_no_rounds_flag_keeps_the_single_run_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            ["run", str(_suite_file(tmp_path)), "--judge", "rules", "--out", str(tmp_path / "out")]
        )
        printed = capsys.readouterr().out
        assert "| scenario |" not in printed
        document = json.loads((tmp_path / "out" / "suite.json").read_text())
        # 1.1 with no `rounds` key: the live-progress fields are written for every
        # suite from M10 on (docs/10 section 2), but only a refinement loop adds the
        # round history (D-112).
        assert document["schema_version"] == "1.1"
        assert document["status"] == "completed"
        assert "rounds" not in document
