"""How `alc` talks to a terminal, and to a pipe.

Two audiences, two contracts. A human gets one scannable line per scenario and a
summary; a machine gets exactly one JSON object on stdout and nothing else. The
second contract is the one that breaks silently — a stray log line makes
`alc run --json | jq` fail in someone's CI, not here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

ANSI = re.compile(r"\x1b\[[0-9;]*m")

SUITE = {
    "scenarios": [
        {
            "id": "tool.drop_key",
            "entrypoint": "tests.fakes.apps:build_naive",
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


@pytest.fixture
def suite_file(tmp_path: Path) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(SUITE))
    return path


class TestJsonIsMachineClean:
    """`--json` prints one object and nothing else, so CI can pipe it."""

    @pytest.mark.parametrize("command", ["run", "list-faults", "validate"])
    def test_stdout_parses_as_one_object(
        self,
        command: str,
        suite_file: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from agent_loop_chaos.cli import main

        argv = {
            "run": [
                "run",
                str(suite_file),
                "--json",
                "--judge",
                "rules",
                "--out",
                str(tmp_path / "o"),
            ],
            "list-faults": ["list-faults", "--json"],
            "validate": ["validate", str(suite_file), "--json"],
        }[command]
        main(argv)
        out = capsys.readouterr().out
        parsed = json.loads(out)  # raises if anything else was printed
        assert isinstance(parsed, dict), f"{command} did not print a JSON object"

    def test_json_output_carries_no_ansi(
        self, suite_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agent_loop_chaos.cli import main

        main(["run", str(suite_file), "--json", "--judge", "rules", "--out", str(tmp_path / "o")])
        assert not ANSI.search(capsys.readouterr().out)

    def test_report_json_parses(
        self, suite_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agent_loop_chaos.cli import main

        out = tmp_path / "o"
        main(["run", str(suite_file), "--judge", "rules", "--out", str(out), "--quiet"])
        capsys.readouterr()
        run_dir = next(out.glob("*/run-*"))
        main(["report", str(run_dir), "--format", "json"])
        assert json.loads(capsys.readouterr().out)["run_id"]


class TestColour:
    """Colour when a human is watching, never otherwise."""

    def test_no_ansi_when_not_a_tty(
        self, suite_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agent_loop_chaos.cli import main

        main(["run", str(suite_file), "--judge", "rules", "--out", str(tmp_path / "o")])
        captured = capsys.readouterr()
        assert not ANSI.search(captured.out + captured.err)

    def test_no_color_env_var_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_loop_chaos.cli import use_colour

        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        assert use_colour() is False

    def test_colour_on_a_tty_without_no_color(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_loop_chaos.cli import use_colour

        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        assert use_colour() is True

    def test_a_pipe_gets_no_colour(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_loop_chaos.cli import use_colour

        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        assert use_colour() is False


class TestHumanOutput:
    def test_the_summary_names_the_work_orders(
        self, suite_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agent_loop_chaos.cli import main

        main(["run", str(suite_file), "--judge", "rules", "--out", str(tmp_path / "o")])
        printed = capsys.readouterr().out
        assert "AGENT_TASK.md" in printed, "the work order is the product; name it"
        assert "1 failed" in printed or "failed 1" in printed

    def test_the_histogram_is_printed(
        self, suite_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agent_loop_chaos.cli import main

        main(["run", str(suite_file), "--judge", "rules", "--out", str(tmp_path / "o")])
        assert "crash_unhandled_exception" in capsys.readouterr().out

    def test_quiet_suppresses_the_per_scenario_lines(
        self, suite_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agent_loop_chaos.cli import main

        main(["run", str(suite_file), "--judge", "rules", "--out", str(tmp_path / "o"), "--quiet"])
        captured = capsys.readouterr()
        assert "tool.drop_key" not in captured.out
        assert captured.err == "" or "warning" in captured.err


class TestExitCodes:
    """All five, per D-26. CI gates on these."""

    def test_zero_when_everything_passes(self, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import EXIT_OK, main

        document = {
            "scenarios": [
                {
                    "id": "control.clean",
                    "entrypoint": "tests.fakes.apps:build_good",
                    "expected_behavior": "ignore_and_continue",
                    "faults": [],
                }
            ]
        }
        path = tmp_path / "clean.json"
        path.write_text(json.dumps(document))
        assert (
            main(["run", str(path), "--judge", "rules", "--out", str(tmp_path / "o"), "--quiet"])
            == EXIT_OK
        )

    def test_one_when_a_scenario_fails(self, suite_file: Path, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import EXIT_FAILED, main

        assert (
            main(
                [
                    "run",
                    str(suite_file),
                    "--judge",
                    "rules",
                    "--out",
                    str(tmp_path / "o"),
                    "--quiet",
                ]
            )
            == EXIT_FAILED
        )

    def test_two_on_a_usage_error(self, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import EXIT_USAGE, main

        assert main(["run", str(tmp_path / "missing.yaml"), "--quiet"]) == EXIT_USAGE

    def test_three_on_an_internal_error(
        self, monkeypatch: pytest.MonkeyPatch, suite_file: Path, tmp_path: Path
    ) -> None:
        from agent_loop_chaos import cli

        def explode(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("something the CLI did not anticipate")

        monkeypatch.setattr(cli, "_run", explode)
        assert cli.main(["run", str(suite_file), "--quiet"]) == cli.EXIT_INTERNAL

    def test_four_on_tampering(self, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import EXIT_TAMPERED, main
        from tests.fakes import swappable

        swappable.reset()
        swappable.BROKEN_SIBLING = True
        document = {
            "scenarios": [
                {
                    "id": "control.dry_run",
                    "entrypoint": "tests.fakes.swappable:build_sibling",
                    "expected_behavior": "ignore_and_continue",
                    "dry_run": True,
                    "faults": [],
                }
            ]
        }
        path = tmp_path / "control.json"
        path.write_text(json.dumps(document))
        try:
            assert (
                main(
                    [
                        "run",
                        str(path),
                        "--rounds",
                        "2",
                        "--judge",
                        "rules",
                        "--out",
                        str(tmp_path / "o"),
                        "--quiet",
                    ]
                )
                == EXIT_TAMPERED
            )
        finally:
            swappable.reset()
