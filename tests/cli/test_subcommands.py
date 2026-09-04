"""One happy path and one error path per subcommand.

`--help` is the documentation most people read, so it is tested like documentation:
every subcommand names its flags and carries a worked example.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loop_chaos.cli import EXIT_OK, EXIT_USAGE, build_parser, main

SUBCOMMANDS = ("run", "replay", "judge", "explain", "report", "validate", "list-faults", "init")


def _subparsers() -> dict[str, object]:
    action = next(a for a in build_parser()._actions if getattr(a, "choices", None))
    return dict(action.choices)  # type: ignore[arg-type,union-attr]


class TestHelp:
    def test_every_documented_subcommand_exists(self) -> None:
        missing = [name for name in SUBCOMMANDS if name not in _subparsers()]
        assert missing == [], f"docs/02-API.md section 10 lists these: {missing}"

    @pytest.mark.parametrize("name", SUBCOMMANDS)
    def test_it_has_a_help_line(self, name: str) -> None:
        parser = _subparsers()[name]
        assert parser.format_help().strip()  # type: ignore[union-attr]

    @pytest.mark.parametrize("name", SUBCOMMANDS)
    def test_it_carries_a_worked_example(self, name: str) -> None:
        """Someone should be able to use the tool from `--help` alone."""
        help_text = _subparsers()[name].format_help()  # type: ignore[union-attr]
        assert "alc " + (name if name != "run" else "run") in help_text, (
            f"`alc {name} --help` shows no example invocation"
        )

    @pytest.mark.parametrize("name", SUBCOMMANDS)
    def test_every_flag_is_described(self, name: str) -> None:
        parser = _subparsers()[name]
        undocumented = [
            a.option_strings[0]
            for a in parser._actions  # type: ignore[union-attr]
            if a.option_strings and not a.help and a.option_strings[0] != "-h"
        ]
        assert undocumented == [], f"alc {name}: undescribed flags {undocumented}"


class TestInit:
    def test_it_scaffolds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert main(["init"]) == EXIT_OK
        assert (tmp_path / "chaos" / "quickstart.yaml").is_file()
        assert (tmp_path / "chaos" / "README.md").is_file()

    def test_the_scaffolded_suite_is_schema_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A scaffold that does not validate teaches the wrong shape on day one.
        monkeypatch.chdir(tmp_path)
        main(["init"])
        assert main(["validate", "chaos/quickstart.yaml"]) == EXIT_OK

    def test_it_refuses_to_overwrite(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        main(["init"])
        (tmp_path / "chaos" / "quickstart.yaml").write_text("# my edits")
        assert main(["init"]) == EXIT_USAGE
        assert "already exists" in capsys.readouterr().err
        assert (tmp_path / "chaos" / "quickstart.yaml").read_text() == "# my edits"

    def test_force_overwrites(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        main(["init"])
        (tmp_path / "chaos" / "quickstart.yaml").write_text("# my edits")
        assert main(["init", "--force"]) == EXIT_OK
        assert "ToolCorruptionFault" in (tmp_path / "chaos" / "quickstart.yaml").read_text()


class TestErrorPaths:
    @pytest.mark.parametrize(
        "argv",
        [
            ["run", "no-such-suite.yaml"],
            ["replay", "no-such-dir"],
            ["judge", "no-such-dir"],
            ["explain", "no-such-dir"],
            ["report", "no-such-dir"],
            ["validate", "no-such-file.json"],
        ],
    )
    def test_a_missing_input_is_a_usage_error(self, argv: list[str]) -> None:
        assert main(argv) == EXIT_USAGE

    def test_an_unknown_subcommand_is_a_usage_error(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["teleport"])
        assert excinfo.value.code == 2

    def test_no_subcommand_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == EXIT_USAGE
        assert "COMMAND" in capsys.readouterr().out


class TestExplainAndReport:
    @staticmethod
    def _a_run(tmp_path: Path) -> Path:
        suite = tmp_path / "s.json"
        suite.write_text(
            json.dumps(
                {
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
            )
        )
        out = tmp_path / "o"
        main(["run", str(suite), "--judge", "rules", "--out", str(out), "--quiet"])
        return next(out.glob("*/run-*"))

    def test_explain_prints_the_narrative_and_the_hint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_dir = self._a_run(tmp_path)
        capsys.readouterr()
        assert main(["explain", str(run_dir)]) == EXIT_OK
        printed = capsys.readouterr().out
        assert "FAIL" in printed
        assert "Seed 1337" in printed, "the narrative should be there"
        assert "hint" in printed.lower(), "explain is the 'what happened' command"

    def test_explain_shows_the_fault_diff(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_dir = self._a_run(tmp_path)
        capsys.readouterr()
        main(["explain", str(run_dir)])
        assert "temp_c" in capsys.readouterr().out

    def test_report_renders_markdown(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_dir = self._a_run(tmp_path)
        capsys.readouterr()
        assert main(["report", str(run_dir)]) == EXIT_OK
        assert "# Chaos finding" in capsys.readouterr().out
