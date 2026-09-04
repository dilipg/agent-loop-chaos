"""`alc run`, `report`, `validate` and `explain` against real runs.

`report`, `judge` and `explain` must work **from disk with no re-run**: the post-run
pipeline is pure over `(trace, plan)` precisely so a stored run can be re-read
(`docs/01` §5).

Exit codes are the contract CI pipes on: 0 all passed, 1 a scenario failed, 2
config or usage error, 3 internal error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, main

SUITE: dict[str, Any] = {
    "scenarios": [
        {
            "id": "clean",
            "entrypoint": "tests.fakes.strict_json_agent:run",
            "inputs": '{"ok": true}',
            "faults": [],
            "expected_behavior": "ignore_and_continue",
        }
    ]
}


def write(tmp_path: Path, body: dict[str, Any], name: str = "suite.json") -> Path:
    """Write a suite file.

    Args:
        tmp_path: pytest's temp directory.
        body: The suite document.
        name: The filename.

    Returns:
        The path.
    """
    path = tmp_path / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_run_a_passing_suite_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exit 0 means every scenario passed."""
    code = main(["run", str(write(tmp_path, SUITE)), "--out", str(tmp_path / ".chaos")])
    assert code == EXIT_OK
    assert "clean" in capsys.readouterr().out


def test_run_a_failing_suite_exits_one(tmp_path: Path) -> None:
    """Exit 1 is what makes `alc run` usable as a CI gate."""
    body = {
        "scenarios": [
            {
                "id": "broken",
                "entrypoint": "tests.fakes.naive_tool_agent:run",
                "inputs": {"data": []},
                "faults": [],
            }
        ]
    }
    assert main(["run", str(write(tmp_path, body)), "--out", str(tmp_path / ".chaos")]) == (
        EXIT_FAILED
    )


def test_run_writes_the_bundle_and_suite_json(tmp_path: Path) -> None:
    """A run directory per scenario, plus the file CI polls."""
    out = tmp_path / ".chaos"
    main(["run", str(write(tmp_path, SUITE)), "--out", str(out)])
    assert (out / "suite.json").exists()
    assert list(out.glob("clean/*/report.json")), "no run directory was written"


def test_run_json_prints_exactly_one_object(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "--json prints one JSON object to stdout and nothing else, so CI can pipe it"."""
    main(["run", str(write(tmp_path, SUITE)), "--out", str(tmp_path / ".chaos"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] == 1


def test_run_filter_selects_scenarios(tmp_path: Path) -> None:
    """A glob, so a work order can name one scenario to re-run."""
    body = {"scenarios": [dict(SUITE["scenarios"][0]), {**SUITE["scenarios"][0], "id": "other"}]}
    out = tmp_path / ".chaos"
    main(["run", str(write(tmp_path, body)), "--out", str(out), "--filter", "clean"])
    assert list(out.glob("clean/*")) and not list(out.glob("other/*"))


def test_run_a_missing_suite_is_a_usage_error(tmp_path: Path) -> None:
    """Exit 2 is config or usage, distinct from a scenario failing."""
    assert main(["run", str(tmp_path / "nope.json")]) == EXIT_USAGE


def test_report_reads_a_run_directory_without_re_running(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of keeping the pipeline pure over `(trace, plan)`."""
    out = tmp_path / ".chaos"
    main(["run", str(write(tmp_path, SUITE)), "--out", str(out)])
    run_dir = next(out.glob("clean/*"))
    capsys.readouterr()
    assert main(["report", str(run_dir), "--format", "json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["run_id"]


def test_report_renders_the_work_order_for_a_failing_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--format md` is the human view: the work order itself."""
    body = {
        "scenarios": [
            {
                "id": "broken",
                "entrypoint": "tests.fakes.naive_tool_agent:run",
                "inputs": {"data": []},
                "faults": [],
            }
        ]
    }
    out = tmp_path / ".chaos"
    main(["run", str(write(tmp_path, body)), "--out", str(out)])
    run_dir = next(out.glob("broken/*"))
    capsys.readouterr()
    main(["report", str(run_dir), "--format", "md"])
    assert "Chaos finding" in capsys.readouterr().out


def test_report_says_so_when_a_run_passed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A passing run has no work order, and the command must say that plainly."""
    out = tmp_path / ".chaos"
    main(["run", str(write(tmp_path, SUITE)), "--out", str(out)])
    run_dir = next(out.glob("clean/*"))
    capsys.readouterr()
    main(["report", str(run_dir), "--format", "md"])
    assert "no work order" in capsys.readouterr().out


def test_validate_accepts_a_written_report(tmp_path: Path) -> None:
    """`alc validate` is how a consumer checks a report it was handed."""
    out = tmp_path / ".chaos"
    main(["run", str(write(tmp_path, SUITE)), "--out", str(out)])
    report = next(out.glob("clean/*/report.json"))
    assert main(["validate", str(report)]) == EXIT_OK


def test_validate_rejects_a_broken_report(tmp_path: Path) -> None:
    """And says what is wrong, rather than exiting 0 on nonsense."""
    broken = tmp_path / "broken.json"
    broken.write_text('{"schema_version": "1.1"}', encoding="utf-8")
    assert main(["validate", str(broken)]) == EXIT_FAILED


def test_validate_accepts_a_suite_file(tmp_path: Path) -> None:
    """`alc validate <suite.yaml>` catches a malformed scenario before a long run."""
    assert main(["validate", str(write(tmp_path, SUITE))]) == EXIT_OK


def test_explain_narrates_a_stored_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A human-readable narrative to stdout, from disk."""
    out = tmp_path / ".chaos"
    main(["run", str(write(tmp_path, SUITE)), "--out", str(out)])
    run_dir = next(out.glob("clean/*"))
    capsys.readouterr()
    assert main(["explain", str(run_dir)]) == EXIT_OK
    text = capsys.readouterr().out
    assert "clean" in text
    assert "assertion" in text.lower() or "probe" in text.lower()


def test_explain_shows_which_assertions_ran(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """§4.4: auto-assertions are visible so an author can see what was checked."""
    body = {
        "scenarios": [
            {
                "id": "corrupt",
                "entrypoint": "tests.fakes.naive_tool_agent:run",
                "inputs": {"data": [{"temp_c": 21}]},
                "faults": [],
            }
        ]
    }
    out = tmp_path / ".chaos"
    main(["run", str(write(tmp_path, body)), "--out", str(out)])
    run_dir = next(out.glob("corrupt/*"))
    capsys.readouterr()
    main(["explain", str(run_dir)])
    assert "output_non_empty" in capsys.readouterr().out
