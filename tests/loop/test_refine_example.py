"""`examples/refine_with_claude_code.py` — the value proposition, made runnable.

The script is the one place in the repo that can edit a user's source, so most of
what is tested here is the refusal path: no clean tree, no default branch, no
`--apply` unless it was asked for in words.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fakes import swappable

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "refine_with_claude_code.py"

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


@pytest.fixture
def suite_file(tmp_path: Path) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(SUITE))
    return path


def _module() -> object:
    import importlib.util

    spec = importlib.util.spec_from_file_location("refine_example", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDefaultMode:
    def test_it_reports_findings_and_task_paths(
        self, suite_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        module = _module()
        code = module.main(  # type: ignore[attr-defined]
            ["--suite", str(suite_file), "--out", str(tmp_path / "out")]
        )
        printed = capsys.readouterr().out
        assert code == 0
        assert "1 failure" in printed or "1 finding" in printed
        assert "AGENT_TASK.md" in printed

    def test_it_prints_one_full_work_order(
        self, suite_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        module = _module()
        module.main(["--suite", str(suite_file), "--out", str(tmp_path / "out")])  # type: ignore[attr-defined]
        printed = capsys.readouterr().out
        assert "# Chaos finding" in printed, "docs/09 section 7: print one finding in full"

    def test_it_points_at_the_reference_destination(
        self, suite_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        module = _module()
        module.main(["--suite", str(suite_file), "--out", str(tmp_path / "out")])  # type: ignore[attr-defined]
        assert "_fixed" in capsys.readouterr().out

    def test_it_does_not_touch_git(
        self, suite_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Default mode must never shell out at all, so there is nothing to get wrong.
        def refuse(*args: object, **kwargs: object) -> None:
            raise AssertionError(f"default mode shelled out: {args}")

        monkeypatch.setattr(subprocess, "run", refuse)
        module = _module()
        monkeypatch.setattr(module, "subprocess", subprocess)  # type: ignore[arg-type]
        assert module.main(["--suite", str(suite_file), "--out", str(tmp_path / "out")]) == 0  # type: ignore[attr-defined]

    def test_it_exits_zero_even_with_failures(self, suite_file: Path, tmp_path: Path) -> None:
        # Finding bugs is the job; it is not an error for the script itself.
        module = _module()
        assert module.main(["--suite", str(suite_file), "--out", str(tmp_path / "out")]) == 0  # type: ignore[attr-defined]


class TestApplyGuards:
    """`--apply` lets something else edit source files. Everything here is a refusal."""

    def test_a_dirty_tree_is_refused(
        self,
        suite_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        module = _module()
        monkeypatch.setattr(module, "git_status", lambda: " M src/thing.py")  # type: ignore[attr-defined]
        monkeypatch.setattr(module, "git_branch", lambda: "wip")  # type: ignore[attr-defined]
        code = module.main(  # type: ignore[attr-defined]
            ["--suite", str(suite_file), "--out", str(tmp_path / "out"), "--apply"]
        )
        assert code == 2
        assert "uncommitted" in capsys.readouterr().err.lower()

    @pytest.mark.parametrize("branch", ["main", "master"])
    def test_the_default_branch_is_refused(
        self,
        suite_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        branch: str,
    ) -> None:
        module = _module()
        monkeypatch.setattr(module, "git_status", lambda: "")  # type: ignore[attr-defined]
        monkeypatch.setattr(module, "git_branch", lambda: branch)  # type: ignore[attr-defined]
        code = module.main(  # type: ignore[attr-defined]
            ["--suite", str(suite_file), "--out", str(tmp_path / "out"), "--apply"]
        )
        assert code == 2
        assert "branch" in capsys.readouterr().err.lower()

    def test_a_missing_agent_cli_prints_the_tasks_and_exits_zero(
        self,
        suite_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        module = _module()
        monkeypatch.setattr(module, "git_status", lambda: "")  # type: ignore[attr-defined]
        monkeypatch.setattr(module, "git_branch", lambda: "wip")  # type: ignore[attr-defined]
        monkeypatch.setattr(module.shutil, "which", lambda _name: None)  # type: ignore[attr-defined]
        code = module.main(  # type: ignore[attr-defined]
            ["--suite", str(suite_file), "--out", str(tmp_path / "out"), "--apply"]
        )
        assert code == 0
        printed = capsys.readouterr().out
        assert "AGENT_TASK.md" in printed
        assert "would have" in printed.lower()

    def test_the_warning_is_loud_and_precedes_any_work(
        self,
        suite_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        module = _module()
        monkeypatch.setattr(module, "git_status", lambda: "")  # type: ignore[attr-defined]
        monkeypatch.setattr(module, "git_branch", lambda: "wip")  # type: ignore[attr-defined]
        monkeypatch.setattr(module.shutil, "which", lambda _name: None)  # type: ignore[attr-defined]
        module.main(  # type: ignore[attr-defined]
            ["--suite", str(suite_file), "--out", str(tmp_path / "out"), "--apply"]
        )
        printed = capsys.readouterr().out
        assert "WILL EDIT YOUR SOURCE FILES" in printed
        assert printed.index("WILL EDIT YOUR SOURCE FILES") < printed.index("AGENT_TASK.md")


class TestApplyRuns:
    def test_a_fake_agent_cli_is_invoked_once_per_task(
        self, suite_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = _module()
        calls: list[list[str]] = []

        monkeypatch.setattr(module, "git_status", lambda: "")  # type: ignore[attr-defined]
        monkeypatch.setattr(module, "git_branch", lambda: "wip")  # type: ignore[attr-defined]
        monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/fake-agent")  # type: ignore[attr-defined]

        def fake_run(cmd: list[str], **_kwargs: object) -> object:
            calls.append(cmd)
            swappable.FIXED = True  # the "coding agent" fixed it
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(module.subprocess, "run", fake_run)  # type: ignore[attr-defined]
        module.main(  # type: ignore[attr-defined]
            ["--suite", str(suite_file), "--out", str(tmp_path / "out"), "--apply", "--rounds", "2"]
        )
        assert len(calls) == 1
        # Invoked by name, not by the resolved path: that is what the user typed and
        # what appears in the printed command, so it stays debuggable.
        assert calls[0][0] == "claude"
        assert "# Chaos finding" in calls[0][-1], "the work order is passed in full"

    def test_the_round_table_is_printed_after_applying(
        self,
        suite_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        module = _module()
        monkeypatch.setattr(module, "git_status", lambda: "")  # type: ignore[attr-defined]
        monkeypatch.setattr(module, "git_branch", lambda: "wip")  # type: ignore[attr-defined]
        monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/fake-agent")  # type: ignore[attr-defined]

        def fake_run(cmd: list[str], **_kwargs: object) -> object:
            swappable.FIXED = True
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(module.subprocess, "run", fake_run)  # type: ignore[attr-defined]
        module.main(  # type: ignore[attr-defined]
            ["--suite", str(suite_file), "--out", str(tmp_path / "out"), "--apply", "--rounds", "2"]
        )
        printed = capsys.readouterr().out
        assert "| scenario | r1 | r2 | outcome |" in printed
        assert "fixed in r2" in printed


def test_the_script_runs_as_a_subprocess(suite_file: Path, tmp_path: Path) -> None:
    """It is documented as `python examples/refine_with_claude_code.py`, so run it that way."""
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--suite", str(suite_file), "--out", str(tmp_path / "out")],
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "AGENT_TASK.md" in proc.stdout
