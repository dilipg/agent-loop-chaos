"""`alc dashboard`, `alc run --dashboard`, and `alc report --format html`.

The one property that matters more than any feature here: **the dashboard cannot
change the outcome of a run**. `docs/10` §1 -- the engine does not know it exists, and
if it falls over the suite keeps going. So the exit code with `--dashboard` is
asserted equal to the exit code without it, on the same suite.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from agent_loop_chaos.cli import main

SUITE = {
    "version": "1.0",
    "defaults": {"seed": 1337},
    "scenarios": [
        {
            "id": f"tool.drop_key_{i}",
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
        for i in range(3)
    ],
}


@pytest.fixture
def suite_file(tmp_path: Path) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(SUITE))
    return path


def _run(suite_file: Path, out: Path, *extra: str) -> int:
    return main(["run", str(suite_file), "--out", str(out), "--judge", "rules", "--quiet", *extra])


class TestRunWithDashboard:
    def test_the_exit_code_is_unchanged(
        self, suite_file: Path, tmp_path: Path, loopback: None
    ) -> None:
        plain = _run(suite_file, tmp_path / "a")
        served = _run(suite_file, tmp_path / "b", "--dashboard", "--port", "0")
        assert served == plain

    def test_a_bind_failure_does_not_fail_the_suite(
        self, suite_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_loop_chaos.dashboard import server as server_mod

        def explode(self: object) -> None:
            raise OSError("address in use")

        monkeypatch.setattr(server_mod.DashboardServer, "start", explode)
        plain_out = tmp_path / "a"
        assert _run(suite_file, plain_out) == _run(
            suite_file, tmp_path / "b", "--dashboard", "--port", "0"
        )


class TestLiveSuiteJson:
    def test_status_goes_running_then_completed(
        self, suite_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every published document is complete, parseable, and correctly staged.

        Read back through the same path a viewer uses -- the file -- rather than from
        the arguments, so a half-written or non-atomic write would fail here.
        """
        from agent_loop_chaos import bundle

        real = bundle.write_suite_json
        seen: list[dict] = []

        def spy(out_dir: Path, results: object, **kwargs: object) -> str:
            path = real(out_dir, results, **kwargs)  # type: ignore[arg-type]
            seen.append(json.loads(Path(path).read_text()))
            return path

        monkeypatch.setattr(bundle, "write_suite_json", spy)

        out = tmp_path / ".chaos"
        _run(suite_file, out)

        assert len(seen) >= 4, "suite.json was not updated per scenario"
        assert seen[0]["status"] == "running"
        assert seen[0]["planned"] == [s["id"] for s in SUITE["scenarios"]]
        assert seen[0]["passed"] + seen[0]["failed"] == 0
        assert all(d["schema_version"] == "1.1" for d in seen)
        assert [d["status"] for d in seen].count("running") >= 3

        final = json.loads((out / "suite.json").read_text())
        assert final["status"] == "completed"
        assert final["current"] is None
        assert final["finished_at"]
        assert final["passed"] + final["failed"] == len(final["planned"]) == 3


class TestDashboardCommand:
    def test_once_renders_and_exits(
        self, suite_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / ".chaos"
        _run(suite_file, out)
        capsys.readouterr()
        assert main(["dashboard", "--out", str(out), "--once"]) == 0
        document = json.loads(capsys.readouterr().out)
        assert len(document["runs"]) == 3

    def test_a_missing_directory_is_a_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["dashboard", "--out", str(tmp_path / "nope")]) == 2

    def test_it_serves_the_directory(
        self, suite_file: Path, tmp_path: Path, loopback: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / ".chaos"
        _run(suite_file, out)
        from agent_loop_chaos.dashboard.server import DashboardServer

        server = DashboardServer(out, port=0, poll_ms=25)
        server.start()
        try:
            with urllib.request.urlopen(server.url + "/api/runs", timeout=5) as fh:
                assert len(json.loads(fh.read())) == 3
        finally:
            server.stop()


class TestReportHtml:
    def test_it_writes_a_self_contained_file(self, suite_file: Path, tmp_path: Path) -> None:
        out = tmp_path / ".chaos"
        _run(suite_file, out)
        target = tmp_path / "report.html"
        assert main(["report", str(out), "--format", "html", "-o", str(target)]) == 0
        page = target.read_text()
        assert "window.__ALC__" in page
        assert "http" + "://" not in page and "http" + "s://" not in page

    def test_a_run_directory_exports_its_whole_suite(
        self, suite_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / ".chaos"
        _run(suite_file, out)
        run_dir = next(p for p in out.rglob("report.json")).parent
        target = tmp_path / "one.html"
        assert main(["report", str(run_dir), "--format", "html", "-o", str(target)]) == 0
        assert "window.__ALC__" in target.read_text()
