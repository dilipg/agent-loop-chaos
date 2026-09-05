"""Getting the work orders out of the browser and into a coding agent.

`AGENT_TASK.md` is the product. A reader who has just seen twenty-one failures needs
the whole set as one file they can hand over -- not twenty-one clicks, and not a
copy-to-clipboard that loses everything after the first one.

Two endpoints, both read-only like everything else here:

- `?download=1` on an artifact, which only adds a `Content-Disposition` header.
- `/api/tasks.md`, every failing run's work order concatenated, newest problem first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loop_chaos.dashboard import api
from agent_loop_chaos.dashboard.watcher import RunDirWatcher


def _run(root: Path, scenario: str, run_id: str, *, failed: bool, task: str = "") -> None:
    run = root / scenario / run_id
    run.mkdir(parents=True)
    (run / "trace.jsonl").write_text(
        json.dumps({"seq": 1, "kind": "run_started", "layer": "engine"}) + "\n"
    )
    # A schema-complete report, so the per-run render path is exercised for real
    # rather than against a stub that only the listing endpoints would accept.
    document = json.loads(
        (Path(__file__).resolve().parents[1] / "data" / "report_failing.json").read_text()
    )
    document |= {
        "run_id": run_id,
        "scenario_id": scenario,
        "scenario_title": scenario.replace(".", " "),
        "success": not failed,
        "failure_mode": "silent_wrong_answer" if failed else "none",
        "severity": "high" if failed else "info",
    }
    (run / "report.json").write_text(json.dumps(document))
    if task:
        (run / "AGENT_TASK.md").write_text(task)


@pytest.fixture
def chaos(tmp_path: Path) -> Path:
    root = tmp_path / ".chaos"
    _run(root, "tool.drop_key", "run-a", failed=True, task="# Chaos finding — one\n\nfix me\n")
    _run(root, "llm.empty", "run-b", failed=True, task="# Chaos finding — two\n\nfix me too\n")
    _run(root, "control.clean", "run-c", failed=False)
    return root


@pytest.fixture
def watcher(chaos: Path) -> RunDirWatcher:
    watcher = RunDirWatcher(chaos)
    watcher.poll()
    return watcher


class TestTheBundle:
    def test_it_carries_every_failing_run(self, watcher: RunDirWatcher) -> None:
        body, ctype = api.tasks(watcher)
        assert "fix me" in body and "fix me too" in body
        assert ctype.startswith("text/markdown")

    def test_a_passing_run_contributes_nothing(self, watcher: RunDirWatcher) -> None:
        """Work orders are written for failures only. A pass has nothing to hand over."""
        assert "control.clean" not in api.tasks(watcher)[0]

    def test_it_leads_with_what_to_do_with_it(self, watcher: RunDirWatcher) -> None:
        body, _ = api.tasks(watcher)
        head = body.split("\n---\n")[0]
        assert "2 findings" in head
        assert "agent-loop-chaos" in head

    def test_each_finding_says_which_run_it_came_from(self, watcher: RunDirWatcher) -> None:
        body, _ = api.tasks(watcher)
        assert "tool.drop_key/run-a" in body
        assert "llm.empty/run-b" in body

    def test_the_order_is_stable(self, watcher: RunDirWatcher) -> None:
        assert api.tasks(watcher)[0] == api.tasks(watcher)[0]

    def test_an_empty_directory_says_so_rather_than_returning_nothing(self, tmp_path: Path) -> None:
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        body, _ = api.tasks(watcher)
        assert "0 findings" in body or "nothing" in body.lower()


class TestOverHttp:
    @pytest.fixture
    def server(self, chaos: Path, loopback: None) -> object:
        from agent_loop_chaos.dashboard.server import DashboardServer

        srv = DashboardServer(chaos, port=0, poll_ms=25)
        srv.start()
        yield srv
        srv.stop()

    def test_the_bundle_downloads_as_a_file(self, server: object) -> None:
        import urllib.request

        with urllib.request.urlopen(server.url + "/api/tasks.md", timeout=5) as fh:  # type: ignore[attr-defined]
            headers = dict(fh.headers)
            body = fh.read().decode()
        assert "attachment" in headers["Content-Disposition"]
        assert "chaos-findings" in headers["Content-Disposition"]
        assert "fix me" in body

    def test_one_artifact_downloads_when_asked(self, server: object) -> None:
        import urllib.request

        url = server.url + "/api/run/run-a/artifact/AGENT_TASK.md?download=1"  # type: ignore[attr-defined]
        with urllib.request.urlopen(url, timeout=5) as fh:
            headers = dict(fh.headers)
        assert "attachment" in headers["Content-Disposition"]
        assert "AGENT_TASK.md" in headers["Content-Disposition"]

    def test_without_the_flag_it_renders_inline(self, server: object) -> None:
        import urllib.request

        url = server.url + "/api/run/run-a/artifact/AGENT_TASK.md"  # type: ignore[attr-defined]
        with urllib.request.urlopen(url, timeout=5) as fh:
            assert "Content-Disposition" not in dict(fh.headers)

    def test_the_download_flag_cannot_smuggle_a_filename(self, server: object) -> None:
        """The name comes from the allow-list, never from the request."""
        import urllib.request

        url = (
            server.url  # type: ignore[attr-defined]
            + "/api/run/run-a/artifact/AGENT_TASK.md?download=%22%3B+rm+-rf+/"
        )
        with urllib.request.urlopen(url, timeout=5) as fh:
            disposition = dict(fh.headers)["Content-Disposition"]
        assert disposition == 'attachment; filename="AGENT_TASK.md"'


class TestThePageOffersIt:
    def test_the_report_view_has_a_download_button(self, repo_root: Path) -> None:
        page = (repo_root / "src/agent_loop_chaos/dashboard/static/index.html").read_text()
        assert "Download all work orders" in page
        assert "/api/tasks.md" in page


class TestTheTerminalPath:
    """`alc report <out_dir> --format md` — the same bundle, without a browser."""

    def test_a_suite_directory_renders_every_work_order(
        self, chaos: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agent_loop_chaos.cli import main

        assert main(["report", str(chaos), "--format", "md"]) == 0
        printed = capsys.readouterr().out
        assert "2 findings" in printed
        assert "fix me" in printed and "fix me too" in printed

    def test_a_single_run_still_renders_just_that_one(
        self, chaos: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agent_loop_chaos.cli import main

        assert main(["report", str(chaos / "tool.drop_key" / "run-a"), "--format", "md"]) == 0
        printed = capsys.readouterr().out
        # One run renders its own work order from `report.json`, as it always has --
        # the suite path is chosen only when there is no report.json to render.
        assert "2 findings" not in printed
        assert "tool.drop_key" in printed
        assert "llm.empty" not in printed

    def test_it_writes_to_a_file_when_asked(self, chaos: Path, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import main

        target = tmp_path / "findings.md"
        assert main(["report", str(chaos), "--format", "md", "-o", str(target)]) == 0
        assert "fix me too" in target.read_text()


class TestTheJsonHandoff:
    """`--json` is how a harness reads a run without parsing prose."""

    def _run(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict:
        from agent_loop_chaos.cli import main

        suite = tmp_path / "suite.json"
        suite.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "scenarios": [
                        {
                            "id": "tool.drop_key",
                            "title": "A tool leaves out a field the agent needs",
                            "entrypoint": "tests.fakes.apps:build_naive",
                            "faults": [
                                {
                                    "type": "ToolCorruptionFault",
                                    "params": {"mutation_type": "drop_key", "keys": ["temp_c"]},
                                    "trigger": {"on_call": 1},
                                }
                            ],
                        }
                    ],
                }
            )
        )
        main(["run", str(suite), "--out", str(tmp_path / ".chaos"), "--judge", "rules", "--json"])
        return json.loads(capsys.readouterr().out)

    def test_it_names_the_work_order_to_act_on(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The path to `AGENT_TASK.md` is the single most useful field for a harness."""
        row = self._run(tmp_path, capsys)["results"][0]
        assert row["agent_task"] is None or row["agent_task"].endswith("AGENT_TASK.md")

    def test_it_carries_severity(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        row = self._run(tmp_path, capsys)["results"][0]
        assert row["severity"] in {"critical", "high", "medium", "low", "info"}

    def test_it_carries_the_human_title(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        row = self._run(tmp_path, capsys)["results"][0]
        assert row["title"] == "A tool leaves out a field the agent needs"

    def test_the_documented_keys_are_the_real_ones(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], repo_root: Path
    ) -> None:
        """The README shows this object. A field it names has to exist."""
        row = self._run(tmp_path, capsys)["results"][0]
        readme = (repo_root / "README.md").read_text()
        block = readme.split("```jsonc")[1].split("```")[0]
        for key in ("scenario_id", "success", "failure_mode", "severity", "agent_task"):
            assert f'"{key}"' in block, f"the README stopped showing {key}"
            assert key in row, f"the README documents {key}, which --json does not emit"


class TestHead:
    """A read-only server that cannot answer HEAD is not read-only, it is GET-only.

    Every HTTP client checks a download's size and type with HEAD before fetching it,
    and `BaseHTTPRequestHandler`'s default is a 501 with an HTML error body -- which is
    what the browser's own download machinery would have hit.
    """

    @pytest.fixture
    def server(self, chaos: Path, loopback: None) -> object:
        from agent_loop_chaos.dashboard.server import DashboardServer

        srv = DashboardServer(chaos, port=0, poll_ms=25)
        srv.start()
        yield srv
        srv.stop()

    def _head(self, url: str) -> tuple[int, dict[str, str]]:
        import urllib.request

        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=5) as fh:
            assert fh.read() == b"", "HEAD must not send a body"
            return fh.status, dict(fh.headers)

    def test_it_answers_head_on_the_bundle(self, server: object) -> None:
        status, headers = self._head(server.url + "/api/tasks.md")  # type: ignore[attr-defined]
        assert status == 200
        assert headers["Content-Type"].startswith("text/markdown")
        assert "attachment" in headers["Content-Disposition"]
        assert int(headers["Content-Length"]) > 0

    def test_it_answers_head_on_the_page(self, server: object) -> None:
        status, headers = self._head(server.url + "/")  # type: ignore[attr-defined]
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")

    def test_head_on_a_missing_run_is_still_404(self, server: object) -> None:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            server.url + "/api/run/nope/artifact/report.json",
            method="HEAD",  # type: ignore[attr-defined]
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=5)
        with exc.value:
            assert exc.value.code == 404
