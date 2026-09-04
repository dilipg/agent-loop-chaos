"""The JSON endpoints, and the path safety that has to hold.

`docs/10` §5: `run_id` is matched against a pattern **and** resolved by lookup in the
watcher's discovered map — never by joining request input onto a path. Artifact names
come from a literal allow-list. `payloads/` is never serveable, because it is the
verbose-mode overflow directory and holds raw payloads.

Pure functions over the watcher plus the filesystem, so all of this is testable with
no server running.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loop_chaos.dashboard import api
from agent_loop_chaos.dashboard.watcher import RunDirWatcher


@pytest.fixture
def chaos(tmp_path: Path) -> Path:
    """A fixture `.chaos/`: one finished run, one mid-run, one aborted."""
    root = tmp_path / ".chaos"

    done = root / "tool.drop_key" / "run-done"
    done.mkdir(parents=True)
    (done / "trace.jsonl").write_text(
        json.dumps({"seq": 1, "kind": "run_started", "layer": "engine"})
        + "\n"
        + json.dumps({"seq": 2, "kind": "fault_fired", "layer": "tool", "name": "w"})
        + "\n"
        + json.dumps({"seq": 3, "kind": "run_finished", "layer": "engine"})
        + "\n"
    )
    (done / "report.json").write_text(
        json.dumps(
            {
                "run_id": "run-done",
                "scenario_id": "tool.drop_key",
                "success": False,
                "failure_mode": "crash_unhandled_exception",
                "severity": "high",
                "started_at": "1970-01-01T00:00:00Z",
                "duration_ms": 12,
            }
        )
    )
    (done / "plan.json").write_text(json.dumps({"seed": 1337, "faults": []}))
    (done / "AGENT_TASK.md").write_text("# Chaos finding\n")
    (done / "payloads").mkdir()
    (done / "payloads" / "secret.json").write_text(json.dumps({"api_key": "sk-nope"}))

    live = root / "tool.unit_swap" / "run-live"
    live.mkdir(parents=True)
    (live / "trace.jsonl").write_text(
        json.dumps({"seq": 1, "kind": "run_started", "layer": "engine"}) + "\n"
    )
    (live / "plan.json").write_text(json.dumps({"seed": 1337, "faults": []}))

    (root / "suite.json").write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "status": "running",
                "planned": ["a", "b"],
                "passed": 1,
                "failed": 0,
                "seed": 1337,
            }
        )
    )
    return root


@pytest.fixture
def watcher(chaos: Path) -> RunDirWatcher:
    w = RunDirWatcher(chaos)
    w.poll()
    return w


class TestSuiteAndRuns:
    def test_suite_returns_the_document(self, watcher: RunDirWatcher) -> None:
        body = api.suite(watcher)
        assert body["status"] == "running"
        assert body["planned"] == ["a", "b"]

    def test_suite_synthesizes_when_absent(self, tmp_path: Path) -> None:
        """Pointing at a directory with no suite.json must still render."""
        run = tmp_path / "s1" / "run-a"
        run.mkdir(parents=True)
        (run / "trace.jsonl").write_text("")
        w = RunDirWatcher(tmp_path)
        w.poll()
        body = api.suite(w)
        assert body["synthesized"] is True
        assert body["planned"] == ["s1"]

    def test_runs_lists_both(self, watcher: RunDirWatcher) -> None:
        ids = {r["run_id"] for r in api.runs(watcher)}
        assert ids == {"run-done", "run-live"}

    def test_a_finished_run_carries_its_verdict(self, watcher: RunDirWatcher) -> None:
        row = next(r for r in api.runs(watcher) if r["run_id"] == "run-done")
        assert row["status"] == "completed"
        assert row["success"] is False
        assert row["failure_mode"] == "crash_unhandled_exception"
        assert row["artifacts"]["AGENT_TASK.md"] is True
        assert row["artifacts"]["judge.json"] is False

    def test_a_mid_run_has_no_verdict_yet(self, watcher: RunDirWatcher) -> None:
        row = next(r for r in api.runs(watcher) if r["run_id"] == "run-live")
        assert row["status"] == "running"
        assert row["success"] is None
        assert row["events"] == 1

    def test_runs_are_ordered_stably(self, watcher: RunDirWatcher) -> None:
        assert [r["run_id"] for r in api.runs(watcher)] == sorted(
            r["run_id"] for r in api.runs(watcher)
        )


class TestRunDetail:
    def test_a_finished_run_returns_its_report(self, watcher: RunDirWatcher) -> None:
        body = api.run(watcher, "run-done")
        assert body["report"]["failure_mode"] == "crash_unhandled_exception"
        assert body["partial"] is False

    def test_a_mid_run_returns_a_partial(self, watcher: RunDirWatcher) -> None:
        body = api.run(watcher, "run-live")
        assert body["partial"] is True
        assert body["plan"]["seed"] == 1337
        assert body["counters"]["events"] == 1

    def test_an_unknown_run_is_none(self, watcher: RunDirWatcher) -> None:
        assert api.run(watcher, "run-nope") is None

    def test_events_page(self, watcher: RunDirWatcher) -> None:
        body = api.events(watcher, "run-done", from_seq=2, limit=10)
        assert [e["seq"] for e in body["events"]] == [2, 3]
        assert body["next_from"] == 4
        assert body["has_more"] is False


class TestArtifacts:
    @pytest.mark.parametrize("name", ["report.json", "plan.json", "trace.jsonl", "AGENT_TASK.md"])
    def test_allowed_artifacts_are_served(self, watcher: RunDirWatcher, name: str) -> None:
        body, content_type = api.artifact(watcher, "run-done", name)
        assert body, name
        assert content_type

    def test_a_missing_but_allowed_artifact_is_none(self, watcher: RunDirWatcher) -> None:
        assert api.artifact(watcher, "run-done", "judge.json") == (None, None)

    @pytest.mark.parametrize(
        "name",
        [
            "payloads/secret.json",
            "../report.json",
            "../../etc/passwd",
            "%2e%2e%2freport.json",
            "secret.json",
            ".env",
            "trace.jsonl/../report.json",
        ],
    )
    def test_anything_off_the_allow_list_is_refused(
        self, watcher: RunDirWatcher, name: str
    ) -> None:
        assert api.artifact(watcher, "run-done", name) == (None, None)

    def test_payloads_is_never_reachable(self, watcher: RunDirWatcher) -> None:
        """It holds raw payloads and is not on the list, in any spelling."""
        for spelling in ("payloads", "payloads/", "./payloads/secret.json", "PAYLOADS"):
            assert api.artifact(watcher, "run-done", spelling) == (None, None)


class TestRunIdSafety:
    @pytest.mark.parametrize(
        "run_id",
        [
            "../../etc/passwd",
            "%2e%2e%2f",
            "..",
            "run-done/../../..",
            "a" * 65,
            "",
            "run done",
            "run\x00done",
            "/absolute",
        ],
    )
    def test_a_bad_run_id_is_refused(self, watcher: RunDirWatcher, run_id: str) -> None:
        assert api.run(watcher, run_id) is None
        assert api.artifact(watcher, run_id, "report.json") == (None, None)
        assert api.events(watcher, run_id)["events"] == []

    def test_resolution_is_by_lookup_not_by_joining(
        self, watcher: RunDirWatcher, chaos: Path
    ) -> None:
        """A directory that exists but was never discovered is still not reachable."""
        sneaky = chaos / "s3" / "run-sneaky"
        sneaky.mkdir(parents=True)
        (sneaky / "report.json").write_text("{}")
        # Not polled, so not in the discovered map.
        assert api.run(watcher, "run-sneaky") is None

    def test_a_symlink_escape_reads_nothing(self, watcher: RunDirWatcher, chaos: Path) -> None:
        target = chaos.parent / "outside.txt"
        target.write_text("secret")
        link = chaos / "tool.drop_key" / "run-done" / "report.json.link"
        link.symlink_to(target)
        assert api.artifact(watcher, "run-done", "report.json.link") == (None, None)


class TestHealth:
    def test_it_reports_the_watcher_state(self, watcher: RunDirWatcher) -> None:
        body = api.health(watcher)
        assert body["version"]
        assert body["out_dir"]
        assert body["runs"] == 2
        assert body["parse_errors"] == 0
