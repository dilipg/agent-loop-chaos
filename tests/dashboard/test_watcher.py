"""`RunDirWatcher` — tailing a file that is still being written.

`docs/10` §4 spells out the rules because this is where a naive implementation
breaks: a line without a terminating newline is *incomplete*, not corrupt, and
parsing it loses an event and desynchronises everything after it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loop_chaos.dashboard.watcher import RunDirWatcher


def _event(seq: int, kind: str = "log", **kw: Any) -> str:
    return json.dumps(
        {
            "schema_version": "1.1",
            "run_id": "run-a",
            "seq": seq,
            "ts": "1970-01-01T00:00:00Z",
            "kind": kind,
            **kw,
        }
    )


def _run_dir(root: Path, scenario: str = "s1", run_id: str = "run-a") -> Path:
    path = root / scenario / run_id
    path.mkdir(parents=True)
    return path


class TestTailing:
    def test_it_reads_what_is_there(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        (run / "trace.jsonl").write_text(_event(1) + "\n" + _event(2) + "\n")
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        assert [e["seq"] for e in watcher.events("run-a")] == [1, 2]

    def test_appends_arrive_on_the_next_poll(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        trace = run / "trace.jsonl"
        trace.write_text(_event(1) + "\n")
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        with trace.open("a") as handle:
            handle.write(_event(2) + "\n")
        watcher.poll()
        assert [e["seq"] for e in watcher.events("run-a")] == [1, 2]

    def test_a_partial_last_line_is_buffered_not_parsed(self, tmp_path: Path) -> None:
        """The single most important rule: half a line is not an event yet."""
        run = _run_dir(tmp_path)
        trace = run / "trace.jsonl"
        full = _event(1)
        trace.write_text(full + "\n" + full[: len(full) // 2])
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        assert [e["seq"] for e in watcher.events("run-a")] == [1]
        assert watcher.stats()["parse_errors"] == 0, "a partial line is not corrupt"

    def test_the_buffered_line_parses_once_it_completes(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        trace = run / "trace.jsonl"
        second = _event(2)
        trace.write_text(_event(1) + "\n" + second[:20])
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        with trace.open("a") as handle:
            handle.write(second[20:] + "\n")
        watcher.poll()
        assert [e["seq"] for e in watcher.events("run-a")] == [1, 2]

    def test_a_corrupt_line_yields_a_marker_and_the_stream_continues(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        (run / "trace.jsonl").write_text(_event(1) + "\n{not json\n" + _event(3) + "\n")
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        events = watcher.events("run-a")
        assert [e.get("kind") for e in events] == ["log", "__unparseable__", "log"]
        assert events[1]["raw"].startswith("{not json")
        assert watcher.stats()["parse_errors"] == 1

    def test_a_shrunk_file_resets(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        trace = run / "trace.jsonl"
        trace.write_text(_event(1) + "\n" + _event(2) + "\n")
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        trace.write_text(_event(9) + "\n")  # directory replaced
        frames = watcher.poll()  # `poll` returns and drains its own frames
        assert [e["seq"] for e in watcher.events("run-a")] == [9]
        assert any(f["kind"] == "reset" for f in frames), (
            "a reset must be announced so the page clears"
        )

    def test_an_empty_file_is_fine(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        (run / "trace.jsonl").write_text("")
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        assert watcher.events("run-a") == []


class TestDiscovery:
    def test_a_run_started_mid_watch_is_found(self, tmp_path: Path) -> None:
        _run_dir(tmp_path, "s1", "run-a")
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        assert set(watcher.runs()) == {"run-a"}
        second = _run_dir(tmp_path, "s2", "run-b")
        (second / "trace.jsonl").write_text(_event(1) + "\n")
        watcher.poll()
        assert set(watcher.runs()) == {"run-a", "run-b"}

    def test_dotfiles_and_payloads_are_skipped(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden" / "run-x").mkdir(parents=True)
        (tmp_path / "s1" / "payloads").mkdir(parents=True)
        _run_dir(tmp_path, "s1", "run-a")
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        assert set(watcher.runs()) == {"run-a"}

    def test_it_records_the_scenario_id(self, tmp_path: Path) -> None:
        _run_dir(tmp_path, "tool.drop_key", "run-a")
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        assert watcher.runs()["run-a"].scenario_id == "tool.drop_key"

    def test_a_missing_out_dir_is_not_an_error(self, tmp_path: Path) -> None:
        watcher = RunDirWatcher(tmp_path / "nothing-here")
        watcher.poll()
        assert watcher.runs() == {}


class TestStatus:
    def test_a_run_with_no_report_is_running(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        (run / "trace.jsonl").write_text(_event(1) + "\n")
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        assert watcher.runs()["run-a"].status == "running"

    def test_a_report_marks_it_completed(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        (run / "trace.jsonl").write_text(_event(1) + "\n")
        (run / "report.json").write_text(json.dumps({"success": True}))
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        assert watcher.runs()["run-a"].status == "completed"

    def test_a_run_finished_event_marks_it_completed(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        (run / "trace.jsonl").write_text(_event(1) + "\n" + _event(2, "run_finished") + "\n")
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        assert watcher.runs()["run-a"].status == "completed"

    def test_silence_makes_it_stale(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        (run / "trace.jsonl").write_text(_event(1) + "\n")
        watcher = RunDirWatcher(tmp_path, stale_after_s=0.0)
        watcher.poll()
        assert watcher.runs()["run-a"].status == "stale"


class TestSeekingBySeq:
    """`from=<seq>` must not re-read the file (`docs/10` §4)."""

    @staticmethod
    def _big(tmp_path: Path, count: int = 4000) -> Path:
        run = _run_dir(tmp_path)
        trace = run / "trace.jsonl"
        with trace.open("w") as handle:
            for seq in range(1, count + 1):
                handle.write(_event(seq, payload={"filler": "x" * 200}) + "\n")
        return trace

    def test_it_returns_the_right_slice(self, tmp_path: Path) -> None:
        self._big(tmp_path)
        watcher = RunDirWatcher(tmp_path, max_events=1000)
        watcher.poll()
        page = watcher.page("run-a", from_seq=3500, limit=10)
        assert [e["seq"] for e in page.events] == list(range(3500, 3510))
        assert page.next_from == 3510
        assert page.has_more is True

    def test_the_last_page_says_so(self, tmp_path: Path) -> None:
        self._big(tmp_path, count=50)
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        page = watcher.page("run-a", from_seq=45, limit=100)
        assert [e["seq"] for e in page.events] == list(range(45, 51))
        assert page.has_more is False

    def test_it_seeks_rather_than_re_reading(self, tmp_path: Path) -> None:
        """Assert the bytes read, not the wall time: the point is the offset index."""
        trace = self._big(tmp_path)
        watcher = RunDirWatcher(tmp_path, max_events=500)
        watcher.poll()
        before = watcher.stats()["bytes_read"]
        watcher.page("run-a", from_seq=3900, limit=10)
        read = watcher.stats()["bytes_read"] - before
        assert read < trace.stat().st_size // 4, (
            f"read {read} bytes of a {trace.stat().st_size}-byte file to fetch 10 events"
        )

    def test_retention_drops_the_head_with_a_marker(self, tmp_path: Path) -> None:
        self._big(tmp_path, count=300)
        watcher = RunDirWatcher(tmp_path, max_events=100)
        watcher.poll()
        events = watcher.events("run-a")
        assert len(events) <= 101, "retention cap not applied"
        assert events[0]["kind"] == "truncated_head", "dropping events must be visible"


class TestFiltering:
    def test_kinds_narrow_the_page(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        (run / "trace.jsonl").write_text(
            _event(1, "log") + "\n" + _event(2, "fault_fired") + "\n" + _event(3, "log") + "\n"
        )
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        page = watcher.page("run-a", kinds=["fault_fired"])
        assert [e["seq"] for e in page.events] == [2]

    def test_layers_narrow_the_page(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        (run / "trace.jsonl").write_text(
            _event(1, "log", layer="tool") + "\n" + _event(2, "log", layer="llm") + "\n"
        )
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        assert [e["seq"] for e in watcher.page("run-a", layers=["llm"]).events] == [2]
