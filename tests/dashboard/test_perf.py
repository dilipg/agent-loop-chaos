"""The numbers from `docs/10` §10.

A `LoopTrapFault` scenario really does produce a 20 000-event trace, and the whole
point of the offset index and the windowed row list is that such a run stays usable.
These are wall-clock assertions with generous headroom -- they catch an accidental
"read the whole file to answer a page query", not a 20% regression.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent_loop_chaos.dashboard import api
from agent_loop_chaos.dashboard.watcher import RunDirWatcher

EVENTS = 20_000


@pytest.fixture(scope="module")
def big(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("perf") / ".chaos"
    run = root / "state.loop_trap" / "run-big"
    run.mkdir(parents=True)
    with (run / "trace.jsonl").open("w", encoding="utf-8") as fh:
        for seq in range(1, EVENTS + 1):
            fh.write(
                json.dumps(
                    {
                        "seq": seq,
                        "kind": "tool_call_requested",
                        "layer": "tool",
                        "name": "search",
                        "t_rel_ms": seq * 3,
                        "payload": {"query": "a" * 40, "note": None},
                    }
                )
                + "\n"
            )
    (run / "report.json").write_text(json.dumps({"run_id": "run-big", "success": False}))
    return root


@pytest.mark.slow
def test_the_run_list_answers_fast(big: Path) -> None:
    watcher = RunDirWatcher(big, max_events=EVENTS)
    watcher.poll()
    start = time.perf_counter()
    rows = api.runs(watcher)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert rows[0]["events"] == EVENTS
    assert elapsed_ms < 200, f"/api/runs took {elapsed_ms:.0f}ms"


@pytest.mark.slow
def test_the_first_page_answers_fast(big: Path) -> None:
    watcher = RunDirWatcher(big, max_events=EVENTS)
    watcher.poll()
    start = time.perf_counter()
    page = api.events(watcher, "run-big", from_seq=0, limit=500)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(page["events"]) == 500
    assert elapsed_ms < 300, f"first events page took {elapsed_ms:.0f}ms"


@pytest.mark.slow
def test_a_deep_page_does_not_re_read_the_file(big: Path) -> None:
    watcher = RunDirWatcher(big, max_events=100)
    watcher.poll()
    before = watcher.stats()["bytes_read"]
    page = api.events(watcher, "run-big", from_seq=19_000, limit=100)
    read = watcher.stats()["bytes_read"] - before
    assert [e["seq"] for e in page["events"]][:1] == [19_000]
    # Retention dropped these events from memory, so the answer comes off disk -- by
    # seeking to the nearest indexed offset, not by re-reading 20 000 lines.
    size = (big / "state.loop_trap" / "run-big" / "trace.jsonl").stat().st_size
    assert read < size / 4, f"read {read} of {size} bytes to answer one page"
