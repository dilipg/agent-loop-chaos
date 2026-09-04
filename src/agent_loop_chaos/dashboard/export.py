"""A single self-contained HTML file: the same page, with the data inlined.

This is the CI-artifact and share-a-finding path. No server, no network, no external
asset -- it opens from a file path on a machine that has never heard of this library.

Two rules it does not get to skip (`docs/10` §9, D-38):

- The blob goes through `redact()` again before it is written. The engine already
  redacts what it writes, but an exported file gets attached to tickets and pasted
  into chats, so a second pass is cheap insurance.
- `</` inside the blob is escaped as `<\\/`, because a payload containing
  `</script>` would otherwise end the script element and start rendering as markup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..redact import redact
from . import api
from .watcher import RunDirWatcher

__all__ = ["build_blob", "export_html"]

_STATIC = Path(__file__).parent / "static" / "index.html"
_MARKER = "<script>\n"


def build_blob(out_dir: str | Path, *, max_trace_events: int = 5_000) -> dict[str, Any]:
    """Collect everything the page needs from a finished output directory.

    Args:
        out_dir: The `.chaos` directory.
        max_trace_events: Events kept per run. A `LoopTrapFault` trace can be
            enormous and the export has to stay openable.

    Returns:
        The blob, already redacted.
    """
    watcher = RunDirWatcher(out_dir, max_events=max(1, max_trace_events))
    watcher.poll()

    runs: list[dict[str, Any]] = []
    for row in api.runs(watcher):
        run_id = row["run_id"]
        state = watcher.runs()[run_id]
        events = watcher.events(run_id)
        detail = api.run(watcher, run_id) or {}
        task, _ = api.artifact(watcher, run_id, "AGENT_TASK.md")
        runs.append(
            {
                "row": row,
                "report": detail.get("report"),
                "events": events,
                "truncated": state.dropped,
                "task": task.decode("utf-8", errors="replace") if task else "",
            }
        )
    blob = {"suite": api.suite(watcher), "runs": runs}
    redacted = redact(blob)
    return redacted if isinstance(redacted, dict) else blob


def export_html(out_dir: str | Path, *, max_trace_events: int = 5_000) -> str:
    """Render the whole output directory as one HTML document.

    Args:
        out_dir: The `.chaos` directory.
        max_trace_events: Events kept per run.

    Returns:
        The document.

    Raises:
        OSError: The packaged page is missing.
    """
    blob = build_blob(out_dir, max_trace_events=max_trace_events)
    # ensure_ascii keeps U+2028/U+2029 escaped, so only the `</` case is left.
    payload = json.dumps(blob, default=str).replace("</", "<\\/")
    note = "".join(
        f"<!-- trace truncated for {r['row']['run_id']}: {r['truncated']} earlier events "
        "dropped -->\n"
        for r in blob["runs"]
        if r.get("truncated")
    )
    page = _STATIC.read_text(encoding="utf-8")
    inline = f"{note}<script>\nwindow.__ALC__ = {payload};\n</script>\n"
    return page.replace(_MARKER, inline + _MARKER, 1)
