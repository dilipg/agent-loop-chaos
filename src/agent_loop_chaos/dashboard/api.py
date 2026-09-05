"""The JSON endpoints, as pure functions over the watcher and the filesystem.

No server needed to call any of these, which is what makes the path-safety rules
testable directly rather than through HTTP.

Two rules are not negotiable (`docs/10` §5, §9):

- A `run_id` is matched against a pattern **and** resolved by lookup in the watcher's
  discovered map. It is never joined onto a path. A directory that exists but was
  never discovered is not reachable.
- An artifact name comes from a literal allow-list. `payloads/` -- the verbose-mode
  overflow directory, which holds raw payloads -- is not on it and never will be.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..version import __version__
from .watcher import RunDirWatcher

__all__ = [
    "ARTIFACTS",
    "artifact",
    "events",
    "glossary",
    "health",
    "run",
    "runs",
    "suite",
    "tasks",
]

#: The only names ever served, and the content type each gets.
ARTIFACTS: dict[str, str] = {
    "report.json": "application/json",
    "plan.json": "application/json",
    "judge.json": "application/json",
    "trace.jsonl": "application/x-ndjson",
    "baseline.diff": "text/plain; charset=utf-8",
    "AGENT_TASK.md": "text/markdown; charset=utf-8",
}

_RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _resolve(watcher: RunDirWatcher, run_id: str) -> Any:
    """Look a run up by id, refusing anything that is not one.

    Args:
        watcher: The watcher holding the discovered map.
        run_id: The requested id.

    Returns:
        The `RunState`, or `None` when the id is malformed or unknown. Both cases
        return `None` on purpose: telling them apart would confirm what exists.
    """
    if not _RUN_ID.match(run_id):
        return None
    return watcher.runs().get(run_id)


def _load(path: Path) -> Any:
    """Read a JSON file, tolerating one that is being written.

    Args:
        path: The file.

    Returns:
        The parsed document, or `None`.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def suite(watcher: RunDirWatcher) -> dict[str, Any]:
    """The suite summary.

    Args:
        watcher: The watcher.

    Returns:
        `suite.json` when present, else a summary synthesized from the run
        directories found -- pointing the dashboard at a directory that was produced
        by something other than `alc run` must still render.
    """
    document = _load(watcher.out_dir / "suite.json")
    if isinstance(document, dict):
        return {**document, "synthesized": False}

    states = watcher.runs().values()
    reports = [_load(s.path / "report.json") for s in states]
    finished = [r for r in reports if isinstance(r, dict)]
    return {
        "schema_version": "1.1",
        "synthesized": True,
        "status": "running" if len(finished) < len(list(states)) else "completed",
        "planned": sorted({s.scenario_id for s in states}),
        "current": None,
        "passed": sum(1 for r in finished if r.get("success")),
        "failed": sum(1 for r in finished if r.get("success") is False),
        "skipped": 0,
        "seed": next((r.get("seed", 0) for r in finished), 0),
        "results": [],
    }


def runs(watcher: RunDirWatcher) -> list[dict[str, Any]]:
    """One row per discovered run.

    Args:
        watcher: The watcher.

    Returns:
        Rows sorted by run id, so the list does not reshuffle between polls.
    """
    out: list[dict[str, Any]] = []
    for state in sorted(watcher.runs().values(), key=lambda s: s.run_id):
        report = _load(state.path / "report.json")
        report = report if isinstance(report, dict) else {}
        out.append(
            {
                "run_id": state.run_id,
                "scenario_id": report.get("scenario_id") or state.scenario_id,
                "title": report.get("scenario_title"),
                "description": report.get("scenario_description"),
                "status": state.status,
                "success": report.get("success"),
                "failure_mode": report.get("failure_mode"),
                "severity": report.get("severity"),
                "events": len([e for e in state.events if e.get("kind") != "truncated_head"]),
                "faults_fired": sum(1 for e in state.events if e.get("kind") == "fault_fired"),
                "started_at": report.get("started_at"),
                "duration_ms": report.get("duration_ms"),
                "artifacts": {name: (state.path / name).is_file() for name in ARTIFACTS},
            }
        )
    return out


def run(watcher: RunDirWatcher, run_id: str) -> dict[str, Any] | None:
    """One run's detail.

    Args:
        watcher: The watcher.
        run_id: Which run.

    Returns:
        The report when it has been written, else a partial built from the plan and
        what the trace shows so far. `None` for an unknown or malformed id.
    """
    state = _resolve(watcher, run_id)
    if state is None:
        return None
    report = _load(state.path / "report.json")
    if isinstance(report, dict):
        return {"partial": False, "report": report, "status": state.status}

    fired = [e for e in state.events if e.get("kind") == "fault_fired"]
    return {
        "partial": True,
        "status": state.status,
        "run_id": state.run_id,
        "scenario_id": state.scenario_id,
        "plan": _load(state.path / "plan.json"),
        "counters": {
            "events": len([e for e in state.events if e.get("kind") != "truncated_head"]),
            "faults_fired": len(fired),
            "tool_calls": sum(1 for e in state.events if e.get("kind") == "tool_call_requested"),
            "llm_calls": sum(1 for e in state.events if e.get("kind") == "llm_request"),
        },
        "faults": [
            {
                "fault_id": e.get("fault_id"),
                "name": e.get("name"),
                "note": (e.get("payload") or {}).get("note"),
            }
            for e in fired
        ],
    }


def events(
    watcher: RunDirWatcher,
    run_id: str,
    *,
    from_seq: int = 0,
    limit: int = 500,
    kinds: list[str] | None = None,
    layers: list[str] | None = None,
) -> dict[str, Any]:
    """A page of trace events.

    Args:
        watcher: The watcher.
        run_id: Which run.
        from_seq: The first `seq` wanted.
        limit: Maximum events.
        kinds: Restrict to these kinds.
        layers: Restrict to these layers.

    Returns:
        `{events, next_from, has_more}`. Empty for an unknown id.
    """
    state = _resolve(watcher, run_id)
    if state is None:
        return {"events": [], "next_from": from_seq, "has_more": False}
    page = watcher.page(run_id, from_seq=from_seq, limit=limit, kinds=kinds, layers=layers)
    return {"events": page.events, "next_from": page.next_from, "has_more": page.has_more}


def artifact(watcher: RunDirWatcher, run_id: str, name: str) -> tuple[bytes | None, str | None]:
    """One artifact's bytes.

    Args:
        watcher: The watcher.
        run_id: Which run.
        name: The artifact, which must be in `ARTIFACTS` exactly.

    Returns:
        `(body, content_type)`, or `(None, None)`. The name is checked against the
        allow-list *by equality* -- no normalisation, no joining, no `..` handling,
        because there is nothing to handle when only six literal strings are valid.
        A symlink inside the run directory is refused too: the file must be a real
        file whose resolved parent is still the run directory.
    """
    state = _resolve(watcher, run_id)
    if state is None or name not in ARTIFACTS:
        return None, None
    path = state.path / name
    try:
        if path.is_symlink() or not path.is_file():
            return None, None
        if path.resolve().parent != state.path.resolve():
            return None, None
        return path.read_bytes(), ARTIFACTS[name]
    except OSError:
        return None, None


def tasks(watcher: RunDirWatcher) -> tuple[str, str]:
    """Every failing run's work order, concatenated into one file.

    `AGENT_TASK.md` is the product, and a reader looking at twenty-one failures needs
    the whole set as one thing they can hand to a coding agent -- not twenty-one
    clicks. Passing runs contribute nothing: a work order is written for a failure, and
    a pass has nothing to hand over.

    Args:
        watcher: The watcher.

    Returns:
        `(markdown, content_type)`.
    """
    found: list[tuple[str, str]] = []
    for state in sorted(watcher.runs().values(), key=lambda s: (s.scenario_id, s.run_id)):
        body, _ = artifact(watcher, state.run_id, "AGENT_TASK.md")
        if body:
            found.append((f"{state.scenario_id}/{state.run_id}", body.decode("utf-8", "replace")))

    header = [
        f"# Chaos findings — {len(found)} finding{'' if len(found) == 1 else 's'}",
        "",
        "Generated by agent-loop-chaos. Each section below is one complete work order,",
        "exactly as written to `AGENT_TASK.md`. Every value in them is read from",
        "`report.json`; nothing is invented at render time.",
        "",
        "Work through them one at a time. Do not edit the scenarios, the `must_not`",
        "lists or the probes to make a run pass -- `alc run --rounds N` hashes all three",
        "between rounds and reports a pass that follows such an edit as a regression.",
    ]
    if not found:
        header.append("")
        header.append("Nothing failed, so there is nothing to do.")

    parts = ["\n".join(header)]
    for where, text in found:
        parts.append(f"<!-- {where} -->\n\n{text.rstrip()}")
    return "\n\n---\n\n".join(parts) + "\n", "text/markdown; charset=utf-8"


def glossary(watcher: RunDirWatcher) -> dict[str, Any]:
    """Plain English for every code a report can emit.

    Args:
        watcher: Unused; kept so every endpoint has one signature.

    Returns:
        The glossary bundle, which the report view renders instead of raw enum names.
    """
    from ..glossary import bundle

    return bundle()


def health(watcher: RunDirWatcher) -> dict[str, Any]:
    """Version and watcher counters.

    Args:
        watcher: The watcher.

    Returns:
        The health document.
    """
    return {"version": __version__, **watcher.stats()}
