"""Discover run directories and tail their traces while they are being written.

This is the part that breaks when done casually. A line without a terminating
newline is **incomplete**, not corrupt: parsing it loses an event and desynchronises
everything after it. So the reader tracks a byte offset, keeps a trailing fragment in
a buffer, and advances the offset only past newline-terminated bytes (`docs/10` §4).

Importable and fully testable with no server running -- `server.py` is a thin
transport over this.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["EventPage", "RunDirWatcher", "RunState"]

#: One offset-index entry per this many events. Enough that `from=<seq>` seeks
#: instead of re-reading, cheap enough that the index never matters for memory.
_INDEX_EVERY = 500

#: Directory names that are never a run.
_SKIP = {"payloads"}


@dataclass(slots=True)
class EventPage:
    """One page of trace events.

    Attributes:
        events: The events, `seq` ascending.
        next_from: The `seq` to ask for next.
        has_more: Whether anything follows.
    """

    events: list[dict[str, Any]]
    next_from: int
    has_more: bool


@dataclass(slots=True)
class RunState:
    """What the watcher knows about one run directory.

    Attributes:
        run_id: The directory name.
        scenario_id: Its parent directory's name.
        path: The run directory.
        status: `running`, `completed`, `aborted` or `stale`.
        offset: Bytes of `trace.jsonl` already consumed, newline-terminated only.
        buffer: The trailing fragment, held until its newline arrives.
        events: Retained events, oldest first.
        index: Sparse `seq -> byte offset`, so a `from=` query seeks.
        last_seen: `time.monotonic()` of the last new event.
        dropped: How many events retention discarded.
    """

    run_id: str
    scenario_id: str
    path: Path
    status: str = "running"
    offset: int = 0
    buffer: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    index: dict[int, int] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.monotonic)
    dropped: int = 0

    @property
    def trace(self) -> Path:
        """Where this run's events are written.

        Returns:
            The `trace.jsonl` path, whether or not it exists yet.
        """
        return self.path / "trace.jsonl"


class RunDirWatcher:
    """Tails every run directory under an output directory.

    Polling rather than filesystem events: a poll is portable, and the cost is one
    `stat` per known run plus a shallow scan. `docs/10` §4 sets the cadence.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        max_events: int = 20_000,
        stale_after_s: float = 120.0,
    ) -> None:
        """Initialise.

        Args:
            out_dir: The `.chaos` directory to watch.
            max_events: Per-run retention. A `LoopTrapFault` scenario can produce a
                very long trace; the dashboard must degrade, not die.
            stale_after_s: Silence after which a run with no report is `stale`.
        """
        self.out_dir = Path(out_dir)
        self.max_events = max(1, int(max_events))
        self.stale_after_s = stale_after_s
        self._runs: dict[str, RunState] = {}
        self._frames: list[dict[str, Any]] = []
        self._parse_errors = 0
        self._bytes_read = 0

    # -- introspection ---------------------------------------------------------

    def runs(self) -> dict[str, RunState]:
        """Every run discovered so far.

        Returns:
            Run id to state.
        """
        return dict(self._runs)

    def events(self, run_id: str) -> list[dict[str, Any]]:
        """Retained events for one run.

        Args:
            run_id: Which run.

        Returns:
            The events, oldest first. Empty for an unknown run.
        """
        state = self._runs.get(run_id)
        return list(state.events) if state else []

    def stats(self) -> dict[str, Any]:
        """Counters for `/api/health`.

        Returns:
            Runs known, events retained, parse errors, bytes read.
        """
        return {
            "runs": len(self._runs),
            "events": sum(len(r.events) for r in self._runs.values()),
            "parse_errors": self._parse_errors,
            "bytes_read": self._bytes_read,
            "out_dir": str(self.out_dir),
        }

    def take_frames(self) -> list[dict[str, Any]]:
        """Drain the frames produced since the last call.

        Returns:
            `reset`, `run_started`, `run_finished` and `trace` frames, in order.
        """
        frames, self._frames = self._frames, []
        return frames

    # -- polling ---------------------------------------------------------------

    def poll(self) -> list[dict[str, Any]]:
        """Discover new runs and read whatever has been appended.

        Never raises: a run directory can be deleted or replaced underneath us, and
        the dashboard must not die because a file went away.

        Returns:
            The frames produced by this poll.
        """
        self._discover()
        for state in list(self._runs.values()):
            try:
                self._read(state)
                self._refresh_status(state)
            except OSError:  # pragma: no cover - the directory vanished mid-poll
                continue
        return self.take_frames()

    def _discover(self) -> None:
        """Scan `out_dir` two levels deep for run directories."""
        if not self.out_dir.is_dir():
            return
        try:
            scenarios = sorted(self.out_dir.iterdir())
        except OSError:  # pragma: no cover
            return
        for scenario in scenarios:
            if not scenario.is_dir() or scenario.name.startswith(".") or scenario.name in _SKIP:
                continue
            try:
                candidates = sorted(scenario.iterdir())
            except OSError:  # pragma: no cover
                continue
            for run in candidates:
                if not run.is_dir() or run.name.startswith(".") or run.name in _SKIP:
                    continue
                if run.name in self._runs:
                    continue
                self._runs[run.name] = RunState(
                    run_id=run.name, scenario_id=scenario.name, path=run
                )
                self._frames.append(
                    {"kind": "run_started", "run_id": run.name, "scenario_id": scenario.name}
                )

    def _read(self, state: RunState) -> None:
        """Consume newly appended bytes from one trace.

        Args:
            state: The run to read.
        """
        trace = state.trace
        if not trace.is_file():
            return
        size = trace.stat().st_size
        if size < state.offset:
            # The directory was replaced. Everything the client holds is stale.
            state.offset, state.buffer, state.events, state.index = 0, "", [], {}
            state.dropped = 0
            self._frames.append({"kind": "reset", "run_id": state.run_id})
            size = trace.stat().st_size
        if size == state.offset:
            return

        with trace.open("rb") as handle:
            handle.seek(state.offset)
            chunk = handle.read(size - state.offset)
        self._bytes_read += len(chunk)

        lines = chunk.decode("utf-8", errors="replace").split("\n")
        # The last element is whatever followed the final newline: a fragment, or "".
        # It is *not* an event yet -- parsing half a line loses it and desynchronises
        # everything after. Hold it, and advance the offset only past newline-
        # terminated bytes, so the next read picks the fragment up whole.
        state.buffer = lines.pop()

        # Each line's own byte offset is tracked as we go, because that is what the
        # seek index stores. Using the end-of-chunk offset instead would index every
        # line in a bulk read to the same place, and `from=<seq>` would then seek
        # past the events it was asked for.
        pos = state.offset
        for line in lines:
            if line.strip():
                self._append(state, line, pos)
            pos += len(line.encode("utf-8")) + 1
        state.offset = pos

    def _append(self, state: RunState, line: str, offset: int) -> None:
        """Parse one newline-terminated line and retain it.

        Args:
            state: The run.
            line: The line, without its newline.
            offset: Where this line starts in the file, for the seek index.
        """
        try:
            event = json.loads(line)
        except ValueError:
            # Newline-terminated and unparseable is corrupt, not incomplete. One bad
            # line must never kill the stream.
            self._parse_errors += 1
            event = {"kind": "__unparseable__", "raw": line[:2000]}
        if not isinstance(event, dict):
            self._parse_errors += 1
            event = {"kind": "__unparseable__", "raw": line[:2000]}

        seq = event.get("seq")
        if isinstance(seq, int) and seq % _INDEX_EVERY == 0:
            state.index[seq] = offset
        state.events.append(event)
        state.last_seen = time.monotonic()
        self._frames.append({"kind": "trace", "run_id": state.run_id, "event": event})

        if len(state.events) > self.max_events:
            overflow = len(state.events) - self.max_events
            del state.events[:overflow]
            state.dropped += overflow
            marker = {"kind": "truncated_head", "dropped": state.dropped}
            if state.events and state.events[0].get("kind") == "truncated_head":
                state.events[0] = marker
            else:
                state.events.insert(0, marker)

    def _refresh_status(self, state: RunState) -> None:
        """Decide whether a run is still going.

        Args:
            state: The run.
        """
        was = state.status
        state.status = "running"
        if (state.path / "report.json").is_file() or any(
            e.get("kind") == "run_finished" for e in reversed(state.events[-20:])
        ):
            state.status = "completed"
        elif time.monotonic() - state.last_seen >= self.stale_after_s:
            state.status = "stale"
        if state.status != was and state.status == "completed":
            self._frames.append({"kind": "run_finished", "run_id": state.run_id})

    # -- paging ----------------------------------------------------------------

    def page(
        self,
        run_id: str,
        *,
        from_seq: int = 0,
        limit: int = 500,
        kinds: list[str] | None = None,
        layers: list[str] | None = None,
    ) -> EventPage:
        """Return events from `from_seq` onward.

        Reads from the file when the request predates what is retained, seeking via
        the sparse index rather than re-reading from the start (`docs/10` §4).

        Args:
            run_id: Which run.
            from_seq: The first `seq` wanted.
            limit: Maximum events to return.
            kinds: Restrict to these event kinds.
            layers: Restrict to these layers.

        Returns:
            The page.
        """
        state = self._runs.get(run_id)
        if state is None:
            return EventPage([], from_seq, False)

        source = self._source_for(state, from_seq)
        wanted = [
            e
            for e in source
            if int(e.get("seq", 0)) >= from_seq
            and (not kinds or e.get("kind") in kinds)
            and (not layers or e.get("layer") in layers)
        ]
        chosen = wanted[: max(1, limit)]
        has_more = len(wanted) > len(chosen)
        next_from = int(chosen[-1].get("seq", from_seq)) + 1 if chosen else from_seq
        return EventPage(chosen, next_from, has_more)

    def _source_for(self, state: RunState, from_seq: int) -> list[dict[str, Any]]:
        """Pick where to read a page from.

        Args:
            state: The run.
            from_seq: The first `seq` wanted.

        Returns:
            Retained events when they cover the request, else a slice read from the
            file starting at the nearest indexed offset.
        """
        retained = [e for e in state.events if isinstance(e.get("seq"), int)]
        if retained and int(retained[0]["seq"]) <= from_seq:
            return state.events
        if not state.index:
            return state.events
        candidates = [s for s in sorted(state.index) if s <= from_seq]
        if not candidates:
            return state.events
        return self._read_from_offset(state, state.index[candidates[-1]])

    def _read_from_offset(self, state: RunState, offset: int) -> list[dict[str, Any]]:
        """Read events from a byte offset to EOF.

        Args:
            state: The run.
            offset: Where to seek to.

        Returns:
            The parsed events. Unparseable lines are skipped here rather than
            surfaced: this is a re-read of history the live path already reported.
        """
        out: list[dict[str, Any]] = []
        try:
            with state.trace.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
        except OSError:  # pragma: no cover
            return state.events
        self._bytes_read += len(chunk)
        for line in chunk.decode("utf-8", errors="replace").split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                out.append(event)
        return out
