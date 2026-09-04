"""The trace: one JSONL line per event, schema-valid, redacted, flushed as it goes.

`trace.jsonl` is a first-class output, not a debug log. Every line validates against
`trace_event.schema.json`, whose `kind` enum is the single source of truth for event
kinds (D-35). The sink flushes per event so a reader — `alc dashboard`, or a human
with `tail -f` — can follow a run live (`docs/01` §5 step 2).

Redaction runs before truncation and before hashing, so neither a stub's hash nor its
head can carry a secret (`docs/04-SCHEMAS.md` §4).
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

from .redact import redact
from .seeding import sanitize_floats, sha256_of

__all__ = [
    "SCHEMA_VERSION",
    "Event",
    "EventKind",
    "JsonlSink",
    "MemorySink",
    "Sink",
    "TraceLevel",
    "TraceRecorder",
    "truncate_payload",
]

log = logging.getLogger("agent_loop_chaos")

SCHEMA_VERSION = "1.1"

EventKind = Literal[
    "run_started",
    "run_finished",
    "internal_error",
    "step_started",
    "step_finished",
    "tool_call_requested",
    "tool_call_returned",
    "tool_call_failed",
    "llm_request",
    "llm_response",
    "node_entered",
    "node_exited",
    "edge_taken",
    "state_snapshot",
    "state_mutated",
    "checkpoint_written",
    "checkpoint_restored",
    "fault_armed",
    "fault_fired",
    "fault_skipped",
    "mutation_applied",
    "probe_fired",
    "metric",
    "judge_request",
    "judge_response",
    "judge_verdict",
    "assertion_result",
    "limit_exceeded",
    "log",
]

TraceLevel = Literal["minimal", "standard", "verbose"]

_LEVEL_ORDER: dict[str, int] = {"minimal": 0, "standard": 1, "verbose": 2}

# Per-payload budgets, in bytes, by recorder level (`docs/04-SCHEMAS.md` §5).
_BUDGET: dict[str, int] = {"minimal": 2 * 1024, "standard": 8 * 1024, "verbose": 64 * 1024}

_HEAD_CHARS = 512


def _utc_now() -> str:
    """Current UTC time, ISO-8601.

    A timing field only. No decision logic may read it (D-47).

    Returns:
        An ISO-8601 timestamp with a ``Z`` suffix.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class Event:
    """One trace event.

    Field names and types mirror `trace_event.schema.json`, which sets
    ``additionalProperties: false`` — so anything extra belongs in `tags` or
    `payload`, never as a new top-level key.
    """

    kind: EventKind
    level: TraceLevel = "standard"
    step: int | None = None
    layer: str | None = None
    phase: str | None = None
    name: str | None = None
    call_index: int | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    fault_id: str | None = None
    duration_ms: float | None = None
    payload: Any = None
    tags: dict[str, Any] = field(default_factory=dict)


def truncate_payload(
    payload: Any,
    *,
    level: TraceLevel,
    seq: int,
    run_dir: Path | None = None,
    budget_multiplier: int = 1,
) -> Any:
    """Replace an oversized payload with a stub, never a silent cut.

    Args:
        payload: The already-redacted payload.
        level: The recorder's level, which sets the budget.
        seq: The event's sequence number, used for the overflow filename.
        run_dir: Where to spill the full value in `verbose` mode.
        budget_multiplier: `payload_before`/`payload_after` in `injected_faults` get
            double the budget, because those diffs are the point of the report.

    Returns:
        The payload unchanged when it fits, otherwise the truncation stub described
        in `docs/04-SCHEMAS.md` §5.
    """
    if payload is None:
        return None
    try:
        encoded = json.dumps(sanitize_floats(payload), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        encoded = repr(payload)

    size = len(encoded.encode("utf-8"))
    budget = _BUDGET[level] * budget_multiplier
    if size <= budget:
        return payload

    stub: dict[str, Any] = {
        "__truncated__": True,
        "bytes": size,
        "sha256": sha256_of(encoded),
        "head": encoded[:_HEAD_CHARS],
    }
    if level == "verbose" and run_dir is not None:
        spill_dir = run_dir / "payloads"
        spill_dir.mkdir(parents=True, exist_ok=True)
        relative = f"payloads/seq-{seq:04d}.json"
        (run_dir / relative).write_text(encoded, encoding="utf-8")
        stub["path"] = relative
    return stub


class Sink(Protocol):
    """Where serialized events go."""

    def write(self, event: dict[str, Any]) -> None:
        """Accept one event.

        Args:
            event: The serialized event.
        """
        ...

    def close(self) -> None:
        """Release any resource the sink holds."""
        ...


class MemorySink:
    """Collects events in a list. The default for tests."""

    def __init__(self) -> None:
        """Initialise an empty sink."""
        self.events: list[dict[str, Any]] = []

    def write(self, event: dict[str, Any]) -> None:
        """Append an event.

        Args:
            event: The serialized event.
        """
        self.events.append(event)

    def close(self) -> None:
        """No-op; nothing is held open."""


class JsonlSink:
    """Appends one JSON object per line, flushing after each.

    The flush is not incidental: it is what lets a reader follow a run live.
    """

    def __init__(self, path: str | Path) -> None:
        """Open the sink, creating parent directories.

        Args:
            path: Where to write `trace.jsonl`.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        """Write and flush one event.

        Args:
            event: The serialized event.
        """
        # `default=str` because a payload can hold anything the agent passed around --
        # a LangChain message, a dataclass, a connection object. A trace that raised
        # on one of those would drop its sink mid-run and leave the file open, which
        # is a worse failure than a stringified value.
        self._handle.write(
            json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)
            + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying file, if still open."""
        if not self._handle.closed:
            self._handle.close()


class TraceRecorder:
    """Assigns sequence numbers, redacts, truncates, and fans out to sinks.

    `seq` is globally monotonic per run under a lock, so a LangGraph parallel branch
    cannot produce two events with the same number.
    """

    def __init__(
        self,
        run_id: str,
        *,
        level: TraceLevel = "standard",
        sinks: Sequence[Sink] | None = None,
        run_dir: Path | None = None,
        redact_keys: Sequence[str] = (),
        canary: str = "",
        strict: bool = False,
    ) -> None:
        """Initialise a recorder.

        Args:
            run_id: Stamped on every event.
            level: Events above this verbosity are dropped.
            sinks: Where to write. Defaults to a single `MemorySink`.
            run_dir: The run directory, for truncation overflow files.
            redact_keys: Extra key patterns from `ChaosEngine(redact_keys=…)`.
            canary: The run's canary, exempt from redaction so the
                `secret_in_output` probe can see it.
            strict: Validate every event against the schema as it is written. Used
                throughout the test suite; off by default so a production run is
                never slowed by it.
        """
        self.run_id = run_id
        self.level: TraceLevel = level
        self.sinks: list[Sink] = list(sinks) if sinks is not None else [MemorySink()]
        self.run_dir = run_dir
        self.redact_keys = tuple(redact_keys)
        self.canary = canary
        self.strict = strict
        self.schema_errors: list[str] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._t0 = perf_counter()

    @property
    def memory(self) -> MemorySink | None:
        """The first `MemorySink` among the sinks, if any.

        Returns:
            The in-memory sink, for tests to read back.
        """
        for sink in self.sinks:
            if isinstance(sink, MemorySink):
                return sink
        return None

    def _wants(self, level: TraceLevel) -> bool:
        """Report whether an event at `level` should be recorded.

        Args:
            level: The event's level.

        Returns:
            True when the recorder is at least as verbose as the event.
        """
        return _LEVEL_ORDER[level] <= _LEVEL_ORDER[self.level]

    def emit(self, event: Event) -> dict[str, Any] | None:
        """Record one event.

        Never raises: a trace failure must not break the run it is observing. A sink
        that fails is dropped from the fan-out.

        Args:
            event: The event to record.

        Returns:
            The serialized event, or `None` when filtered out by level.
        """
        if not self._wants(event.level):
            return None

        with self._lock:
            self._seq += 1
            seq = self._seq

        payload = redact(event.payload, self.redact_keys, allow=(self.canary,))
        multiplier = 2 if event.kind in {"fault_fired", "mutation_applied"} else 1
        payload = truncate_payload(
            payload,
            level=self.level,
            seq=seq,
            run_dir=self.run_dir,
            budget_multiplier=multiplier,
        )

        serialized: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "seq": seq,
            "ts": _utc_now(),
            "ts_mono_ms": round((perf_counter() - self._t0) * 1000, 3),
            "kind": event.kind,
            "level": event.level,
        }
        optional: dict[str, Any] = {
            "step": event.step,
            "layer": event.layer,
            "phase": event.phase,
            "name": event.name,
            "call_index": event.call_index,
            "span_id": event.span_id,
            "parent_span_id": event.parent_span_id,
            "fault_id": event.fault_id,
            "duration_ms": event.duration_ms,
        }
        serialized.update({k: v for k, v in optional.items() if v is not None})
        if payload is not None:
            serialized["payload"] = sanitize_floats(payload)
        if event.tags:
            serialized["tags"] = redact(event.tags, self.redact_keys, allow=(self.canary,))

        if self.strict:
            # Local import keeps jsonschema out of the package's import graph.
            from .schema import validate_obj

            errors = validate_obj(serialized, "trace")
            if errors:
                self.schema_errors.extend(f"seq {seq}: {e}" for e in errors)
                raise AssertionError(
                    f"trace event {seq} ({event.kind}) is not schema-valid: {errors}"
                )

        for sink in list(self.sinks):
            try:
                sink.write(serialized)
            except Exception:  # a broken sink must not break the run, but it must
                # still be closed: dropping it while its file handle is open leaks it.
                log.exception("trace sink %s failed; dropping it", type(sink).__name__)
                with contextlib.suppress(Exception):
                    sink.close()
                self.sinks.remove(sink)
        return serialized

    def close(self) -> None:
        """Close every sink."""
        for sink in self.sinks:
            # Closing must never raise into the run it was observing.
            with contextlib.suppress(Exception):
                sink.close()

    @staticmethod
    def load(path: str | Path) -> list[dict[str, Any]]:
        """Read a `trace.jsonl` back.

        The post-run pipeline is pure over `(trace, plan)`, so `alc judge` and
        `alc report` can re-judge an old run without re-running the agent
        (`docs/01` §5).

        Args:
            path: Path to a `trace.jsonl`.

        Returns:
            The events in file order. Blank lines are skipped.

        Raises:
            json.JSONDecodeError: If a non-blank line is not valid JSON.
        """
        events: list[dict[str, Any]] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate the in-memory events, when a `MemorySink` is attached.

        Returns:
            An iterator over recorded events.
        """
        sink = self.memory
        return iter(sink.events if sink is not None else [])
