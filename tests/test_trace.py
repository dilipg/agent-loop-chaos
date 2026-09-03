"""The trace recorder.

`trace.jsonl` is a first-class output, so every line must validate against
`trace_event.schema.json` — whose `kind` enum is the single source of truth (D-35).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loop_chaos.schema import validate_obj
from agent_loop_chaos.trace import (
    SCHEMA_VERSION,
    Event,
    EventKind,
    JsonlSink,
    MemorySink,
    TraceRecorder,
    truncate_payload,
)


def test_every_event_kind_validates() -> None:
    """The literal and the schema enum cannot drift apart."""
    from typing import get_args

    recorder = TraceRecorder("run-1", strict=True)
    for kind in get_args(EventKind):
        assert recorder.emit(Event(kind=kind, level="minimal")) is not None


def test_seq_is_monotonic_from_one() -> None:
    """Ids are counter-derived, so a trace is reproducible."""
    recorder = TraceRecorder("run-1", strict=True)
    seqs = [recorder.emit(Event(kind="log", level="minimal"))["seq"] for _ in range(5)]  # type: ignore[index]
    assert seqs == [1, 2, 3, 4, 5]


def test_seq_stays_unique_under_threads() -> None:
    """LangGraph parallel branches interleave; two events must never share a seq."""
    import threading

    recorder = TraceRecorder("run-1")
    results: list[int] = []
    lock = threading.Lock()

    def emit_many() -> None:
        for _ in range(50):
            event = recorder.emit(Event(kind="log", level="minimal"))
            if event:
                with lock:
                    results.append(int(event["seq"]))

    threads = [threading.Thread(target=emit_many) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == len(set(results)) == 200


@pytest.mark.parametrize(
    ("recorder_level", "event_level", "recorded"),
    [
        ("minimal", "minimal", True),
        ("minimal", "standard", False),
        ("minimal", "verbose", False),
        ("standard", "minimal", True),
        ("standard", "standard", True),
        ("standard", "verbose", False),
        ("verbose", "verbose", True),
    ],
)
def test_level_filtering(recorder_level: str, event_level: str, recorded: bool) -> None:
    """A recorder drops anything more verbose than itself."""
    recorder = TraceRecorder("run-1", level=recorder_level)  # type: ignore[arg-type]
    result = recorder.emit(Event(kind="log", level=event_level))  # type: ignore[arg-type]
    assert (result is not None) is recorded


def test_secrets_are_redacted_on_write() -> None:
    """Nothing reaches a sink unfiltered."""
    recorder = TraceRecorder("run-1", strict=True)
    event = recorder.emit(Event(kind="log", level="minimal", payload={"api_key": "hunter2"}))
    assert event is not None
    assert event["payload"]["api_key"] == "<redacted:api_key>"


def test_the_canary_survives_redaction() -> None:
    """The `secret_in_output` probe needs to see it (`docs/04` §4)."""
    canary = "ALC-CANARY-run-1"
    recorder = TraceRecorder("run-1", canary=canary, strict=True)
    event = recorder.emit(Event(kind="log", level="minimal", payload={"answer": canary}))
    assert event is not None
    assert event["payload"]["answer"] == canary


def test_truncation_stub_shape() -> None:
    """Large payloads are replaced by a stub, never silently cut (`docs/04` §5)."""
    stub = truncate_payload("x" * 20_000, level="standard", seq=19)
    assert stub["__truncated__"] is True
    assert stub["bytes"] > 8 * 1024
    assert stub["sha256"].startswith("sha256:")
    assert len(stub["head"]) == 512
    assert "path" not in stub


def test_verbose_truncation_spills_the_full_payload(tmp_path: Path) -> None:
    """In `verbose` the full value is written beside the trace, and `path` is set."""
    stub = truncate_payload("y" * 200_000, level="verbose", seq=19, run_dir=tmp_path)
    assert stub["path"] == "payloads/seq-0019.json"
    assert (tmp_path / stub["path"]).exists()


def test_a_payload_within_budget_is_untouched() -> None:
    """Truncation must not disturb a normal payload."""
    payload = {"a": 1, "b": "short"}
    assert truncate_payload(payload, level="standard", seq=1) == payload


def test_fault_payloads_get_double_the_budget() -> None:
    """`payload_before`/`payload_after` diffs are the point of the report (`docs/04` §5)."""
    value = "z" * 9000
    assert truncate_payload(value, level="standard", seq=1) != value
    assert truncate_payload(value, level="standard", seq=1, budget_multiplier=2) == value


def test_redaction_runs_before_hashing() -> None:
    """Otherwise a stub's hash would be computed over the secret (`docs/04` §4)."""
    secret = "sk-" + "A" * 20_000
    recorder = TraceRecorder("run-1")
    event = recorder.emit(Event(kind="log", level="minimal", payload=secret))
    assert event is not None
    assert "sk-AAAA" not in json.dumps(event)


def test_jsonl_round_trips_through_load(tmp_path: Path) -> None:
    """`load` is what lets `alc judge` re-judge a stored run."""
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder("run-1", sinks=[JsonlSink(path)], strict=True)
    for index in range(3):
        recorder.emit(Event(kind="log", level="minimal", payload={"i": index}))
    recorder.close()
    loaded = TraceRecorder.load(path)
    assert [event["payload"]["i"] for event in loaded] == [0, 1, 2]
    assert all(validate_obj(event, "trace") == [] for event in loaded)


def test_jsonl_flushes_per_event_so_a_reader_can_follow_live(tmp_path: Path) -> None:
    """The flush is the feature, not an implementation detail (`docs/10` §2)."""
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder("run-1", sinks=[JsonlSink(path)])
    recorder.emit(Event(kind="run_started", level="minimal"))
    assert len(path.read_text().splitlines()) == 1
    recorder.close()


def test_load_skips_blank_lines(tmp_path: Path) -> None:
    """A partially written file must not crash the post-run pipeline."""
    path = tmp_path / "trace.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert len(TraceRecorder.load(path)) == 2


def test_a_broken_sink_never_breaks_the_run() -> None:
    """The observer must not break the observed."""

    class Exploding:
        def write(self, event: dict) -> None:
            raise RuntimeError("boom")

        def close(self) -> None:
            raise RuntimeError("boom")

    memory = MemorySink()
    recorder = TraceRecorder("run-1", sinks=[Exploding(), memory])  # type: ignore[list-item]
    recorder.emit(Event(kind="log", level="minimal"))
    recorder.emit(Event(kind="log", level="minimal"))
    recorder.close()
    assert len(memory.events) == 2


def test_schema_version_matches_the_fixtures() -> None:
    """A silent version skew would desynchronize every stored trace."""
    assert SCHEMA_VERSION == "1.1"
