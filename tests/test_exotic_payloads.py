"""Payload shapes a real agent passes around, which a fixture-backed one never does.

The library serializes whatever the agent had in its hands. `default=str` covers a
*value* it cannot encode -- a LangChain message, a connection object, an `ObjectId`.
It does not cover a **key**, and `json.dumps` refuses a non-string key outright:

    TypeError: keys must be str, int, float, bool or None, not ObjectId

`sort_keys=True` compounds it: mixed key types cannot be compared either. A
Mongo-backed agent keys dicts by `ObjectId` as a matter of course --
`signals_by_id: dict[PyObjectId, Signal]` in the codebase this was found in -- so the
trace sink died mid-run on the first real agent it met (D-140).

The rule this file defends: **the library must never fail on the shape of a payload.**
A stringified key is a small loss; a dropped sink is a lost run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import ToolCorruptionFault


class Oid:
    """Stands in for `bson.ObjectId`: hashable, not a string, not JSON-encodable."""

    def __init__(self, raw: str) -> None:
        self.raw = raw

    def __hash__(self) -> int:
        return hash(self.raw)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Oid) and other.raw == self.raw

    def __repr__(self) -> str:
        return f"Oid({self.raw})"


class Unencodable:
    """A value with no JSON form and a deliberately hostile `__repr__`."""

    def __repr__(self) -> str:
        return "<a database connection>"


PAYLOADS: dict[str, Any] = {
    "objectid keys": {Oid("a1"): {"score": 3}, Oid("a2"): {"score": 4}},
    "mixed keys": {"ok": 1, 2: "two", Oid("a3"): "three"},
    "tuple keys": {("a", 1): "compound"},
    "unencodable value": {"conn": Unencodable(), "score": 3},
    "nested objectid keys": {"rows": [{Oid("b1"): {"n": 1}}]},
    "bytes value": {"blob": b"\\x00\\x01binary"},
    "set value": {"tags": {"a", "b"}},
    "objectid key with a secret": {Oid("c1"): {"authorization": "Bearer sk-live-abcdefghijkl"}},
}


def _run(payload: Any, tmp_path: Path) -> Any:
    engine = ChaosEngine(
        seed=1337, out_dir=tmp_path, judge="rules", trace_level="verbose", strict_schema=False
    )

    @engine.tool
    def fetch(key: str) -> Any:
        return payload

    @engine.intercept_tools()
    def agent(question: str) -> str:
        return f"got {type(fetch('k')).__name__}"

    engine.register_fault(ToolCorruptionFault(mutation_type="empty_json"), target_tool="fetch")
    return engine.run(agent, inputs="q")


@pytest.mark.parametrize("label", sorted(PAYLOADS))
class TestTheLibraryNeverFailsOnAShape:
    def test_the_run_completes(self, label: str, tmp_path: Path) -> None:
        result = _run(PAYLOADS[label], tmp_path)
        assert result.run_id, f"{label} killed the run"

    def test_the_trace_is_written_and_parseable(self, label: str, tmp_path: Path) -> None:
        """A dropped sink is a lost run -- the whole point of the trace."""
        _run(PAYLOADS[label], tmp_path)
        trace = next(tmp_path.rglob("trace.jsonl")).read_text(encoding="utf-8")
        lines = [line for line in trace.splitlines() if line.strip()]
        assert lines, f"{label}: the trace is empty, so the sink was dropped"
        for line in lines:
            json.loads(line)

    def test_no_internal_error_was_recorded(self, label: str, tmp_path: Path) -> None:
        _run(PAYLOADS[label], tmp_path)
        trace = next(tmp_path.rglob("trace.jsonl")).read_text(encoding="utf-8")
        kinds = [json.loads(line).get("kind") for line in trace.splitlines() if line.strip()]
        assert "internal_error" not in kinds, f"{label} produced an internal error"

    def test_the_report_serializes(self, label: str, tmp_path: Path) -> None:
        result = _run(PAYLOADS[label], tmp_path)
        assert json.loads(result.to_json())["run_id"] == result.run_id


def test_a_secret_under_an_exotic_key_is_still_redacted(tmp_path: Path) -> None:
    """Key coercion must not route around redaction on the way past."""
    result = _run(PAYLOADS["objectid key with a secret"], tmp_path)
    blob = result.to_json() + next(tmp_path.rglob("trace.jsonl")).read_text(encoding="utf-8")
    assert "sk-live-abcdefghijkl" not in blob


def test_the_coerced_key_is_recognisable(tmp_path: Path) -> None:
    """Stringified, not dropped: a reader has to be able to tell what it was."""
    _run(PAYLOADS["objectid keys"], tmp_path)
    trace = next(tmp_path.rglob("trace.jsonl")).read_text(encoding="utf-8")
    assert "Oid(a1)" in trace or "a1" in trace


def test_a_bare_exotic_dict_survives_canonical_json() -> None:
    """`plan_hash` runs through this, so it must not raise either."""
    from agent_loop_chaos.seeding import canonical_json

    assert canonical_json({Oid("a1"): 1, "b": 2})


def test_nothing_regressed_for_ordinary_payloads(tmp_path: Path) -> None:
    result = _run({"rows": [{"temp_c": 24}]}, tmp_path)
    trace = next(tmp_path.rglob("trace.jsonl")).read_text(encoding="utf-8")
    assert '"temp_c"' in trace
    assert result.tool_calls
