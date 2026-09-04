"""`resume_from_checkpoint` — replaying a committed node.

D-106 wired the checkpoint *crossing*; the resume itself stayed unimplemented, so the
fault recorded an honest skip and the demo had to cover the same observable with
`DuplicateSideEffectFault`.

The replay happens at the **node** boundary, not inside the checkpointer's `put`.
LangGraph calls `put` after the node has already returned, so a same-node replay
cannot be driven from there without re-entering the graph. `route_node` still holds
the node's function and the state it entered with — which is exactly a checkpoint —
so restoring that state and calling the function again *is* the rollback, and it
re-runs the side effects, which is the whole point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import CheckpointRollbackFault
from agent_loop_chaos.targeting import Target, Trigger


def _agent(engine: ChaosEngine, calls: list[str], *, idempotent: bool) -> Any:
    """A two-node pipeline whose second node books a seat."""
    holds: dict[str, str] = {}

    @engine.tool(name="hold_booking", side_effecting=True)
    def hold_booking(flight: str, key: str | None = None) -> dict[str, Any]:
        calls.append(flight)
        token = key or f"{flight}:{len(holds) + 1}"
        first = token not in holds
        if first:
            holds[token] = f"H-{len(holds) + 1:03d}"
        return {"hold_id": holds[token], "duplicate": not first}

    def book(state: dict[str, Any]) -> dict[str, Any]:
        result = hold_booking("AI-143", key="trip-1") if idempotent else hold_booking("AI-143")
        return {**state, "hold": result["hold_id"]}

    def run(question: Any = None) -> str:
        state = engine.route_node(book, name="book", state={"query": "book it"})
        return f"Held {state['hold']}."

    return run


def _run(tmp_path: Path, *, idempotent: bool, times: int = 1) -> tuple[Any, list[str]]:
    calls: list[str] = []
    engine = ChaosEngine(
        seed=1337,
        out_dir=tmp_path,
        write_bundle=False,
        strict_schema=False,
        judge="rules",
        allow_side_effects=["hold_booking"],
    )
    engine.register_fault(
        CheckpointRollbackFault(rollback_steps=1, times=times),
        target=Target(layer="node", node="book", phase="post"),
        trigger=Trigger(on_call=1),
    )
    result = engine.run(
        _agent(engine, calls, idempotent=idempotent),
        inputs={"question": "?"},
        scenario_id="resume",
        expected_behavior="graceful_degradation",
        allow_side_effects=["hold_booking"],
    )
    return result, calls


class TestTheNodeIsReplayed:
    def test_the_fault_fires(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path, idempotent=False)
        record = result.injected_faults[0]
        assert record["fired"] is True, f"still skipped: {record.get('skipped_reason')!r}"

    def test_the_node_body_runs_again(self, tmp_path: Path) -> None:
        _result, calls = _run(tmp_path, idempotent=False)
        assert len(calls) == 2, f"the committed node was not replayed: {calls}"

    def test_times_controls_how_many_replays(self, tmp_path: Path) -> None:
        _result, calls = _run(tmp_path, idempotent=False, times=2)
        assert len(calls) == 3, f"expected one original plus two replays: {calls}"

    def test_the_replay_is_attributed_to_the_harness(self, tmp_path: Path) -> None:
        """R1: the replayed call is ours, so `duplicate_side_effect` must not fire."""
        result, _ = _run(tmp_path, idempotent=False)
        assert "duplicate_side_effect" not in [s["code"] for s in result.symptoms]

    def test_it_emits_a_checkpoint_restored_event(self, tmp_path: Path) -> None:
        engine = ChaosEngine(
            seed=1337,
            out_dir=tmp_path,
            strict_schema=False,
            judge="rules",
            allow_side_effects=["hold_booking"],
        )
        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=1, times=1),
            target=Target(layer="node", node="book", phase="post"),
            trigger=Trigger(on_call=1),
        )
        result = engine.run(
            _agent(engine, [], idempotent=False),
            inputs={"question": "?"},
            scenario_id="resume-evt",
            allow_side_effects=["hold_booking"],
        )
        events = [
            json.loads(line) for line in Path(result.artifacts["trace"]).read_text().splitlines()
        ]
        assert any(e["kind"] == "checkpoint_restored" for e in events)


class TestTheFindingIsDetectable:
    """The replay is the harness's; the *two effects* are the agent's design."""

    def test_an_agent_without_an_idempotency_key_is_caught(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path, idempotent=False)
        failed = [a["check"] for a in result.assertions if not a["ok"]]
        assert "idempotent_effects" in failed, f"failures were {failed}"

    def test_it_classifies_as_a_duplicate_side_effect(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path, idempotent=False)
        assert result.failure_mode == "duplicate_side_effect"

    def test_an_agent_with_a_key_survives(self, tmp_path: Path) -> None:
        result, calls = _run(tmp_path, idempotent=True)
        assert len(calls) == 2, "the node must still be replayed"
        failed = [(a["check"], a["detail"]) for a in result.assertions if not a["ok"]]
        assert failed == [], f"the negative control failed: {failed}"
        assert result.success is True


class TestItStaysContained:
    def test_a_replay_that_raises_does_not_escape_as_a_library_bug(self, tmp_path: Path) -> None:
        engine = ChaosEngine(
            seed=1337, out_dir=tmp_path, write_bundle=False, strict_schema=False, judge="rules"
        )
        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=1, times=1),
            target=Target(layer="node", node="flaky", phase="post"),
            trigger=Trigger(on_call=1),
        )
        seen: list[int] = []

        def flaky(state: dict[str, Any]) -> dict[str, Any]:
            seen.append(1)
            if len(seen) > 1:
                raise RuntimeError("the replay found a bug the first pass did not")
            return state

        def run(question: Any = None) -> str:
            engine.route_node(flaky, name="flaky", state={})
            return "done"

        result = engine.run(run, inputs={"question": "?"}, scenario_id="flaky")
        assert (result.error or {}).get("raised_in") != "harness", result.error

    def test_the_checkpoint_layer_still_accepts_the_fault(self) -> None:
        """The `(checkpoint, post)` pair stays: it is where a real resume belongs."""
        assert ("checkpoint", "post") in CheckpointRollbackFault.accepts
        assert ("node", "post") in CheckpointRollbackFault.accepts
