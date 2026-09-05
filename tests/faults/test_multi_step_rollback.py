"""`rollback_steps` — replaying more than the node that just committed.

A crash-and-resume in production rarely redoes exactly one step. The scheduler
restarts from the last durable checkpoint, and everything after it runs again. That is
what `rollback_steps: 3` is asking for, and until now the parameter was accepted and
ignored: a scenario asking for three got a one-node replay and no indication that its
request had been quietly reduced.

The replay stays at the `(node, post)` crossing where it is performable. The engine
keeps a bounded history of `(node, function, entry state)` and re-runs the last N of
them in order, exactly as a resume would. `(checkpoint, post)` is still a skip: the
checkpointer's `put` runs after the node returned, so replaying from there means
re-entering the graph, and the engine does not drive a user's graph (D-111).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import CheckpointRollbackFault
from agent_loop_chaos.targeting import Target, Trigger


def _engine(tmp_path: Path, *, bundle: bool = False) -> ChaosEngine:
    return ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=bundle)


def _trace(tmp_path: Path, result: Any) -> list[dict[str, Any]]:
    """Read the run's events back off disk.

    Args:
        tmp_path: The output directory.
        result: The run's result, for its id.

    Returns:
        The trace events.
    """
    import json

    path = next(tmp_path.rglob("trace.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pipeline(engine: ChaosEngine, log: list[str]) -> Any:
    """Three nodes that each record that they ran."""

    def make(name: str) -> Any:
        def node(state: dict[str, Any]) -> dict[str, Any]:
            log.append(name)
            return {**state, name: state.get(name, 0) + 1}

        return node

    nodes = {name: make(name) for name in ("alpha", "beta", "gamma")}

    def agent(inputs: Any) -> Any:
        state: dict[str, Any] = {"inputs": inputs}
        for name, fn in nodes.items():
            state = engine.route_node(fn, name=name, state=state)
        return state

    return agent


class TestRollbackSteps:
    def test_one_step_replays_only_the_committed_node(self, tmp_path: Path) -> None:
        log: list[str] = []
        engine = _engine(tmp_path)
        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=1, times=1),
            target=Target(layer="node", node="gamma", phase="post"),
            trigger=Trigger(on_call=1),
        )
        engine.run(_pipeline(engine, log), inputs="x")
        assert log == ["alpha", "beta", "gamma", "gamma"]

    def test_two_steps_replay_the_node_before_it_too(self, tmp_path: Path) -> None:
        """The resume redoes everything after the checkpoint, in order."""
        log: list[str] = []
        engine = _engine(tmp_path)
        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=2, times=1),
            target=Target(layer="node", node="gamma", phase="post"),
            trigger=Trigger(on_call=1),
        )
        engine.run(_pipeline(engine, log), inputs="x")
        assert log == ["alpha", "beta", "gamma", "beta", "gamma"]

    def test_three_steps_replay_all_of_them(self, tmp_path: Path) -> None:
        log: list[str] = []
        engine = _engine(tmp_path)
        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=3, times=1),
            target=Target(layer="node", node="gamma", phase="post"),
            trigger=Trigger(on_call=1),
        )
        engine.run(_pipeline(engine, log), inputs="x")
        assert log == ["alpha", "beta", "gamma", "alpha", "beta", "gamma"]

    def test_asking_for_more_than_ran_replays_what_there_is(self, tmp_path: Path) -> None:
        """A rollback cannot go behind the start of the run, and must not raise."""
        log: list[str] = []
        engine = _engine(tmp_path)
        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=99, times=1),
            target=Target(layer="node", node="gamma", phase="post"),
            trigger=Trigger(on_call=1),
        )
        result = engine.run(_pipeline(engine, log), inputs="x")
        assert log == ["alpha", "beta", "gamma", "alpha", "beta", "gamma"]
        assert result.error is None

    def test_times_multiplies_the_whole_window(self, tmp_path: Path) -> None:
        log: list[str] = []
        engine = _engine(tmp_path)
        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=2, times=2),
            target=Target(layer="node", node="gamma", phase="post"),
            trigger=Trigger(on_call=1),
        )
        engine.run(_pipeline(engine, log), inputs="x")
        assert log == ["alpha", "beta", "gamma", "beta", "gamma", "beta", "gamma"]


class TestItIsRecorded:
    def test_the_window_is_in_the_restore_events(self, tmp_path: Path) -> None:
        log: list[str] = []
        engine = _engine(tmp_path, bundle=True)
        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=2, times=1),
            target=Target(layer="node", node="gamma", phase="post"),
            trigger=Trigger(on_call=1),
        )
        result = engine.run(_pipeline(engine, log), inputs="x")
        restored = [e for e in _trace(tmp_path, result) if e.get("kind") == "checkpoint_restored"]
        assert restored, "no checkpoint_restored event"
        assert [e["payload"]["node"] for e in restored] == ["beta", "gamma"]
        assert all(e["payload"]["rollback_steps"] == 2 for e in restored)

    def test_the_note_says_how_far_it_rolled_back(self, tmp_path: Path) -> None:
        log: list[str] = []
        engine = _engine(tmp_path)
        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=2, times=1),
            target=Target(layer="node", node="gamma", phase="post"),
            trigger=Trigger(on_call=1),
        )
        result = engine.run(_pipeline(engine, log), inputs="x")
        fired = [f for f in result.injected_faults if f["fired"]]
        assert fired and "2" in fired[0]["fires"][0]["note"]


class TestTheHarnessStillOwnsTheReplay:
    def test_every_replayed_call_is_attributed_to_the_harness(self, tmp_path: Path) -> None:
        """R1: a probe never fires on what the engine itself caused.

        Without this a multi-step rollback would make `duplicate_side_effect` fire on
        a correct agent, since the harness called the node three times.
        """
        engine = _engine(tmp_path)
        log: list[str] = []

        @engine.tool(side_effecting=True)
        def book(ref: str) -> dict[str, Any]:
            log.append(ref)
            return {"held": ref, "id": f"h{len(log)}"}

        def make(name: str) -> Any:
            def node(state: dict[str, Any]) -> dict[str, Any]:
                # A reference per node: the agent never repeats itself, so any
                # duplicate the probe sees would be one the harness caused.
                book(f"R-{name}")
                return state

            return node

        def agent(inputs: Any) -> Any:
            state: dict[str, Any] = {}
            for name in ("alpha", "beta"):
                state = engine.route_node(make(name), name=name, state=state)
            return "done"

        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=2, times=1),
            target=Target(layer="node", node="beta", phase="post"),
            trigger=Trigger(on_call=1),
        )
        result = engine.run(
            engine.intercept_tools()(agent), inputs="x", allow_side_effects=["book"]
        )
        assert log == ["R-alpha", "R-beta", "R-alpha", "R-beta"], log
        assert "duplicate_side_effect" not in {s["code"] for s in result.symptoms}


def test_the_checkpoint_crossing_is_still_an_honest_skip() -> None:
    """D-109/D-111: replaying from `put` means re-entering the graph, which the
    engine does not do. Asking there records a skip naming the action."""
    from agent_loop_chaos.engine import _action_is_performable

    assert _action_is_performable("resume_from_checkpoint", "node") is True
    assert _action_is_performable("resume_from_checkpoint", "checkpoint") is False
