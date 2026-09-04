"""Checkpoint crossings under LangGraph.

`instrument_graph(..., intercept_checkpoints=True)` was accepted as a parameter and
never used: the adapter created no checkpoint crossing, so `CheckpointRollbackFault`
could not fire and `resume.checkpoint_rollback` had to be removed from the demo suite
(D-82). A documented option that does nothing is the same failure the library exists
to find.

The crossing goes where the checkpointer writes, because that is the only place the
adapter can see a node commit -- and a rollback has to happen *after* a commit or
there is nothing to roll back to.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph")

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import CheckpointRollbackFault, NoopFault
from agent_loop_chaos.targeting import Target, Trigger


def _graph(engine: ChaosEngine, *, intercept: bool = True) -> Any:
    from langgraph.checkpoint.memory import MemorySaver

    from agent_loop_chaos.adapters.langgraph import instrument_graph
    from tests.fakes.lg_agent import build

    compiled = build(lambda prompt: "warm").compile(checkpointer=MemorySaver())
    return instrument_graph(compiled, engine, intercept_checkpoints=intercept)


class TestCheckpointCrossingsExist:
    def test_a_checkpoint_crossing_is_recorded(self, tmp_path: Path) -> None:
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        result = engine.run(
            _graph(engine),
            inputs={"query": "x"},
            scenario_id="cp",
            expected_behavior="ignore_and_continue",
        )
        events = [
            json.loads(line) for line in Path(result.artifacts["trace"]).read_text().splitlines()
        ]
        assert any(e.get("layer") == "checkpoint" for e in events), (
            "intercept_checkpoints=True created no checkpoint crossing"
        )

    def test_none_are_recorded_when_it_is_off(self, tmp_path: Path) -> None:
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        result = engine.run(
            _graph(engine, intercept=False),
            inputs={"query": "x"},
            scenario_id="cp-off",
            expected_behavior="ignore_and_continue",
        )
        events = [
            json.loads(line) for line in Path(result.artifacts["trace"]).read_text().splitlines()
        ]
        assert not any(e.get("layer") == "checkpoint" for e in events)

    def test_a_fault_can_target_the_checkpoint_layer(self, tmp_path: Path) -> None:
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        engine.register_fault(NoopFault(), target=Target(layer="checkpoint", phase="post"))
        result = engine.run(
            _graph(engine),
            inputs={"query": "x"},
            scenario_id="cp-target",
            expected_behavior="ignore_and_continue",
        )
        assert any(f["fired"] for f in result.injected_faults), (
            "a checkpoint-layer fault still cannot reach a crossing"
        )

    def test_the_graph_still_produces_its_answer(self, tmp_path: Path) -> None:
        """Instrumenting the checkpointer must not break the run it observes."""
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        result = engine.run(
            _graph(engine),
            inputs={"query": "x"},
            scenario_id="cp-clean",
            expected_behavior="ignore_and_continue",
        )
        assert result.error is None, result.error
        assert result.success


class TestTheRollbackFaultReachesIt:
    """`CheckpointRollbackFault` is what the layer exists for (D-82)."""

    def test_it_is_registrable_against_the_checkpoint_layer(self, tmp_path: Path) -> None:
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        # No `allow_side_effects` needed: nothing here is declared side-effecting, so
        # the D-23 gate has nothing to refuse.
        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=1, times=1),
            trigger=Trigger(on_call=1),
        )
        result = engine.run(
            _graph(engine),
            inputs={"query": "x"},
            scenario_id="rollback",
            expected_behavior="graceful_degradation",
        )
        record = result.injected_faults[0]
        assert record["fired"] or record.get("skipped_reason"), (
            "the fault neither fired nor recorded why not -- it never saw a crossing"
        )

    def test_it_reaches_the_crossing_and_says_what_it_could_not_do(self, tmp_path: Path) -> None:
        """The crossing exists; the resume does not, and the record says so.

        Before D-106 the record was `fired=False` with **no reason at all** -- the
        fault never saw a crossing. Now it sees one, asks for
        `resume_from_checkpoint`, and the engine records that no adapter implements
        that action (D-109). Two different silences, and only the first was a bug: a
        documented option doing nothing, versus one honestly reporting an
        unimplemented action.
        """
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        # The checkpoint layer specifically: since D-111 the fault also accepts
        # `(node, post)`, where the replay *is* performable.
        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=1, times=1),
            target=Target(layer="checkpoint", phase="post"),
            trigger=Trigger(on_call=1),
        )
        result = engine.run(
            _graph(engine),
            inputs={"query": "x"},
            scenario_id="rollback2",
            expected_behavior="graceful_degradation",
        )
        record = result.injected_faults[0]
        assert record["fired"] is False, "nothing was replayed, so nothing fired"
        assert "resume_from_checkpoint" in (record.get("skipped_reason") or ""), (
            "the record has to name the action the adapter cannot perform"
        )
