"""The fault base machinery: registry, keys, mutation logs, and `NoopFault`.

`NoopFault` is the plumbing double: it must fire, record, and change nothing. A
probe that fires on a `NoopFault` run has a false positive.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from agent_loop_chaos.context import Crossing, Layer, Phase
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.faults.base import (
    FAULT_REGISTRY,
    TERMINAL_ACTIONS,
    VALUE_ACTIONS,
    Fault,
    FaultOutcome,
    FaultRecord,
    MutationLog,
    NoopFault,
    fault_from_dict,
    fault_key_for,
    list_faults,
    register_fault,
)


def test_noop_fault_is_registered_under_its_kind() -> None:
    """`kind` is how a YAML scenario names a fault, so it is public contract."""
    assert FAULT_REGISTRY["NoopFault"] is NoopFault


def test_noop_fault_changes_nothing_and_records_an_empty_patch() -> None:
    """Its whole purpose: exercise the plumbing without perturbing the agent."""
    crossing = Crossing(layer="tool", phase="post", name="t", result={"a": 1})
    outcome = NoopFault().apply(crossing, ctx=None)  # type: ignore[arg-type]
    assert outcome.action == "noop"
    assert outcome.mutation is not None
    assert outcome.mutation.json_patch == []
    assert outcome.mutation.payload_before == outcome.mutation.payload_after


def test_noop_fault_accepts_every_layer_and_phase() -> None:
    """So it can be pointed anywhere while testing the engine."""
    assert ("tool", "post") in NoopFault.accepts
    assert ("checkpoint", "error") in NoopFault.accepts
    assert len(NoopFault.accepts) == 18


def test_the_action_sets_partition_cleanly() -> None:
    """D-13 depends on value-replacing and terminal actions being disjoint."""
    assert not (VALUE_ACTIONS & TERMINAL_ACTIONS)


# ------------------------------------------------------------------- fault_key


def test_fault_key_is_stable_for_the_same_spec() -> None:
    """The RNG stream must not move when nothing about the fault moved."""
    args = ("NoopFault", {"a": 1}, {"tool": "t"}, {"on_call": [1]})
    assert fault_key_for(*args) == fault_key_for(*args)


@pytest.mark.parametrize(
    "changed",
    [
        ("OtherFault", {"a": 1}, {"tool": "t"}, {"on_call": [1]}),
        ("NoopFault", {"a": 2}, {"tool": "t"}, {"on_call": [1]}),
        ("NoopFault", {"a": 1}, {"tool": "u"}, {"on_call": [1]}),
        ("NoopFault", {"a": 1}, {"tool": "t"}, {"on_call": [2]}),
    ],
)
def test_fault_key_changes_when_any_part_of_the_spec_changes(changed: tuple[Any, ...]) -> None:
    """The key is a hash of the whole spec (D-03)."""
    base = fault_key_for("NoopFault", {"a": 1}, {"tool": "t"}, {"on_call": [1]})
    assert fault_key_for(*changed) != base


def test_fault_key_ignores_parameter_ordering() -> None:
    """Canonical JSON means key order cannot shift a stream."""
    left = fault_key_for("NoopFault", {"a": 1, "b": 2}, {}, {})
    right = fault_key_for("NoopFault", {"b": 2, "a": 1}, {}, {})
    assert left == right


# ------------------------------------------------------------------ registry


def test_fault_from_dict_builds_a_registered_fault() -> None:
    """This is the YAML loading path."""
    assert isinstance(fault_from_dict({"type": "NoopFault"}), NoopFault)


def test_fault_from_dict_requires_a_type() -> None:
    """A spec without `type` is a config error, named as such."""
    with pytest.raises(ConfigError, match="missing `type`"):
        fault_from_dict({"params": {}})


def test_fault_from_dict_lists_the_known_kinds_on_a_typo() -> None:
    """The error should be enough to fix the scenario file without reading source."""
    with pytest.raises(ConfigError) as exc:
        fault_from_dict({"type": "ToolCorruptionFualt"})
    message = str(exc.value)
    assert "registered kinds:" in message
    assert "ToolCorruptionFault" in message, "the near-miss must be visible in the list"


def test_register_fault_rejects_a_missing_kind() -> None:
    """An unnamed fault could never be referenced from YAML."""

    class Unnamed(Fault):
        accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "post")})

        def apply(self, crossing: Crossing, ctx: Any) -> FaultOutcome:
            return FaultOutcome()

    with pytest.raises(ConfigError, match="non-empty class-level `kind`"):
        register_fault(Unnamed)


def test_register_fault_rejects_a_duplicate_kind() -> None:
    """Two classes under one name would make a stored report ambiguous."""

    class Clashing(Fault):
        kind: ClassVar[str] = "NoopFault"
        accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "post")})

        def apply(self, crossing: Crossing, ctx: Any) -> FaultOutcome:
            return FaultOutcome()

    with pytest.raises(ConfigError, match="already registered"):
        register_fault(Clashing)


def test_register_fault_is_idempotent_for_the_same_class() -> None:
    """Re-importing a module must not explode."""
    assert register_fault(NoopFault) is NoopFault


def test_list_faults_describes_every_kind() -> None:
    """This powers `alc list-faults`."""
    infos = {info.kind: info for info in list_faults()}
    assert infos["NoopFault"].severity_hint == "info"
    assert "tool.post" in infos["NoopFault"].layer_phases
    assert infos["NoopFault"].summary.startswith("Fires and changes nothing")


# ---------------------------------------------------------------- MutationLog


def test_mutation_log_diffs_two_values() -> None:
    """`json_patch` is half of what makes a finding actionable."""
    log = MutationLog.of({"a": 1}, {"a": 2})
    assert log.json_patch == [{"op": "replace", "path": "/a", "value": 2}]
    assert log.unrepresentable is False


def test_mutation_log_flags_an_unrepresentable_payload() -> None:
    """So an empty patch is not mistaken for "nothing changed"."""
    log = MutationLog.of({"s": {1, 2}}, {"s": {1}})
    assert log.json_patch == []
    assert log.unrepresentable is True


def test_mutation_log_serializes_every_field() -> None:
    """The trace and the report both consume this shape."""
    assert set(MutationLog.of(1, 2).to_dict()) == {
        "payload_before",
        "payload_after",
        "json_patch",
        "unrepresentable",
        "mutation_skipped_uncopyable",
    }


# ---------------------------------------------------------------- FaultRecord


def test_fault_record_omits_skipped_reason_once_something_fired() -> None:
    """D-36: `skipped_reason` is set only when `fire_count == 0`."""
    record = FaultRecord(
        fault_id="f1",
        fault_key="abc",
        type="NoopFault",
        params={},
        target={},
        trigger={},
        fired=True,
        fire_count=1,
        skipped_reason="cooldown",
    )
    assert "skipped_reason" not in record.to_dict()


def test_fault_record_includes_skipped_reason_when_nothing_fired() -> None:
    """The complement: "why didn't my fault fire" needs an answer."""
    record = FaultRecord(
        fault_id="f1",
        fault_key="abc",
        type="NoopFault",
        params={},
        target={},
        trigger={},
        skipped_reason="target_never_called",
    )
    assert record.to_dict()["skipped_reason"] == "target_never_called"


# ------------------------------------------------------------------ base class


def test_params_returns_a_copy_so_the_plan_cannot_be_mutated() -> None:
    """A caller holding a reference to the plan could otherwise change the hash."""
    fault = NoopFault(a=1)
    fault.params()["a"] = 999
    assert fault.params() == {"a": 1}


def test_repr_shows_sorted_parameters() -> None:
    """Deterministic, because it appears in error messages."""
    assert repr(NoopFault(b=2, a=1)) == "NoopFault(a=1, b=2)"


def test_a_fault_can_reject_its_parameters_eagerly() -> None:
    """Validation belongs at construction, not mid-run."""

    class Picky(Fault):
        kind: ClassVar[str] = "TestPickyFault"
        accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "post")})

        def validate(self) -> None:
            if "required" not in self.params():
                raise ConfigError("TestPickyFault needs `required`")

        def apply(self, crossing: Crossing, ctx: Any) -> FaultOutcome:
            return FaultOutcome()

    with pytest.raises(ConfigError, match="needs `required`"):
        Picky()
    assert Picky(required=1).params() == {"required": 1}


class TestANoopOutcomeIsNotAFire:
    """A fault that decided to do nothing did not fire.

    Several faults evaluate their trigger, look at the crossing, and correctly decide
    there is nothing to do -- `RateLimitFault` inside its budget, `NodeSkipFault` at a
    node it does not target, `MalformedToolCallFault` on a response with no tool call.
    They recorded an honest note and then set `fired: True`.

    That inflates `suite.json`'s `coverage`, which is read as "this fault kind was
    exercised", and it suppresses the `no fault fired` warning for a scenario that
    genuinely proved nothing. D-36 already says `skipped_reason` carries the reason a
    fault did not fire; a no-op outcome is exactly that case, and its note is exactly
    that reason.
    """

    @staticmethod
    def _run(tmp_path: Any, fault: Any, **kw: Any) -> Any:
        from agent_loop_chaos import ChaosEngine

        engine = ChaosEngine(
            seed=1337, out_dir=tmp_path, write_bundle=False, strict_schema=False, judge="rules"
        )

        @engine.tool(name="lookup", side_effecting=False)
        def lookup() -> dict[str, Any]:
            return {"value": 1}

        def agent(question: Any = None) -> str:
            lookup()
            return "done"

        engine.register_fault(fault, **kw)
        return engine.run(agent, inputs={"question": "?"}, scenario_id="noop")

    def test_a_noop_outcome_does_not_count_as_fired(self, tmp_path: Any) -> None:
        from agent_loop_chaos.faults import RateLimitFault
        from agent_loop_chaos.targeting import Target

        # `after_calls: 5` with one call: the fault runs and correctly does nothing.
        result = self._run(tmp_path, RateLimitFault(after_calls=5), target=Target(tool="lookup"))
        record = result.injected_faults[0]
        assert record["fired"] is False, "a no-op is not a fire"

    def test_the_reason_is_kept(self, tmp_path: Any) -> None:
        from agent_loop_chaos.faults import RateLimitFault
        from agent_loop_chaos.targeting import Target

        result = self._run(tmp_path, RateLimitFault(after_calls=5), target=Target(tool="lookup"))
        reason = result.injected_faults[0]["skipped_reason"]
        assert reason, "the fault's own note explains why nothing happened; keep it"
        assert "budget" in reason or "rate" in reason.lower()

    def test_a_real_mutation_still_counts_as_fired(self, tmp_path: Any) -> None:
        from agent_loop_chaos.faults import ToolCorruptionFault
        from agent_loop_chaos.targeting import Target

        result = self._run(
            tmp_path,
            ToolCorruptionFault(mutation_type="drop_key", keys=["value"]),
            target=Target(tool="lookup"),
        )
        record = result.injected_faults[0]
        assert record["fired"] is True
        # D-36: the key is present only when the fault did not fire.
        assert record.get("skipped_reason") is None

    def test_a_delay_counts_as_fired(self, tmp_path: Any) -> None:
        """Nothing changed shape, but the run really was slowed."""
        from agent_loop_chaos.faults import ToolLatencyFault
        from agent_loop_chaos.targeting import Target

        result = self._run(tmp_path, ToolLatencyFault(delay_ms=1), target=Target(tool="lookup"))
        assert result.injected_faults[0]["fired"] is True

    def test_a_raise_counts_as_fired(self, tmp_path: Any) -> None:
        from agent_loop_chaos.faults import ToolErrorFault
        from agent_loop_chaos.targeting import Target

        result = self._run(
            tmp_path,
            ToolErrorFault(error_type="http_500", message="boom"),
            target=Target(tool="lookup"),
        )
        assert result.injected_faults[0]["fired"] is True


class TestAnUnsupportedActionIsNotAFire:
    """An action the adapter cannot perform did not happen, so it did not fire.

    `resume_from_checkpoint` has no implementation: the engine emits `fault_skipped`
    with `action_not_supported_by_adapter` and carries on. The fault record still said
    `fired: true`, so `suite.json` counted coverage for an injection that did nothing
    -- the same dishonesty D-95 fixed for a no-op outcome, arriving by another path.

    Needs a real checkpoint crossing to reach the action at all, so it needs a graph
    with a checkpointer.
    """

    @staticmethod
    def _run(tmp_path: Any) -> Any:
        pytest.importorskip("langgraph")
        from langgraph.checkpoint.memory import MemorySaver

        from agent_loop_chaos import ChaosEngine
        from agent_loop_chaos.adapters.langgraph import instrument_graph
        from agent_loop_chaos.faults import CheckpointRollbackFault
        from agent_loop_chaos.targeting import Target, Trigger
        from tests.fakes.lg_agent import build

        engine = ChaosEngine(
            seed=1337, out_dir=tmp_path, write_bundle=False, strict_schema=False, judge="rules"
        )
        # Targeted at the *checkpoint* layer specifically. Since D-111 the fault also
        # accepts `(node, post)`, where the replay is performable -- so an untargeted
        # registration fires there and this would be testing nothing.
        engine.register_fault(
            CheckpointRollbackFault(rollback_steps=1),
            target=Target(layer="checkpoint", phase="post"),
            trigger=Trigger(on_call=1),
        )
        compiled = build(lambda prompt: "warm").compile(checkpointer=MemorySaver())
        graph = instrument_graph(compiled, engine, intercept_checkpoints=True)
        return engine.run(graph, inputs={"query": "x"}, scenario_id="unsupported")

    def test_it_is_recorded_as_skipped(self, tmp_path: Any) -> None:
        record = self._run(tmp_path).injected_faults[0]
        assert record["fired"] is False, "the adapter did nothing; that is not a fire"
        assert record["skipped_reason"], "the reason has to say the adapter cannot do it"
        assert "adapter" in record["skipped_reason"]

    def test_the_reason_names_the_action(self, tmp_path: Any) -> None:
        reason = self._run(tmp_path).injected_faults[0]["skipped_reason"]
        assert "resume_from_checkpoint" in reason
