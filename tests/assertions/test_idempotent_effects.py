"""`idempotent_effects` — catching a missing idempotency key.

`duplicate_side_effect` counts **agent-issued** invocations only (R1), so when a
`CheckpointRollbackFault` replays a committed node the second `hold_booking` is the
harness's and the probe correctly declines. That is right, and it left the actual
finding undetectable: the agent booked twice because it supplied no idempotency key.

The finding is not "the tool was called twice" — the harness did that on purpose. It
is "calling it twice produced **two distinct effects**", which is a property of the
agent's design and is visible in the results. An agent that passes an idempotency key
gets the same hold back both times; one that does not gets two holds.

That is why this is an assertion and not a probe: it does not care who issued the
calls, only what came back.
"""

from __future__ import annotations

from typing import Any

from agent_loop_chaos.assertions import EvidenceContext, Expect, evaluate
from agent_loop_chaos.context import ToolInfo


def _tool_registry(**effects: bool) -> dict[str, ToolInfo]:
    return {name: ToolInfo(name=name, side_effecting=value) for name, value in effects.items()}


def _evidence(invocations: list[dict[str, Any]], **effects: bool) -> EvidenceContext:
    return EvidenceContext(
        final_output="done",
        tool_invocations=invocations,
        tool_registry=_tool_registry(**effects),
    )


def _check(expect: Expect, evidence: EvidenceContext) -> Any:
    results = [
        r for r in evaluate(expect, evidence, source="scenario") if r.check == "idempotent_effects"
    ]
    assert results, "the check did not run"
    return results[0]


class TestItCatchesTheRealFinding:
    def test_two_calls_two_effects_fails(self) -> None:
        evidence = _evidence(
            [
                {"name": "hold", "kwargs": {"flight": "AI-143"}, "result": {"hold_id": "H-1"}},
                {"name": "hold", "kwargs": {"flight": "AI-143"}, "result": {"hold_id": "H-2"}},
            ],
            hold=True,
        )
        result = _check(Expect(idempotent_effects=True), evidence)
        assert result.ok is False
        assert "hold" in result.detail

    def test_the_detail_names_what_differed(self) -> None:
        evidence = _evidence(
            [
                {"name": "hold", "kwargs": {"flight": "AI-143"}, "result": {"hold_id": "H-1"}},
                {"name": "hold", "kwargs": {"flight": "AI-143"}, "result": {"hold_id": "H-2"}},
            ],
            hold=True,
        )
        detail = _check(Expect(idempotent_effects=True), evidence).detail
        assert "2 distinct" in detail or "distinct result" in detail

    def test_two_calls_one_effect_passes(self) -> None:
        """An idempotency key makes the second call return the first hold."""
        evidence = _evidence(
            [
                {
                    "name": "hold",
                    "kwargs": {"flight": "AI-143", "key": "k1"},
                    "result": {"hold_id": "H-1", "duplicate": False},
                },
                {
                    "name": "hold",
                    "kwargs": {"flight": "AI-143", "key": "k1"},
                    "result": {"hold_id": "H-1", "duplicate": True},
                },
            ],
            hold=True,
        )
        assert _check(Expect(idempotent_effects=True), evidence).ok is True


class TestItDoesNotFireWhereItShouldNot:
    def test_a_read_only_tool_is_ignored(self) -> None:
        """Two `search` calls returning different results is not a side effect."""
        evidence = _evidence(
            [
                {"name": "search", "kwargs": {"q": "paris"}, "result": {"hits": 3}},
                {"name": "search", "kwargs": {"q": "paris"}, "result": {"hits": 5}},
            ],
            search=False,
        )
        assert _check(Expect(idempotent_effects=True), evidence).ok is True

    def test_an_undeclared_tool_is_ignored(self) -> None:
        """Undeclared is not False, but guessing which methods move money is worse."""
        evidence = EvidenceContext(
            final_output="done",
            tool_invocations=[
                {"name": "mystery", "kwargs": {}, "result": {"id": 1}},
                {"name": "mystery", "kwargs": {}, "result": {"id": 2}},
            ],
            tool_registry={"mystery": ToolInfo(name="mystery", side_effecting=None)},
        )
        assert _check(Expect(idempotent_effects=True), evidence).ok is True

    def test_different_arguments_are_different_operations(self) -> None:
        """Booking two different flights is two bookings, correctly."""
        evidence = _evidence(
            [
                {"name": "hold", "kwargs": {"flight": "AI-143"}, "result": {"hold_id": "H-1"}},
                {"name": "hold", "kwargs": {"flight": "LH-761"}, "result": {"hold_id": "H-2"}},
            ],
            hold=True,
        )
        assert _check(Expect(idempotent_effects=True), evidence).ok is True

    def test_a_single_call_passes(self) -> None:
        evidence = _evidence(
            [{"name": "hold", "kwargs": {"flight": "AI-143"}, "result": {"hold_id": "H-1"}}],
            hold=True,
        )
        assert _check(Expect(idempotent_effects=True), evidence).ok is True

    def test_no_calls_at_all_passes(self) -> None:
        assert _check(Expect(idempotent_effects=True), _evidence([], hold=True)).ok is True


class TestScoping:
    def test_a_list_restricts_to_named_tools(self) -> None:
        evidence = _evidence(
            [
                {"name": "hold", "kwargs": {"f": 1}, "result": {"id": "H-1"}},
                {"name": "hold", "kwargs": {"f": 1}, "result": {"id": "H-2"}},
                {"name": "charge", "kwargs": {"a": 1}, "result": {"id": "C-1"}},
                {"name": "charge", "kwargs": {"a": 1}, "result": {"id": "C-2"}},
            ],
            hold=True,
            charge=True,
        )
        result = _check(Expect(idempotent_effects=["charge"]), evidence)
        assert result.ok is False
        assert "charge" in result.detail
        assert "hold" not in result.detail, "the list should scope the check"

    def test_it_does_not_run_when_unset(self) -> None:
        evidence = _evidence(
            [
                {"name": "hold", "kwargs": {"f": 1}, "result": {"id": "H-1"}},
                {"name": "hold", "kwargs": {"f": 1}, "result": {"id": "H-2"}},
            ],
            hold=True,
        )
        codes = [r.check for r in evaluate(Expect(), evidence, source="scenario")]
        assert "idempotent_effects" not in codes


class TestItIsSynthesizedForARollback:
    """A rollback scenario gets the check for free, because that is what it is for."""

    def test_a_checkpoint_rollback_synthesizes_it(self) -> None:
        from agent_loop_chaos.assertions import synthesize_auto_expect

        expect = synthesize_auto_expect(
            [{"action": "resume_from_checkpoint", "type": "CheckpointRollbackFault"}],
            max_steps=25,
            recovered=False,
        )
        assert expect.idempotent_effects is True

    def test_a_duplicate_side_effect_fault_synthesizes_it(self) -> None:
        from agent_loop_chaos.assertions import synthesize_auto_expect

        expect = synthesize_auto_expect(
            [{"action": "invoke_target", "type": "DuplicateSideEffectFault"}],
            max_steps=25,
            recovered=False,
        )
        assert expect.idempotent_effects is True

    def test_an_ordinary_fault_does_not(self) -> None:
        from agent_loop_chaos.assertions import synthesize_auto_expect

        expect = synthesize_auto_expect(
            [{"action": "replace_result", "type": "ToolCorruptionFault", "json_patch": []}],
            max_steps=25,
            recovered=False,
        )
        assert expect.idempotent_effects is not True


class TestEndToEnd:
    """The rollback scenario that D-106 could wire but not detect."""

    @staticmethod
    def _run(tmp_path: Any, *, idempotent: bool) -> Any:
        from agent_loop_chaos import ChaosEngine
        from agent_loop_chaos.faults import DuplicateSideEffectFault
        from agent_loop_chaos.targeting import Target, Trigger

        holds: dict[str, str] = {}

        engine = ChaosEngine(
            seed=1337,
            out_dir=tmp_path,
            write_bundle=False,
            judge="rules",
            strict_schema=False,
            allow_side_effects=["hold_booking"],
        )

        @engine.tool(name="hold_booking", side_effecting=True)
        def hold_booking(flight_id: str, key: str | None = None) -> dict[str, Any]:
            token = key or f"{flight_id}:{len(holds) + 1}"
            first = token not in holds
            if first:
                holds[token] = f"H-{len(holds) + 1:03d}"
            return {"hold_id": holds[token], "duplicate": not first}

        def agent(question: Any = None) -> str:
            # The only difference between the two trees.
            result = hold_booking("AI-143", key="trip-1") if idempotent else hold_booking("AI-143")
            return f"Held {result['hold_id']}."

        engine.register_fault(
            DuplicateSideEffectFault(times=2),
            target=Target(tool="hold_booking"),
            trigger=Trigger(on_call=1),
        )
        return engine.run(
            agent,
            inputs={"question": "book it"},
            scenario_id="rollback",
            expected_behavior="graceful_degradation",
            allow_side_effects=["hold_booking"],
        )

    def test_the_agent_without_a_key_is_caught(self, tmp_path: Any) -> None:
        result = self._run(tmp_path, idempotent=False)
        failed = [a["check"] for a in result.assertions if not a["ok"]]
        assert "idempotent_effects" in failed, (
            f"the double booking went undetected; failures were {failed}"
        )

    def test_the_agent_with_a_key_passes(self, tmp_path: Any) -> None:
        result = self._run(tmp_path, idempotent=True)
        failed = [(a["check"], a["detail"]) for a in result.assertions if not a["ok"]]
        assert failed == [], f"the negative control failed: {failed}"
        assert result.success is True

    def test_it_is_synthesized_without_being_declared(self, tmp_path: Any) -> None:
        """The scenario says nothing; the fault implies the question."""
        result = self._run(tmp_path, idempotent=False)
        auto = [
            a
            for a in result.assertions
            if a["check"] == "idempotent_effects" and a["source"] == "auto"
        ]
        assert auto, "a duplicate-effect fault should synthesize the check"

    def test_it_classifies_as_a_duplicate_side_effect(self, tmp_path: Any) -> None:
        assert self._run(tmp_path, idempotent=False).failure_mode == "duplicate_side_effect"
