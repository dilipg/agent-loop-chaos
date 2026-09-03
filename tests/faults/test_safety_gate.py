"""The D-23 side-effect gate.

Three faults perform real operations the agent never requested. Pointed at
`charge_card` or `delete_rows` those are a double charge and a delete with no
predicate. `SAFETY.md` §1 says the gate is enforced in code, not documented, so
these are the tests that make that true.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.context import Crossing, Layer, Phase
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.faults.base import Fault, FaultOutcome
from agent_loop_chaos.targeting import Trigger


class DangerousFault(Fault):
    """A stand-in for the three faults that perform real operations.

    **What agent weakness it proves:** nothing; it exists so the gate can be tested
    without depending on any particular fault being implemented yet.

    **What graceful behaviour looks like:** not applicable.
    """

    kind: ClassVar[str] = "TestDangerousFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "pre")})
    performs_real_action: ClassVar[bool] = True

    def apply(self, crossing: Crossing, ctx: Any) -> FaultOutcome:
        """Do nothing; the gate is what is under test.

        Args:
            crossing: Ignored.
            ctx: Ignored.

        Returns:
            A noop outcome.
        """
        return FaultOutcome(action="noop")


def test_a_dangerous_fault_refuses_a_side_effecting_tool() -> None:
    """The core refusal: no opt-in, no targeting a tool that does real work."""
    engine = ChaosEngine(write_bundle=False)

    @engine.tool(side_effecting=True)
    def charge_card(amount: int) -> str:
        return "charged"

    with pytest.raises(ConfigError, match="side_effecting"):
        engine.register_fault(DangerousFault(), target_tool="charge_card")


def test_an_explicit_opt_in_permits_the_tool() -> None:
    """`allow_side_effects` is the deliberate, per-tool escape hatch."""
    engine = ChaosEngine(write_bundle=False, allow_side_effects=["charge_card"])

    @engine.tool(side_effecting=True)
    def charge_card(amount: int) -> str:
        return "charged"

    assert engine.register_fault(DangerousFault(), target_tool="charge_card") == "f1"


def test_the_opt_in_is_per_tool_not_global() -> None:
    """Allowing one tool must not quietly allow its neighbour."""
    engine = ChaosEngine(write_bundle=False, allow_side_effects=["hold_booking"])

    @engine.tool(side_effecting=True)
    def charge_card(amount: int) -> str:
        return "charged"

    with pytest.raises(ConfigError, match="charge_card"):
        engine.register_fault(DangerousFault(), target_tool="charge_card")


def test_a_glob_target_never_matches_a_side_effecting_tool() -> None:
    """`SAFETY.md` §1 item 2, and it is not limited to the real-action faults.

    A wildcard should never quietly reach something that moves money or deletes
    rows. Hitting one requires naming it.
    """
    from agent_loop_chaos.faults.base import NoopFault

    engine = ChaosEngine(write_bundle=False, strict_trace=True)
    engine.register_fault(NoopFault(), target_tool="*", trigger=Trigger(max_fires=9))

    @engine.tool
    def read_rows() -> str:
        return "rows"

    @engine.tool(side_effecting=True)
    def delete_rows() -> str:
        return "deleted"

    result = engine.run(lambda: [read_rows(), delete_rows()])
    fired_on = {fire["name"] for fire in result.injected_faults[0]["fires"]}
    assert fired_on == {"read_rows"}, "the glob must skip the side-effecting tool"


def test_the_glob_skip_is_recorded_with_a_reason() -> None:
    """A silent skip would be indistinguishable from a targeting mistake."""
    from agent_loop_chaos.faults.base import NoopFault
    from agent_loop_chaos.trace import TraceRecorder

    engine = ChaosEngine(seed=1, out_dir="/tmp/.chaos-gate", strict_trace=True)
    engine.register_fault(NoopFault(), target_tool="*", trigger=Trigger(max_fires=9))

    @engine.tool(side_effecting=True)
    def delete_rows() -> str:
        return "deleted"

    result = engine.run(lambda: delete_rows(), scenario_id="glob-skip")
    reasons = {
        event["payload"]["reason"]
        for event in TraceRecorder.load(result.artifacts["trace"])
        if event["kind"] == "fault_skipped"
    }
    assert "side_effecting_tool_not_named" in reasons


def test_an_explicitly_named_side_effecting_tool_is_still_reachable() -> None:
    """The rule is about wildcards, not about forbidding the tool outright."""
    from agent_loop_chaos.faults.base import NoopFault

    engine = ChaosEngine(write_bundle=False, strict_trace=True)
    engine.register_fault(NoopFault(), target_tool="delete_rows")

    @engine.tool(side_effecting=True)
    def delete_rows() -> str:
        return "deleted"

    assert engine.run(lambda: delete_rows()).injected_faults[0]["fire_count"] == 1


def test_the_gate_refuses_a_broad_preset_while_a_tool_is_undeclared() -> None:
    """`SAFETY.md` §1 item 3.

    A broad preset points faults at everything, so it must not run while the engine
    cannot tell which tools are dangerous. Omitting `side_effecting` is "unknown",
    not "safe".
    """
    engine = ChaosEngine(write_bundle=False)

    @engine.tool
    def might_write_somewhere() -> str:
        return "?"

    with pytest.raises(ConfigError, match="might_write_somewhere"):
        engine.require_declared_side_effects()


def test_declaring_false_is_a_deliberate_statement_that_satisfies_the_gate() -> None:
    """ "Declaring `side_effecting=False` is the opt-out, and it is a deliberate
    statement" — `SAFETY.md` §1. So it must be distinguishable from silence.
    """
    engine = ChaosEngine(write_bundle=False)

    @engine.tool(side_effecting=False)
    def reads_only() -> str:
        return "rows"

    engine.require_declared_side_effects()


def test_declaring_true_also_satisfies_the_gate() -> None:
    """The check is about knowing, not about being harmless."""
    engine = ChaosEngine(write_bundle=False)

    @engine.tool(side_effecting=True)
    def charge_card() -> str:
        return "charged"

    engine.require_declared_side_effects()


def test_an_undeclared_tool_is_not_treated_as_side_effecting() -> None:
    """Undeclared must not silently behave as `True` either, or globs would
    stop matching ordinary tools and the harness would go quiet.
    """
    from agent_loop_chaos.faults.base import NoopFault

    engine = ChaosEngine(write_bundle=False, strict_trace=True)
    engine.register_fault(NoopFault(), target_tool="*")

    @engine.tool
    def undeclared() -> str:
        return "x"

    assert engine.run(lambda: undeclared()).injected_faults[0]["fire_count"] == 1
