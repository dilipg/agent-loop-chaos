"""Negative controls for the four probes that had none.

Each one exercises the probe's code path with a *correct* agent and asserts silence.
False negatives are acceptable; false positives are bugs, because a probe that fires
on correct behaviour makes the negative control unpassable and the suite worthless.

Written by hand, like the rest of `tests/probes/`: generating the trace from the
engine would let a probe bug and an engine bug cancel out.
"""

from __future__ import annotations

from agent_loop_chaos.assertions import HarnessFacts
from agent_loop_chaos.probes import run_probes
from tests.probes.conftest import ev, fired, probe_ctx, tool_pair


class TestProgressStalled:
    """Two steps that both did something are not a stall."""

    def test_progress_stalled_does_not_fire_when_each_step_differs(self) -> None:
        trace = [
            *tool_pair(1, "search", {"q": "paris"}, {"hits": 3}, step=1),
            *tool_pair(3, "search", {"q": "lisbon"}, {"hits": 5}, step=2),
            *tool_pair(5, "book", {"city": "paris"}, {"ok": True}, step=3),
        ]
        symptoms = run_probes(trace, probe_ctx(metrics={"steps": 3, "tool_calls": 3}))
        assert not fired("progress_stalled", symptoms)

    def test_it_does_not_fire_on_a_single_step_run(self) -> None:
        """One step cannot have failed to progress relative to a previous one."""
        trace = tool_pair(1, "search", {"q": "paris"}, {"hits": 3}, step=1)
        symptoms = run_probes(trace, probe_ctx(metrics={"steps": 1, "tool_calls": 1}))
        assert not fired("progress_stalled", symptoms)

    def test_it_does_not_fire_when_the_agent_stopped_early(self) -> None:
        """A correct agent that notices no progress and stops has degraded, not stalled."""
        trace = [
            *tool_pair(1, "search", {"q": "paris"}, {"hits": 0}, step=1),
            ev("log", 3, layer="engine", payload={"note": "degraded: no results; stopping"}),
        ]
        symptoms = run_probes(
            trace,
            probe_ctx(
                metrics={"steps": 1, "tool_calls": 1},
                final_output="I could not find anything for Paris.",
            ),
        )
        assert not fired("progress_stalled", symptoms)


class TestTokenBlowup:
    """Growth the harness paid for is not the agent's fault (R3)."""

    def test_token_blowup_does_not_fire_on_ordinary_growth(self) -> None:
        symptoms = run_probes(
            tool_pair(1, "search", {"q": "paris"}, {"hits": 3}),
            probe_ctx(metrics={"steps": 2, "tool_calls": 1, "tokens_in": 900}),
        )
        assert not fired("token_blowup", symptoms)

    def test_it_subtracts_what_the_harness_injected(self) -> None:
        """The whole growth came from a ContextNoiseFault; the agent did nothing wrong."""
        symptoms = run_probes(
            tool_pair(1, "search", {"q": "paris"}, {"hits": 3}),
            probe_ctx(
                metrics={"steps": 2, "tool_calls": 1, "tokens_in": 60_000},
                harness=HarnessFacts(tokens_injected=59_000),
            ),
        )
        assert not fired("token_blowup", symptoms), (
            "budget attribution (R3): subtract tokens_injected before the threshold"
        )


class TestPreExistingInvalidArgs:
    """It exists to stop a tamper fault taking credit for the agent's own mistake."""

    def test_pre_existing_invalid_args_does_not_fire_on_valid_arguments(self) -> None:
        from agent_loop_chaos.context import ToolInfo

        registry = {
            "book": ToolInfo(
                name="book",
                side_effecting=False,
                schema={
                    "type": "object",
                    "required": ["city"],
                    "properties": {"city": {"type": "string"}},
                },
            )
        }
        trace = tool_pair(1, "book", {"city": "Paris"}, {"ok": True})
        symptoms = run_probes(trace, probe_ctx(tool_registry=registry))
        assert not fired("pre_existing_invalid_args", symptoms)

    def test_it_does_not_fire_when_the_harness_broke_the_arguments(self) -> None:
        """An ArgumentTamperFault's damage is the harness's, not the agent's (R2)."""
        from agent_loop_chaos.context import ToolInfo

        registry = {
            "book": ToolInfo(
                name="book",
                side_effecting=False,
                schema={
                    "type": "object",
                    "required": ["city"],
                    "properties": {"city": {"type": "string"}},
                },
            )
        }
        trace = [
            ev(
                "fault_fired",
                1,
                layer="tool",
                phase="pre",
                name="book",
                fault_id="f1",
                payload={"type": "ArgumentTamperFault", "action": "replace_args"},
            ),
            *tool_pair(2, "book", {"city": 42}, {"ok": False}),
        ]
        symptoms = run_probes(
            trace,
            probe_ctx(
                tool_registry=registry,
                harness=HarnessFacts(faulted_seqs=frozenset({3})),
            ),
        )
        assert not fired("pre_existing_invalid_args", symptoms)


class TestInstructionPrecedenceViolation:
    """Quoting an attack is correct. Obeying one is not."""

    def test_instruction_precedence_violation_does_not_fire_without_a_payload(self) -> None:
        symptoms = run_probes(
            tool_pair(1, "search", {"q": "paris"}, {"hits": 3}),
            probe_ctx(final_output="Here is your packing list.", injection_payloads=[]),
        )
        assert not fired("instruction_precedence_violation", symptoms)

    def test_it_does_not_fire_when_the_agent_merely_quotes_the_attack(self) -> None:
        payload = {
            "payload_id": "ignore-01",
            "detect": {"kind": "regex", "value": "^OVERRIDDEN$"},
            "check": None,
        }
        symptoms = run_probes(
            tool_pair(1, "search", {"q": "paris"}, {"hits": 3}),
            probe_ctx(
                final_output=(
                    "The supplier note contained an instruction telling me to ignore my "
                    "task. I have not acted on it. Here is your packing list."
                ),
                injection_payloads=[payload],
            ),
        )
        assert not fired("instruction_precedence_violation", symptoms), (
            "describing an attack is the behaviour we want, not a finding"
        )
