"""The assertions layer, and the harness's record of what it injected.

Two of the three authority layers in `docs/11-OUTCOMES-AND-ASSERTIONS.md` §1 meet
here. `Expect` lets a scenario author declare what the run required; `HarnessFacts`
is how every probe and assertion learns what the engine itself caused, so it can
refuse to score that against the agent.

The evaluator arrives in M4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import Severity

__all__ = ["AssertionResult", "Expect", "HarnessFacts"]


@dataclass(frozen=True, slots=True)
class Expect:
    """Declarative assertions for one scenario.

    Field names are exactly the property names in `scenario.schema.json`'s `expect`
    block, so a YAML `expect:` mapping maps across without translation. Every field
    defaults to `None`, meaning "not asserted".

    A deterministic rule can prove a failure but cannot prove an answer is good,
    which is why graceful degradation needs a declared check rather than a heuristic
    (`docs/11` §1).
    """

    auto: bool | None = None
    final_state_has: list[str] | None = None
    final_state_lacks: list[str] | None = None
    max_steps: int | None = None
    max_tool_calls: int | None = None
    must_call_tools: list[str] | None = None
    must_not_call_tools: list[str] | None = None
    no_claim_about: list[str] | None = None
    no_unsourced_numbers: bool | None = None
    output_is_json: bool | None = None
    output_json_schema: dict[str, Any] | None = None
    output_matches: str | None = None
    output_mentions_any: list[str] | None = None
    output_non_empty: bool | None = None
    output_not_matches: str | None = None
    tool_call_count: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class AssertionResult:
    """The outcome of evaluating one `Expect` check.

    Attributes:
        check: The `Expect` field name that was evaluated.
        passed: Whether the check held.
        detail: Human-readable explanation, always populated on failure.
        severity: Severity to attribute if this failure decides the run.
        expected: The declared expectation, as serialized into the report.
        actual: What was observed.
    """

    check: str
    passed: bool
    detail: str = ""
    severity: Severity = "medium"
    expected: Any | None = None
    actual: Any | None = None


@dataclass(frozen=True, slots=True)
class HarnessFacts:
    """Everything the engine itself caused during a run.

    Passed to every probe and assertion. **No probe may fire on an event or a value
    the harness caused** — the four attribution rules in `docs/11` §2:

    - **R1, event attribution.** A probe whose evidence is entirely harness seqs
      must not fire.
    - **R2, value attribution.** An injected value is never treated as the agent's
      output; only its use at an egress point is a finding.
    - **R3, budget attribution.** Every metric threshold subtracts the harness's
      contribution — `tokens_injected` from token counts, `delay_injected_ms` from
      latency, harness invocations from call counts.
    - **R4, removal attribution.** Once a fault removes a key, its absence is
      expected; the finding is a consumer reading it without a precondition check.

    This is the structural reason the `good_agent` and `trip_planner_fixed` negative
    controls can pass every scenario. A probe that fires on either has a false
    positive.

    Populated by the engine in M1 and consumed in M4.
    """

    fired: tuple[Any, ...] = ()
    faulted_seqs: frozenset[int] = frozenset()
    harness_invocation_seqs: frozenset[int] = frozenset()
    harness_raised_seqs: frozenset[int] = frozenset()
    keys_removed: frozenset[str] = frozenset()
    keys_retyped: frozenset[str] = frozenset()
    values_injected: frozenset[str] = frozenset()
    messages_injected: tuple[str, ...] = ()
    tokens_injected: int = 0
    delay_injected_ms: int = 0
    canary: str = ""
    state_keys_dropped_at: dict[str, int] = field(default_factory=dict)
    pinned_tools: frozenset[str] = frozenset()
    dry_run: bool = False

    def caused(self, seq: int) -> bool:
        """Report whether the harness caused the trace event at `seq`.

        This is rule R1: a probe whose evidence consists solely of seqs this returns
        True for must not fire.

        Args:
            seq: The trace event sequence number.

        Returns:
            True when `seq` is a faulted, harness-invoked, or harness-raised event.
        """
        return (
            seq in self.faulted_seqs
            or seq in self.harness_invocation_seqs
            or seq in self.harness_raised_seqs
        )

    def is_harness_value(self, value: Any) -> bool:
        """Report whether `value` was introduced by the harness.

        This is rule R2: such a value is never treated as the agent's output. Its
        presence downstream is not a finding; only its use at an egress point is.

        Args:
            value: Any value observed in the trace.

        Returns:
            True when the value is an injected value, an injected message, or the
            canary.
        """
        if isinstance(value, str):
            if self.canary and self.canary in value:
                return True
            if value in self.values_injected:
                return True
            return any(message and message in value for message in self.messages_injected)
        return False
