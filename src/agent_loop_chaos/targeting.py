"""Targeting and triggering.

`Target` answers *where* a fault applies; `Trigger` answers *when* and *how often*.
Keeping them separate is what makes the scenario matrix expressible
(`docs/01-ARCHITECTURE.md` §4).

`matches()` and `should_fire()` arrive in M1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .context import Crossing, Layer, Phase

__all__ = ["Target", "Trigger"]


@dataclass(frozen=True, slots=True)
class Target:
    """Where a fault applies.

    `tool`, `node` and `llm` accept `fnmatch` globs. `state_key` is a dotted path
    that may contain ``*`` wildcards, e.g. ``"messages.*.content"``.

    A glob never matches a tool declared ``side_effecting=True`` — that is part of
    the D-23 safety gate, not a convenience (`SAFETY.md` §1).

    Attributes:
        layer: Restrict to one layer, or `None` for any.
        tool: Tool-name glob.
        node: Graph node-name glob.
        llm: LLM alias glob.
        state_key: Dotted state path, ``*`` allowed.
        phase: Restrict to one phase, or `None` for any.
        predicate: Final arbitrary filter over the crossing. Must be pure and must
            not read the clock.
    """

    layer: Layer | None = None
    tool: str | None = None
    node: str | None = None
    llm: str | None = None
    state_key: str | None = None
    phase: Phase | None = None
    predicate: Callable[[Crossing], bool] | None = None


@dataclass(frozen=True, slots=True)
class Trigger:
    """When, and how often, a fault fires.

    All randomness is drawn from the run's seeded RNG keyed on
    ``f"{fault_key}:{purpose}"`` (D-02, D-03), so `probability` is reproducible and
    adding an unrelated fault does not shift this one's stream.

    Attributes:
        on_call: Fire on exactly this 1-based call index of the target.
        on_step: Fire on exactly this step.
        after_step: Fire on every eligible crossing after this step.
        probability: Fire chance in ``[0, 1]``. Outside that range raises
            `ConfigError` at registration.
        max_fires: Stop after this many fires.
        cooldown_calls: Minimum target calls between two fires.
        stop_after_step: Never fire beyond this step.
    """

    on_call: int | None = None
    on_step: int | None = None
    after_step: int | None = None
    probability: float = 1.0
    max_fires: int = 1
    cooldown_calls: int = 0
    stop_after_step: int | None = None


def matches(target: Target, crossing: Crossing) -> bool:
    """Report whether `target` selects `crossing`.

    Args:
        target: The target to test.
        crossing: The crossing to test it against.

    Returns:
        True when every constraint the target sets is satisfied.

    Raises:
        NotImplementedError: Until M1.
    """
    raise NotImplementedError("targeting.matches arrives in M1 (prompts/01-core-engine.md)")


def should_fire(trigger: Trigger, ctx: Any) -> bool:
    """Report whether `trigger` fires on the current crossing.

    Args:
        trigger: The trigger to evaluate.
        ctx: The `FaultContext` carrying counters and the seeded RNG.

    Returns:
        True when the fault should fire here.

    Raises:
        NotImplementedError: Until M1.
    """
    raise NotImplementedError("targeting.should_fire arrives in M1 (prompts/01-core-engine.md)")
