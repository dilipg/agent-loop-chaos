"""Targeting and triggering.

`Target` answers *where* a fault applies; `Trigger` answers *when* and *how often*.
Keeping them separate is what makes the scenario matrix expressible
(`docs/01-ARCHITECTURE.md` §4).

Both predicates return a reason on every branch, because "why didn't my fault fire"
is the first question anyone asks, and `injected_faults[].skipped_reason` needs an
answer (D-36).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

from .context import Crossing, FaultContext, Layer, Phase

__all__ = ["Target", "Trigger", "matches", "should_fire", "state_key_matches"]

_LAYER_ATTR: dict[str, str] = {"tool": "tool", "llm": "llm", "node": "node"}


@dataclass(frozen=True, slots=True)
class Target:
    """Where a fault applies.

    `tool`, `node` and `llm` accept `fnmatch` globs. `state_key` is a dotted path
    that may contain ``*`` wildcards, e.g. ``"messages.*.content"``.

    A glob never matches a tool declared ``side_effecting=True``; that is part of
    the D-23 safety gate and is enforced at registration, not here (`SAFETY.md` §1).

    Attributes:
        layer: Restrict to one layer, or `None` for any.
        tool: Tool-name glob.
        node: Graph node-name glob.
        llm: LLM alias glob.
        state_key: Dotted state path, ``*`` allowed as a whole segment.
        phase: Restrict to one phase, or `None` for any.
        predicate: Final arbitrary filter over the crossing. Must be pure, and must
            not read the clock (D-47).
    """

    layer: Layer | None = None
    tool: str | None = None
    node: str | None = None
    llm: str | None = None
    state_key: str | None = None
    phase: Phase | None = None
    predicate: Callable[[Crossing], bool] | None = None

    def implied_layer(self) -> Layer | None:
        """Infer the layer from whichever name field is set.

        Returns:
            The implied layer, or `None` when nothing implies one.
        """
        if self.layer is not None:
            return self.layer
        if self.tool is not None:
            return "tool"
        if self.llm is not None:
            return "llm"
        # `state_key` outranks `node`: `{state_key: location, node: summarize}` is a
        # state target scoped to one node's boundary, not a node target. Reading it
        # the other way made every such target unmatchable.
        if self.state_key is not None:
            return "state"
        if self.node is not None:
            return "node"
        return None


@dataclass(frozen=True, slots=True)
class Trigger:
    """When, and how often, a fault fires.

    `probability` is drawn from the run's seeded RNG keyed on
    ``f"{fault_key}:trigger"`` (D-02, D-03). **The RNG is consumed only when
    `probability < 1.0`** — that is what keeps a fully deterministic scenario's
    streams stable as a suite grows, and it is asserted by a test.

    Attributes:
        on_call: Fire on this 1-based call index of the target, or any of several.
        on_step: Fire on exactly this step.
        after_step: Fire on every eligible crossing after this step.
        probability: Fire chance in ``[0, 1]``. Outside that range raises
            `ConfigError` at registration.
        max_fires: Stop after this many fires. `None` means unlimited, which is
            what `scenario.schema.json` types it as and what the loop and context
            faults in the demo suite use.
        cooldown_calls: Minimum target calls between two fires.
        stop_after_step: Never fire beyond this step.
    """

    on_call: int | Sequence[int] | None = None
    on_step: int | None = None
    after_step: int | None = None
    probability: float = 1.0
    max_fires: int | None = 1
    cooldown_calls: int = 0
    stop_after_step: int | None = None

    def call_indices(self) -> tuple[int, ...] | None:
        """Normalize `on_call` to a tuple.

        Returns:
            The accepted call indices, or `None` when `on_call` is unset.
        """
        if self.on_call is None:
            return None
        if isinstance(self.on_call, int):
            return (self.on_call,)
        return tuple(self.on_call)


def state_key_matches(pattern: str, key: str) -> bool:
    """Match a dotted state path against a pattern with ``*`` segments.

    A ``*`` matches exactly one segment; a trailing ``**`` matches the remainder.

    Args:
        pattern: The target pattern, e.g. ``"messages.*.content"``.
        key: The concrete dotted path.

    Returns:
        True when the pattern selects the key.
    """
    pattern_parts = pattern.split(".")
    key_parts = key.split(".")
    for index, part in enumerate(pattern_parts):
        if part == "**":
            return True
        if index >= len(key_parts):
            return False
        if part != "*" and not fnmatchcase(key_parts[index], part):
            return False
    return len(pattern_parts) == len(key_parts)


def matches(target: Target, crossing: Crossing) -> bool:
    """Report whether `target` selects `crossing`.

    Args:
        target: The target to test.
        crossing: The crossing to test it against.

    Returns:
        True when every constraint the target sets is satisfied. A target that sets
        nothing matches everything.
    """
    layer = target.implied_layer()
    if layer is not None and crossing.layer != layer:
        return False
    if target.phase is not None and crossing.phase != target.phase:
        return False

    for attr, expected in (
        ("tool", target.tool),
        ("llm", target.llm),
        ("node", target.node),
    ):
        if expected is None:
            continue
        # `node:` also narrows a state crossing, which carries the node's name: a
        # state fault is routinely scoped to one node's boundary.
        if crossing.layer != _LAYER_ATTR[attr] and not (
            attr == "node" and crossing.layer == "state"
        ):
            return False
        if not fnmatchcase(crossing.name, expected):
            return False

    if target.state_key is not None:
        if crossing.layer != "state":
            return False
        if not _selects_state(target.state_key, crossing):
            return False

    return not (target.predicate is not None and not target.predicate(crossing))


def _state_paths(value: Any, depth: int, prefix: str = "") -> Iterator[str]:
    """Enumerate dotted key paths in a state value, down to `depth` segments.

    Bounded by the pattern's own segment count, so a large state costs no more than
    the pattern asks for.

    Args:
        value: The state, or a nested part of it.
        depth: How many more segments to descend.
        prefix: The path so far.

    Yields:
        Each dotted path.
    """
    if depth <= 0:
        return
    if isinstance(value, Mapping):
        items: Iterable[tuple[str, Any]] = ((str(k), v) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        items = ((str(i), v) for i, v in enumerate(value))
    elif isinstance(fields := getattr(type(value), "model_fields", None), dict):
        # A Pydantic model is a state shape too -- `StateGraph(MyModel)` is
        # LangGraph's own recommended pattern. Without this branch no `state_key`
        # could ever select a crossing on such a state, so every state fault armed
        # and never fired, with no reason recorded (D-141).
        items = ((str(k), getattr(value, k, None)) for k in fields)
    else:
        return
    for key, child in items:
        path = f"{prefix}.{key}" if prefix else key
        yield path
        yield from _state_paths(child, depth - 1, path)


def _selects_state(pattern: str, crossing: Crossing) -> bool:
    """Report whether a `state_key` pattern selects this state crossing.

    The engine builds one state crossing per node entry and names it after the node,
    because a node boundary is the only place the whole state is visible and a
    partial update has not yet been merged. So the key has to be looked for *in the
    state*, not in the crossing's name -- matching the name was the reason no state
    fault could fire.

    Args:
        pattern: The `state_key` glob.
        crossing: The state crossing.

    Returns:
        True when the pattern matches a path present in the state, or the crossing's
        own name (the shape a per-key crossing would have).
    """
    if state_key_matches(pattern, crossing.name):
        return True
    depth = pattern.count(".") + 1
    return any(state_key_matches(pattern, path) for path in _state_paths(crossing.state, depth))


def should_fire(
    trigger: Trigger,
    crossing: Crossing,
    ctx: FaultContext,
    fault_id: str,
) -> tuple[bool, str]:
    """Decide whether `trigger` fires at this crossing.

    Checks run cheapest-first, and the seeded RNG is touched last and only when
    `probability < 1.0`.

    Args:
        trigger: The trigger to evaluate.
        crossing: The crossing under consideration.
        ctx: The fault context, for counters and the seeded RNG.
        fault_id: Which fault this is, for its fire counters.

    Returns:
        ``(fired, reason)``. The reason is populated on both branches: ``"fired"``,
        or one of ``"max_fires_reached"``, ``"stopped_after_step"``,
        ``"call_index_mismatch"``, ``"step_mismatch"``, ``"after_step_not_reached"``,
        ``"cooldown"``, ``"probability_not_met"``.
    """
    fires = ctx.counters.fires.get(fault_id, 0)
    if trigger.max_fires is not None and fires >= trigger.max_fires:
        return False, "max_fires_reached"

    if trigger.stop_after_step is not None and crossing.step > trigger.stop_after_step:
        return False, "stopped_after_step"

    indices = trigger.call_indices()
    if indices is not None and crossing.call_index not in indices:
        return False, "call_index_mismatch"

    if trigger.on_step is not None and crossing.step != trigger.on_step:
        return False, "step_mismatch"

    if trigger.after_step is not None and crossing.step <= trigger.after_step:
        return False, "after_step_not_reached"

    if trigger.cooldown_calls:
        last = ctx.counters.last_fire_call.get(fault_id)
        if last is not None and (crossing.call_index - last) <= trigger.cooldown_calls:
            return False, "cooldown"

    # Touched last, and only when it can change the answer, so a probability of 1.0
    # never advances a stream (asserted by a test).
    if trigger.probability < 1.0 and ctx.rng("trigger").random() >= trigger.probability:
        return False, "probability_not_met"

    return True, "fired"
