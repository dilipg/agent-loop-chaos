"""Graph and state faults (catalog section C).

These operate on the agent's state rather than on any framework's objects, so they
behave identically under both adapters and are testable with no framework installed.

Two rules from `docs/06` §1.3 shape the whole module:

- **Never mutate the caller's state.** The agent is still using that object; a fault
  that edited it in place would corrupt the run rather than perturb it, and would
  make `payload_before` a lie.
- **A node returns a partial update, not a whole state.** A key therefore cannot be
  *removed* by returning a dict without it -- the reducer merges the update in and
  the key survives. `StateDropFault(mode="remove")` at a `post` phase is refused at
  registration rather than silently doing nothing.
"""

from __future__ import annotations

import copy
from fnmatch import fnmatchcase
from typing import Any, ClassVar

from ..context import Crossing, FaultContext, Layer, Phase
from ..enums import Severity
from ..errors import ConfigError
from ..mutations import MUTATIONS
from .base import Fault, FaultOutcome, MutationLog, register_fault

__all__ = [
    "CheckpointRollbackFault",
    "EdgeMisrouteFault",
    "NodeSkipFault",
    "StateDropFault",
    "StateStaleFault",
    "StateTypeFault",
]

_STATE_PAIRS: frozenset[tuple[Layer, Phase]] = frozenset({("state", "pre"), ("state", "post")})

# Mutations that replace the whole payload and ignore `keys`. Wrapping a value for
# these would hand back the wrapper rather than the mutated value.
_WHOLE_PAYLOAD_MUTATIONS: frozenset[str] = frozenset(
    {
        "empty_json",
        "empty_list",
        "null_result",
        "json_as_string",
        "malformed_json_string",
        "whitespace_only",
        "deep_nest",
        "wrong_schema",
        "truncate_list",
        "duplicate_items",
        "reorder_list",
    }
)


def _expand(state: Any, pattern: str, prefix: str = "") -> list[str]:
    """Expand a dotted path that may contain globs into concrete paths.

    Args:
        state: The state to walk.
        pattern: A dotted path; any segment may be a glob.
        prefix: Accumulated path, used in recursion.

    Returns:
        Every concrete path the pattern selects.
    """
    head, _, rest = pattern.partition(".")
    if isinstance(state, dict):
        keys = [k for k in state if fnmatchcase(str(k), head)]
    elif isinstance(state, list):
        keys = [str(i) for i in range(len(state)) if head == "*" or fnmatchcase(str(i), head)]
    else:
        return []

    out: list[str] = []
    for key in keys:
        path = f"{prefix}{key}"
        if not rest:
            out.append(path)
        else:
            child = state[int(key)] if isinstance(state, list) else state[key]
            out.extend(_expand(child, rest, f"{path}."))
    return out


def _parent(state: Any, path: str) -> tuple[Any, str] | None:
    """Walk to the container holding a path's final segment.

    Args:
        state: The state to walk.
        path: A concrete dotted path.

    Returns:
        ``(container, leaf)``, or `None` when the path is unreachable.
    """
    parts = path.split(".")
    node = state
    for part in parts[:-1]:
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    leaf = parts[-1]
    if isinstance(node, dict) and leaf in node:
        return node, leaf
    if isinstance(node, list) and leaf.isdigit() and int(leaf) < len(node):
        return node, leaf
    return None


def _state_of(crossing: Crossing) -> Any:
    """The state a crossing carries.

    Args:
        crossing: The crossing.

    Returns:
        The state, preferring an explicit `state` over the phase payload.
    """
    if crossing.state is not None:
        return crossing.state
    return crossing.result if crossing.phase == "post" else None


def _copy_state(state: Any) -> tuple[Any, bool]:
    """Deep-copy the state, reporting failure rather than raising.

    Args:
        state: The state to copy.

    Returns:
        ``(copy, copied)``. A state that cannot be copied is passed through and the
        fault records `mutation_skipped_uncopyable`.
    """
    try:
        return copy.deepcopy(state), True
    except Exception:
        return state, False


@register_fault
class StateDropFault(Fault):
    """Remove or null a key from the graph's state.

    **What agent weakness it proves:** downstream nodes assume the keys their
    upstream neighbours were supposed to set actually exist.

    **What graceful behaviour looks like:** each node validates its required state
    slice and routes to an error branch rather than indexing blindly.

    Expected failure modes: `crash_unhandled_exception`,
    `state_corruption_propagated`.
    """

    kind: ClassVar[str] = "StateDropFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = _STATE_PAIRS | {("node", "pre")}
    severity_hint: ClassVar[Severity] = "high"

    def validate(self) -> None:
        """Check the key list and the mode.

        Raises:
            ConfigError: With the offending value.
        """
        params = self.params()
        keys = params.get("keys")
        if keys is not None and not isinstance(keys, (list, tuple)):
            raise ConfigError(f"{self.kind}.keys must be a list of dotted paths; got {keys!r}")
        mode = str(params.get("mode", "remove"))
        if mode not in {"remove", "null"}:
            raise ConfigError(f"{self.kind}.mode must be 'remove' or 'null'; got {mode!r}")

    def check_target(self, phase: str | None) -> None:
        """Refuse a combination that cannot express itself.

        A LangGraph node returns a **partial update**, so returning a dict without a
        key does not remove it -- the reducer merges the update and the key survives.
        Registering `mode="remove"` at `post` would produce a fault that silently
        does nothing (`docs/06` §1.3).

        Args:
            phase: The target's phase, or `None` for any.

        Raises:
            ConfigError: For `remove` at a `post` phase.
        """
        if phase == "post" and str(self.params().get("mode", "remove")) == "remove":
            raise ConfigError(
                f"{self.kind}(mode='remove') cannot fire at a `post` phase: a node returns "
                "a partial update, so a key omitted from it is merged over rather than "
                "removed and the fault would do nothing. Use mode='null', or target `pre`."
            )

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Remove or null the selected keys in a copy of the state.

        Args:
            crossing: The crossing carrying the state.
            ctx: The fault context.

        Returns:
            A `replace_state` outcome recording which keys went, so R4 can tell a
            later unguarded read from the removal itself.
        """
        params = self.params()
        before = _state_of(crossing)
        if before is None:
            return FaultOutcome(action="noop", note="no visible state to drop from")

        working, copied = _copy_state(before)
        mode = str(params.get("mode", "remove"))
        patterns = list(params.get("keys") or [])
        removed: list[str] = []

        for pattern in patterns:
            for path in _expand(working, pattern):
                found = _parent(working, path)
                if found is None:
                    continue
                container, leaf = found
                if mode == "null":
                    container[int(leaf) if isinstance(container, list) else leaf] = None
                elif isinstance(container, list):
                    del container[int(leaf)]
                else:
                    container.pop(leaf, None)
                removed.append(path)

        log = MutationLog.of(before, working)
        log.mutation_skipped_uncopyable = not copied
        verb = "nulled" if mode == "null" else "dropped"
        return FaultOutcome(
            action="replace_state",
            value=working,
            note=f"{verb} state key(s) {removed or patterns} before {crossing.name!r}",
            mutation=log,
            params={"mode": mode, "keys_removed": removed},
        )


@register_fault
class StateTypeFault(Fault):
    """Retype a state value using the shared mutation registry.

    **What agent weakness it proves:** unvalidated state schema, and reducer
    assumptions -- a list reducer such as `add_messages` receiving a bare string is
    the classic LangGraph break.

    **What graceful behaviour looks like:** validate the slice's types at the node
    boundary rather than trusting whatever the previous node wrote.

    Expected failure modes: `crash_unhandled_exception`,
    `state_corruption_propagated`.
    """

    kind: ClassVar[str] = "StateTypeFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = _STATE_PAIRS
    severity_hint: ClassVar[Severity] = "high"

    def validate(self) -> None:
        """Check the mutation name.

        Raises:
            ConfigError: When the mutation is unknown.
        """
        mutation_type = str(self.params().get("mutation_type", "type_flip"))
        if mutation_type not in MUTATIONS:
            known = ", ".join(sorted(MUTATIONS))
            raise ConfigError(
                f"{self.kind}.mutation_type {mutation_type!r} is not a mutation; one of: {known}"
            )

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Apply the mutation to the selected state keys.

        Args:
            crossing: The crossing carrying the state.
            ctx: The fault context.

        Returns:
            A `replace_state` outcome.
        """
        params = self.params()
        before = _state_of(crossing)
        if before is None:
            return FaultOutcome(action="noop", note="no visible state to retype")

        working, copied = _copy_state(before)
        mutation_type = str(params.get("mutation_type", "type_flip"))
        mutate = MUTATIONS[mutation_type]
        touched: list[str] = []

        for pattern in list(params.get("keys") or []):
            for path in _expand(working, pattern):
                found = _parent(working, path)
                if found is None:
                    continue
                container, leaf = found
                index: Any = int(leaf) if isinstance(container, list) else leaf
                current = container[index]
                # Keyed mutations expect a mapping, so the value is wrapped and
                # unwrapped. Whole-payload mutations ignore `keys` and replace the
                # object outright -- wrapping those would hand back the wrapper.
                if mutation_type in _WHOLE_PAYLOAD_MUTATIONS:
                    container[index] = mutate(current, ctx.rng(mutation_type))
                else:
                    wrapped = mutate({"v": current}, ctx.rng(mutation_type), keys=["v"])
                    container[index] = wrapped["v"] if isinstance(wrapped, dict) else wrapped
                touched.append(path)

        log = MutationLog.of(before, working)
        log.mutation_skipped_uncopyable = not copied
        return FaultOutcome(
            action="replace_state",
            value=working,
            note=(
                f"applied {mutation_type} to state key(s) "
                f"{touched or list(params.get('keys') or [])}"
            ),
            mutation=log,
            params={"mutation_type": mutation_type, "keys_retyped": touched},
        )


@register_fault
class StateStaleFault(Fault):
    """Revert selected state keys to an earlier snapshot.

    Simulates a lost update or a racing branch overwriting fresher work.

    **What agent weakness it proves:** no version or idempotency guard around state
    updates, so a node's progress can silently vanish.

    **What graceful behaviour looks like:** version the state slice, or make the
    update idempotent so a replay is harmless.

    Expected failure modes: `infinite_loop`, `state_corruption_propagated`.
    """

    kind: ClassVar[str] = "StateStaleFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("state", "pre")})
    severity_hint: ClassVar[Severity] = "medium"

    def __init__(self, **params: Any) -> None:
        """Initialise, and set up the per-instance snapshot history.

        Args:
            **params: `keys`, `revert_versions`.
        """
        self._history: list[dict[str, Any]] = []
        super().__init__(**params)

    def validate(self) -> None:
        """Check the revert depth.

        Raises:
            ConfigError: With the offending value.
        """
        if int(self.params().get("revert_versions", 1)) < 1:
            raise ConfigError(f"{self.kind}.revert_versions must be >= 1")

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Serve an earlier value for the selected keys.

        Args:
            crossing: The crossing carrying the state.
            ctx: The fault context.

        Returns:
            A `replace_state` outcome, or `noop` on the first crossing -- there is no
            earlier version to revert to yet.
        """
        params = self.params()
        before = _state_of(crossing)
        if before is None:
            return FaultOutcome(action="noop", note="no visible state to revert")

        keys = list(params.get("keys") or [])
        snapshot = {key: copy.deepcopy(_read(before, key)) for key in keys}
        depth = int(params.get("revert_versions", 1))

        if len(self._history) < depth:
            self._history.append(snapshot)
            return FaultOutcome(action="noop", note="no earlier state version to revert to yet")

        older = self._history[-depth]
        self._history.append(snapshot)
        working, copied = _copy_state(before)
        for key, value in older.items():
            found = _parent(working, key)
            if found is not None:
                container, leaf = found
                container[int(leaf) if isinstance(container, list) else leaf] = value

        log = MutationLog.of(before, working)
        log.mutation_skipped_uncopyable = not copied
        return FaultOutcome(
            action="replace_state",
            value=working,
            note=f"reverted state key(s) {keys} by {depth} version(s)",
            mutation=log,
            params={"revert_versions": depth, "keys": keys},
        )


def _read(state: Any, path: str) -> Any:
    """Read a dotted path from a state object.

    Args:
        state: The state.
        path: A concrete dotted path.

    Returns:
        The value, or `None` when unreachable.
    """
    found = _parent(state, path)
    if found is None:
        return None
    container, leaf = found
    return container[int(leaf)] if isinstance(container, list) else container[leaf]


@register_fault
class NodeSkipFault(Fault):
    """Prevent a node from executing; the graph proceeds as if it had.

    **What agent weakness it proves:** implicit ordering assumptions, and nodes with
    no precondition checks on what their predecessors were supposed to produce.

    **What graceful behaviour looks like:** each node asserts its inputs are present
    before using them, rather than trusting the graph's shape.

    Expected failure modes: `state_corruption_propagated`, `silent_wrong_answer`.
    """

    kind: ClassVar[str] = "NodeSkipFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("node", "pre")})
    severity_hint: ClassVar[Severity] = "high"

    def validate(self) -> None:
        """Check the node name and the return mode.

        Raises:
            ConfigError: With the offending value.
        """
        params = self.params()
        mode = str(params.get("return_state", "unchanged"))
        if mode not in {"unchanged", "partial"}:
            raise ConfigError(
                f"{self.kind}.return_state must be 'unchanged' or 'partial'; got {mode!r}"
            )

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Replace the node's execution with an empty or partial update.

        Args:
            crossing: The `(node, pre)` crossing.
            ctx: The fault context.

        Returns:
            A `replace_result` outcome standing in for the node's return value, or
            `noop` when the crossing is a different node.
        """
        params = self.params()
        target = params.get("node")
        if target and crossing.name != str(target):
            return FaultOutcome(action="noop", note=f"not the targeted node ({target!r})")

        mode = str(params.get("return_state", "unchanged"))
        update: dict[str, Any] = {}
        if mode == "partial":
            state = _state_of(crossing) or {}
            for key in list(params.get("partial_keys") or []):
                value = _read(state, key)
                if value is not None:
                    update[key] = value

        return FaultOutcome(
            action="replace_result",
            value=update,
            note=f"skipped node {crossing.name!r}; the graph proceeds as if it had run",
            mutation=MutationLog.of(None, update),
            params={"skipped_node": crossing.name, "return_state": mode},
        )


@register_fault
class EdgeMisrouteFault(Fault):
    """Override a conditional edge's routing decision.

    **What agent weakness it proves:** nodes are reachable in states they do not
    expect, and the guards that would catch it are missing.

    **What graceful behaviour looks like:** every node validates the state it was
    entered with, whichever edge delivered it.

    Expected failure modes: `crash_unhandled_exception`, `infinite_loop`.
    """

    kind: ClassVar[str] = "EdgeMisrouteFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("edge", "pre")})
    severity_hint: ClassVar[Severity] = "high"

    def __init__(self, **params: Any) -> None:
        """Initialise, and set up the per-instance misroute counter.

        Args:
            **params: `from_node`, `force_to`, `times`.
        """
        self._used = 0
        super().__init__(**params)

    def validate(self) -> None:
        """Check the destination and the budget.

        Raises:
            ConfigError: When `force_to` is missing or `times` is below one.
        """
        params = self.params()
        if not params.get("force_to"):
            raise ConfigError(f"{self.kind} needs a `force_to` destination node")
        if int(params.get("times", 1)) < 1:
            raise ConfigError(f"{self.kind}.times must be >= 1")

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Force the routing decision to a different node.

        Args:
            crossing: The `(edge, pre)` crossing, whose `result` is the chosen node.
            ctx: The fault context.

        Returns:
            A `replace_result` outcome carrying the forced destination, or `noop`
            when this is a different edge or the budget is spent. Forcing the same
            edge forever would guarantee a loop rather than test for one.
        """
        params = self.params()
        source = params.get("from_node")
        if source and crossing.name != str(source):
            return FaultOutcome(action="noop", note=f"not the targeted edge ({source!r})")
        if self._used >= int(params.get("times", 1)):
            return FaultOutcome(action="noop", note="misroute budget spent")

        self._used += 1
        destination = str(params["force_to"])
        return FaultOutcome(
            action="replace_result",
            value=destination,
            note=(
                f"misrouted the edge out of {crossing.name!r} to {destination!r} "
                f"instead of {crossing.result!r}"
            ),
            mutation=MutationLog.of(crossing.result, destination),
            params={"from_node": crossing.name, "force_to": destination},
        )


@register_fault
class CheckpointRollbackFault(Fault):
    """Restore an older checkpoint after a node commits, so work is replayed.

    **This performs a real operation the agent never requested.** Replaying a
    committed node re-runs its side effects, which is why the D-23 gate covers it.
    The **adapter** performs the resume via the `resume_from_checkpoint` action
    (D-10): `apply()` cannot, and for an async node could not await.

    **What agent weakness it proves:** non-idempotent nodes, duplicated side effects
    on resume, and the resume path that otherwise only ever runs in production.

    **What graceful behaviour looks like:** nodes are idempotent, or they carry a
    key that makes a replay a no-op.

    Expected failure modes: `duplicate_side_effect`, `infinite_loop`. Without a
    checkpointer the fault records `no_checkpointer` and does not fail the run.
    """

    kind: ClassVar[str] = "CheckpointRollbackFault"
    # `(node, post)` is where the replay is actually performable: LangGraph calls the
    # checkpointer's `put` *after* the node has returned, so a same-node replay cannot
    # be driven from there without re-entering the graph. `route_node` still holds the
    # node's function and the state it entered with -- which is exactly a checkpoint --
    # so restoring that state and calling the function again is the rollback (D-111).
    # `(checkpoint, post)` stays: it is where a true multi-step resume belongs when an
    # adapter can perform one.
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset(
        {("checkpoint", "post"), ("node", "post")}
    )
    severity_hint: ClassVar[Severity] = "critical"
    performs_real_action: ClassVar[bool] = True

    def validate(self) -> None:
        """Check the rollback depth and the budget.

        Raises:
            ConfigError: With the offending value.
        """
        params = self.params()
        for name in ("rollback_steps", "times"):
            if name in params and int(params[name]) < 1:
                raise ConfigError(f"{self.kind}.{name} must be >= 1; got {params[name]!r}")

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Ask the adapter to resume from an earlier checkpoint.

        Args:
            crossing: The `(checkpoint, post)` crossing.
            ctx: The fault context.

        Returns:
            A `resume_from_checkpoint` outcome, or `noop` when the graph has no
            checkpointer -- an absent checkpointer is a configuration fact, not a
            failure, so it must not fail the run.
        """
        params = self.params()
        if params.get("requires_checkpointer") and not crossing.state:
            return FaultOutcome(action="noop", note="no_checkpointer")

        steps = int(params.get("rollback_steps", 1))
        return FaultOutcome(
            action="resume_from_checkpoint",
            note=f"rolled back {steps} checkpoint(s) after {crossing.name!r} committed",
            params={"rollback_steps": steps, "times": int(params.get("times", 1))},
        )
