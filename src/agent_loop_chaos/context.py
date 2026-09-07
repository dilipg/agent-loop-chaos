"""Run-scoped context objects.

`Crossing` is the keystone: every interception point — tool, LLM, graph node, edge,
state key, checkpoint — produces one, which is why the six layers share a single
engine instead of needing six (`docs/01-ARCHITECTURE.md` §4).

`FaultContext` is specified by `docs/DECISIONS.md` D-01 (amended by D-53); phases
02/03/05 may not add fields to it without amending D-01 first.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .seeding import rng as _rng

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance, types only
    from .trace import TraceRecorder

__all__ = [
    "BaselineRef",
    "Counters",
    "Crossing",
    "FaultContext",
    "Layer",
    "Limits",
    "Phase",
    "RunContext",
    "StateView",
    "ToolInfo",
]

Layer = Literal["tool", "llm", "state", "node", "edge", "checkpoint"]
"""Which interception point produced a crossing."""

Phase = Literal["pre", "post", "error"]
"""When in the call it was produced: before, after, or on exception."""


@dataclass(frozen=True, slots=True)
class Limits:
    """Run guard rails, checked at every crossing.

    Defaults are mutually consistent under D-05's definition of a step: one agent
    iteration — a node entry under LangGraph, one LLM call under vanilla. Tool calls
    increment `max_tool_calls`, not `max_steps`.

    Timeouts are cooperative (D-08): there is no portable way to interrupt a
    blocking user callable, so a deadline is compared at crossings. `arun`
    additionally wraps the agent in `asyncio.wait_for`, which does cancel.
    """

    max_steps: int = 25
    max_tool_calls: int = 50
    max_llm_calls: int = 25
    timeout_s: float = 120.0
    max_tokens: int | None = None
    max_injected_delay_ms: int = 2000


@dataclass(slots=True)
class Counters:
    """Mutable per-run tallies.

    Attributes:
        steps: Agent iterations so far (D-05).
        tool_calls: Total tool invocations.
        llm_calls: Total LLM invocations.
        retries: Observed retries.
        injected_delay_ms: Delay the harness itself added, subtracted from any
            latency comparison (R3).
        injected_tokens: Tokens the harness itself added (D-43, R3).
        fires: Fire count per `fault_id`.
        last_fire_call: Per `fault_id`, the target's `call_index` at its last fire.
            `Trigger.cooldown_calls` is measured against this.
        calls_by_name: Call count per crossing name, for `call_index`.
    """

    steps: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    retries: int = 0
    injected_delay_ms: int = 0
    injected_tokens: int = 0
    fires: dict[str, int] = field(default_factory=dict)
    last_fire_call: dict[str, int] = field(default_factory=dict)
    calls_by_name: dict[str, int] = field(default_factory=dict)

    def next_call_index(self, name: str) -> int:
        """Increment and return the 1-based call index for `name`.

        Args:
            name: The crossing name.

        Returns:
            How many times this name has now been called.
        """
        self.calls_by_name[name] = self.calls_by_name.get(name, 0) + 1
        return self.calls_by_name[name]


@dataclass(frozen=True, slots=True)
class ToolInfo:
    """What the engine knows about a registered tool.

    Attributes:
        name: The tool's registered name.
        schema: JSON Schema for its arguments, if supplied.
        side_effecting: Whether it performs a real action. Tri-state on purpose:
            `True` and `False` are both deliberate declarations, and `None` means
            the user never said. `SAFETY.md` §1 item 3 needs that distinction --
            a broad preset refuses to run while any tool is undeclared, because
            silence is "unknown", not "safe". Undeclared is treated as not
            side-effecting everywhere else, so the harness does not go quiet.
        idempotency_arg: Argument that makes a repeat call safe, if any.
        is_async: Whether the underlying callable is a coroutine function.
    """

    name: str
    schema: dict[str, Any] | None = None
    side_effecting: bool | None = None
    idempotency_arg: str | None = None
    is_async: bool = False


@dataclass(frozen=True, slots=True)
class BaselineRef:
    """A pointer to a prior unfaulted run, for comparison.

    Attributes:
        run_id: The baseline run's id.
        plan_hash: Its plan hash, which proves it was the same experiment unfaulted.
        final_output_sha256: Hash of its final output.
        report_path: Where its report lives, if written.
    """

    run_id: str
    plan_hash: str
    final_output_sha256: str | None = None
    report_path: str | None = None


@dataclass(slots=True)
class Crossing:
    """One interception point, in flight.

    A fault declares which ``(layer, phase)`` pairs it accepts; the engine refuses a
    mismatched registration with `ConfigError` at register time, not run time.

    Attributes:
        layer: Which interception point this is.
        phase: ``"pre"`` exposes args, ``"post"`` exposes the result, ``"error"``
            exposes the exception.
        name: Tool name, node name, LLM alias, or state key.
        step: The engine's step counter at this crossing.
        call_index: 1-based count of calls to *this* name in this run.
        args: Positional arguments, when ``phase == "pre"``.
        kwargs: Keyword arguments, when ``phase == "pre"``.
        result: The return value, when ``phase == "post"``.
        exception: The raised exception, when ``phase == "error"``.
        state: Graph state or user-supplied state snapshot, when available.
        span_id: Counter-derived span id (``s1``, ``s2``, …); never a UUID, so the
            report stays deterministic.
        parent_span_id: Enclosing span, or `None` at the root.
        messages: Normalized LLM messages, when the payload could be normalized.
        messages_unavailable: True when an LLM payload could not be normalized, which
            disables `pre`-phase LLM faults for that call with reason
            ``unnormalizable_payload`` (`docs/06` §2.1).
        has_substitute: Set when a `pre`-phase fault supplied a result instead of
            letting the real callable run, so the engine short-circuits the call
            (D-57). This is how `ToolErrorFault(error_type="error_payload")` and
            `RateLimitFault` return an error envelope without the tool executing.
        substitute_result: The value to return in place of the call.
        replay_times: How many times to re-run the rollback window.
        rollback_steps: How many committed nodes the window covers. `1` is the
            node that just committed; `3` is it and the two before it, which is
            what a resume from a durable checkpoint actually redoes.
        invoke_times: How many times the adapter should invoke the real callable.
            `DuplicateSideEffectFault` sets this via the `invoke_target` action
            (D-10): only the adapter knows whether to `await`, so `apply()` cannot
            perform the repeat itself.
        invoke_return_from: Which of the repeated responses the agent receives.
    """

    layer: Layer
    phase: Phase
    name: str
    step: int = 0
    call_index: int = 1
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    result: Any | None = None
    exception: BaseException | None = None
    state: Any | None = None
    span_id: str = "s0"
    parent_span_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    messages_unavailable: bool = False
    has_substitute: bool = False
    substitute_result: Any = None
    #: How many times to re-run the node body from the state it entered with. Set by
    #: a `resume_from_checkpoint` action at a `(node, post)` crossing; 0 normally.
    replay_times: int = 0
    rollback_steps: int = 1
    invoke_times: int = 1
    invoke_return_from: str = "first"


def _is_field(node: Any, name: str) -> bool:
    """Whether `name` is a declared field on a model-like state object.

    Declared fields only, never arbitrary attributes: a write that invents a field
    would corrupt the state shape, and a read that reaches a method would report a
    bound method as state.

    Args:
        node: The candidate container.
        name: The field name.

    Returns:
        True when `node` declares `name` as a field.
    """
    if isinstance(node, (dict, list, str, bytes)) or node is None:
        return False
    fields = getattr(type(node), "model_fields", None)
    if isinstance(fields, dict):
        return name in fields
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return name in {f.name for f in dataclasses.fields(node)}
    return False


class StateView:
    """Get, set and delete over the agent's visible state by dotted path.

    A vanilla agent only has visible state if the caller passed a mutable mapping as
    `initial_state` and the agent uses it as its working state; otherwise state
    faults record ``fault_skipped: no_visible_state`` (`docs/06` §2.4). Graph state
    is better served by the LangGraph adapter.
    """

    def __init__(self, state: Any) -> None:
        """Wrap a state object.

        Args:
            state: The agent's visible state, usually a mutable mapping.
        """
        self._state = state

    @property
    def raw(self) -> Any:
        """The underlying state object.

        Returns:
            The wrapped state, by reference.
        """
        return self._state

    def keys(self) -> list[str]:
        """List the top-level state keys.

        Returns:
            Sorted key names, or an empty list when the state is not a mapping.
        """
        if isinstance(self._state, dict):
            return sorted(str(k) for k in self._state)
        # A Pydantic model is what `StateGraph(MyModel)` is built on and what real
        # projects use, so its fields are state keys too. Without this every state
        # fault was silently inert against such an agent -- armed, never fired, no
        # crossing to attach to (D-141).
        fields = getattr(type(self._state), "model_fields", None)
        if isinstance(fields, dict):
            return sorted(str(k) for k in fields)
        return []

    def get(self, path: str, default: Any = None) -> Any:
        """Read a dotted path.

        Args:
            path: Dotted path, e.g. ``"messages.0.content"``.
            default: Returned when the path is absent.

        Returns:
            The value at `path`, or `default`.
        """
        node: Any = self._state
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            elif _is_field(node, part):
                node = getattr(node, part)
            else:
                return default
        return node

    def set(self, path: str, value: Any) -> bool:
        """Write a dotted path.

        Args:
            path: Dotted path to write.
            value: The new value.

        Returns:
            True when the write landed, False when the path is unreachable.
        """
        parts = path.split(".")
        node = self._parent_of(parts)
        if node is None:
            return False
        last = parts[-1]
        if isinstance(node, dict):
            node[last] = value
            return True
        if isinstance(node, list) and last.isdigit() and int(last) < len(node):
            node[int(last)] = value
            return True
        if _is_field(node, last):
            try:
                setattr(node, last, value)
            except Exception:
                # A frozen or validating model refuses the write. Reporting that the
                # write did not land is the honest answer; raising into the agent over
                # the shape of its own state never is.
                return False
            return True
        return False

    def delete(self, path: str) -> bool:
        """Remove a dotted path.

        Args:
            path: Dotted path to delete.

        Returns:
            True when something was removed.
        """
        parts = path.split(".")
        node = self._parent_of(parts)
        last = parts[-1]
        if isinstance(node, dict) and last in node:
            del node[last]
            return True
        if isinstance(node, list) and last.isdigit() and int(last) < len(node):
            del node[int(last)]
            return True
        return False

    def _parent_of(self, parts: list[str]) -> Any:
        """Walk to the container holding the final path segment.

        Args:
            parts: The split dotted path.

        Returns:
            The container, or `None` when unreachable.
        """
        node: Any = self._state
        for part in parts[:-1]:
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            elif _is_field(node, part):
                node = getattr(node, part)
            else:
                return None
        return node


@dataclass(slots=True)
class RunContext:
    """Created once per run. Immutable except for the counters.

    Attributes:
        run_id: ``"run-" + sha256(f"{seed}|{scenario_id}|{plan_hash}|{attempt}")[:8]``
            (D-04). Never derived by scanning the filesystem, which would make it
            path-dependent.
        scenario_id: The scenario's id, when running under a suite.
        scenario_title: A short human name for the experiment (D-120).
        scenario_description: Why the scenario exists.
        seed: The root seed.
        started_at: ISO-8601 UTC. A timing field only — never read by decision logic.
        limits: The run's guard rails.
        trace: The recorder every event goes through.
        rng_registry: The **only** RNG cache there is (D-02). Per-run, so two runs in
            one process never share a stream.
        counters: Mutable tallies.
        attempt: Which attempt this is; part of `run_id` (D-04).
        dry_run: Faults are armed and recorded but never applied (D-12).
        canary: The planted canary for this run (D-16), exempt from redaction.
        baseline: A prior unfaulted run to compare against.
        tags: Free-form labels recorded in the report.
        deadline_mono: `time.perf_counter()` value past which the run is over.
            Compared at crossings only (D-08).
    """

    run_id: str
    seed: int
    started_at: str
    limits: Limits
    trace: TraceRecorder
    scenario_id: str | None = None
    scenario_title: str | None = None
    scenario_description: str | None = None
    rng_registry: dict[str, random.Random] = field(default_factory=dict)
    counters: Counters = field(default_factory=Counters)
    attempt: int = 1
    dry_run: bool = False
    canary: str = ""
    baseline: BaselineRef | None = None
    tags: dict[str, str] = field(default_factory=dict)
    deadline_mono: float | None = None
    span_counter: int = 0
    draws: dict[str, int] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def rng(self, purpose: str) -> random.Random:
        """Return the generator for `purpose`, creating it on first use.

        Args:
            purpose: Canonically ``f"{fault_key}:{purpose}"`` (D-02).

        Returns:
            The cached generator for this run and purpose.
        """
        stream = self.rng_registry.get(purpose)
        if stream is None:
            stream = _rng(self.seed, purpose)
            self.rng_registry[purpose] = stream
        return stream

    def next_span_id(self) -> str:
        """Allocate the next span id.

        Returns:
            ``s1``, ``s2``, … — counter-derived so the trace is reproducible.
        """
        self.span_counter += 1
        return f"s{self.span_counter}"


@dataclass(slots=True)
class FaultContext:
    """What a fault sees when it fires. Specified by D-01, amended by D-53.

    Attributes:
        fault_id: Ordinal label (``f1``, ``f2``, …) for display and cross-reference.
        fault_key: Hash of the fault spec. Keys the RNG stream, so inserting or
            reordering a fault never re-keys another one's stream (D-03, D-53).
        run: The enclosing run.
        limits: The run's guard rails.
        counters: Live tallies.
        canary: The run's planted canary.
        history: Tool name to results in call order, **post-fault** (D-18).
        pre_fault_history: The same, as the tool actually returned it.
        tool_registry: Registered tools by name.
        baseline: A prior unfaulted run, if any.
        state_view: Access to the agent's visible state, or `None`.
        objective: The scenario's inputs rendered to text, for goal faults.
    """

    fault_id: str
    fault_key: str
    run: RunContext
    limits: Limits
    counters: Counters
    canary: str = ""
    history: dict[str, list[Any]] = field(default_factory=dict)
    pre_fault_history: dict[str, list[Any]] = field(default_factory=dict)
    tool_registry: dict[str, ToolInfo] = field(default_factory=dict)
    baseline: BaselineRef | None = None
    state_view: StateView | None = None
    objective: str | None = None

    def rng(self, purpose: str) -> random.Random:
        """Return this fault's generator for `purpose`.

        Args:
            purpose: A short label such as ``"trigger"`` or ``"key_choice"``.

        Returns:
            The generator for ``f"{fault_key}:{purpose}"``, and records a draw so
            `randomness.streams` can list only purposes actually used (D-36).
        """
        key = f"{self.fault_key}:{purpose}"
        self.run.draws[key] = self.run.draws.get(key, 0) + 1
        return self.run.rng(key)

    def record(self, key: str, value: Any) -> None:
        """Record an RNG-derived choice into `randomness.decisions`.

        Only genuinely random choices belong here: an explicitly configured key is
        not a decision (D-36).

        Args:
            key: What was decided, e.g. ``"dropped_key"``.
            value: The choice made.
        """
        self.run.decisions.append({"fault_id": self.fault_id, "key": key, "value": value})
