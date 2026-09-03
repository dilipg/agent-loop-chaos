"""Run-scoped context objects.

`Crossing` is the keystone abstraction: every interception point — tool, LLM, graph
node, edge, state key, checkpoint — produces one, which is why the six layers share
a single engine instead of needing six (`docs/01-ARCHITECTURE.md` §4).

`RunContext` and `FaultContext` (D-01) arrive in M1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["Crossing", "Layer", "Limits", "Phase"]

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

    Exceeding any of these raises `LimitExceeded` at the next crossing. Timeouts are
    cooperative for the same reason (D-08): there is no portable way to interrupt a
    blocking user callable, so a deadline is compared at crossings rather than
    enforced by a signal.
    """

    max_steps: int = 25
    max_tool_calls: int = 50
    max_llm_calls: int = 25
    timeout_s: float = 120.0
    max_tokens: int | None = None
    max_injected_delay_ms: int = 2000


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
