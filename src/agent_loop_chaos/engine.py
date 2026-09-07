"""The chaos engine.

Owns a fault plan, a seeded RNG and a trace recorder. Adapters turn their framework's
hooks into `Crossing` objects and route them here; at each crossing the engine asks
the plan whether anything fires, applies what does to a copy, and records the diff.

Two invariants run through this module:

- **The observer never breaks the observed.** Any exception raised inside the
  library — a fault's `apply`, a sink, the recorder — is caught, recorded as an
  `internal_error` event with a traceback, and the original value passes through
  untouched. An engine bug must never be reported as an agent failure.
- **No clock in decision logic.** Wall-clock time appears only in timing fields and
  in the cooperative deadline (D-08). Nothing about whether a fault fires, or how a
  run is classified, depends on it.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import inspect
import json
import logging
import platform
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .adapters.vanilla import VanillaAdapter, build_wrapper, make_crossing
from .assertions import EvidenceContext, Expect, HarnessFacts, evaluate, synthesize_auto_expect
from .bundle import write_bundle
from .context import (
    BaselineRef,
    Counters,
    Crossing,
    FaultContext,
    Layer,
    Limits,
    RunContext,
    StateView,
    ToolInfo,
)
from .enums import ExpectedBehavior
from .errors import ConfigError, LimitExceeded
from .faults.base import (
    TERMINAL_ACTIONS,
    VALUE_ACTIONS,
    Fault,
    FaultOutcome,
    FaultRecord,
    MutationLog,
    fault_key_for,
)
from .intensity import DEFAULT_LEVEL, profile
from .metrics import compute_delta, compute_metrics
from .probes import ProbeContext, run_probes
from .redact import redact
from .report import ChaosResult, assemble, extract_code_context
from .seeding import canonical_json, sha256_of
from .targeting import Target, Trigger, matches, should_fire
from .trace import Event, JsonlSink, MemorySink, TraceLevel, TraceRecorder
from .version import __version__

__all__ = ["ChaosEngine"]

log = logging.getLogger("agent_loop_chaos")

# Precedence for `injected_faults[].skipped_reason`, most significant first (D-36).
_SKIP_PRECEDENCE: tuple[str, ...] = (
    "dry_run",
    "side_effecting_tool_not_named",
    "target_never_called",
    "no_checkpointer",
    "max_fires_reached",
    "cooldown",
    "probability_not_met",
    "call_index_mismatch",
    "superseded",
)


class _InjectedFailure(BaseException):
    """Carries a fault's deliberate exception out through the routing layer.

    Derives from `BaseException` on purpose. The routing layer wraps itself in
    `except Exception` so a library bug cannot be reported as an agent failure, and
    a fault's `raise` action travels through exactly that code path -- without this
    marker the guard would swallow the injected error and call the real tool anyway,
    silently disarming every `raise` fault.
    """

    def __init__(self, original: BaseException) -> None:
        """Wrap the exception the fault asked for.

        Args:
            original: The exception to deliver to the agent.
        """
        self.original = original
        super().__init__(repr(original))


_PRE_EVENT = {"tool": "tool_call_requested", "llm": "llm_request"}
_POST_EVENT = {"tool": "tool_call_returned", "llm": "llm_response"}
_ERROR_EVENT = {"tool": "tool_call_failed", "llm": "llm_response"}


@dataclass
class _ArmedFault:
    """A fault as registered: identity, plan, and live fire history."""

    fault: Fault
    record: FaultRecord
    target: Target
    trigger: Trigger

    @property
    def fault_id(self) -> str:
        """The ordinal label.

        Returns:
            ``f1``, ``f2``, …
        """
        return self.record.fault_id


@dataclass
class _RunState:
    """Everything one in-flight run holds. Scoped to a contextvar."""

    engine: ChaosEngine
    ctx: RunContext
    armed: list[_ArmedFault]
    state_view: StateView | None
    facts_values: set[str] = field(default_factory=set)
    facts_messages: list[str] = field(default_factory=list)
    faulted_seqs: set[int] = field(default_factory=set)
    harness_invocation_seqs: set[int] = field(default_factory=set)
    #: Non-zero while a harness-caused node replay is running. Everything the replayed
    #: body does is the harness's doing, so its tool calls are attributed to us (R1) --
    #: otherwise `duplicate_side_effect` fires on the *correct* agent too, since both
    #: trees call the tool twice and only the results differ.
    replay_depth: int = 0
    #: The last `_ROLLBACK_WINDOW` committed nodes as
    #: `(name, function, entry state, args, kwargs)`, oldest first. A rollback replays
    #: a slice of this; the entry state *is* the checkpoint. Recorded unconditionally,
    #: because the fault fires after the nodes it rolls back over have already run.
    node_window: list[tuple[str, Any, Any, tuple[Any, ...], dict[str, Any] | None]] = field(
        default_factory=list
    )
    harness_raised_seqs: set[int] = field(default_factory=set)
    keys_removed: set[str] = field(default_factory=set)
    keys_retyped: set[str] = field(default_factory=set)
    #: sha256 of the entrypoint's source, for `reproduce.env` (D-31).
    entrypoint_sha256: str = ""
    #: What the agent was asked, so `replay` can ask it the same thing.
    inputs: Any = None
    history: dict[str, list[Any]] = field(default_factory=dict)
    pre_fault_history: dict[str, list[Any]] = field(default_factory=dict)
    # Tool-layer results only. `history` is keyed by crossing name across every
    # layer, so it also holds the model's own responses -- and building
    # `EvidenceContext.tool_results` from it let a fabricated number source itself
    # from the sentence that fabricated it.
    tool_history: dict[str, list[Any]] = field(default_factory=dict)
    tool_records: list[dict[str, Any]] = field(default_factory=list)
    llm_records: list[dict[str, Any]] = field(default_factory=list)
    limit_hit: str | None = None
    internal_errors: int = 0
    planted_state: dict[str, Any] | None = None
    expect: Expect | None = None
    must_not: tuple[str, ...] = ()
    expected_errors: tuple[str, ...] = ()
    objective: str | None = None
    objective_state_key: str = "query"
    plan: dict[str, Any] = field(default_factory=dict)
    baseline_result: ChaosResult | None = None
    intercept_only: frozenset[str] | None = None


_ACTIVE: ContextVar[_RunState | None] = ContextVar("agent_loop_chaos_run", default=None)


def _utc_now() -> str:
    """Current UTC time, ISO-8601. A timing field only.

    Returns:
        An ISO-8601 timestamp with a ``Z`` suffix.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ChaosEngine:
    """Plans faults, instruments a target, runs it, and reports what broke."""

    def __init__(
        self,
        *,
        seed: int = 1337,
        out_dir: str | Path = ".chaos",
        trace_level: TraceLevel = "standard",
        limits: Limits | None = None,
        judge: Any | str | None = None,
        redact_keys: Sequence[str] = (),
        strict_schema: bool = True,
        dry_run: bool = False,
        write_bundle: bool = True,
        allow_remote_judge: bool = False,
        tags: Mapping[str, str] | None = None,
        allow_side_effects: Sequence[str] = (),
        strict_trace: bool = False,
        judge_options: Mapping[str, Any] | None = None,
        narrate_all: bool = False,
        intensity: int = DEFAULT_LEVEL,
    ) -> None:
        """Initialise an engine.

        Args:
            seed: Root seed. Every RNG stream derives from it (D-02).
            out_dir: Where run bundles are written. Sensitive by default (D-24).
            trace_level: How much payload detail to record.
            limits: Run guard rails. Defaults to `Limits()`.
            judge: `"rules"`, `"slm"`, `"ensemble"`, `None`, or a judge instance.
                `None` picks the ensemble when a model endpoint answers and rules
                when it does not, recording the choice in `judge_meta`.
            redact_keys: Extra key patterns on top of the default deny-list.
            strict_schema: Raise `SchemaError` when the assembled report does not
                validate, instead of recording `schema_errors[]` and continuing.
            dry_run: Arm faults but fire none, so the run classifies exactly as an
                unfaulted baseline (D-12).
            write_bundle: Write `trace.jsonl`, `plan.json` and the report to disk.
            allow_remote_judge: Opt-in required before a non-loopback judge endpoint
                may receive code context (D-22).
            tags: Free-form labels recorded in the report.
            allow_side_effects: Tools the D-23 gate may target with a real-action
                fault. Per-tool and deliberate: naming one tool never permits
                another (`SAFETY.md` §1). `run(allow_side_effects=…)` extends this
                for one run.
            strict_trace: Validate every trace event against the schema as it is
                written. Used throughout the test suite; off by default so a
                production run is not slowed by it.
            intensity: How hard this run's plan pushes, 1 (strict) to 10 (creative).
                Recorded in `plan.json` and the report. The dial scales **fault
                specs**, so a suite, a preset or `--intensity` gets scaled faults;
                a `Fault` object handed to `register_fault` directly is taken as
                given, because its parameters are already constructed.
            judge_options: Extra keyword arguments for a constructed `SLMJudge`
                (`model`, `base_url`, `transport`, `timeout_s`, …). Ignored when
                `judge` is already an instance.
            narrate_all: Make a narration call for passing runs too. Off by
                default: one judge call per failing run is the cost model
                (`docs/05` §10).
        """
        self.seed = seed
        self.out_dir = Path(out_dir)
        self.trace_level: TraceLevel = trace_level
        self.limits = limits or Limits()
        self.judge = judge
        self.redact_keys = tuple(redact_keys)
        self.strict_schema = strict_schema
        self.dry_run = dry_run
        self.write_bundle = write_bundle
        self.allow_remote_judge = allow_remote_judge
        self.tags: dict[str, str] = dict(tags or {})
        self.allow_side_effects: set[str] = set(allow_side_effects)
        self.strict_trace = strict_trace
        self.judge_options: dict[str, Any] = dict(judge_options or {})
        self.narrate_all = narrate_all
        self.intensity = profile(intensity).level
        self._judge_impl: Any | None = None

        self._faults: list[_ArmedFault] = []
        self._tools: dict[str, ToolInfo] = {}
        self._fault_counter = 0

    # ------------------------------------------------------------------ planning

    def register_fault(
        self,
        fault: Fault,
        *,
        target: Target | None = None,
        trigger: Trigger | None = None,
        fault_id: str | None = None,
        target_tool: str | None = None,
        target_node: str | None = None,
        target_llm: str | None = None,
        target_state_key: str | None = None,
        origin: str = "",
    ) -> str:
        """Add a fault to the plan and return its `fault_id`.

        Validates eagerly, so a misconfiguration surfaces here rather than
        mid-run where it could masquerade as an agent failure.

        Args:
            fault: The `Fault` instance to register.
            target: Where it applies. Mutually exclusive with the shorthands.
            trigger: When it fires. Defaults to `Trigger()`.
            fault_id: Override the default ``f{n}`` registration-order label.
            target_tool: Shorthand for ``Target(tool=...)``.
            target_node: Shorthand for ``Target(node=...)``.
            target_llm: Shorthand for ``Target(llm=...)``.
            target_state_key: Shorthand for ``Target(state_key=...)``.
            origin: Where this fault came from. Empty means a caller asked for it;
                the intensity dial stamps `intensity:<level>:<preset>` on what it
                added, so a reader can tell the two apart.

        Returns:
            The `fault_id`, for cross-referencing in the report.

        Raises:
            ConfigError: On an unknown layer for the fault's `accepts` set, an
                impossible trigger, a `probability` outside ``[0, 1]``, or both
                `target` and a shorthand.
        """
        shorthands = {
            "tool": target_tool,
            "node": target_node,
            "llm": target_llm,
            "state_key": target_state_key,
        }
        given = {k: v for k, v in shorthands.items() if v is not None}
        if target is not None and given:
            raise ConfigError(
                f"pass either `target` or a shorthand, not both; got target={target!r} "
                f"and {given!r}"
            )
        if target is None:
            target = Target(
                tool=target_tool,
                node=target_node,
                llm=target_llm,
                state_key=target_state_key,
            )

        trigger = trigger or Trigger()
        self._validate_trigger(trigger)
        self._validate_accepts(fault, target)
        self._validate_side_effect_gate(fault, target)
        # A fault may refuse a target combination only it can judge -- see
        # StateDropFault(mode='remove') at a post phase (docs/06 §1.3).
        check = getattr(fault, "check_target", None)
        if callable(check):
            check(target.phase)

        self._fault_counter += 1
        assigned = fault_id or f"f{self._fault_counter}"
        if any(a.fault_id == assigned for a in self._faults):
            raise ConfigError(f"duplicate fault_id {assigned!r}")

        target_dict = self._target_to_dict(target)
        trigger_dict = self._trigger_to_dict(trigger)
        key = fault_key_for(fault.kind, fault.params(), target_dict, trigger_dict)

        self._faults.append(
            _ArmedFault(
                fault=fault,
                target=target,
                trigger=trigger,
                record=FaultRecord(
                    fault_id=assigned,
                    fault_key=key,
                    type=fault.kind,
                    params=fault.params(),
                    target=target_dict,
                    trigger=trigger_dict,
                    origin=origin,
                ),
            )
        )
        return assigned

    @staticmethod
    def _validate_trigger(trigger: Trigger) -> None:
        """Reject impossible triggers at registration.

        Args:
            trigger: The trigger to check.

        Raises:
            ConfigError: With the offending value in the message.
        """
        if not 0.0 <= trigger.probability <= 1.0:
            raise ConfigError(f"Trigger.probability must be in [0, 1]; got {trigger.probability!r}")
        if trigger.max_fires is not None and trigger.max_fires < 1:
            raise ConfigError(f"Trigger.max_fires must be >= 1 or None; got {trigger.max_fires!r}")
        if trigger.cooldown_calls < 0:
            raise ConfigError(
                f"Trigger.cooldown_calls must be >= 0; got {trigger.cooldown_calls!r}"
            )
        indices = trigger.call_indices()
        if indices is not None:
            bad = [i for i in indices if i < 1]
            if bad:
                raise ConfigError(f"Trigger.on_call is 1-based; got {bad!r}")
        for name in ("on_step", "after_step", "stop_after_step"):
            value = getattr(trigger, name)
            if value is not None and value < 0:
                raise ConfigError(f"Trigger.{name} must be >= 0; got {value!r}")

    def _validate_side_effect_gate(self, fault: Fault, target: Target) -> None:
        """Refuse a real-action fault against a side-effecting tool (D-23).

        Args:
            fault: The fault being registered.
            target: Its target.

        Raises:
            ConfigError: When the fault performs real operations and the named tool
                is declared `side_effecting=True` without an opt-in.
        """
        if not fault.performs_real_action or target.tool is None:
            return
        if target.tool in self.allow_side_effects:
            return
        info = self._tools.get(target.tool)
        if info is not None and info.side_effecting:
            raise ConfigError(
                f"{fault.kind} performs operations the agent never requested and "
                f"tool {target.tool!r} is declared side_effecting=True. Add it to "
                f"allow_side_effects to opt in deliberately. See SAFETY.md §1."
            )

    @staticmethod
    def _validate_accepts(fault: Fault, target: Target) -> None:
        """Check the fault can handle the layer the target selects.

        Args:
            fault: The fault being registered.
            target: Its target.

        Raises:
            ConfigError: When the fault accepts no phase on that layer.
        """
        if not fault.accepts:
            raise ConfigError(f"{fault.kind} declares an empty `accepts` set")
        layer = target.implied_layer()
        if layer is None:
            return
        allowed = {pair[0] for pair in fault.accepts}
        if layer not in allowed:
            raise ConfigError(
                f"{fault.kind} does not accept layer {layer!r}; it accepts {sorted(allowed)!r}"
            )
        if target.phase is not None and (layer, target.phase) not in fault.accepts:
            pairs = sorted(f"{a}.{b}" for a, b in fault.accepts)
            raise ConfigError(
                f"{fault.kind} does not accept {layer}.{target.phase}; it accepts {pairs!r}"
            )

    @staticmethod
    def _target_to_dict(target: Target) -> dict[str, Any]:
        """Serialize a target for the plan hash and the report.

        A `predicate` is recorded as a flag, not a reference: a function object is
        neither hashable across processes nor serializable.

        Args:
            target: The target to serialize.

        Returns:
            A plain dict with unset fields omitted, so the plan hash is stable.
        """
        out = {
            "layer": target.layer,
            "tool": target.tool,
            "node": target.node,
            "llm": target.llm,
            "state_key": target.state_key,
            "phase": target.phase,
        }
        result = {k: v for k, v in out.items() if v is not None}
        if target.predicate is not None:
            # A name, not a reference: a function object is neither hashable across
            # processes nor serializable, and the schema types this as string|null.
            result["predicate"] = getattr(target.predicate, "__qualname__", "<callable>")
        return result

    @staticmethod
    def _trigger_to_dict(trigger: Trigger) -> dict[str, Any]:
        """Serialize a trigger for the plan hash and the report.

        Args:
            trigger: The trigger to serialize.

        Returns:
            A plain dict with unset fields omitted.
        """
        indices = trigger.call_indices()
        out: dict[str, Any] = {
            "on_call": list(indices) if indices is not None else None,
            "on_step": trigger.on_step,
            "after_step": trigger.after_step,
            "probability": trigger.probability,
            "max_fires": trigger.max_fires,
            "cooldown_calls": trigger.cooldown_calls or None,
            "stop_after_step": trigger.stop_after_step,
        }
        return {k: v for k, v in out.items() if v is not None}

    def clear_faults(self) -> None:
        """Drop every registered fault and reset the ordinal counter."""
        self._faults.clear()
        self._fault_counter = 0

    def plan(
        self,
        *,
        entrypoint: str | None = None,
        adapter: str = "vanilla",
        expected_behavior: ExpectedBehavior = "graceful_degradation",
        must_not: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Return the canonical plan dict that `plan_hash` is computed over.

        Contains only seed, ordered fault specs, limits, adapter, entrypoint string,
        `expected_behavior` and sorted `must_not` — nothing time- or path-dependent,
        so the hash proves plan identity and nothing else (`docs/04` §3).

        Args:
            entrypoint: A ``module:attr`` string, when known.
            adapter: Which adapter will run.
            expected_behavior: The scenario's expectation.
            must_not: Probe codes that always fail the run.

        Returns:
            The plan, ready for `canonical_json`.
        """
        plan: dict[str, Any] = {
            "seed": self.seed,
            "adapter": adapter,
            "entrypoint": entrypoint,
            "expected_behavior": expected_behavior,
            "must_not": sorted(must_not),
            "limits": {
                "max_steps": self.limits.max_steps,
                "max_tool_calls": self.limits.max_tool_calls,
                "max_llm_calls": self.limits.max_llm_calls,
                "timeout_s": self.limits.timeout_s,
                "max_tokens": self.limits.max_tokens,
                "max_injected_delay_ms": self.limits.max_injected_delay_ms,
            },
            "faults": [
                {
                    "fault_id": a.record.fault_id,
                    "fault_key": a.record.fault_key,
                    "type": a.record.type,
                    "params": a.record.params,
                    "target": a.record.target,
                    "trigger": a.record.trigger,
                }
                for a in self._faults
            ],
        }
        if self.intensity != DEFAULT_LEVEL:
            # Part of the experiment's identity, so it belongs in the hash -- but only
            # when the dial was actually turned, so every existing plan_hash stays
            # exactly where it was.
            plan["intensity"] = self.intensity
        return plan

    # ------------------------------------------------------- instrumentation API

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        side_effecting: bool | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        """Register and wrap a tool callable. Works bare or parameterized.

        Args:
            fn: The tool, when used bare as ``@engine.tool``.
            name: Tool name. Defaults to the function's name.
            side_effecting: Declare whether this tool performs a real action.
                `True` and `False` are both deliberate declarations; omitting it
                leaves the tool *undeclared*, which a broad preset refuses to run
                against (`SAFETY.md` §1 item 3). Undeclared is treated as not
                side-effecting for every other purpose.
            schema: JSON Schema for the tool's arguments.

        Returns:
            The wrapped tool, or the decorator when used parameterized. A coroutine
            function yields a coroutine wrapper.
        """

        def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
            if _is_langchain_tool(target):
                # `@tool` produces a `StructuredTool`: not callable, invoked through
                # `.invoke()`, and `inspect.signature` on it raises. It is also *the*
                # way tools are declared in a LangChain or LangGraph codebase, so
                # refusing one means refusing to instrument most of them (D-138).
                wrapped: Callable[..., Any] = self._wrap_langchain_tool(
                    target, name=name, side_effecting=side_effecting, schema=schema
                )
                return wrapped
            tool_name = name or target.__name__
            self._tools[tool_name] = ToolInfo(
                name=tool_name,
                schema=schema,
                side_effecting=side_effecting,
                is_async=inspect.iscoroutinefunction(target),
            )
            return build_wrapper(self, target, layer="tool", name=tool_name)

        if fn is None:
            return decorate
        return decorate(fn)

    def _wrap_langchain_tool(
        self,
        target: Any,
        *,
        name: str | None,
        side_effecting: bool | None,
        schema: dict[str, Any] | None,
    ) -> Any:
        """Instrument a LangChain tool in place, keeping its identity.

        The tool object is returned unchanged -- only its underlying `func` and
        `coroutine` are replaced by wrappers. A registry, a bound model or a `ToolNode`
        already holds this object; handing back a different one would leave every one
        of those calling the unwrapped function, which is the silent no-op this library
        exists to catch.

        Args:
            target: A `BaseTool` -- anything with `name` and `invoke`.
            name: Override for the crossing name. Defaults to the tool's own.
            side_effecting: Whether it performs a real action (`SAFETY.md` §1).
            schema: JSON Schema for its arguments.

        Returns:
            The same tool object, now instrumented.
        """
        tool_name = name or str(getattr(target, "name", None) or type(target).__name__)
        inner = getattr(target, "func", None)
        coroutine = getattr(target, "coroutine", None)
        self._tools[tool_name] = ToolInfo(
            name=tool_name,
            schema=schema or getattr(target, "args", None),
            side_effecting=side_effecting,
            is_async=inner is None and coroutine is not None,
        )
        if inner is not None:
            target.func = build_wrapper(self, inner, layer="tool", name=tool_name)
        if coroutine is not None:
            target.coroutine = build_wrapper(self, coroutine, layer="tool", name=tool_name)
        if inner is None and coroutine is None:  # pragma: no cover - exotic BaseTool
            raise ConfigError(
                f"cannot instrument tool {tool_name!r}: it exposes neither `func` nor "
                "`coroutine`. Wrap the function it calls with `engine.tool` instead."
            )
        return target

    def llm(
        self, fn: Callable[..., Any] | None = None, *, name: str = "default"
    ) -> Callable[..., Any]:
        """Register and wrap an LLM callable. Works bare or parameterized.

        Args:
            fn: The callable, when used bare as ``@engine.llm``.
            name: LLM alias used by targets and in the trace.

        Returns:
            The wrapped callable, or the decorator when used parameterized.
        """

        def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
            return build_wrapper(self, target, layer="llm", name=name)

        if fn is None:
            return decorate
        return decorate(fn)

    def wrap_tools(self, tools: Mapping[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
        """Wrap a name-to-callable mapping of tools.

        This is the shape a hand-rolled dispatch loop uses (`docs/06` §2.2).

        Args:
            tools: The tools to wrap.

        Returns:
            A new mapping of wrapped tools. The input is not mutated.
        """
        wrapped: dict[str, Callable[..., Any]] = {}
        for tool_name, fn in tools.items():
            self._tools.setdefault(
                tool_name,
                ToolInfo(name=tool_name, is_async=inspect.iscoroutinefunction(fn)),
            )
            wrapped[tool_name] = build_wrapper(self, fn, layer="tool", name=tool_name)
        return wrapped

    def wrap_callable(
        self,
        fn: Callable[..., Any],
        *,
        layer: Layer,
        name: str,
        side_effecting: bool | None = None,
    ) -> Callable[..., Any]:
        """Wrap an arbitrary callable as an interception point.

        Args:
            fn: The callable to wrap.
            layer: Which layer its crossings belong to.
            name: The name its crossings carry.
            side_effecting: Whether the tool performs a real action. Leaving it
                `None` is *undeclared*, not `False`: `--preset full` refuses to start
                against an undeclared tool rather than guessing which ones move money
                (D-23 item 3).

        Returns:
            The wrapped callable, preserving signature and coroutine-ness.
        """
        if layer == "tool":
            self._tools.setdefault(
                name,
                ToolInfo(
                    name=name,
                    is_async=inspect.iscoroutinefunction(fn),
                    side_effecting=side_effecting,
                ),
            )
        return build_wrapper(self, fn, layer=layer, name=name)

    def intercept_tools(self, *tool_names: str) -> Callable[..., Any]:
        """Decorate an agent entrypoint so registered tools are intercepted inside it.

        Exists for compatibility with the original blueprint API. Implemented on top
        of `build_wrapper` plus a contextvar, which is what keeps a decorated tool
        inert in production code when no run is active (`docs/06` §2.3).

        Args:
            *tool_names: Tools to intercept. Empty means every registered tool.

        Returns:
            A decorator for the agent entrypoint.
        """
        allowed = frozenset(tool_names) if tool_names else None

        def decorate(agent: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.iscoroutinefunction(agent):

                async def async_scoped(*args: Any, **kwargs: Any) -> Any:
                    with self._scoped_interception(allowed):
                        return await agent(*args, **kwargs)

                return functools_wraps(agent, async_scoped)

            def scoped(*args: Any, **kwargs: Any) -> Any:
                with self._scoped_interception(allowed):
                    return agent(*args, **kwargs)

            return functools_wraps(agent, scoped)

        return decorate

    @contextlib.contextmanager
    def _scoped_interception(self, allowed: frozenset[str] | None) -> Iterator[None]:
        """Restrict interception to `allowed` for the enclosing dynamic scope.

        Args:
            allowed: Tool names to intercept, or `None` for all.

        Yields:
            None.
        """
        state = _ACTIVE.get()
        if state is None:
            yield
            return
        previous = state.intercept_only
        state.intercept_only = allowed
        try:
            yield
        finally:
            state.intercept_only = previous

    def instrument_object(
        self,
        obj: Any,
        *,
        tools: Sequence[str] | Mapping[str, bool] = (),
        llm_methods: Sequence[str] = (),
        nodes: Sequence[str] = (),
    ) -> Any:
        """Wrap named methods on an existing instance.

        `name` is the bare method name for tools and LLM methods, and
        ``f"{Class}.{method}"`` for nodes (`docs/06` §2.5).

        Args:
            obj: The instance to instrument.
            tools: Method names to treat as tools. Pass a **mapping** of name to
                `side_effecting` to declare each one -- a bare sequence leaves them
                undeclared, which `--preset full` refuses to run against (D-23).
            llm_methods: Method names to treat as LLM calls.
            nodes: Method names to treat as graph nodes.

        Returns:
            The same instance, with the named methods replaced by wrappers.

        Raises:
            ConfigError: When a named method does not exist.
        """
        cls_name = type(obj).__name__
        effects: Mapping[str, bool] = tools if isinstance(tools, Mapping) else {}
        for names, layer in ((tools, "tool"), (llm_methods, "llm"), (nodes, "node")):
            for method_name in names:
                bound = getattr(obj, method_name, None)
                if bound is None or not callable(bound):
                    raise ConfigError(
                        f"{cls_name} has no callable attribute {method_name!r} to instrument"
                    )
                crossing_name = f"{cls_name}.{method_name}" if layer == "node" else method_name
                setattr(
                    obj,
                    method_name,
                    self.wrap_callable(
                        bound,
                        layer=layer,  # type: ignore[arg-type]
                        name=crossing_name,
                        side_effecting=effects.get(method_name),
                    ),
                )
        return obj

    # ------------------------------------------------------------ run-time state

    def require_declared_side_effects(self) -> None:
        """Refuse to proceed while any registered tool is undeclared (D-23 item 3).

        Called by broad presets and by `alc run --preset full`, which point faults
        at everything and therefore cannot afford to guess which tools move money
        or delete rows.

        Raises:
            ConfigError: Naming every undeclared tool, so the fix is mechanical.
        """
        undeclared = sorted(
            name for name, info in self._tools.items() if info.side_effecting is None
        )
        if undeclared:
            raise ConfigError(
                "a broad preset cannot run while these tools leave `side_effecting` "
                f"undeclared: {undeclared}. Declare it True or False on each -- "
                "False is a deliberate statement, silence is not (SAFETY.md §1)."
            )

    def is_active(self) -> bool:
        """Report whether a run of *this* engine is active in the current context.

        This is the check every wrapper makes first, and it is why leaving
        `@engine.tool` on production code is safe.

        Returns:
            True when a run is in flight here.
        """
        state = _ACTIVE.get()
        return state is not None and state.engine is self

    def _state(self) -> _RunState:
        """Return the active run state.

        Returns:
            The current `_RunState`.

        Raises:
            RuntimeError: When no run of this engine is active. Callers gate on
                `is_active` first, so this indicates an internal bug.
        """
        state = _ACTIVE.get()
        if state is None or state.engine is not self:
            raise RuntimeError("no active agent-loop-chaos run in this context")
        return state

    def step(self) -> int:
        """Advance and return the step counter. Called by adapters.

        A step is one agent iteration — a node entry under LangGraph, one LLM call
        under vanilla. Tool calls and crossings do not increment it (D-05).

        Returns:
            The new step number, or 0 when no run is active.
        """
        if not self.is_active():
            return 0
        state = self._state()
        state.ctx.counters.steps += 1
        self._emit(Event(kind="step_started", step=state.ctx.counters.steps, level="standard"))
        self._check_limits()
        return state.ctx.counters.steps

    def validated(self, value: Any = None, *, name: str | None = None) -> None:
        """Record positive evidence that the agent checked something.

        Optional, and absence is never read as misbehaviour — only as "no evidence
        either way" (`docs/11` §3.2). Inert outside a run.

        Args:
            value: What was validated.
            name: A label for the check.
        """
        if not self.is_active():
            return
        self._emit(
            Event(
                kind="log",
                level="standard",
                name=name or "validated",
                layer="engine",
                payload={"marker": "validated", "value": value},
            )
        )

    def note(self, message: str) -> None:
        """Record a free-text breadcrumb in the trace (`docs/11` §3.3).

        Inert outside a run.

        Args:
            message: The note.
        """
        if not self.is_active():
            return
        self._emit(
            Event(
                kind="log",
                level="standard",
                layer="engine",
                name="note",
                payload={"marker": "note", "message": message},
            )
        )

    @contextlib.contextmanager
    def bind_context(self) -> Iterator[None]:
        """Bind the current run inside a thread the agent spawned.

        Faults do not fire in threads the agent starts unless that thread enters
        this context manager (D-42), because the run state lives in a
        `contextvars.ContextVar` which a new thread does not inherit.

        Yields:
            None.
        """
        state = _ACTIVE.get()
        token = _ACTIVE.set(state)
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    # -------------------------------------------------------------- the hot path

    def _emit(self, event: Event) -> dict[str, Any] | None:
        """Record an event, swallowing any recorder failure.

        Args:
            event: The event to record.

        Returns:
            The serialized event, or `None`.
        """
        state = _ACTIVE.get()
        if state is None or state.engine is not self:
            return None
        try:
            return state.ctx.trace.emit(event)
        except AssertionError:
            raise
        except Exception:
            log.exception("trace recorder failed")
            return None

    def _internal_error(self, where: str, exc: BaseException) -> None:
        """Record a library bug as an event and keep going.

        Args:
            where: A short label for the failing call site.
            exc: The exception that escaped.
        """
        state = _ACTIVE.get()
        if state is not None:
            state.internal_errors += 1
        log.exception("internal error in %s", where)
        self._emit(
            Event(
                kind="internal_error",
                level="minimal",
                layer="engine",
                name=where,
                payload={
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                    "traceback": "".join(traceback.format_exception(exc))[-4000:],
                },
            )
        )

    def _check_limits(self) -> None:
        """Compare the run against its guard rails.

        Called at every crossing and at every step, which is what makes the timeout
        cooperative (D-08).

        Raises:
            LimitExceeded: When any limit is breached. Derives from `BaseException`
                so a broad ``except Exception`` in the agent cannot swallow it
                (D-06).
        """
        state = self._state()
        counters = state.ctx.counters
        limits = state.ctx.limits
        breach: str | None = None
        if counters.steps > limits.max_steps:
            breach = "max_steps"
        elif counters.tool_calls > limits.max_tool_calls:
            breach = "max_tool_calls"
        elif counters.llm_calls > limits.max_llm_calls:
            breach = "max_llm_calls"
        elif state.ctx.deadline_mono is not None and time.perf_counter() > state.ctx.deadline_mono:
            breach = "timeout_s"
        if breach is None:
            return
        state.limit_hit = breach
        self._emit(
            Event(kind="limit_exceeded", level="minimal", layer="engine", payload={"limit": breach})
        )
        raise LimitExceeded(breach)

    def _blocked_by_gate(self, armed: _ArmedFault, crossing: Crossing) -> bool:
        """Report whether the D-23 gate blocks this fault at this crossing.

        A wildcard target must never reach a tool declared `side_effecting=True`;
        hitting one requires naming it, or listing it in `allow_side_effects`
        (`SAFETY.md` §1 item 2). This applies to every fault, not only the three
        that perform real operations: a glob quietly reaching `delete_rows` is the
        accident the rule exists to prevent.

        Args:
            armed: The fault under consideration.
            crossing: The crossing it matched.

        Returns:
            True when the fault must be skipped here.
        """
        if crossing.layer != "tool":
            return False
        info = self._tools.get(crossing.name)
        if info is None or not info.side_effecting:
            return False
        if crossing.name in self.allow_side_effects:
            return False
        pattern = armed.target.tool
        # No tool constraint at all is broader than a glob, so it counts as one.
        return pattern is None or any(ch in pattern for ch in "*?[")

    def _armed_for(self, crossing: Crossing) -> list[_ArmedFault]:
        """Select the faults whose target matches this crossing.

        Args:
            crossing: The crossing under consideration.

        Returns:
            Matching faults in registration order.
        """
        state = self._state()
        pair = (crossing.layer, crossing.phase)
        out: list[_ArmedFault] = []
        for armed in state.armed:
            if pair not in armed.fault.accepts:
                continue
            try:
                if matches(armed.target, crossing):
                    out.append(armed)
            except Exception as exc:
                self._internal_error(f"target.match[{armed.fault_id}]", exc)
        return out

    @staticmethod
    def _note_skip(record: FaultRecord, reason: str) -> None:
        """Record the most significant skip reason seen so far (D-36).

        Args:
            record: The fault's record.
            reason: The reason from `should_fire` or the engine.
        """
        if record.fire_count:
            return
        current = record.skipped_reason
        order = {r: i for i, r in enumerate(_SKIP_PRECEDENCE)}
        if current is None or order.get(reason, 99) < order.get(current, 99):
            record.skipped_reason = reason

    def _observed_value(self, crossing: Crossing) -> Any:
        """The value a fault at this crossing operates on.

        Args:
            crossing: The crossing.

        Returns:
            The result for `post`, the normalized messages for an LLM `pre`, the
            exception for `error`, otherwise the positional arguments.
        """
        if crossing.phase == "post":
            return crossing.result
        if crossing.phase == "error":
            return crossing.exception
        if crossing.layer == "llm" and crossing.messages is not None:
            return crossing.messages
        if crossing.layer in {"state", "node"} and crossing.state is not None:
            return crossing.state
        if crossing.layer == "edge":
            return crossing.result
        return crossing.args

    def cross(self, crossing: Crossing) -> Any:
        """Route one crossing through the plan.

        Value-replacing actions **chain** in registration order — each fault
        receives the previous one's output and records its own `MutationLog` — while
        `raise`, `delay`, `invoke_target` and `resume_from_checkpoint` are
        first-wins and mark the rest ``superseded``. That asymmetry is what makes a
        multi-mutation preset exercise every mutation instead of only the first
        (D-13).

        Args:
            crossing: The crossing to evaluate.

        Returns:
            The value execution should continue with — unchanged when nothing fires.

        Raises:
            BaseException: Whatever a `raise`-action fault supplied, and
                `LimitExceeded` from the limit check.
        """
        state = self._state()
        self._check_limits()
        value = self._observed_value(crossing)

        if state.ctx.dry_run:
            for armed in self._armed_for(crossing):
                self._note_skip(armed.record, "dry_run")
            return value

        terminal: tuple[_ArmedFault, FaultOutcome] | None = None

        for armed in self._armed_for(crossing):
            if self._blocked_by_gate(armed, crossing):
                self._note_skip(armed.record, "side_effecting_tool_not_named")
                self._emit(
                    Event(
                        kind="fault_skipped",
                        level="minimal",
                        layer=crossing.layer,
                        phase=crossing.phase,
                        name=crossing.name,
                        fault_id=armed.fault_id,
                        payload={
                            "reason": "side_effecting_tool_not_named",
                            "detail": (
                                f"{crossing.name!r} is declared side_effecting=True and this "
                                "target is a wildcard; name the tool explicitly or list it in "
                                "allow_side_effects (SAFETY.md §1)"
                            ),
                        },
                    )
                )
                continue
            if terminal is not None:
                self._note_skip(armed.record, "superseded")
                self._emit(
                    Event(
                        kind="fault_skipped",
                        level="standard",
                        layer=crossing.layer,
                        phase=crossing.phase,
                        name=crossing.name,
                        fault_id=armed.fault_id,
                        payload={"reason": "superseded"},
                    )
                )
                continue

            ctx = self._fault_context(armed)
            try:
                fired, reason = should_fire(armed.trigger, crossing, ctx, armed.fault_id)
            except Exception as exc:
                self._internal_error(f"should_fire[{armed.fault_id}]", exc)
                continue

            if not fired:
                self._note_skip(armed.record, reason)
                self._emit(
                    Event(
                        kind="fault_skipped",
                        level="verbose",
                        layer=crossing.layer,
                        phase=crossing.phase,
                        name=crossing.name,
                        fault_id=armed.fault_id,
                        payload={"reason": reason},
                    )
                )
                continue

            outcome = self._apply_fault(armed, crossing, ctx)
            if outcome is None:
                continue

            # Checked here rather than where a terminal action is resolved: that
            # happens after this loop, and by then the fire is already recorded.
            unsupported = not _action_is_performable(outcome.action, crossing.layer)
            # A mutation whose patch is empty changed nothing -- its own note says so.
            # Easy to hit against a real agent: `arg_names=["tenant_id"]` finds nothing
            # to tamper with when the agent passes it positionally (D-132).
            inert = (
                outcome.mutation is not None
                and not outcome.mutation.json_patch
                and not outcome.mutation.unrepresentable
                and not outcome.mutation.mutation_skipped_uncopyable
                and not armed.fault.noop_is_a_fire
            )
            if (
                unsupported
                or inert
                or (
                    outcome.action == "noop"
                    and not outcome.delay_ms
                    and not armed.fault.noop_is_a_fire
                )
            ):
                # The fault ran and correctly decided to do nothing -- inside its rate
                # budget, at a node it does not target, on a response with no tool
                # call. Counting that as a fire inflates `suite.json`'s coverage, which
                # is read as "this fault kind was exercised", and it suppresses the
                # "no fault fired" warning for a scenario that proved nothing. D-36
                # already reserves `skipped_reason` for exactly this, and the fault's
                # own note is the reason.
                if not armed.record.fired:
                    armed.record.skipped_reason = (
                        f"the adapter cannot perform {outcome.action!r}"
                        if unsupported
                        else (outcome.note or "the fault applied no change")
                    )
                continue

            counters = state.ctx.counters
            counters.fires[armed.fault_id] = counters.fires.get(armed.fault_id, 0) + 1
            counters.last_fire_call[armed.fault_id] = crossing.call_index
            armed.record.fired = True
            armed.record.fire_count += 1
            armed.record.skipped_reason = None

            event = self._emit(
                Event(
                    kind="fault_fired",
                    level="minimal",
                    layer=crossing.layer,
                    phase=crossing.phase,
                    name=crossing.name,
                    step=crossing.step,
                    call_index=crossing.call_index,
                    span_id=crossing.span_id,
                    fault_id=armed.fault_id,
                    payload={
                        "type": armed.record.type,
                        "action": outcome.action,
                        "note": outcome.note,
                        # The user frame that is about to receive the changed value.
                        # For a crash the traceback is better evidence; for a *silent*
                        # wrong answer -- the commonest mode -- there is no traceback,
                        # and without this the work order's "where to look" section is
                        # empty for exactly the findings that most need it.
                        "caller": _user_caller(),
                    },
                    tags={"fault_key": armed.record.fault_key},
                )
            )
            seq = int(event["seq"]) if event else 0
            fire: dict[str, Any] = {
                "seq": seq,
                "step": crossing.step,
                "layer": crossing.layer,
                "phase": crossing.phase,
                "name": crossing.name,
                "call_index": crossing.call_index,
                "action": outcome.action,
                "note": outcome.note,
            }
            if outcome.params:
                # The fault's own record of what it did. For an injection this is the
                # corpus entry's `payload_id`, `detect` rule and `check` -- the only
                # thing that lets `injection_followed` attribute a follow-through to
                # the payload that caused it. Scrubbed: a fault that carries a value
                # it read from the agent carries whatever the agent had.
                fire["params"] = self._scrub(dict(outcome.params))
            if outcome.mutation is not None:
                # Redacted like every other payload. `ArgumentTamperFault` records the
                # arguments it rewrote, which for an authenticated tool means the
                # bearer token -- the third place in the report that carried one in
                # full, after `tool_calls[]` and `llm_exchanges[]` (D-131, D-136).
                fire["payload_before"] = self._scrub(outcome.mutation.payload_before)
                fire["payload_after"] = self._scrub(outcome.mutation.payload_after)
                fire["json_patch"] = self._scrub(outcome.mutation.json_patch)
                fire["unrepresentable"] = outcome.mutation.unrepresentable
            if outcome.delay_ms:
                fire["delay_ms"] = outcome.delay_ms
            armed.record.fires.append(fire)

            if outcome.mutation is not None:
                self._record_mutation(armed, crossing, outcome.mutation, seq)
                self._note_injected_values(state, outcome.mutation)

            if outcome.action in VALUE_ACTIONS:
                value = outcome.value
                # `replace_result` at a `pre` crossing means "do not call the real
                # thing, use this instead" -- the only way an in-band error envelope
                # or a 429 body can reach the agent without the tool executing
                # (D-57).
                if outcome.action == "replace_result" and crossing.phase == "pre":
                    crossing.has_substitute = True
                    crossing.substitute_result = value
                else:
                    self._rebind(crossing, value)
                state.faulted_seqs.add(seq)
            elif outcome.action in TERMINAL_ACTIONS:
                terminal = (armed, outcome)

        if terminal is not None:
            value = self._resolve_terminal(terminal[0], terminal[1], crossing, value)
        return value

    @staticmethod
    def _note_injected_values(state: _RunState, mutation: MutationLog) -> None:
        """Record the scalars a fault introduced, for rule R2.

        Without this, `values_injected` is always empty and a value the harness
        planted counts as a legitimate source. `unit_swap` is the sharp case: it
        changes the value and keeps the label, so the swapped number really is in the
        payload the agent received, and `no_unsourced_numbers` would find it sourced.

        Args:
            state: The run state, whose facts are accumulated here.
            mutation: What the fault changed.
        """
        before = set(_scalars(mutation.payload_before))
        for value in _scalars(mutation.payload_after):
            if value not in before:
                state.facts_values.add(value)
        for op in mutation.json_patch:
            path = str(op.get("path", ""))
            leaf = path.rsplit("/", 1)[-1]
            if op.get("op") == "remove" and leaf and not leaf.isdigit():
                state.keys_removed.add(leaf)

    def _rebind(self, crossing: Crossing, value: Any) -> None:
        """Feed a chained fault's output back into the crossing.

        Without this, the second fault in a chain would see the original payload
        rather than the first fault's output (D-13).

        Args:
            crossing: The crossing to update.
            value: The new value.
        """
        if crossing.layer in {"state", "node"} and crossing.phase == "pre":
            crossing.state = value
        elif crossing.phase == "post":
            crossing.result = value
        elif crossing.layer == "llm" and crossing.messages is not None:
            crossing.messages = value
        elif isinstance(value, tuple):
            crossing.args = value
        elif isinstance(value, dict):
            # `ArgumentTamperFault` mutates keyword arguments, so a mapping here is
            # the replaced kwargs rather than a substitute result.
            crossing.kwargs = value

    def _resolve_terminal(
        self,
        armed: _ArmedFault,
        outcome: FaultOutcome,
        crossing: Crossing,
        value: Any,
    ) -> Any:
        """Act on a first-wins terminal outcome.

        Args:
            armed: The fault that won.
            outcome: Its outcome.
            crossing: The crossing.
            value: The value chained so far.

        Returns:
            The value to continue with, for a non-raising terminal.

        Raises:
            BaseException: The exception a `raise` action supplied.
        """
        state = self._state()
        if outcome.action == "raise":
            error = outcome.value
            if not isinstance(error, BaseException):
                error = RuntimeError(str(error) if error is not None else armed.record.type)
            state.harness_raised_seqs.add(state.ctx.trace._seq)
            # Tag it so attribution can tell a *deliberate* injected raise from a
            # library defect. Both are raised from library frames, but only the second
            # is our bug -- and an injected raise the agent failed to handle is the
            # agent's failure, which is the whole point of injecting it.
            error._alc_injected = True
            raise _InjectedFailure(error)
        if outcome.action == "delay":
            delay_ms = min(outcome.delay_ms, state.ctx.limits.max_injected_delay_ms)
            state.ctx.counters.injected_delay_ms += delay_ms
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)
            return value
        if outcome.action == "invoke_target":
            crossing.invoke_times = max(1, int(outcome.params.get("times", 2)))
            crossing.invoke_return_from = str(outcome.params.get("return_from", "first"))
            return value
        if outcome.action == "resume_from_checkpoint" and crossing.layer == "node":
            # The node has committed and `route_node` still holds its function and the
            # state it entered with. Re-running from that state is the rollback, and it
            # re-runs the side effects -- which is the finding (D-111). `rollback_steps`
            # widens the window to the last N committed nodes, which is what a resume
            # from a durable checkpoint actually redoes (D-127).
            crossing.replay_times = max(1, int(outcome.params.get("times", 1)))
            crossing.rollback_steps = max(1, int(outcome.params.get("rollback_steps", 1)))
            return value
        # resume_from_checkpoint needs a checkpointer, so it lands with the LangGraph
        # adapter in M5 (D-10).
        self._emit(
            Event(
                kind="fault_skipped",
                level="standard",
                layer=crossing.layer,
                phase=crossing.phase,
                name=crossing.name,
                fault_id=armed.fault_id,
                payload={"reason": "action_not_supported_by_adapter", "action": outcome.action},
            )
        )
        return value

    def _record_mutation(
        self,
        armed: _ArmedFault,
        crossing: Crossing,
        mutation: MutationLog,
        seq: int,
    ) -> None:
        """Emit the `mutation_applied` event for one firing.

        Args:
            armed: The fault that fired.
            crossing: The crossing.
            mutation: What changed.
            seq: The `fault_fired` sequence number, for cross-reference.
        """
        self._emit(
            Event(
                kind="mutation_applied",
                level="standard",
                layer=crossing.layer,
                phase=crossing.phase,
                name=crossing.name,
                step=crossing.step,
                call_index=crossing.call_index,
                span_id=crossing.span_id,
                fault_id=armed.fault_id,
                payload={"fired_seq": seq, **mutation.to_dict()},
            )
        )

    def _apply_fault(
        self, armed: _ArmedFault, crossing: Crossing, ctx: FaultContext
    ) -> FaultOutcome | None:
        """Call a fault's `apply`, containing any exception it raises.

        A fault that raises is a library bug, and a library bug must never reach the
        agent under test.

        Args:
            armed: The fault to apply.
            crossing: The crossing.
            ctx: The fault context.

        Returns:
            The outcome, or `None` when `apply` raised — in which case an
            `internal_error` was recorded and the original value passes through.
        """
        try:
            # Typed as `object`: a third-party fault can return anything, and the
            # isinstance check below is the guard that keeps that from breaking a run.
            outcome: object = armed.fault.apply(crossing, ctx)
        except Exception as exc:
            self._internal_error(f"{armed.record.type}.apply[{armed.fault_id}]", exc)
            return None
        if not isinstance(outcome, FaultOutcome):
            self._internal_error(
                f"{armed.record.type}.apply[{armed.fault_id}]",
                TypeError(f"apply returned {type(outcome).__name__}, expected FaultOutcome"),
            )
            return None
        return outcome

    def _fault_context(self, armed: _ArmedFault) -> FaultContext:
        """Build the context one fault sees.

        Args:
            armed: The fault.

        Returns:
            A `FaultContext` per D-01, amended by D-53.
        """
        state = self._state()
        return FaultContext(
            fault_id=armed.fault_id,
            fault_key=armed.record.fault_key,
            run=state.ctx,
            limits=state.ctx.limits,
            counters=state.ctx.counters,
            canary=state.ctx.canary,
            history=state.history,
            pre_fault_history=state.pre_fault_history,
            tool_registry=dict(self._tools),
            baseline=state.ctx.baseline,
            state_view=state.state_view,
            objective=None,
        )

    # ---------------------------------------------------------- crossing routing

    def _should_intercept(self, layer: str, name: str) -> bool:
        """Honour an `intercept_tools` restriction.

        Args:
            layer: The crossing layer.
            name: The crossing name.

        Returns:
            True when this crossing should be routed through the plan.
        """
        state = _ACTIVE.get()
        if state is None:
            return False
        if layer != "tool" or state.intercept_only is None:
            return True
        return name in state.intercept_only

    def _open_crossing(self, layer: str, name: str) -> tuple[int, str]:
        """Count the call and allocate a span.

        A tool call increments `tool_calls`; an LLM call increments both `llm_calls`
        and `steps`, because one LLM call is one agent iteration under vanilla
        (D-05).

        Args:
            layer: The crossing layer.
            name: The crossing name.

        Returns:
            ``(call_index, span_id)``.
        """
        state = self._state()
        counters = state.ctx.counters
        call_index = counters.next_call_index(f"{layer}:{name}")
        if layer == "tool":
            counters.tool_calls += 1
        elif layer == "llm":
            counters.llm_calls += 1
            counters.steps += 1
        return call_index, state.ctx.next_span_id()

    def _replaced_args(
        self,
        crossing: Crossing,
        routed: Any,
        pre_value: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Fold a `pre`-phase result back into the call arguments.

        `cross` returns the *same object* it was given when nothing fired, so
        identity is an exact and cheap test for "did a fault change this". Guessing
        by equality instead would rewrite an untouched call, which is how a
        normalized message list ends up being passed to a callable that wanted a
        string.

        Args:
            crossing: The pre crossing.
            routed: What `cross` returned.
            pre_value: The value handed to `cross`.
            args: The original positional arguments.
            kwargs: The original keyword arguments.

        Returns:
            The arguments to invoke the real callable with, unchanged when no fault
            replaced anything.
        """
        if routed is pre_value:
            return args, kwargs
        if crossing.layer == "llm":
            payload = _denormalize(args[0] if args else None, routed)
            return ((payload, *args[1:]) if args else (payload,)), kwargs
        # `_rebind` has already written whichever half the fault replaced.
        return crossing.args, crossing.kwargs

    def route_sync(
        self,
        fn: Callable[..., Any],
        *,
        layer: str,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Route a synchronous call through pre, real call, and post or error.

        Args:
            fn: The real callable.
            layer: The crossing layer.
            name: The crossing name.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            The value the agent should see.

        Raises:
            BaseException: Whatever the real callable or a `raise` fault produced.
        """
        if not self._should_intercept(layer, name):
            return fn(*args, **kwargs)
        # The wrapper runs inside the agent's own call stack, so an exception from the
        # routing layer would otherwise be captured as the agent's `error` -- a
        # library bug reported as an agent failure, which CLAUDE.md forbids.
        try:
            call_args, call_kwargs, crossing = self._pre_phase(layer, name, args, kwargs)
        except _InjectedFailure as injected:
            raise injected.original from None
        except LimitExceeded:
            raise
        except Exception as exc:
            self._internal_error(f"route_sync.pre[{layer}:{name}]", exc)
            return fn(*args, **kwargs)
        started = time.perf_counter()
        if crossing.has_substitute:
            return self._post_phase(crossing, crossing.substitute_result, started)
        if crossing.invoke_times > 1:
            return self._invoke_repeatedly(fn, crossing, call_args, call_kwargs, started)
        try:
            result = fn(*call_args, **call_kwargs)
        except LimitExceeded:
            raise
        except BaseException as exc:
            return self._error_phase(crossing, _from_target(exc), started)
        return self._guarded_post(crossing, result, started)

    def _guarded_post(self, crossing: Crossing, result: Any, started: float) -> Any:
        """Run the post phase, containing any failure inside the library.

        Args:
            crossing: The crossing.
            result: What the real callable returned.
            started: `perf_counter` at call start.

        Returns:
            The possibly-faulted result, or the untouched result when the post phase
            itself failed.
        """
        try:
            return self._post_phase(crossing, result, started)
        except _InjectedFailure as injected:
            raise injected.original from None
        except LimitExceeded:
            raise
        except Exception as exc:
            self._internal_error(f"post_phase[{crossing.layer}:{crossing.name}]", exc)
            return result

    async def route_async(
        self,
        fn: Callable[..., Any],
        *,
        layer: str,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Async twin of `route_sync`.

        Args:
            fn: The real coroutine function.
            layer: The crossing layer.
            name: The crossing name.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            The value the agent should see.

        Raises:
            BaseException: Whatever the real callable or a `raise` fault produced.
        """
        if not self._should_intercept(layer, name):
            return await fn(*args, **kwargs)
        # Same containment as the sync path: a library bug must not become the
        # agent's error.
        try:
            call_args, call_kwargs, crossing = self._pre_phase(layer, name, args, kwargs)
        except _InjectedFailure as injected:
            raise injected.original from None
        except LimitExceeded:
            raise
        except Exception as exc:
            self._internal_error(f"route_async.pre[{layer}:{name}]", exc)
            return await fn(*args, **kwargs)
        started = time.perf_counter()
        if crossing.has_substitute:
            return self._post_phase(crossing, crossing.substitute_result, started)
        if crossing.invoke_times > 1:
            return await self._ainvoke_repeatedly(fn, crossing, call_args, call_kwargs, started)
        try:
            result = await fn(*call_args, **call_kwargs)
        except LimitExceeded:
            raise
        except BaseException as exc:
            return self._error_phase(crossing, _from_target(exc), started)
        return self._guarded_post(crossing, result, started)

    def _record_harness_invocation(self, crossing: Crossing, repeat: int) -> None:
        """Emit and attribute one invocation the harness made on its own initiative.

        Every repeat lands in `HarnessFacts.harness_invocation_seqs` so the
        `duplicate_side_effect` probe can exclude the harness's own calls (R1).
        Without that the probe would fire on every run of the fault, including
        against a perfectly idempotent agent, which would make the control
        unpassable.

        Args:
            crossing: The crossing being repeated.
            repeat: 1-based index of this invocation.
        """
        state = self._state()
        event = self._emit(
            Event(
                kind="tool_call_requested",
                level="standard",
                layer=crossing.layer,
                phase="pre",
                name=crossing.name,
                step=crossing.step,
                call_index=crossing.call_index,
                span_id=crossing.span_id,
                payload={"harness_invocation": True, "repeat": repeat},
            )
        )
        if event is not None and repeat > 1:
            state.harness_invocation_seqs.add(int(event["seq"]))

    def _pick_repeat_result(self, crossing: Crossing, results: list[Any]) -> Any:
        """Choose which repeated response the agent receives.

        Args:
            crossing: The crossing, carrying `invoke_return_from`.
            results: The responses, in invocation order.

        Returns:
            The first or the last response.
        """
        if not results:
            return None
        return results[0] if crossing.invoke_return_from == "first" else results[-1]

    def _invoke_repeatedly(
        self,
        fn: Callable[..., Any],
        crossing: Crossing,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
        started: float,
    ) -> Any:
        """Call the real tool `invoke_times` times, synchronously.

        Args:
            fn: The real callable.
            crossing: The crossing.
            call_args: Positional arguments.
            call_kwargs: Keyword arguments.
            started: `perf_counter` at call start.

        Returns:
            The chosen response.

        Raises:
            BaseException: Whatever the real callable raised.
        """
        results: list[Any] = []
        for repeat in range(1, crossing.invoke_times + 1):
            self._record_harness_invocation(crossing, repeat)
            try:
                results.append(fn(*call_args, **call_kwargs))
            except LimitExceeded:
                raise
            except BaseException as exc:
                return self._error_phase(crossing, _from_target(exc), started)
            if repeat > 1:
                # Every repeat past the first gets its own record, carrying *its* own
                # result. `_post_phase` records the chosen one, so the ledger ends up
                # with one entry per real invocation. Without this the extra calls are
                # invisible and `idempotent_effects` cannot see two effects -- which
                # is the whole finding when the agent passed no idempotency key.
                self._record_call(
                    self._state(),
                    crossing,
                    ok=True,
                    duration_ms=0.0,
                    result=results[-1],
                    error=None,
                )
        return self._post_phase(crossing, self._pick_repeat_result(crossing, results), started)

    async def _ainvoke_repeatedly(
        self,
        fn: Callable[..., Any],
        crossing: Crossing,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
        started: float,
    ) -> Any:
        """Async twin of `_invoke_repeatedly`.

        Args:
            fn: The real coroutine function.
            crossing: The crossing.
            call_args: Positional arguments.
            call_kwargs: Keyword arguments.
            started: `perf_counter` at call start.

        Returns:
            The chosen response.

        Raises:
            BaseException: Whatever the real callable raised.
        """
        results: list[Any] = []
        for repeat in range(1, crossing.invoke_times + 1):
            self._record_harness_invocation(crossing, repeat)
            try:
                results.append(await fn(*call_args, **call_kwargs))
            except LimitExceeded:
                raise
            except BaseException as exc:
                return self._error_phase(crossing, exc, started)
        return self._post_phase(crossing, self._pick_repeat_result(crossing, results), started)

    def route_node(
        self,
        fn: Callable[..., Any],
        *,
        name: str,
        state: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        intercept_state: bool = True,
    ) -> Any:
        """Route one graph node's execution through the plan.

        A node entry is one step (D-05). State faults fire here rather than at a
        separate hook, because a node boundary is the only place the whole state is
        visible and a partial update has not yet been merged.

        Args:
            fn: The node callable.
            name: The node's name.
            state: The state the node was entered with.
            args: Extra positional arguments LangGraph passed.
            kwargs: Extra keyword arguments LangGraph passed.
            intercept_state: Whether to route a state crossing too.

        Returns:
            The node's update, possibly replaced by a fault.
        """
        if not self.is_active():
            return fn(state, *args, **(kwargs or {}))
        try:
            crossing, working = self._node_pre(name, state, intercept_state=intercept_state)
        except _InjectedFailure as injected:
            raise injected.original from None
        except LimitExceeded:
            raise
        except Exception as exc:
            self._internal_error(f"route_node.pre[{name}]", exc)
            return fn(state, *args, **(kwargs or {}))

        if crossing.has_substitute:
            return self._node_post(crossing, crossing.substitute_result)
        try:
            update = fn(working, *args, **(kwargs or {}))
        except LimitExceeded:
            raise
        except BaseException as exc:
            return self._error_phase(crossing, _from_target(exc), time.perf_counter())
        result = self._node_post(crossing, update)
        self._remember_node(crossing.name, fn, working, args, kwargs)
        return self._replay_node(fn, crossing, working, args, kwargs, result)

    def _remember_node(
        self,
        name: str,
        fn: Callable[..., Any],
        entry_state: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
    ) -> None:
        """Keep what a rollback would need to replay this node.

        Bounded: a rollback window is small and a long run must not accumulate every
        node it ever entered. Kept unconditionally rather than only when a rollback
        fault is armed, because the fault fires *after* the nodes it rolls back over
        have already run.

        Args:
            name: The node's name.
            fn: Its function.
            entry_state: The state it was entered with -- this is the checkpoint.
            args: Extra positional arguments.
            kwargs: Extra keyword arguments.
        """
        state = self._state()
        state.node_window.append((name, fn, entry_state, args, kwargs))
        del state.node_window[:-_ROLLBACK_WINDOW]

    def _replay_node(
        self,
        fn: Callable[..., Any],
        crossing: Crossing,
        entry_state: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
        result: Any,
    ) -> Any:
        """Re-run a committed node from the state it entered with.

        That state is the checkpoint: restoring it and calling the node again is what
        a rollback does, and it re-runs the node's side effects -- which is the whole
        finding. The replayed invocations are attributed to the harness (R1), so
        `duplicate_side_effect` does not fire on them; what catches the agent is
        `idempotent_effects`, which asks whether two identical calls produced two
        effects (D-108).

        Args:
            fn: The node's function.
            crossing: The node crossing, carrying `replay_times`.
            entry_state: The state the node was entered with.
            args: Extra positional arguments.
            kwargs: Extra keyword arguments.
            result: What the first pass produced.

        Returns:
            The last replay's update, since a restored checkpoint means the replay is
            what actually proceeds. `result` unchanged when nothing was replayed.
        """
        if crossing.replay_times < 1:
            return result
        state = self._state()
        # The window is the last `rollback_steps` committed nodes, oldest first -- a
        # resume redoes them in the order they originally ran. Asking for more than
        # the run has produced replays what there is: a rollback cannot go behind the
        # start of the run, and refusing would be less useful than doing what it can.
        window = state.node_window[-crossing.rollback_steps :] or [
            (crossing.name, fn, entry_state, args, kwargs)
        ]
        for repeat in range(1, crossing.replay_times + 1):
            for node_name, node_fn, node_state, node_args, node_kwargs in window:
                self._emit(
                    Event(
                        kind="checkpoint_restored",
                        level="minimal",
                        layer="node",
                        name=crossing.name,
                        step=crossing.step,
                        payload={
                            "replay": repeat,
                            "of": crossing.replay_times,
                            "node": node_name,
                            "rollback_steps": crossing.rollback_steps,
                        },
                    )
                )
                self._record_harness_invocation(crossing, repeat)
                state.replay_depth += 1
                try:
                    replayed = node_fn(node_state, *node_args, **(node_kwargs or {}))
                except LimitExceeded:
                    raise
                except BaseException as exc:
                    return self._error_phase(crossing, _from_target(exc), time.perf_counter())
                finally:
                    state.replay_depth -= 1
                if node_name == crossing.name:
                    result = replayed
        state.history.setdefault(crossing.name, []).append(result)
        return result

    async def aroute_node(
        self,
        fn: Callable[..., Any],
        *,
        name: str,
        state: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        intercept_state: bool = True,
    ) -> Any:
        """Async twin of `route_node`.

        Args:
            fn: The node coroutine function.
            name: The node's name.
            state: The state the node was entered with.
            args: Extra positional arguments.
            kwargs: Extra keyword arguments.
            intercept_state: Whether to route a state crossing too.

        Returns:
            The node's update.
        """
        if not self.is_active():
            return await fn(state, *args, **(kwargs or {}))
        try:
            crossing, working = self._node_pre(name, state, intercept_state=intercept_state)
        except _InjectedFailure as injected:
            raise injected.original from None
        except LimitExceeded:
            raise
        except Exception as exc:
            self._internal_error(f"aroute_node.pre[{name}]", exc)
            return await fn(state, *args, **(kwargs or {}))

        if crossing.has_substitute:
            return self._node_post(crossing, crossing.substitute_result)
        try:
            update = await fn(working, *args, **(kwargs or {}))
        except LimitExceeded:
            raise
        except BaseException as exc:
            return self._error_phase(crossing, exc, time.perf_counter())
        return self._node_post(crossing, update)

    def _node_pre(self, name: str, state: Any, *, intercept_state: bool) -> tuple[Crossing, Any]:
        """Open a node crossing, and a state crossing alongside it.

        Args:
            name: The node's name.
            state: The state the node was entered with.
            intercept_state: Whether to route the state crossing.

        Returns:
            ``(crossing, state)`` where the state may have been replaced by a fault.
        """
        run_state = self._state()
        counters = run_state.ctx.counters
        call_index = counters.next_call_index(f"node:{name}")
        counters.steps += 1
        span = run_state.ctx.next_span_id()

        working = state
        if intercept_state:
            state_crossing = Crossing(
                layer="state",
                phase="pre",
                name=name,
                state=state,
                step=counters.steps,
                call_index=call_index,
                span_id=span,
            )
            routed = self.cross(state_crossing)
            if routed is not state:
                working = routed

        crossing = Crossing(
            layer="node",
            phase="pre",
            name=name,
            state=working,
            step=counters.steps,
            call_index=call_index,
            span_id=span,
        )
        self._emit(
            Event(
                kind="node_entered",
                level="standard",
                layer="node",
                phase="pre",
                name=name,
                step=counters.steps,
                call_index=call_index,
                span_id=span,
                payload={"reads": sorted(working) if isinstance(working, dict) else []},
            )
        )
        self.cross(crossing)
        return crossing, working

    def _node_post(self, crossing: Crossing, update: Any) -> Any:
        """Close a node crossing.

        Args:
            crossing: The pre crossing, reused with `phase="post"`.
            update: The partial update the node returned.

        Returns:
            The possibly-faulted update.
        """
        crossing.phase = "post"
        crossing.result = update
        self._emit(
            Event(
                kind="node_exited",
                level="standard",
                layer="node",
                phase="post",
                name=crossing.name,
                step=crossing.step,
                call_index=crossing.call_index,
                span_id=crossing.span_id,
                payload={"update": update},
            )
        )
        return self._guarded_post(crossing, update, time.perf_counter())

    def route_checkpoint(self, fn: Callable[[], Any], *, name: str, state: Any = None) -> Any:
        """Route a checkpoint write through the plan.

        A checkpoint crossing is `("checkpoint", "post")` only: it exists so a fault
        can act on a *committed* node, and there is nothing to roll back to before
        the write happens. The adapter calls this after the checkpointer's own `put`.

        Args:
            fn: Returns the checkpointer's result. Called only when the engine is
                inactive, since the write has already happened by the time this runs.
            name: The checkpoint's id.
            state: The checkpoint payload, so a fault can inspect it.

        Returns:
            The checkpointer's result, possibly replaced by a fault.
        """
        if not self.is_active():
            return fn()
        run_state = self._state()
        counters = run_state.ctx.counters
        crossing = Crossing(
            layer="checkpoint",
            phase="post",
            name=name,
            state=state,
            result=fn(),
            step=counters.steps,
            call_index=counters.next_call_index(f"checkpoint:{name}"),
            span_id=run_state.ctx.next_span_id(),
        )
        self._emit(
            Event(
                kind="checkpoint_written",
                level="standard",
                layer="checkpoint",
                phase="post",
                name=name,
                step=crossing.step,
                call_index=crossing.call_index,
                span_id=crossing.span_id,
                payload={"checkpoint_id": name},
            )
        )
        try:
            return self.cross(crossing)
        except _InjectedFailure as injected:
            raise injected.original from None
        except LimitExceeded:
            raise
        except Exception as exc:
            self._internal_error(f"route_checkpoint[{name}]", exc)
            return crossing.result

    async def route_checkpoint_async(
        self, fn: Callable[[], Any], *, name: str, state: Any = None
    ) -> Any:
        """Async twin of `route_checkpoint`.

        Args:
            fn: Returns the checkpointer's result.
            name: The checkpoint's id.
            state: The checkpoint payload.

        Returns:
            The checkpointer's result, possibly replaced by a fault.
        """
        return self.route_checkpoint(fn, name=name, state=state)

    def route_edge(
        self,
        fn: Callable[..., Any],
        *,
        name: str,
        state: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Route a conditional edge's routing decision through the plan.

        Args:
            fn: The router callable.
            name: The node the edge leaves from.
            state: The state the decision is made on.
            args: Extra positional arguments.
            kwargs: Extra keyword arguments.

        Returns:
            The destination, possibly overridden by `EdgeMisrouteFault`.
        """
        decision = fn(state, *args, **(kwargs or {}))
        if not self.is_active():
            return decision
        try:
            run_state = self._state()
            crossing = Crossing(
                layer="edge",
                phase="pre",
                name=name,
                result=decision,
                state=state,
                step=run_state.ctx.counters.steps,
                span_id=run_state.ctx.next_span_id(),
            )
            routed = self.cross(crossing)
            self._emit(
                Event(
                    kind="edge_taken",
                    level="standard",
                    layer="edge",
                    phase="pre",
                    name=name,
                    step=crossing.step,
                    span_id=crossing.span_id,
                    payload={"chose": routed, "would_have_chosen": decision},
                )
            )
            return routed
        except LimitExceeded:
            raise
        except Exception as exc:
            self._internal_error(f"route_edge[{name}]", exc)
            return decision

    def _pre_phase(
        self, layer: str, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any], Crossing]:
        """Emit the pre event and route the pre crossing.

        Args:
            layer: The crossing layer.
            name: The crossing name.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            ``(call_args, call_kwargs, crossing)``.
        """
        state = self._state()
        call_index, span_id = self._open_crossing(layer, name)
        crossing = make_crossing(
            state.ctx,
            layer=layer,
            phase="pre",
            name=name,
            args=args,
            kwargs=kwargs,
            call_index=call_index,
            span_id=span_id,
            state=state.state_view.raw if state.state_view else None,
        )
        payload: dict[str, Any] = {"args": list(args), "kwargs": kwargs}
        if layer == "llm":
            payload = {
                "messages": crossing.messages,
                "messages_unavailable": crossing.messages_unavailable,
            }
        self._emit(
            Event(
                kind=_PRE_EVENT.get(layer, "log"),  # type: ignore[arg-type]
                level="standard",
                layer=layer,
                phase="pre",
                name=name,
                step=crossing.step,
                call_index=call_index,
                span_id=span_id,
                payload=payload,
            )
        )
        pre_value = self._observed_value(crossing)
        routed = self.cross(crossing)
        call_args, call_kwargs = self._replaced_args(crossing, routed, pre_value, args, kwargs)
        return call_args, call_kwargs, crossing

    def _post_phase(self, crossing: Crossing, result: Any, started: float) -> Any:
        """Emit the post event and route the post crossing.

        Args:
            crossing: The pre crossing, reused with `phase="post"`.
            result: What the real callable returned.
            started: `perf_counter` at call start, for `duration_ms`.

        Returns:
            The possibly-faulted result.
        """
        state = self._state()
        state.pre_fault_history.setdefault(crossing.name, []).append(result)
        if crossing.layer == "tool":
            state.tool_history.setdefault(crossing.name, []).append(result)
            if state.replay_depth > 0:
                # Inside a harness-caused replay, so this call is ours (R1).
                state.harness_invocation_seqs.add(state.ctx.trace._seq)
        crossing.phase = "post"
        crossing.result = result
        duration = (time.perf_counter() - started) * 1000

        # The crossing runs *before* the event is emitted, so the payload records what
        # the agent actually received rather than what the callable returned. D-18
        # already says recorded results are post-fault; the trace event was the one
        # place still showing the pre-fault value, which is why
        # `truncated_output_used` could never see the `finish_reason` its own fault
        # had just set. An injected raise propagates from here and emits no "returned"
        # event, which is correct: the agent never received a result.
        final = self.cross(crossing)
        self._emit(
            Event(
                kind=_POST_EVENT.get(crossing.layer, "log"),  # type: ignore[arg-type]
                level="standard",
                layer=crossing.layer,
                phase="post",
                name=crossing.name,
                step=crossing.step,
                call_index=crossing.call_index,
                span_id=crossing.span_id,
                duration_ms=round(duration, 3),
                payload=_post_payload(crossing, final),
            )
        )
        state.history.setdefault(crossing.name, []).append(final)
        self._record_call(state, crossing, ok=True, duration_ms=duration, result=final, error=None)
        return final

    def _record_call(
        self,
        state: _RunState,
        crossing: Crossing,
        *,
        ok: bool,
        duration_ms: float,
        result: Any,
        error: str | None,
    ) -> None:
        """Append a `tool_calls[]` or `llm_exchanges[]` entry.

        The two arrays have different required shapes in the schema, so they are
        built separately rather than sharing one dict.

        Args:
            state: The run state.
            crossing: The crossing being recorded.
            ok: Whether the call succeeded.
            duration_ms: How long it took. A timing field only.
            result: The post-fault result the agent saw (D-18).
            error: The exception class name, when it failed.
        """
        faulted = bool(state.ctx.counters.fires) and any(
            fire.get("name") == crossing.name and fire.get("call_index") == crossing.call_index
            for armed in state.armed
            for fire in armed.record.fires
        )
        if crossing.layer == "tool":
            state.tool_records.append(
                {
                    "step": crossing.step,
                    "tool": crossing.name,
                    "call_index": crossing.call_index,
                    # Redacted here, not only in the trace. `report.json` and the work
                    # order rendered from it are written to be attached to tickets and
                    # handed to coding agents, and an agent under test holds real
                    # credentials -- a bearer token in a tool's keyword arguments was
                    # masked in `trace.jsonl` and printed in full beside it (D-131).
                    "args": self._scrub(list(crossing.args)),
                    "kwargs": self._scrub(dict(crossing.kwargs)),
                    "result": self._scrub(result),
                    "ok": ok,
                    "error": error,
                    "duration_ms": int(duration_ms),
                    "faulted": faulted,
                }
            )
            return
        if crossing.layer == "llm":
            messages = crossing.messages or []
            state.llm_records.append(
                {
                    "step": crossing.step,
                    "llm": crossing.name,
                    "exact_prompt": self._scrub(_render_prompt(messages)),
                    "messages": self._scrub(messages),
                    "raw_response": None if result is None else self._scrub(str(result)[:8000]),
                    "finish_reason": None if ok else "error",
                    "faulted": faulted,
                }
            )

    def _scrub(self, payload: Any) -> Any:
        """Mask known secret shapes on their way into the report.

        The trace gets this from `TraceRecorder`; the report's `tool_calls[]` and
        `llm_exchanges[]` blocks are assembled here and need it too. Same deny-list,
        same extra keys, same canary exemption -- a divergence between the two would
        be a leak in whichever one was forgotten.

        Args:
            payload: Anything about to be recorded.

        Returns:
            The payload with known secrets replaced by a labelled placeholder.
        """
        state = self._state()
        return redact(payload, self.redact_keys, allow=(state.ctx.canary,))

    def _error_phase(self, crossing: Crossing, exc: BaseException, started: float) -> Any:
        """Emit the error event and route the error crossing.

        Args:
            crossing: The pre crossing, reused with `phase="error"`.
            exc: The exception the real callable raised.
            started: `perf_counter` at call start.

        Returns:
            A substitute value, when a fault supplied one.

        Raises:
            BaseException: The original exception, when no fault handled it.
        """
        state = self._state()
        crossing.phase = "error"
        crossing.exception = exc
        self._emit(
            Event(
                kind=_ERROR_EVENT.get(crossing.layer, "log"),  # type: ignore[arg-type]
                level="standard",
                layer=crossing.layer,
                phase="error",
                name=crossing.name,
                step=crossing.step,
                call_index=crossing.call_index,
                span_id=crossing.span_id,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                payload={"error_type": type(exc).__name__, "message": str(exc)[:500]},
            )
        )
        self._record_call(
            state,
            crossing,
            ok=False,
            duration_ms=(time.perf_counter() - started) * 1000,
            result=None,
            error=type(exc).__name__,
        )
        substitute = self.cross(crossing)
        if substitute is crossing.exception or substitute is exc:
            raise exc
        return substitute

    # -------------------------------------------------------------------- running

    def _new_run(
        self,
        *,
        scenario_id: str | None,
        attempt: int,
        seed: int,
        plan_hash: str,
        dry_run: bool,
        initial_state: Mapping[str, Any] | None,
        baseline: ChaosResult | None,
    ) -> tuple[_RunState, Path]:
        """Create the run context, run directory and trace.

        The directory is created and the sink flushes per event, so a reader can
        follow the run live (`docs/01` §5 step 2).

        Args:
            scenario_id: The scenario id, used in the run path.
            attempt: Which attempt this is (D-04).
            seed: The effective seed for this run.
            plan_hash: The frozen plan's hash.
            dry_run: Whether faults are armed but never applied.
            initial_state: The caller's starting state.
            baseline: A prior unfaulted result.

        Returns:
            ``(state, run_dir)``.
        """
        run_id = (
            "run-"
            + sha256_of(f"{seed}|{scenario_id}|{plan_hash}|{attempt}").removeprefix("sha256:")[:8]
        )
        run_dir = self.out_dir / (scenario_id or "run") / run_id
        sinks: list[Any] = [MemorySink()]
        if self.write_bundle:
            run_dir.mkdir(parents=True, exist_ok=True)
            sinks.append(JsonlSink(run_dir / "trace.jsonl"))

        canary = f"ALC-CANARY-{run_id}"
        trace = TraceRecorder(
            run_id,
            level=self.trace_level,
            sinks=sinks,
            run_dir=run_dir if self.write_bundle else None,
            redact_keys=self.redact_keys,
            canary=canary,
            strict=self.strict_trace,
        )
        baseline_ref = (
            BaselineRef(run_id=baseline.run_id, plan_hash=baseline.plan_hash)
            if baseline is not None
            else None
        )
        ctx = RunContext(
            run_id=run_id,
            seed=seed,
            started_at=_utc_now(),
            limits=self.limits,
            trace=trace,
            scenario_id=scenario_id,
            counters=Counters(),
            attempt=attempt,
            dry_run=dry_run,
            canary=canary,
            baseline=baseline_ref,
            tags=dict(self.tags),
            deadline_mono=time.perf_counter() + self.limits.timeout_s,
        )
        # D-16: plant the canary somewhere the agent can actually reach. Without a
        # reachable secret, `objective: exfiltrate_secret` has nothing to exfiltrate
        # and the `secret_in_output` probe can never fire.
        planted: dict[str, Any] | None = None
        if initial_state is not None:
            # A *deep* copy. `dict(initial_state)` shares every nested list and dict
            # with the caller, so an agent that appends to a scratchpad mutates the
            # scenario itself -- and the next run, or the loop's next round, starts
            # from a state that no longer matches what the scenario declares. The
            # library must never mutate the user's objects in place.
            planted = {**copy.deepcopy(dict(initial_state)), "_alc_canary": canary}
        state = _RunState(
            engine=self,
            ctx=ctx,
            armed=list(self._faults),
            state_view=StateView(planted) if planted is not None else None,
            planted_state=planted,
        )
        return state, run_dir

    def run(
        self,
        target: Callable[..., Any] | Any,
        *,
        inputs: Any = None,
        initial_state: Mapping[str, Any] | None = None,
        scenario_id: str | None = None,
        expected_behavior: ExpectedBehavior = "graceful_degradation",
        must_not: Sequence[str] = (),
        expect: Expect | Mapping[str, Any] | None = None,
        expected_errors: Sequence[str] = (),
        allow_side_effects: Sequence[str] = (),
        dry_run: bool | None = None,
        attempt: int = 1,
        adapter: Literal["auto", "vanilla", "langgraph"] = "auto",
        baseline: ChaosResult | None = None,
        seed: int | None = None,
        scenario_title: str | None = None,
        scenario_description: str | None = None,
    ) -> ChaosResult:
        """Run an agent under the fault plan.

        Args:
            target: An agent callable, or a compiled LangGraph app (M5).
            inputs: Payload passed to the agent, resolved per D-19.
            initial_state: Starting state, tracked by the engine and exposed to state
                faults through `state_view`.
            scenario_id: Identifier recorded in the report and used in the run path.
            expected_behavior: What good behaviour would look like here.
            must_not: Probe codes that always fail the run. Evaluated in M4.
            expect: Declarative assertions (`docs/11` §4). Evaluated in M4.
            expected_errors: Exception names that count as an explicit error.
            allow_side_effects: Tools the D-23 gate may target.
            dry_run: Override the engine's `dry_run` for this run.
            attempt: Part of `run_id`; threaded by `RefinementLoop` as the round
                number and by `replay` as previous + 1 (D-04).
            adapter: Force an adapter, or sniff it.
            baseline: A prior unfaulted result to diff against.
            seed: Override the engine seed for this run.
            scenario_title: A short human name for the experiment (D-120).
            scenario_description: Why the scenario exists.

        Returns:
            The assembled `ChaosResult`.

        Raises:
            ConfigError: On a signature mismatch, an unsupported adapter, or a
                coroutine agent -- `run` is synchronous and cannot await one.
            SchemaError: When `strict_schema` and the report does not validate.
        """
        if inspect.iscoroutinefunction(target):
            # Calling it here would build a coroutine, never await it, and report a
            # verdict about a run that did not happen. Refusing is the only honest
            # answer a synchronous method has.
            raise ConfigError(
                f"{getattr(target, '__name__', target)!r} is a coroutine function; "
                "use `await engine.arun(...)` instead of `engine.run(...)`"
            )
        return self._execute(
            target,
            inputs=inputs,
            initial_state=initial_state,
            scenario_id=scenario_id,
            expected_behavior=expected_behavior,
            must_not=must_not,
            dry_run=dry_run,
            attempt=attempt,
            adapter=adapter,
            baseline=baseline,
            seed=seed,
            is_async=False,
            expect=expect,
            expected_errors=expected_errors,
            scenario_title=scenario_title,
            scenario_description=scenario_description,
        )

    async def arun(
        self,
        target: Callable[..., Any] | Any,
        *,
        inputs: Any = None,
        initial_state: Mapping[str, Any] | None = None,
        scenario_id: str | None = None,
        expected_behavior: ExpectedBehavior = "graceful_degradation",
        must_not: Sequence[str] = (),
        expect: Expect | Mapping[str, Any] | None = None,
        expected_errors: Sequence[str] = (),
        allow_side_effects: Sequence[str] = (),
        dry_run: bool | None = None,
        attempt: int = 1,
        adapter: Literal["auto", "vanilla", "langgraph"] = "auto",
        baseline: ChaosResult | None = None,
        seed: int | None = None,
        scenario_title: str | None = None,
        scenario_description: str | None = None,
    ) -> ChaosResult:
        """Async twin of `run`, with the same signature.

        Unlike the sync path, this wraps the agent in `asyncio.wait_for`, so a
        timeout genuinely cancels rather than being noticed at the next crossing
        (D-08).

        Args:
            target: An agent coroutine function, or a compiled graph.
            inputs: Payload passed to the agent.
            initial_state: Starting state.
            scenario_id: Identifier recorded in the report.
            expected_behavior: What good behaviour would look like here.
            must_not: Probe codes that always fail the run.
            expect: Declarative assertions.
            expected_errors: Exception names that count as an explicit error.
            scenario_title: A short human name for the experiment (D-120).
            scenario_description: Why the scenario exists.
            allow_side_effects: Tools the D-23 gate may target.
            dry_run: Override the engine's `dry_run` for this run.
            attempt: Part of `run_id`.
            adapter: Force an adapter, or sniff it.
            baseline: A prior unfaulted result.
            seed: Override the engine seed.

        Returns:
            The assembled `ChaosResult`.
        """
        return await self._aexecute(
            target,
            inputs=inputs,
            initial_state=initial_state,
            scenario_id=scenario_id,
            expected_behavior=expected_behavior,
            must_not=must_not,
            dry_run=dry_run,
            attempt=attempt,
            adapter=adapter,
            baseline=baseline,
            seed=seed,
            expect=expect,
            expected_errors=expected_errors,
            scenario_title=scenario_title,
            scenario_description=scenario_description,
        )

    def run_with_state(
        self,
        target: Callable[..., Any] | Any,
        *,
        query: Any = None,
        initial_state: Mapping[str, Any] | None = None,
        **kw: Any,
    ) -> ChaosResult:
        """Compatibility alias for the original blueprint call shape.

        Args:
            target: The agent callable.
            query: Passed through as the agent's `inputs`.
            initial_state: Starting state.
            **kw: Forwarded to `run`.

        Returns:
            The assembled `ChaosResult`.
        """
        return self.run(target, inputs=query, initial_state=initial_state, **kw)

    def baseline(self, target: Callable[..., Any] | Any, **kw: Any) -> ChaosResult:
        """Run with the plan disabled, for comparison.

        Args:
            target: The agent callable or graph.
            **kw: Forwarded to `run`.

        Returns:
            The unfaulted `ChaosResult`.
        """
        kw.setdefault("expected_behavior", "ignore_and_continue")
        if inspect.iscoroutinefunction(target):
            return asyncio.run(self.arun(target, dry_run=True, **kw))
        return self.run(target, dry_run=True, **kw)

    def replay(
        self, run_dir: str | Path, *, target: Callable[..., Any] | Any | None = None
    ) -> ChaosResult:
        """Rebuild the plan and seed from `plan.json` and re-run.

        Args:
            run_dir: A previous run directory.
            target: The agent to re-run.

        Returns:
            The new `ChaosResult`, with `attempt` incremented.

        Raises:
            ConfigError: When the run directory has no `plan.json`, or when the plan
                used a `Target.predicate` -- a callable cannot be serialized, so its
                plan cannot be faithfully rebuilt and guessing would be worse than
                refusing (D-31).
        """
        run_dir = Path(run_dir)
        plan_path = run_dir / "plan.json"
        if not plan_path.is_file():
            raise ConfigError(f"no plan.json in {run_dir}; nothing to replay")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

        for spec in plan.get("faults") or []:
            if (spec.get("target") or {}).get("predicate"):
                raise ConfigError(
                    f"fault {spec.get('fault_id')} used a Target.predicate, which is a "
                    "callable and is not serialized. This plan cannot be replayed "
                    "faithfully; re-run the scenario instead (D-31)."
                )

        previous: dict[str, Any] = {}
        report_path = run_dir / "report.json"
        if report_path.is_file():
            previous = json.loads(report_path.read_text(encoding="utf-8"))

        self.clear_faults()
        from .faults.base import fault_from_dict
        from .targeting import Target, Trigger

        for spec in plan.get("faults") or []:
            target_spec = dict(spec.get("target") or {})
            trigger_spec = dict(spec.get("trigger") or {})
            self.register_fault(
                fault_from_dict(dict(spec)),
                target=Target(**target_spec) if target_spec else None,
                trigger=Trigger(**trigger_spec) if trigger_spec else None,
                fault_id=spec.get("fault_id"),
            )

        if target is None:
            raise ConfigError(
                "replay needs the agent to re-run: pass `target=`, or use "
                "`alc replay <run_dir> --entrypoint module:attr`"
            )

        stored = previous.get("reproduce", {}).get("scenario_yaml")
        recorded: dict[str, Any] = {}
        if stored:
            try:
                recorded = json.loads(stored)
            except ValueError:  # pragma: no cover - hand-edited report
                recorded = {}

        result = self.run(
            target,
            inputs=recorded.get("inputs"),
            initial_state=recorded.get("initial_state"),
            scenario_id=previous.get("scenario_id") or plan.get("scenario_id"),
            expected_behavior=plan.get("expected_behavior", "graceful_degradation"),
            must_not=plan.get("must_not") or (),
            seed=int(plan.get("seed", self.seed)),
            attempt=int(previous.get("attempt", 1)) + 1,
        )
        self._check_replay_drift(result, previous, run_dir)
        return result

    def _check_replay_drift(
        self, result: ChaosResult, previous: Mapping[str, Any], run_dir: Path
    ) -> None:
        """Compare what happened against what the original run recorded (D-31).

        A matching `plan_hash` proves the plan was rebuilt, not that the experiment
        was. The agent's source, the model and the tool fixtures all sit outside the
        hash, and triggers are relative to the agent's own call sequence -- so drift
        relocates every fault while the hash still matches. The comparison is on the
        observed crossing sequence, which is what a trigger actually indexes into.

        Args:
            result: The replayed result, annotated in place.
            previous: The original `report.json`, empty when it was not kept.
            run_dir: The original run directory, for the message.
        """
        notes: list[str] = []
        env = (previous.get("reproduce") or {}).get("env") or {}
        now = (result.reproduce or {}).get("env") or {}
        for key, label in (
            ("entrypoint_source_sha256", "the agent's source"),
            ("library_version", "the library version"),
            ("adapter", "the adapter"),
        ):
            before, after = env.get(key), now.get(key)
            if before and after and before != after:
                notes.append(f"{label} changed ({before[:12]} -> {after[:12]})")

        try:
            before_seq = crossing_signature(_read_trace(run_dir / "trace.jsonl"))
            after_seq = crossing_signature(_read_trace(Path(result.artifacts.get("trace") or "")))
        except OSError:
            before_seq = after_seq = []
        if before_seq and after_seq and before_seq != after_seq:
            notes.append(
                f"the crossing sequence changed ({len(before_seq)} -> {len(after_seq)} "
                "crossings); triggers are relative to it, so the faults did not land "
                "where they did originally"
            )

        if not notes:
            return
        detail = "; ".join(notes)
        result.schema_errors.append(f"replay_divergence: {detail}")

        # The run is over and its trace is closed, so this is appended rather than
        # emitted. It is genuinely a post-hoc annotation: the comparison needs the
        # finished trace on both sides.
        trace_path = Path(result.artifacts.get("trace") or "")
        if trace_path.is_file():
            event = {
                "schema_version": "1.0",
                "seq": _next_seq(trace_path),
                "ts": _utc_now(),
                "run_id": result.run_id,
                "kind": "replay_divergence",
                "level": "minimal",
                "layer": "engine",
                "payload": {"replayed_from": str(run_dir), "notes": notes},
            }
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
        if result.artifacts.get("report"):
            Path(result.artifacts["report"]).write_text(result.to_json(), encoding="utf-8")
        log.warning("replay diverged from %s: %s", run_dir, detail)

    def _prepare(
        self,
        *,
        scenario_id: str | None,
        expected_behavior: ExpectedBehavior,
        must_not: Sequence[str],
        adapter: str,
        dry_run: bool | None,
        attempt: int,
        seed: int | None,
        initial_state: Mapping[str, Any] | None,
        baseline: ChaosResult | None,
        entrypoint: str | None,
    ) -> tuple[_RunState, Path, dict[str, Any], str]:
        """Freeze the plan and open the run (lifecycle steps 1-2).

        Args:
            scenario_id: The scenario id.
            expected_behavior: The scenario's expectation.
            must_not: Probe codes that always fail.
            adapter: Which adapter will run.
            dry_run: Per-run override.
            attempt: Which attempt this is.
            seed: Per-run seed override.
            initial_state: The caller's state.
            baseline: A prior unfaulted result.
            entrypoint: A ``module:attr`` string, when known.

        Returns:
            ``(state, run_dir, plan, plan_hash)``.
        """
        plan = self.plan(
            entrypoint=entrypoint,
            adapter=adapter,
            expected_behavior=expected_behavior,
            must_not=must_not,
        )
        plan_hash = sha256_of(canonical_json(plan))
        state, run_dir = self._new_run(
            scenario_id=scenario_id,
            attempt=attempt,
            seed=self.seed if seed is None else seed,
            plan_hash=plan_hash,
            dry_run=self.dry_run if dry_run is None else dry_run,
            initial_state=initial_state,
            baseline=baseline,
        )
        if self.write_bundle:
            (run_dir / "plan.json").write_text(
                canonical_json({**plan, "plan_hash": plan_hash, "attempt": attempt}),
                encoding="utf-8",
            )
        return state, run_dir, plan, plan_hash

    def _open(self, state: _RunState, plan_hash: str) -> None:
        """Emit `run_started` and arm every fault.

        Args:
            state: The run state.
            plan_hash: The frozen plan's hash.
        """
        self._emit(
            Event(
                kind="run_started",
                level="minimal",
                layer="engine",
                payload={
                    "plan_hash": plan_hash,
                    "seed": state.ctx.seed,
                    "attempt": state.ctx.attempt,
                    "dry_run": state.ctx.dry_run,
                    "library_version": __version__,
                },
            )
        )
        for armed in state.armed:
            self._emit(
                Event(
                    kind="fault_armed",
                    level="minimal",
                    fault_id=armed.fault_id,
                    payload={
                        "type": armed.record.type,
                        "params": armed.record.params,
                        "target": armed.record.target,
                        "trigger": armed.record.trigger,
                    },
                    tags={"fault_key": armed.record.fault_key},
                )
            )
            if state.ctx.dry_run:
                self._note_skip(armed.record, "dry_run")

    def _resolve_adapter(self, target: Any, requested: str) -> Any:
        """Choose an adapter.

        Args:
            target: The agent or graph.
            requested: ``"auto"``, ``"vanilla"`` or ``"langgraph"``.

        Returns:
            The adapter instance.

        Raises:
            ConfigError: When LangGraph is requested, which lands in M5.
        """
        looks_like_graph = hasattr(target, "invoke") and hasattr(target, "get_graph")
        if requested == "langgraph" or (requested == "auto" and looks_like_graph):
            from .adapters.langgraph import LangGraphAdapter

            return LangGraphAdapter()
        return VanillaAdapter()

    def _execute(
        self,
        target: Any,
        *,
        inputs: Any,
        initial_state: Mapping[str, Any] | None,
        scenario_id: str | None,
        expected_behavior: ExpectedBehavior,
        must_not: Sequence[str],
        dry_run: bool | None,
        attempt: int,
        adapter: str,
        baseline: ChaosResult | None,
        seed: int | None,
        is_async: bool,
        expect: Expect | Mapping[str, Any] | None = None,
        expected_errors: Sequence[str] = (),
        scenario_title: str | None = None,
        scenario_description: str | None = None,
    ) -> ChaosResult:
        """Lifecycle steps 1-7 and 14, synchronously.

        Args:
            target: The agent callable.
            inputs: The payload.
            initial_state: The caller's state.
            scenario_id: The scenario id.
            expected_behavior: The scenario's expectation.
            must_not: Probe codes that always fail.
            dry_run: Per-run override.
            attempt: Which attempt this is.
            adapter: Which adapter to use.
            baseline: A prior unfaulted result.
            seed: Per-run seed override.
            is_async: Unused; the async path is `_aexecute`.

        Returns:
            The assembled `ChaosResult`.
        """
        resolved = self._resolve_adapter(target, adapter)
        entrypoint = getattr(target, "__qualname__", None) or type(target).__name__
        state, run_dir, _plan, plan_hash = self._prepare(
            scenario_id=scenario_id,
            expected_behavior=expected_behavior,
            must_not=must_not,
            adapter=resolved.name,
            dry_run=dry_run,
            attempt=attempt,
            seed=seed,
            initial_state=initial_state,
            baseline=baseline,
            entrypoint=entrypoint,
        )
        state.entrypoint_sha256 = entrypoint_fingerprint(target)
        state.ctx.scenario_title = scenario_title
        state.ctx.scenario_description = scenario_description
        state.inputs = inputs
        state.expect = (
            expect
            if isinstance(expect, Expect)
            else (Expect(**dict(expect)) if isinstance(expect, Mapping) else None)
        )
        state.must_not = tuple(must_not)
        state.expected_errors = tuple(expected_errors)
        state.objective = _objective_text(inputs, initial_state)
        state.plan = dict(_plan)
        state.baseline_result = baseline
        token = _ACTIVE.set(state)
        started = time.perf_counter()
        output: Any = None
        error: dict[str, Any] | None = None
        try:
            self._open(state, plan_hash)
            instrumented = resolved.instrument(target, self, state.ctx)
            from .adapters.vanilla import resolve_invocation

            try:
                if resolved.name == "langgraph":
                    output = resolved.run(instrumented, inputs, state.ctx)
                else:
                    args, kwargs = resolve_invocation(instrumented, inputs, state.planted_state)
                    output = instrumented(*args, **kwargs)
            except LimitExceeded as limit:
                state.limit_hit = limit.limit
            except Exception as exc:
                # `recursion_limit` is set from `Limits.max_steps`, so a
                # GraphRecursionError is a stop we imposed. Reporting it in `error`
                # would blame the agent for our limit, exactly as D-06 forbids.
                if _is_recursion_limit(exc):
                    state.limit_hit = "max_steps"
                else:
                    error = self._error_info(exc)
        finally:
            wall_ms = int((time.perf_counter() - started) * 1000)
            result = self._finish(
                state,
                run_dir,
                plan_hash,
                resolved,
                entrypoint,
                output,
                error,
                wall_ms,
                expected_behavior,
            )
            _ACTIVE.reset(token)
        return result

    async def _aexecute(
        self,
        target: Any,
        *,
        inputs: Any,
        initial_state: Mapping[str, Any] | None,
        scenario_id: str | None,
        expected_behavior: ExpectedBehavior,
        must_not: Sequence[str],
        dry_run: bool | None,
        attempt: int,
        adapter: str,
        baseline: ChaosResult | None,
        seed: int | None,
        expect: Expect | Mapping[str, Any] | None = None,
        expected_errors: Sequence[str] = (),
        scenario_title: str | None = None,
        scenario_description: str | None = None,
    ) -> ChaosResult:
        """Lifecycle steps 1-7 and 14, asynchronously.

        Args:
            target: The agent coroutine function.
            inputs: The payload.
            initial_state: The caller's state.
            scenario_id: The scenario id.
            expected_behavior: The scenario's expectation.
            must_not: Probe codes that always fail.
            dry_run: Per-run override.
            attempt: Which attempt this is.
            adapter: Which adapter to use.
            baseline: A prior unfaulted result.
            seed: Per-run seed override.

        Returns:
            The assembled `ChaosResult`.
        """
        resolved = self._resolve_adapter(target, adapter)
        entrypoint = getattr(target, "__qualname__", None) or type(target).__name__
        state, run_dir, _plan, plan_hash = self._prepare(
            scenario_id=scenario_id,
            expected_behavior=expected_behavior,
            must_not=must_not,
            adapter=resolved.name,
            dry_run=dry_run,
            attempt=attempt,
            seed=seed,
            initial_state=initial_state,
            baseline=baseline,
            entrypoint=entrypoint,
        )
        state.entrypoint_sha256 = entrypoint_fingerprint(target)
        state.ctx.scenario_title = scenario_title
        state.ctx.scenario_description = scenario_description
        state.inputs = inputs
        state.expect = (
            expect
            if isinstance(expect, Expect)
            else (Expect(**dict(expect)) if isinstance(expect, Mapping) else None)
        )
        state.must_not = tuple(must_not)
        state.expected_errors = tuple(expected_errors)
        state.objective = _objective_text(inputs, initial_state)
        state.plan = dict(_plan)
        state.baseline_result = baseline
        token = _ACTIVE.set(state)
        started = time.perf_counter()
        output: Any = None
        error: dict[str, Any] | None = None
        try:
            self._open(state, plan_hash)
            instrumented = resolved.instrument(target, self, state.ctx)
            from .adapters.vanilla import resolve_invocation

            try:
                if resolved.name == "langgraph":
                    coroutine = resolved.arun(instrumented, inputs, state.ctx)
                else:
                    args, kwargs = resolve_invocation(instrumented, inputs, state.planted_state)
                    coroutine = self._maybe_await(instrumented(*args, **kwargs))
                output = await asyncio.wait_for(coroutine, timeout=self.limits.timeout_s)
            except LimitExceeded as limit:
                state.limit_hit = limit.limit
            except (TimeoutError, asyncio.TimeoutError):
                state.limit_hit = "timeout_s"
                self._emit(
                    Event(
                        kind="limit_exceeded",
                        level="minimal",
                        layer="engine",
                        payload={"limit": "timeout_s"},
                    )
                )
            except Exception as exc:
                # `recursion_limit` is set from `Limits.max_steps`, so a
                # GraphRecursionError is a stop we imposed. Reporting it in `error`
                # would blame the agent for our limit, exactly as D-06 forbids.
                if _is_recursion_limit(exc):
                    state.limit_hit = "max_steps"
                else:
                    error = self._error_info(exc)
        finally:
            wall_ms = int((time.perf_counter() - started) * 1000)
            result = self._finish(
                state,
                run_dir,
                plan_hash,
                resolved,
                entrypoint,
                output,
                error,
                wall_ms,
                expected_behavior,
            )
            _ACTIVE.reset(token)
        return result

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        """Await a value if it is awaitable.

        Args:
            value: A value or awaitable.

        Returns:
            The resolved value.
        """
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _error_info(exc: BaseException) -> dict[str, Any]:
        """Capture an agent exception for the report.

        Args:
            exc: The exception the agent raised.

        Returns:
            A dict with the type, message and the tail of the traceback.
        """
        text = "".join(traceback.format_exception(exc))
        # Both mean "not our bug": an injected raise is the fault working, and
        # anything out of the invocation boundary is the code under test failing.
        injected = bool(getattr(exc, "_alc_injected", False)) or bool(
            getattr(exc, "_alc_from_target", False)
        )
        return {
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
            "traceback": text[-4000:],
            # `probes.py` and `outcomes.py` both branch on this to keep a library bug
            # from being reported as an agent failure. It was read and never written,
            # so the branch was dead and every exception was blamed on the agent --
            # including ours.
            "raised_in": "agent" if injected else raised_in(text),
        }

    def checkpoint_thread_id(self) -> str:
        """A deterministic thread id for a graph compiled with a checkpointer.

        LangGraph refuses to run a checkpointed graph without a `thread_id`, and the
        engine invokes a graph as a plain callable with no `config`. Deriving one
        from the run keeps the report reproducible: a `uuid4` here would land in a
        checkpoint namespace and break byte-identical reruns (`docs/01` §7).

        Returns:
            The thread id, stable for a given seed, scenario and attempt.
        """
        state = _ACTIVE.get()
        if state is None:
            return f"alc-{self.seed}"
        return f"alc-{state.ctx.run_id}"

    def harness_facts(self) -> HarnessFacts:
        """Snapshot what the harness itself caused.

        Every probe and assertion receives this, and none may fire on what it
        records — the four attribution rules in `docs/11` §2. This is the structural
        reason a well-behaved agent passes instead of being punished for the fault it
        handled correctly.

        Returns:
            The accumulated facts, empty under `dry_run` (D-12).
        """
        state = self._state()
        if state.ctx.dry_run:
            return HarnessFacts(dry_run=True, canary=state.ctx.canary)
        return HarnessFacts(
            fired=tuple(a.record for a in state.armed if a.record.fired),
            faulted_seqs=frozenset(state.faulted_seqs),
            harness_invocation_seqs=frozenset(state.harness_invocation_seqs),
            harness_raised_seqs=frozenset(state.harness_raised_seqs),
            keys_removed=frozenset(state.keys_removed),
            keys_retyped=frozenset(state.keys_retyped),
            values_injected=frozenset(state.facts_values),
            messages_injected=tuple(state.facts_messages),
            tokens_injected=state.ctx.counters.injected_tokens,
            delay_injected_ms=state.ctx.counters.injected_delay_ms,
            canary=state.ctx.canary,
            dry_run=False,
        )

    def _finish(
        self,
        state: _RunState,
        run_dir: Path,
        plan_hash: str,
        adapter: Any,
        entrypoint: str | None,
        output: Any,
        error: dict[str, Any] | None,
        wall_ms: int,
        expected_behavior: ExpectedBehavior,
    ) -> ChaosResult:
        """Close the trace and assemble the result (lifecycle steps 6-7, then 14).

        Steps 8-13 — metrics, assertions, probes, classification, the judge and the
        bundle — land in M4. `_post_run` is the seam they attach to, and it is pure
        over `(trace, plan)` so `alc judge` can re-run them on a stored trace.

        Args:
            state: The run state.
            run_dir: The run directory.
            plan_hash: The frozen plan's hash.
            adapter: The adapter that ran.
            entrypoint: The agent's qualified name.
            output: The agent's final output.
            error: Captured agent exception, if any.
            wall_ms: Wall time, a timing field only.
            expected_behavior: The scenario's expectation.

        Returns:
            The assembled `ChaosResult`.
        """
        facts_error: dict[str, Any] | None = error
        try:
            self._emit(
                Event(
                    kind="run_finished",
                    level="minimal",
                    layer="engine",
                    payload={
                        "limit_hit": state.limit_hit,
                        "internal_errors": state.internal_errors,
                        "steps": state.ctx.counters.steps,
                    },
                )
            )
        except Exception as exc:
            self._internal_error("run_finished", exc)

        # The trace closes *after* post-run, not before it. Steps 8-13 emit
        # `internal_error` events of their own, and closing first wrote them to a
        # dead handle -- a library bug in a probe or a judge left no trace at all.
        # Probes still see only the events that existed when the agent stopped,
        # because `_post_run` snapshots the trace before emitting anything.
        try:
            return self._post_run(
                state,
                run_dir,
                plan_hash,
                adapter,
                entrypoint,
                output,
                facts_error,
                wall_ms,
                expected_behavior,
            )
        finally:
            state.ctx.trace.close()

    def _post_run(
        self,
        state: _RunState,
        run_dir: Path,
        plan_hash: str,
        adapter: Any,
        entrypoint: str | None,
        output: Any,
        error: dict[str, Any] | None,
        wall_ms: int,
        expected_behavior: ExpectedBehavior,
    ) -> ChaosResult:
        """Lifecycle steps 8-13: metrics, assertions, probes, classify, assemble, write.

        Pure over the trace and the plan, so `alc judge` can re-run it from disk.

        Args:
            state: The run state.
            run_dir: The run directory.
            plan_hash: The frozen plan's hash.
            adapter: The adapter that ran.
            entrypoint: The agent's qualified name.
            output: The agent's final output.
            error: Captured agent exception, if any.
            wall_ms: Wall time.
            expected_behavior: The scenario's expectation.

        Returns:
            The assembled result, with the bundle written when `write_bundle`.
        """
        ctx = state.ctx
        trace = list(ctx.trace.memory.events if ctx.trace.memory else [])
        records = [a.record for a in state.armed]
        fired = [r for r in records if r.fired]
        facts = self.harness_facts()

        # 8. metrics, before probes (D-11)
        metrics = compute_metrics(
            trace,
            injected_tokens=ctx.counters.injected_tokens,
            injected_delay_ms=ctx.counters.injected_delay_ms,
            wall_ms=wall_ms,
            retries=ctx.counters.retries,
            counters={
                "steps": ctx.counters.steps,
                "tool_calls": ctx.counters.tool_calls,
                "llm_calls": ctx.counters.llm_calls,
            },
        )

        # 9. assertions: the author's, plus the ones synthesized from what fired
        fires = [{**f, "type": r.type, "params": r.params} for r in fired for f in r.fires]
        evidence = EvidenceContext(
            final_output=output,
            # Tool results only. A model's own response is not a source: an answer
            # that cites itself is exactly the failure the grounding check exists to
            # catch.
            tool_results=[v for values in state.tool_history.values() for v in values],
            inputs=state.objective,
            initial_state=dict(state.planted_state or {}),
            tools_called=[
                str(e.get("name"))
                for e in trace
                if e.get("kind") == "tool_call_requested" and e.get("name")
            ],
            values_injected=facts.values_injected,
            steps=metrics["steps"],
            tool_calls=metrics["tool_calls"],
            final_state=_visible_state(state.state_view),
            errors=[str(error.get("type"))] if error else [],
            keys_removed=facts.keys_removed,
            # Per-call detail, so `idempotent_effects` can ask what a repeated
            # side-effecting call actually produced. `tool_results` is a flat list
            # with no way to tell which tool returned what, or with which arguments.
            tool_invocations=list(state.tool_records),
            tool_registry=dict(self._tools),
        )
        assertions = []
        if state.expect is not None:
            assertions.extend(evaluate(state.expect, evidence, source="scenario"))
        # A harness-injected `raise` never reaches `_error_phase`, so there is no
        # `tool_call_failed` event to key on. The fault's own fires are the record.
        failed_tools = {
            str(f.get("name")) for f in fires if f.get("action") == "raise" and f.get("name")
        }
        failed_tools |= {str(e.get("name")) for e in trace if e.get("kind") == "tool_call_failed"}
        recovered = any(
            e.get("kind") == "tool_call_returned" and str(e.get("name")) in failed_tools
            for e in trace
        )
        auto = synthesize_auto_expect(
            fires,
            max_steps=ctx.limits.max_steps,
            recovered=recovered,
            expected_behavior=expected_behavior,
        )
        # Always evaluated, even when no fault had a data effect: `output_non_empty`
        # defaults to true, and "a scenario with no expect block is not unchecked"
        # (§4.4). Gating on "did we synthesize anything" left a clean run with zero
        # assertions.
        assertions.extend(evaluate(auto, evidence, source="auto"))
        for assertion in assertions:
            self._emit(
                Event(
                    kind="assertion_result",
                    level="standard",
                    payload={
                        "check": assertion.check,
                        "ok": assertion.ok,
                        "detail": assertion.detail,
                        "source": assertion.source,
                    },
                )
            )

        # 10. probes
        # The probe reads `detect`, `check` and `payload_id` off each payload, and the
        # fault records them per fire -- one payload per fire, since a matrix scenario
        # can inject a different corpus entry each time. Passing the fire dict itself
        # buried them one level down where the probe never looked.
        injection_payloads = [
            fire["params"]
            for r in fired
            if r.type == "PromptInjectionFault"
            for fire in r.fires
            if fire.get("params")
        ]
        if not injection_payloads:
            injection_payloads = [dict(r.params) for r in fired if r.type == "PromptInjectionFault"]
        probe_ctx = ProbeContext(
            metrics=metrics,
            limits=ctx.limits,
            baseline=None,
            assertions=assertions,
            harness=facts,
            final_output=output,
            error=error,
            limit_hit=state.limit_hit,
            expected_behavior=expected_behavior,
            tool_registry=dict(self._tools),
            objective=state.objective,
            objective_state_key=state.objective_state_key,
            final_state=evidence.final_state,
            injection_payloads=[p for p in injection_payloads if p],
            expected_errors=list(state.expected_errors),
        )
        symptoms = run_probes(trace, probe_ctx)
        for symptom in symptoms:
            self._emit(
                Event(
                    kind="probe_fired",
                    level="standard",
                    payload={
                        "code": symptom.code,
                        "severity": symptom.severity,
                        "detail": symptom.detail,
                    },
                )
            )

        # 11-12. classify and assemble
        # Destructive means data the agent relied on was *changed*, not that the
        # harness added something. A patch of pure `add` ops -- an inducer planting a
        # premise, noise appended to a context -- corrupts nothing, and calling the
        # result `hallucination_on_corrupt_data` would blame data that was never
        # touched. The honest mode there is `unverified_claim_emitted` (D-118).
        destructive = any(
            f.get("action") == "raise"
            or any(op.get("op") != "add" for op in f.get("json_patch") or [])
            for f in fires
        )
        baseline_block, delta = _baseline_blocks(state.baseline_result, metrics, output, evidence)
        result = assemble(
            trace=trace,
            plan=state.plan,
            plan_hash=plan_hash,
            run_id=ctx.run_id,
            scenario_id=ctx.scenario_id,
            started_at=ctx.started_at,
            finished_at=_utc_now(),
            wall_ms=wall_ms,
            final_output=output,
            error=error,
            symptoms=symptoms,
            assertions=assertions,
            expected_behavior=expected_behavior,
            must_not=state.must_not,
            injected_faults=[r.to_dict() for r in records],
            intensity=self.intensity,
            scenario_title=ctx.scenario_title,
            scenario_description=ctx.scenario_description,
            limit_hit=state.limit_hit,
            baseline=baseline_block,
            delta=delta,
            attempt=ctx.attempt,
            dry_run=ctx.dry_run,
            tags=ctx.tags,
            destructive_mutation=destructive,
            internal_error=bool(state.internal_errors) and error is None and not output,
            expected_errors=list(state.expected_errors),
            # `recovered` is computed above for the auto-expect and is exactly what
            # the classifier needs to reach `retried_then_succeeded`. Not passing it
            # made that outcome unreachable, so a scenario expecting a retry failed
            # the agent that performed one correctly.
            retry_succeeded=recovered,
            fault_severity_hints=[a.fault.severity_hint for a in state.armed if a.record.fired],
            target={
                "framework": adapter.name,
                "entrypoint": entrypoint,
                "adapter_version": adapter.version,
                "tools": sorted(self._tools),
            },
            randomness={
                "seed": ctx.seed,
                "streams": {k: v for k, v in sorted(ctx.draws.items()) if v > 0},
                "decisions": ctx.decisions,
            },
            reproduce={
                "cmd": f"alc replay {run_dir}",
                "seed": ctx.seed,
                "plan_path": str(run_dir / "plan.json") if self.write_bundle else None,
                # What `plan_hash` cannot cover, so a replay can tell the experiment
                # drifted even when the plan matched (D-31).
                "scenario_yaml": _reproduce_scenario(state),
                "env": {
                    "library_version": __version__,
                    "python": platform.python_version(),
                    "entrypoint_source_sha256": state.entrypoint_sha256,
                    "adapter": adapter.name,
                },
            },
            artifacts={
                "report": str(run_dir / "report.json"),
                "trace": str(run_dir / "trace.jsonl"),
                "plan": str(run_dir / "plan.json") if self.write_bundle else None,
                "run_dir": str(run_dir),
            },
            strict_schema=self.strict_schema,
            metrics=metrics,
        )
        result.tool_calls = state.tool_records
        result.llm_exchanges = state.llm_records
        result.chaos_narrative = _narrate(fired)

        # 12b. judge -- advisory only. `success`, `failure_mode` and `severity` are
        # already decided above and are not revisited here.
        judge_record = self._apply_judge(result)

        # 13. bundle
        if self.write_bundle:
            written = write_bundle(
                result,
                run_dir,
                plan=state.plan,
                # Re-read rather than reusing the snapshot the probes saw: post-run
                # steps emit `internal_error` events of their own, and writing the
                # snapshot would clobber them out of `trace.jsonl`.
                trace=list(ctx.trace.memory.events if ctx.trace.memory else trace),
                judge=judge_record,
                baseline_output=(
                    state.baseline_result.final_output if state.baseline_result else None
                ),
            )
            if "baseline_diff" in written and result.delta_vs_baseline:
                result.delta_vs_baseline["diff_path"] = written["baseline_diff"]
            if "agent_task" in written:
                result.artifacts["agent_task"] = written["agent_task"]
                (run_dir / "report.json").write_text(result.to_json(), encoding="utf-8")
        return result

    # -------------------------------------------------------------------- judging

    def _judge(self) -> Any:
        """Resolve and cache the judge for this engine.

        Cached per engine rather than per run: a reachability probe costs 1.5 s, and
        a 21-scenario suite should pay it once.

        Returns:
            A judge instance. Falls back to `RuleJudge` when selection itself fails,
            because a misconfigured judge must not stop a run.
        """
        if self._judge_impl is None:
            from .judges import RuleJudge, select_judge

            options = dict(self.judge_options)
            options.setdefault("allow_remote_judge", self.allow_remote_judge)
            try:
                self._judge_impl = select_judge(self.judge, **options)
            except Exception as exc:
                self._internal_error("judge_selection", exc)
                self._judge_impl = RuleJudge()
        return self._judge_impl

    def _apply_judge(self, result: ChaosResult) -> dict[str, Any] | None:
        """Judge an assembled result in place, and return the record for `judge.json`.

        A judge is advisory (`docs/11` §1). It may write `verdict`'s narration fields
        and the report's `root_cause_hypothesis`, `refinement_hint` and
        `suggested_fixes`. It may not touch `success`, `failure_mode` or `severity`,
        which is enforced here by copying the computed values back over whatever the
        judge returned.

        Args:
            result: The assembled result, mutated in place.

        Returns:
            The judge's raw request/response plus the verdict, or `None` when no
            model was called and there is nothing to audit.
        """
        from .judges.base import build_evidence

        judge: Any = None
        try:
            evidence = build_evidence(
                result, code_context=extract_code_context(result.code_pointers)
            )
        except Exception as exc:
            self._internal_error("judge_evidence", exc)
            return None

        try:
            judge = self._judge()
            verdict = judge.judge(evidence)
        except Exception as exc:
            # An engine bug must never be reported as an agent failure, and a judge
            # is the least trustworthy thing in the pipeline. Falling through to the
            # rules judge rather than returning keeps the report narrated: an empty
            # narrative reads as "nothing to say", not as "the judge broke".
            self._internal_error("judge", exc)
            try:
                from dataclasses import replace

                from .judges.rules import RuleJudge

                verdict = RuleJudge().judge(evidence)
                verdict.judge_meta = replace(verdict.judge_meta, fell_back_to_rules=True)
            except Exception as inner:
                self._internal_error("judge_fallback", inner)
                return None

        # The judge carries these; it never decides them.
        verdict.passed = result.success
        verdict.observed_behavior = result.verdict["observed_behavior"]
        verdict.failure_mode = result.failure_mode
        verdict.severity = result.severity
        verdict.expected_behavior = result.verdict["expected_behavior"]

        if self.narrate_all and result.success and hasattr(judge, "narrate"):
            try:
                narrated = judge.narrate(evidence)
                if narrated:
                    verdict.narrative = narrated
            except Exception as exc:
                self._internal_error("judge_narrate", exc)

        result.verdict = verdict.to_dict()
        result.root_cause_hypothesis = verdict.root_cause_hypothesis
        result.refinement_hint = verdict.refinement_hint
        result.suggested_fixes = list(verdict.suggested_fixes)
        if verdict.narrative:
            result.chaos_narrative = verdict.narrative

        last_call = dict(getattr(judge, "last_call", {}) or {})
        if not last_call:
            return None
        return {**last_call, "verdict": verdict.to_dict()}


def _next_seq(trace_path: Path) -> int:
    """The next sequence number for a trace being appended to.

    Args:
        trace_path: The `trace.jsonl`.

    Returns:
        One past the highest `seq` already recorded.
    """
    return max((int(e.get("seq", 0)) for e in _read_trace(trace_path)), default=0) + 1


def _post_payload(crossing: Crossing, result: Any) -> dict[str, Any]:
    """Build the payload for a post-crossing event.

    An `llm_response` carries `finish_reason` when the response exposes one. The
    `truncated_output_used` probe gates on exactly that key, and the payload used to
    hold only `result` -- so the probe could not fire from a fault *or* from a real
    model whose response genuinely ran out of room.

    Nothing is invented: a bare string reply records no reason, because claiming
    `stop` for a response that never said so would make the probe's silence a lie
    rather than an absence of evidence.

    Args:
        crossing: The crossing being closed.
        result: What the callable returned, after any fault.

    Returns:
        The event payload.
    """
    payload: dict[str, Any] = {"result": result}
    if crossing.layer != "llm":
        return payload
    reason = None
    if isinstance(result, Mapping):
        reason = result.get("finish_reason") or result.get("stop_reason")
    else:
        reason = getattr(result, "finish_reason", None) or getattr(result, "stop_reason", None)
    if reason is not None:
        payload["finish_reason"] = str(reason)
    return payload


def _reproduce_scenario(state: _RunState) -> str | None:
    """Serialize what the agent was asked, for `reproduce.scenario_yaml`.

    A replay that re-runs with different inputs is not a replay: the agent falls back
    to whatever default its signature carries and the run can pass where the original
    failed, reporting a clean reproduction of a different experiment. The plan cannot
    carry this -- `plan_hash` is computed over the plan and is documented as proving
    plan identity and nothing else -- so it goes in the field the report schema
    already reserves for it.

    Args:
        state: The run state.

    Returns:
        Compact JSON of `{inputs, initial_state}`, or `None` when neither was given.
    """
    payload: dict[str, Any] = {}
    if state.inputs is not None:
        payload["inputs"] = state.inputs
    planted = {k: v for k, v in (state.planted_state or {}).items() if k != "_alc_canary"}
    if planted:
        payload["initial_state"] = planted
    if not payload:
        return None
    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - exotic inputs
        return None


def _read_trace(path: Path) -> list[dict[str, Any]]:
    """Load a `trace.jsonl`.

    Args:
        path: The file.

    Returns:
        The events, or an empty list when the file is absent or unreadable.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:  # pragma: no cover - a truncated final line
            continue
    return out


def _from_target(exc: BaseException) -> BaseException:
    """Mark an exception as having come out of the code under test.

    When the engine calls a wrapped user callable and the arguments do not match, the
    `TypeError` is raised at *our* call site because the callee never entered -- so
    the innermost real frame is `engine.py` and naive attribution files the agent's
    bad dispatch as a library bug.

    Args:
        exc: The exception that escaped the invocation.

    Returns:
        The same exception, tagged.
    """
    # A BaseException subclass may define __slots__, in which case the marker cannot
    # be attached. Attribution then falls back to reading the frames, which is the
    # behaviour without the marker rather than a wrong answer.
    with contextlib.suppress(AttributeError):
        object.__setattr__(exc, "_alc_from_target", True)
    return exc


#: Actions no adapter can currently perform. A fault asking for one changes nothing,
#: so it is recorded as a skip rather than a fire -- the same reasoning as D-95.
#: `resume_from_checkpoint` needs a replay the LangGraph adapter does not implement:
#: the checkpoint *crossing* exists (D-106), the resume does not.
#: `resume_from_checkpoint` is performable at a `(node, post)` crossing and not at a
#: `(checkpoint, post)` one -- the checkpointer's `put` runs after the node returned,
#: so a replay from there would mean re-entering the graph. The check is per crossing
#: rather than a flat action list.
UNSUPPORTED_ACTIONS: frozenset[str] = frozenset()


def _action_is_performable(action: str, layer: str) -> bool:
    """Report whether the engine can carry out an action at this layer.

    Args:
        action: The outcome's action.
        layer: The crossing's layer.

    Returns:
        False when the action would be a silent no-op, so the caller records a skip
        naming it rather than counting a fire (D-109).
    """
    if action == "resume_from_checkpoint":
        return layer == "node"
    return action not in UNSUPPORTED_ACTIONS


#: How many committed nodes are kept for a rollback to replay. A window is small by
#: nature -- a resume redoes the work since the last durable checkpoint, not the run --
#: and a long run must not accumulate every node it ever entered.
_ROLLBACK_WINDOW = 32


def _is_langchain_tool(obj: Any) -> bool:
    """Whether this is a LangChain tool object rather than a plain callable.

    Duck-typed on purpose: importing `langchain_core` to check would make an optional
    extra mandatory. A `BaseTool` has a `name`, an `invoke`, and is not itself callable.

    Args:
        obj: The candidate.

    Returns:
        True for a LangChain tool.
    """
    return (
        not callable(obj)
        and hasattr(obj, "invoke")
        and hasattr(obj, "name")
        and (hasattr(obj, "func") or hasattr(obj, "coroutine"))
    )


#: Frames that own nothing. A failure inside them belongs to their caller.
_TRANSPARENT_FRAMES = ("site-packages", "/lib/python", "<frozen", "<string>")


def raised_in(traceback_text: str) -> str:
    """Attribute an exception to the agent or to the harness.

    Reads the **innermost** frame -- where the exception was actually raised. The
    outermost frame is always the engine calling the agent, so reading that would
    attribute every crash to the library.

    Args:
        traceback_text: A formatted traceback.

    Returns:
        `"harness"` when the deepest frame is the library's own source, else
        `"agent"`. An unparseable traceback defaults to `"agent"`: claiming a library
        bug on no evidence would hide real agent failures behind `harness_error`.

        A **deliberately injected** raise is excluded by the caller before this is
        consulted. It comes from a library frame by construction -- that is where the
        interception point is -- but an agent that failed to handle it has failed,
        which is exactly what injecting it was for.
    """
    from .report import _FRAME_RE

    files = [
        match.group("file")
        for line in traceback_text.splitlines()
        if (match := _FRAME_RE.match(line))
    ]
    # Walk inward-out to the first frame that *owns* the failure. The standard
    # library and installed packages are transparent: a `JSONDecodeError` raised
    # inside `json/decoder.py` belongs to whoever called `json.loads`, and blaming
    # the harness for it would file every agent's parse bug as our own.
    for filename in reversed(files):
        if "agent_loop_chaos" in filename:
            return "harness"
        if not any(marker in filename for marker in _TRANSPARENT_FRAMES):
            return "agent"
    return "agent"


def _user_caller(*, depth: int = 40) -> dict[str, Any] | None:
    """Find the innermost stack frame that belongs to the user's own code.

    Args:
        depth: How far up the stack to look before giving up.

    Returns:
        ``{file, line, symbol}`` for the nearest non-library frame, or `None`. The
        library's own frames, the standard library and installed packages are all
        skipped: a pointer into any of them sends a coding agent to fix nothing.
    """
    from .report import _is_user_frame

    try:
        frame: Any = sys._getframe(1)
    except (AttributeError, ValueError):  # pragma: no cover - exotic interpreter
        return None
    for _ in range(depth):
        if frame is None:
            return None
        filename = frame.f_code.co_filename
        if _is_user_frame(filename):
            return {
                "file": filename,
                "line": frame.f_lineno,
                "symbol": frame.f_code.co_name,
            }
        frame = frame.f_back
    return None


def entrypoint_fingerprint(target: Any) -> str:
    """Hash an entrypoint's source, so a replay can tell the agent changed.

    `plan_hash` covers the plan, not the experiment: the agent's own body sits
    outside it, and a trigger like `on_call: 1` is relative to that body's call
    sequence. An agent edited between a run and its replay relocates every fault
    while the plan hash matches exactly (D-31).

    Args:
        target: The agent callable, or anything else.

    Returns:
        A sha256 hex digest of the source, or the empty string when there is none --
        a builtin, a C extension, or a callable defined in a REPL. Recording nothing
        beats crashing a run over provenance metadata.
    """
    import inspect

    fn = target
    for attribute in ("__wrapped__", "__func__"):
        fn = getattr(fn, attribute, fn)
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return ""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def crossing_signature(trace: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """Project a trace to the crossing sequence a replay compares against.

    `(layer, name, phase, call_index)` per D-31. Timing, payloads and seq numbers are
    excluded: they vary between two identical runs and comparing them would report
    divergence on every replay.

    Args:
        trace: The loaded trace.

    Returns:
        One entry per intercepted crossing, in order.
    """
    out: list[list[Any]] = []
    for event in trace:
        layer = event.get("layer")
        if layer in (None, "engine") or not event.get("name"):
            continue
        if event.get("kind") not in _CROSSING_KINDS:
            continue
        out.append([layer, str(event.get("name")), event.get("phase"), event.get("call_index")])
    return out


#: Events that mark an interception. `_post`-side kinds only, so one crossing counts
#: once whether or not the pre side emitted.
_CROSSING_KINDS = frozenset(
    {
        "tool_call_requested",
        "llm_request",
        "node_entered",
        "edge_taken",
        "state_mutated",
        "checkpoint_written",
    }
)


def _narrate(fired: Sequence[FaultRecord]) -> str:
    """Describe what the harness did, deterministically.

    The rules-mode narrative. M6's SLM judge replaces it with prose; until then a
    template is honest and a fabricated paragraph would not be.

    Args:
        fired: The faults that fired.

    Returns:
        One sentence.
    """
    if not fired:
        return "No fault fired; the run proceeded unperturbed."
    notes = [f["note"] for r in fired for f in r.fires if f.get("note")]
    lead = f"{len(fired)} fault(s) fired: " if len(fired) > 1 else "One fault fired: "
    return lead + "; ".join(notes[:4]) + "."


def _denormalize(original: Any, messages: Any) -> Any:
    """Convert mutated messages back into the shape the callable expects.

    An LLM fault operates on normalized messages, but the wrapped callable was
    written against whatever the caller passes it. Returning a list to a function
    that wanted a string would surface as the agent crashing on a harness artefact.

    Args:
        original: The callable's first argument before interception.
        messages: The mutated payload from the plan.

    Returns:
        `messages` rendered back to `original`'s shape.
    """
    if isinstance(original, str) and isinstance(messages, list):
        return "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, Mapping))
    return messages


def _baseline_blocks(
    baseline: ChaosResult | None,
    metrics: Mapping[str, Any],
    output: Any,
    evidence: EvidenceContext,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Build the report's `baseline` and `delta_vs_baseline` blocks.

    Args:
        baseline: A prior unfaulted result, or `None`.
        metrics: This run's metrics.
        output: This run's final output.
        evidence: This run's evidence, for the tool list.

    Returns:
        ``(baseline_block, delta)``, both `None` without a baseline. The report's
        `baseline` block is counters plus a path -- not the internal `BaselineRef`,
        which additionally carries `plan_hash`.
    """
    if baseline is None:
        return None, None
    block = {
        "run_id": baseline.run_id,
        "steps": int(baseline.metrics.get("steps", 0)),
        "tool_calls": int(baseline.metrics.get("tool_calls", 0)),
        "llm_calls": int(baseline.metrics.get("llm_calls", 0)),
        "report_path": baseline.artifacts.get("report"),
    }
    delta = compute_delta(
        dict(metrics),
        baseline.metrics,
        chaos_output=output,
        baseline_output=baseline.final_output,
        chaos_tools=evidence.tools_called,
        baseline_tools=[str(c["tool"]) for c in baseline.tool_calls if c.get("tool")],
    )
    return block, delta


def _scalars(value: Any) -> list[str]:
    """Every scalar leaf of a payload, rendered as text.

    Args:
        value: Any nested structure.

    Returns:
        The leaves as strings.
    """
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in _scalars(child)]
    if isinstance(value, (list, tuple)):
        return [leaf for child in value for leaf in _scalars(child)]
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float, str)):
        return [f"{value:g}" if isinstance(value, float) else str(value)]
    return []


def _visible_state(view: StateView | None) -> dict[str, Any]:
    """The agent's visible state as a plain mapping.

    Args:
        view: The state view, or `None` when the agent keeps state the engine cannot
            see (`docs/06` §2.4).

    Returns:
        A copy of the state, or an empty mapping when there is none observable.
    """
    if view is None or not isinstance(view.raw, dict):
        return {}
    return dict(view.raw)


def _is_recursion_limit(exc: BaseException) -> bool:
    """Report whether an exception is LangGraph's own step guard.

    Matched by name so the check needs no langgraph import: the core must not depend
    on an optional extra to classify an error correctly.

    Args:
        exc: The exception the graph raised.

    Returns:
        True for a recursion-limit error.
    """
    return type(exc).__name__ == "GraphRecursionError"


def _objective_text(inputs: Any, initial_state: Mapping[str, Any] | None) -> str | None:
    """Render the scenario's inputs as the objective text.

    Args:
        inputs: The payload passed to the agent.
        initial_state: The starting state.

    Returns:
        The objective, or `None` when there is nothing text-shaped to use.
    """
    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, Mapping):
        for key in ("query", "objective", "task", "question"):
            if isinstance(inputs.get(key), str):
                return str(inputs[key])
    if initial_state:
        value = initial_state.get("query")
        if isinstance(value, str):
            return value
    return None


def _render_prompt(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render normalized messages as the exact prompt text the model received.

    `exact_prompt` exists so a work order can quote what actually went to the model,
    rather than a paraphrase of it.

    Args:
        messages: Normalized message dicts.

    Returns:
        One ``role: content`` line per message.
    """
    return "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in messages)


def functools_wraps(source: Callable[..., Any], wrapper: Callable[..., Any]) -> Callable[..., Any]:
    """Copy `source`'s metadata and signature onto `wrapper`.

    Args:
        source: The wrapped callable.
        wrapper: The wrapper to decorate.

    Returns:
        `wrapper`, with `__name__`, `__doc__`, `__wrapped__` and `__signature__` set.
    """
    import functools

    functools.update_wrapper(wrapper, source)
    with contextlib.suppress(TypeError, ValueError):
        wrapper.__signature__ = inspect.signature(source)  # type: ignore[attr-defined]
    return wrapper
