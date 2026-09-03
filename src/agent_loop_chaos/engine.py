"""The chaos engine.

Owns a fault plan, a seeded RNG and a trace recorder. Adapters turn their
framework's hooks into `Crossing` objects and route them here; at each crossing the
engine asks the plan whether anything fires, and if so applies the fault to a deep
copy and records the diff.

Everything in this module is a stub until M1 (`prompts/01-core-engine.md`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Literal

from .assertions import Expect
from .context import Crossing, Layer, Limits
from .enums import ExpectedBehavior
from .report import ChaosResult
from .targeting import Target, Trigger

__all__ = ["ChaosEngine"]

_M1 = "arrives in M1 (prompts/01-core-engine.md)"


class ChaosEngine:
    """Plans faults, instruments a target, runs it, and reports what broke.

    The engine never raises into the agent under test on an internal error: a bug in
    a fault, a probe or a judge is caught, recorded as an `internal_error` trace
    event with a traceback, and the run continues. An engine bug must never be
    reported as an agent failure.
    """

    def __init__(
        self,
        *,
        seed: int = 1337,
        out_dir: str | Path = ".chaos",
        trace_level: Literal["minimal", "standard", "verbose"] = "standard",
        limits: Limits | None = None,
        judge: Any | str | None = None,
        redact_keys: Sequence[str] = (),
        strict_schema: bool = True,
        dry_run: bool = False,
        write_bundle: bool = True,
        allow_remote_judge: bool = False,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Initialise an engine.

        Args:
            seed: Root seed. Every RNG stream derives from it, so the same seed
                always tells the same story (D-02).
            out_dir: Where run bundles are written. Sensitive by default (D-24).
            trace_level: How much payload detail to record.
            limits: Run guard rails. Defaults to `Limits()`.
            judge: ``"rules"``, ``"slm"``, ``"ensemble"``, a `Judge` instance, or
                `None` to auto-select.
            redact_keys: Extra key patterns to redact on top of the default
                deny-list.
            strict_schema: Fail the run if the assembled report does not validate.
            dry_run: Register faults but fire none, so the run must classify exactly
                as an unfaulted baseline (D-12).
            write_bundle: Write `report.json`, `trace.jsonl` and friends to disk.
            allow_remote_judge: Required opt-in before a non-loopback judge
                endpoint may receive code context (D-22).
            tags: Free-form labels recorded in the report.
        """
        raise NotImplementedError(f"ChaosEngine {_M1}")

    def register_fault(
        self,
        fault: Any,
        *,
        target: Target | None = None,
        trigger: Trigger | None = None,
        fault_id: str | None = None,
        target_tool: str | None = None,
        target_node: str | None = None,
        target_llm: str | None = None,
        target_state_key: str | None = None,
    ) -> str:
        """Add a fault to the plan and return its `fault_id`.

        Validates eagerly: a target layer outside the fault's `accepts` set, an
        impossible trigger such as ``on_call=0``, or a `probability` outside
        ``[0, 1]`` raises `ConfigError` now rather than mid-run. The D-23
        side-effect gate is enforced here too.

        Args:
            fault: The `Fault` instance to register.
            target: Where it applies. Mutually exclusive with the shorthands.
            trigger: When it fires. Defaults to `Trigger()`.
            fault_id: Override the default ``f{n}`` registration-order label.
            target_tool: Shorthand for ``Target(tool=...)``.
            target_node: Shorthand for ``Target(node=...)``.
            target_llm: Shorthand for ``Target(llm=...)``.
            target_state_key: Shorthand for ``Target(state_key=...)``.

        Returns:
            The `fault_id`, for cross-referencing in the report.

        Raises:
            ConfigError: On any invalid combination.
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.register_fault {_M1}")

    def clear_faults(self) -> None:
        """Drop every registered fault.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.clear_faults {_M1}")

    def plan(self) -> dict[str, Any]:
        """Return the canonical plan dict that `plan_hash` is computed over.

        Contains only seed, ordered fault specs, limits, adapter, entrypoint string,
        `expected_behavior` and sorted `must_not` — nothing time- or path-dependent.

        Returns:
            The plan, ready for canonical JSON serialization.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.plan {_M1}")

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        side_effecting: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        """Register and wrap a tool callable. Works bare or parameterized.

        Args:
            fn: The tool, when used bare as ``@engine.tool``.
            name: Tool name. Defaults to the function's name.
            side_effecting: Declare that this tool performs a real action.
                Declaring `False` is a deliberate statement, and it is the opt-out
                the D-23 gate checks (`SAFETY.md` §1).
            schema: JSON Schema for the tool's arguments.

        Returns:
            The wrapped tool, or the decorator when used parameterized. Coroutine
            functions yield coroutine wrappers.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.tool {_M1}")

    def llm(
        self, fn: Callable[..., Any] | None = None, *, name: str = "default"
    ) -> Callable[..., Any]:
        """Register and wrap an LLM callable. Works bare or parameterized.

        Args:
            fn: The callable, when used bare as ``@engine.llm``.
            name: LLM alias used by targets and in the trace.

        Returns:
            The wrapped callable, or the decorator when used parameterized.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.llm {_M1}")

    def wrap_tools(self, tools: Mapping[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
        """Wrap a name-to-callable mapping of tools.

        Args:
            tools: The tools to wrap.

        Returns:
            A new mapping of wrapped tools. The input is not mutated.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.wrap_tools {_M1}")

    def wrap_callable(
        self, fn: Callable[..., Any], *, layer: Layer, name: str
    ) -> Callable[..., Any]:
        """Wrap an arbitrary callable as an interception point.

        Args:
            fn: The callable to wrap.
            layer: Which layer its crossings belong to.
            name: The name its crossings carry.

        Returns:
            The wrapped callable, preserving signature and coroutine-ness.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.wrap_callable {_M1}")

    def intercept_tools(self, *tool_names: str) -> Callable[..., Any]:
        """Decorate an agent entrypoint so registered tools are intercepted inside it.

        Exists for compatibility with the original blueprint API, and is implemented
        on top of `wrap_callable` plus a contextvar.

        Args:
            *tool_names: Tools to intercept. Empty means every registered tool.

        Returns:
            A decorator for the agent entrypoint.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.intercept_tools {_M1}")

    def instrument_object(
        self,
        obj: Any,
        *,
        tools: Sequence[str] = (),
        llm_methods: Sequence[str] = (),
        nodes: Sequence[str] = (),
    ) -> Any:
        """Wrap named methods on an existing instance.

        Args:
            obj: The instance to instrument.
            tools: Method names to treat as tools.
            llm_methods: Method names to treat as LLM calls.
            nodes: Method names to treat as graph nodes.

        Returns:
            The instrumented object.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.instrument_object {_M1}")

    def step(self) -> int:
        """Advance and return the step counter. Called by adapters.

        A step is one agent iteration — a node entry under LangGraph, one LLM call
        under vanilla. Tool calls and crossings do not increment it (D-05).

        Returns:
            The new step number.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.step {_M1}")

    def validated(self, value: Any = None, *, name: str | None = None) -> None:
        """Record positive evidence that the agent checked something.

        Optional, and absence is never read as misbehaviour — only as "no evidence
        either way" (`docs/11` §3.2).

        Args:
            value: What was validated.
            name: A label for the check.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.validated {_M1}")

    def note(self, message: str) -> None:
        """Record a free-text breadcrumb in the trace (`docs/11` §3.3).

        Args:
            message: The note.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.note {_M1}")

    def bind_context(self) -> AbstractContextManager[None]:
        """Bind the run context inside a thread the agent spawned.

        Faults do not fire in threads the agent starts unless that thread enters
        this context manager (D-42), because the run state lives in a
        `contextvars.ContextVar`.

        Returns:
            A context manager binding the current run.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.bind_context {_M1}")

    def cross(self, crossing: Crossing) -> Any:
        """Route one crossing through the plan.

        Args:
            crossing: The crossing to evaluate.

        Returns:
            The value execution should continue with — unchanged when nothing fires.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.cross {_M1}")

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
    ) -> ChaosResult:
        """Run an agent under the fault plan.

        Args:
            target: An agent callable, or a compiled LangGraph app.
            inputs: Payload passed to the agent.
            initial_state: Starting state, for stateful agents.
            scenario_id: Identifier recorded in the report and used in the run path.
            expected_behavior: What good behaviour would look like here.
            must_not: Probe codes that always fail the run.
            expect: Declarative assertions (`docs/11` §4).
            expected_errors: Exception class names that count as an explicit error
                rather than a crash.
            allow_side_effects: Tools the D-23 gate may target.
            dry_run: Override the engine's `dry_run` for this run.
            attempt: Part of `run_id`; threaded by `RefinementLoop` as the round
                number and by `replay` as previous + 1 (D-04).
            adapter: Force an adapter, or sniff it.
            baseline: A prior unfaulted result to diff against.
            seed: Override the engine seed for this run.

        Returns:
            The assembled `ChaosResult`.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.run {_M1}")

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
    ) -> ChaosResult:
        """Async twin of `run`, with the same signature.

        Unlike the sync path, this wraps the agent in `asyncio.wait_for`, so a
        timeout genuinely cancels (D-08).

        Returns:
            The assembled `ChaosResult`.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.arun {_M1}")

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

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.run_with_state {_M1}")

    def baseline(self, target: Callable[..., Any] | Any, **kw: Any) -> ChaosResult:
        """Run with the plan disabled, for comparison.

        Equivalent to `run` with ``dry_run=True`` and
        ``expected_behavior="ignore_and_continue"``.

        Args:
            target: The agent callable or graph.
            **kw: Forwarded to `run`.

        Returns:
            The unfaulted `ChaosResult`.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.baseline {_M1}")

    def replay(
        self, run_dir: str | Path, *, target: Callable[..., Any] | Any | None = None
    ) -> ChaosResult:
        """Rebuild the plan and seed from `plan.json` and re-run.

        Refuses rather than guessing when it cannot verify the plan it loaded
        (D-31).

        Args:
            run_dir: A previous run directory.
            target: The agent to re-run, when it cannot be resolved from
                `plan.json`'s entrypoint.

        Returns:
            The new `ChaosResult`, with `attempt` incremented.

        Raises:
            NotImplementedError: Until M1.
        """
        raise NotImplementedError(f"ChaosEngine.replay {_M1}")
