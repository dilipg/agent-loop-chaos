"""The vanilla adapter: plain functions, classes, and hand-rolled while-loops.

Wrappers preserve signatures, `functools.wraps` metadata and coroutine-ness, and are
**inert outside a run**. That last property matters more than it sounds: users leave
`@engine.tool` on production code, so an unarmed decorator must cost almost nothing
and change nothing (`docs/06-LANGGRAPH-ADAPTER.md` §2.3).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..context import Crossing, RunContext
from ..errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - types only
    from ..engine import ChaosEngine

__all__ = ["VanillaAdapter", "normalize_messages", "resolve_invocation"]


def normalize_messages(payload: Any) -> tuple[list[dict[str, Any]] | None, bool]:
    """Normalize an outbound LLM payload to a message list.

    A string becomes one user message; a list of message dicts is used as-is;
    anything else is passed through untouched and reported rather than guessed at
    (`docs/06` §2.1). When normalization fails, `pre`-phase LLM faults are disabled
    for that call with reason ``unnormalizable_payload``, and the report says so.

    Args:
        payload: The first positional argument to the LLM callable.

    Returns:
        ``(messages, unavailable)``. `messages` is `None` exactly when `unavailable`
        is True.
    """
    if isinstance(payload, str):
        return [{"role": "user", "content": payload}], False
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        items = list(payload)
        if items and all(isinstance(m, Mapping) and "role" in m for m in items):
            return [dict(m) for m in items], False
        if not items:
            return [], False
    return None, True


def resolve_invocation(
    agent: Callable[..., Any],
    inputs: Any,
    initial_state: Mapping[str, Any] | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Work out how to call a vanilla agent, per D-19.

    Resolution order, decided by `inspect.signature`:

    1. `inputs` is a mapping and every key is a named parameter → ``agent(**inputs)``.
    2. `inputs` is a mapping, the signature has a positional parameter and no
       matching keys → ``agent(inputs)``.
    3. Otherwise → ``agent(inputs)`` when `inputs` is not `None`, else ``agent()``.
    4. `initial_state` is passed by keyword **only** when the signature has a
       parameter named `state` or `initial_state`; otherwise the engine holds it and
       exposes it through `state_view`.

    Args:
        agent: The agent callable.
        inputs: The payload.
        initial_state: The starting state, if any.

    Returns:
        ``(args, kwargs)`` to call the agent with.

    Raises:
        ConfigError: When the signature cannot accept the payload. Never a
            `TypeError` from inside the agent.
    """
    try:
        signature = inspect.signature(agent)
    except (TypeError, ValueError) as exc:  # pragma: no cover - exotic callables
        raise ConfigError(f"cannot inspect the signature of {agent!r}: {exc}") from exc

    parameters = signature.parameters
    names = set(parameters)
    accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in parameters.values())
    positional = [
        p
        for p in parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.name not in {"state", "initial_state", "self"}
    ]

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {}

    if isinstance(inputs, Mapping) and inputs and (set(inputs) <= names or accepts_var_kw):
        kwargs.update(dict(inputs))
    elif inputs is not None:
        if not positional and not accepts_var_kw:
            raise ConfigError(
                f"agent {getattr(agent, '__qualname__', agent)!r} takes no positional "
                f"parameter for `inputs`; its signature is {signature}. Pass a mapping "
                "whose keys are parameter names, or add a parameter."
            )
        args = (inputs,)

    if initial_state is not None:
        for candidate in ("state", "initial_state"):
            if candidate in names:
                kwargs[candidate] = initial_state
                break

    try:
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise ConfigError(
            f"cannot call agent {getattr(agent, '__qualname__', agent)!r} with the given "
            f"inputs: {exc}. Signature is {signature}; see docs/DECISIONS.md D-19."
        ) from exc
    return args, kwargs


class VanillaAdapter:
    """Instruments plain Python callables.

    Attributes:
        name: ``"vanilla"``, as recorded in `target.framework`.
        version: The running Python version, since there is no framework to version.
    """

    name = "vanilla"

    def __init__(self) -> None:
        """Initialise the adapter."""
        import platform

        self.version = f"python-{platform.python_version()}"

    def instrument(self, target: Any, engine: ChaosEngine, ctx: RunContext) -> Any:
        """Return the target unchanged.

        Vanilla instrumentation happens at decoration time — `engine.tool`,
        `engine.llm`, `wrap_tools`, `instrument_object` — so by the time a run
        starts there is nothing left to wrap.

        Args:
            target: The agent callable.
            engine: The engine, unused here.
            ctx: The run context, unused here.

        Returns:
            `target`.
        """
        return target

    def run(self, instrumented: Any, inputs: Any, ctx: RunContext) -> Any:
        """Call the agent, resolving its calling convention per D-19.

        Args:
            instrumented: The agent callable.
            inputs: The payload.
            ctx: The run context, unused here.

        Returns:
            The agent's return value.

        Raises:
            ConfigError: On a signature mismatch.
        """
        args, kwargs = resolve_invocation(instrumented, inputs, None)
        return instrumented(*args, **kwargs)

    def state_of(self, instrumented: Any, ctx: RunContext) -> Any:
        """Vanilla agents expose no state of their own.

        Args:
            instrumented: The agent callable.
            ctx: The run context.

        Returns:
            `None`. The engine holds whatever `initial_state` the caller passed.
        """
        return None


def build_wrapper(
    engine: ChaosEngine,
    fn: Callable[..., Any],
    *,
    layer: str,
    name: str,
) -> Callable[..., Any]:
    """Wrap a callable as an interception point.

    Produces a coroutine wrapper for a coroutine function and a plain wrapper
    otherwise, so async and sync parity is structural rather than remembered.

    Args:
        engine: The engine to route crossings through.
        fn: The callable to wrap.
        layer: Which layer its crossings belong to.
        name: The name its crossings carry.

    Returns:
        The wrapper, with `fn`'s metadata and signature.
    """
    is_async = inspect.iscoroutinefunction(fn)

    if is_async:

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not engine.is_active():
                return await fn(*args, **kwargs)
            return await engine.route_async(fn, layer=layer, name=name, args=args, kwargs=kwargs)

        async_wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        # The hot path when unarmed: one contextvar read, then straight through.
        if not engine.is_active():
            return fn(*args, **kwargs)
        return engine.route_sync(fn, layer=layer, name=name, args=args, kwargs=kwargs)

    sync_wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    return sync_wrapper


def make_crossing(
    ctx: RunContext,
    *,
    layer: str,
    phase: str,
    name: str,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    result: Any = None,
    exception: BaseException | None = None,
    call_index: int = 1,
    span_id: str = "s0",
    state: Any = None,
) -> Crossing:
    """Build a `Crossing`, normalizing LLM messages when relevant.

    Args:
        ctx: The run context, for the step counter.
        layer: The crossing's layer.
        phase: The crossing's phase.
        name: The crossing's name.
        args: Positional arguments, for `pre`.
        kwargs: Keyword arguments, for `pre`.
        result: The return value, for `post`.
        exception: The raised exception, for `error`.
        call_index: 1-based index of this call to this name.
        span_id: The allocated span id.
        state: The agent's visible state, if any.

    Returns:
        The populated crossing.
    """
    crossing = Crossing(
        layer=layer,  # type: ignore[arg-type]  # validated by the engine at registration
        phase=phase,  # type: ignore[arg-type]
        name=name,
        step=ctx.counters.steps,
        call_index=call_index,
        args=tuple(args),
        kwargs=dict(kwargs or {}),
        result=result,
        exception=exception,
        state=state,
        span_id=span_id,
    )
    if layer == "llm" and phase == "pre":
        payload = (
            args[0] if args else crossing.kwargs.get("messages", crossing.kwargs.get("prompt"))
        )
        crossing.messages, crossing.messages_unavailable = normalize_messages(payload)
    return crossing


async def maybe_await(value: Any) -> Any:
    """Await a value if it is awaitable.

    Args:
        value: A value or awaitable.

    Returns:
        The resolved value.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def is_async_callable(fn: Callable[..., Any]) -> bool:
    """Report whether `fn` is a coroutine function.

    Args:
        fn: The callable to test.

    Returns:
        True for a coroutine function, including one wrapped by `functools.wraps`.
    """
    return inspect.iscoroutinefunction(fn) or asyncio.iscoroutinefunction(fn)
