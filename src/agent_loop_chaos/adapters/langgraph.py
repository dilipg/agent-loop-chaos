"""The LangGraph adapter.

Its only job is to turn LangGraph's hooks into `Crossing` objects; the core never
learns which framework is in play. `langgraph` is an optional extra and is imported
locally, so `import agent_loop_chaos` never pulls it in.

Every attribute path here is version-specific. The shims resolve a chain and raise
`AdapterError` naming the installed version and every path they tried, because the
alternative -- instrumenting zero nodes -- looks like a clean run and proves nothing.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from ..errors import AdapterError, MissingExtraError
from ._lc_messages import from_langchain, to_langchain

if TYPE_CHECKING:  # pragma: no cover - types only
    from ..context import RunContext
    from ..engine import ChaosEngine

__all__ = [
    "LangGraphAdapter",
    "branch_slots",
    "instrument_graph",
    "instrument_model",
    "is_compiled",
    "langgraph_version",
    "node_slots",
    "wrap_langchain_tools",
]

# Set on a wrapper so a second `instrument_graph` is a no-op. Double wrapping would
# double every crossing and every step, which silently doubles the metrics a probe
# compares against a baseline.
_MARKER = "__alc_instrumented__"

Slot = tuple[str, Callable[[], Any], Callable[[Any], None]]
BranchSlot = tuple[str, str, Callable[[], Any], Callable[[Any], None]]


def _require_langgraph() -> Any:
    """Import langgraph, or explain how to install it.

    Returns:
        The `langgraph` module.

    Raises:
        MissingExtraError: When the extra is not installed.
    """
    try:
        import langgraph
    except ModuleNotFoundError as exc:
        raise MissingExtraError(
            "the LangGraph adapter needs langgraph: install agent-loop-chaos[langgraph]"
        ) from exc
    return langgraph


def langgraph_version() -> tuple[int, int, int]:
    """The installed LangGraph version.

    Every `AdapterError` names it, so a compat break is diagnosable from a stored
    report without reproducing the environment.

    Returns:
        ``(major, minor, patch)``, zero-padded when the version is unparseable.
    """
    _require_langgraph()
    from importlib.metadata import version

    try:
        parts = version("langgraph").split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2].split("rc")[0]))
    except Exception:
        return (0, 0, 0)


def is_compiled(graph: Any) -> bool:
    """Report whether a graph has been compiled.

    Args:
        graph: A `StateGraph` or a compiled graph.

    Returns:
        True for a compiled graph. Instrumentation differs between the two shapes:
        a builder exposes `runnable`, a compiled graph exposes `bound`.
    """
    return hasattr(graph, "get_graph") and not hasattr(graph, "add_node")


def _slot_paths() -> tuple[str, ...]:
    """The attribute names tried when resolving a node's callable.

    Returns:
        The chain, most current first.
    """
    return ("runnable", "bound", "func", "callable")


def node_slots(graph: Any, names: Sequence[str] | None = None) -> list[Slot]:
    """Resolve every instrumentable node to a getter and a setter.

    Args:
        graph: A `StateGraph` or a compiled graph.
        names: Restrict to these nodes, or `None` for all.

    Returns:
        ``(name, get, set)`` per node. LangGraph's internal nodes -- `__start__` and
        friends -- are excluded: they are plumbing, not agent steps.

    Raises:
        AdapterError: When no node exposes any known slot, naming the installed
            version and every path tried.
    """
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict) or not nodes:
        raise AdapterError(
            f"cannot find nodes on {type(graph).__name__} (langgraph "
            f"{'.'.join(map(str, langgraph_version()))}); tried `.nodes` as a mapping"
        )

    wanted = set(names) if names is not None else None
    out: list[Slot] = []
    for name, spec in nodes.items():
        if str(name).startswith("__"):
            continue
        if wanted is not None and name not in wanted:
            continue
        attribute = next((p for p in _slot_paths() if hasattr(spec, p)), None)
        if attribute is None:
            continue
        out.append(
            (
                str(name),
                functools.partial(getattr, spec, attribute),
                functools.partial(setattr, spec, attribute),
            )
        )

    if not out and wanted is None:
        raise AdapterError(
            f"no node on {type(graph).__name__} exposed a known callable slot "
            f"(langgraph {'.'.join(map(str, langgraph_version()))}); "
            f"tried {list(_slot_paths())}"
        )
    return out


def branch_slots(graph: Any) -> list[BranchSlot]:
    """Resolve every conditional edge's router to a getter and a setter.

    `EdgeMisrouteFault` overrides a routing *decision*, so it needs the router
    callable rather than the edge list.

    Args:
        graph: A `StateGraph`, or a compiled graph exposing its builder.

    Returns:
        ``(source_node, branch_name, get, set)`` per conditional edge. Empty when the
        graph has none, which is not an error.
    """
    branches = getattr(graph, "branches", None)
    if not isinstance(branches, dict):
        builder = getattr(graph, "builder", None)
        branches = getattr(builder, "branches", None)
    if not isinstance(branches, dict):
        return []

    out: list[BranchSlot] = []
    for source, named in branches.items():
        if not isinstance(named, dict):
            continue
        for branch_name, spec in named.items():
            attribute = next((p for p in ("path", "condition", "run") if hasattr(spec, p)), None)
            if attribute is None:
                continue
            out.append(
                (
                    str(source),
                    str(branch_name),
                    functools.partial(getattr, spec, attribute),
                    _branch_setter(named, branch_name, spec, attribute),
                )
            )
    return out


def _branch_setter(
    container: dict[str, Any], key: str, spec: Any, attribute: str
) -> Callable[[Any], None]:
    """Build a setter for a branch's router.

    `BranchSpec` is immutable in current LangGraph, so the setter replaces the whole
    spec in its parent mapping rather than assigning through. Falls back to direct
    assignment for any version that allows it.

    Args:
        container: The mapping holding the spec.
        key: Its key in that mapping.
        spec: The current spec.
        attribute: Which attribute carries the router.

    Returns:
        A setter taking the new router.
    """

    def set_(value: Any) -> None:
        try:
            setattr(spec, attribute, value)
            return
        except AttributeError:
            pass
        replacer = getattr(spec, "_replace", None)  # NamedTuple
        if callable(replacer):
            container[key] = replacer(**{attribute: value})
            return
        import dataclasses

        if dataclasses.is_dataclass(spec) and not isinstance(spec, type):
            container[key] = dataclasses.replace(spec, **{attribute: value})
            return
        raise AdapterError(
            f"cannot replace the router on {type(spec).__name__} "
            f"(langgraph {'.'.join(map(str, langgraph_version()))}); tried assignment, "
            "_replace and dataclasses.replace"
        )

    return set_


def _wrap_node(engine: ChaosEngine, name: str, target: Any, *, intercept_state: bool) -> Any:
    """Wrap one node so its entry and exit become crossings.

    LangGraph stores a node as a `RunnableCallable` holding `func` and/or `afunc`,
    and inspects those signatures to decide what to pass. Replacing the object with a
    plain function loses that, so the wrapper goes *inside*: the inner callables are
    replaced and the container is handed back untouched. The wrapper copies
    `functools.wraps` metadata and `__signature__` for the same reason -- getting it
    wrong makes the graph fail for a reason unrelated to the fault under test.

    Args:
        engine: The engine to route through.
        name: The node's name.
        target: The node callable or `RunnableCallable`.
        intercept_state: Whether to route state crossings as well as node ones.

    Returns:
        The instrumented node, or `target` unchanged when already instrumented.
    """
    if getattr(target, _MARKER, False):
        return target

    def sync_wrapper(inner: Callable[..., Any]) -> Callable[..., Any]:
        def route(state: Any, *args: Any, **kwargs: Any) -> Any:
            return engine.route_node(
                inner,
                name=name,
                state=state,
                args=args,
                kwargs=kwargs,
                intercept_state=intercept_state,
            )

        return _copy_metadata(inner, route)

    def async_wrapper(inner: Callable[..., Any]) -> Callable[..., Any]:
        async def aroute(state: Any, *args: Any, **kwargs: Any) -> Any:
            return await engine.aroute_node(
                inner,
                name=name,
                state=state,
                args=args,
                kwargs=kwargs,
                intercept_state=intercept_state,
            )

        return _copy_metadata(inner, aroute)

    # A RunnableCallable: instrument in place so LangGraph keeps the object it built.
    if hasattr(target, "func") or hasattr(target, "afunc"):
        if getattr(target, "func", None) is not None:
            target.func = sync_wrapper(target.func)
        if getattr(target, "afunc", None) is not None:
            target.afunc = async_wrapper(target.afunc)
        setattr(target, _MARKER, True)
        return target

    inner = target
    wrapped = async_wrapper(inner) if inspect.iscoroutinefunction(inner) else sync_wrapper(inner)
    setattr(wrapped, _MARKER, True)
    return wrapped


def _copy_metadata(source: Any, wrapper: Callable[..., Any]) -> Callable[..., Any]:
    """Copy a node's metadata and signature onto its wrapper.

    Args:
        source: The original callable.
        wrapper: The wrapper.

    Returns:
        The wrapper, with `__signature__` set so LangGraph's introspection still
        sees the parameters the node declared.
    """
    with contextlib.suppress(AttributeError, TypeError):  # exotic callables
        functools.update_wrapper(wrapper, source)
    with contextlib.suppress(TypeError, ValueError):  # unintrospectable callables
        # A plain function has no declared `__signature__`; setting one is how
        # LangGraph's introspection still sees the node's real parameters.
        wrapper.__signature__ = inspect.signature(source)  # type: ignore[attr-defined]
    return wrapper


def _wrap_branch(engine: ChaosEngine, source: str, router: Any) -> Any:
    """Wrap a conditional edge's router so its decision becomes a crossing.

    A router is often a `RunnableCallable` too, so the same in-place treatment as
    `_wrap_node` applies: replacing the container with a plain function makes
    LangGraph call `.invoke` on something that does not have it.

    Args:
        engine: The engine to route through.
        source: The node the edge leaves from.
        router: The routing callable.

    Returns:
        The instrumented router, or `router` unchanged when already instrumented.
    """
    if getattr(router, _MARKER, False):
        return router

    def wrap(inner: Callable[..., Any]) -> Callable[..., Any]:
        def route(state: Any, *args: Any, **kwargs: Any) -> Any:
            return engine.route_edge(inner, name=source, state=state, args=args, kwargs=kwargs)

        return _copy_metadata(inner, route)

    if hasattr(router, "func") or hasattr(router, "afunc"):
        if getattr(router, "func", None) is not None:
            router.func = wrap(router.func)
        setattr(router, _MARKER, True)
        return router

    wrapper = wrap(router)
    setattr(wrapper, _MARKER, True)
    return wrapper


def instrument_graph(
    graph: Any,
    engine: ChaosEngine,
    *,
    nodes: Sequence[str] | None = None,
    intercept_tools: bool = True,
    intercept_state: bool = True,
    intercept_checkpoints: bool = False,
) -> Any:
    """Instrument a LangGraph app and return something runnable.

    Args:
        graph: A `StateGraph` or a compiled graph.
        engine: The engine to route crossings through.
        nodes: Restrict instrumentation to these nodes, or `None` for all.
        intercept_tools: Reserved; tools are wrapped by `engine.wrap_tools`.
        intercept_state: Route state crossings at node boundaries.
        intercept_checkpoints: Route checkpoint crossings. Off by default because it
            requires a checkpointer.

    Returns:
        A compiled graph with its nodes and conditional edges wrapped.

    Raises:
        AdapterError: When the restriction matches no node, or when no node exposes
            a known slot. Instrumenting nothing would look like a clean run.
    """
    _require_langgraph()
    slots = node_slots(graph, nodes)
    if not slots:
        available = [n for n in getattr(graph, "nodes", {}) if not str(n).startswith("__")]
        raise AdapterError(
            f"instrumenting no nodes would prove nothing: {list(nodes or [])} matched none "
            f"of {available} (langgraph {'.'.join(map(str, langgraph_version()))})"
        )

    for name, get, set_ in slots:
        set_(_wrap_node(engine, name, get(), intercept_state=intercept_state))

    for source, _branch, get, set_ in branch_slots(graph):
        if nodes is None or source in set(nodes):
            set_(_wrap_branch(engine, source, get()))

    compiled = graph if is_compiled(graph) else graph.compile()
    setattr(compiled, _MARKER, True)
    return compiled


class LangGraphAdapter:
    """Runs an instrumented LangGraph app.

    Attributes:
        name: ``"langgraph"``, as recorded in `target.framework`.
        version: The installed LangGraph version, so a compat break is diagnosable
            from a stored report.
    """

    name = "langgraph"

    def __init__(self) -> None:
        """Initialise, recording the installed version."""
        self.version = f"langgraph-{'.'.join(map(str, langgraph_version()))}"

    def instrument(self, target: Any, engine: ChaosEngine, ctx: RunContext) -> Any:
        """Return the target, already instrumented by `instrument_graph`.

        Args:
            target: The compiled graph.
            engine: The engine, unused here.
            ctx: The run context, unused here.

        Returns:
            `target`.
        """
        return target

    def run(self, instrumented: Any, inputs: Any, ctx: RunContext) -> Any:
        """Invoke the graph under the run's step limit.

        `recursion_limit` is set from `Limits.max_steps` so LangGraph's own guard and
        ours agree; `GraphRecursionError` is caught by the engine and mapped to
        `limit_hit`, never to `error` -- it is a stop we imposed, not an agent crash.

        Args:
            instrumented: The compiled graph.
            inputs: The initial state.
            ctx: The run context.

        Returns:
            The final state.
        """
        config = {"recursion_limit": max(2, ctx.limits.max_steps)}
        return instrumented.invoke(inputs if inputs is not None else {}, config=config)

    async def arun(self, instrumented: Any, inputs: Any, ctx: RunContext) -> Any:
        """Async twin of `run`.

        Args:
            instrumented: The compiled graph.
            inputs: The initial state.
            ctx: The run context.

        Returns:
            The final state.
        """
        config = {"recursion_limit": max(2, ctx.limits.max_steps)}
        return await instrumented.ainvoke(inputs if inputs is not None else {}, config=config)

    def state_of(self, instrumented: Any, ctx: RunContext) -> Any:
        """LangGraph state is carried through the run rather than held here.

        Args:
            instrumented: The compiled graph.
            ctx: The run context.

        Returns:
            `None`; the engine tracks state at node boundaries instead.
        """
        return None


# The six invocation methods plus `bind_tools`, per `docs/06` §1.5.
_INTERCEPTED = ("invoke", "ainvoke", "stream", "astream", "batch", "abatch")


class _InstrumentedModel:
    """A proxy that forwards everything and intercepts the invocation methods.

    Forwarding matters as much as intercepting: a model that quietly lost a
    provider-specific attribute, or `bind_tools`, would break the agent for a reason
    unrelated to any fault -- and the report would blame the agent.
    """

    def __init__(self, model: Any, engine: ChaosEngine, name: str = "default") -> None:
        """Wrap a chat model.

        Args:
            model: The model to proxy.
            engine: The engine to route crossings through.
            name: The LLM alias targets use and the trace records.
        """
        object.__setattr__(self, "_alc_model", model)
        object.__setattr__(self, "_alc_engine", engine)
        object.__setattr__(self, "_alc_name", name)

    def __getattr__(self, item: str) -> Any:
        """Forward anything not intercepted.

        Args:
            item: The attribute name.

        Returns:
            The underlying model's attribute.
        """
        return getattr(object.__getattribute__(self, "_alc_model"), item)

    def __repr__(self) -> str:
        """Readable representation naming the wrapped model.

        Returns:
            e.g. ``<instrumented FakeChatModel as 'default'>``.
        """
        model = object.__getattribute__(self, "_alc_model")
        name = object.__getattribute__(self, "_alc_name")
        return f"<instrumented {type(model).__name__} as {name!r}>"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Bind tools and keep the result instrumented.

        A bound copy that lost its instrumentation would silently stop being watched,
        and agents bind tools as a matter of course -- this is the common path, not an
        edge case.

        Args:
            tools: The tools to bind.
            **kwargs: Forwarded to the model.

        Returns:
            An instrumented proxy around the bound model.
        """
        model = object.__getattribute__(self, "_alc_model")
        engine = object.__getattribute__(self, "_alc_engine")
        name = object.__getattribute__(self, "_alc_name")
        return _InstrumentedModel(model.bind_tools(tools, **kwargs), engine, name)

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Route a synchronous invocation through the plan.

        Args:
            messages: The conversation.
            *args: Forwarded.
            **kwargs: Forwarded.

        Returns:
            The model's response, possibly faulted.
        """
        return self._route("invoke", messages, args, kwargs)

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Route an asynchronous invocation through the plan.

        Args:
            messages: The conversation.
            *args: Forwarded.
            **kwargs: Forwarded.

        Returns:
            The model's response, possibly faulted.
        """
        return await self._aroute("ainvoke", messages, args, kwargs)

    def batch(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Route a batch invocation.

        Args:
            messages: The batch.
            *args: Forwarded.
            **kwargs: Forwarded.

        Returns:
            The responses.
        """
        return self._route("batch", messages, args, kwargs)

    async def abatch(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Route an asynchronous batch invocation.

        Args:
            messages: The batch.
            *args: Forwarded.
            **kwargs: Forwarded.

        Returns:
            The responses.
        """
        return await self._aroute("abatch", messages, args, kwargs)

    def stream(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Materialize the stream, fault it once, and re-yield it as one chunk.

        v0.1 limitation, recorded in the trace rather than pretended away: per-chunk
        faults would need a streaming-aware fault protocol that does not exist, and
        faulting a partial chunk would produce findings about the harness's own
        chunking rather than about the agent.

        Args:
            messages: The conversation.
            *args: Forwarded.
            **kwargs: Forwarded.

        Yields:
            One chunk carrying the whole faulted response.
        """
        engine = object.__getattribute__(self, "_alc_engine")
        model = object.__getattribute__(self, "_alc_model")
        name = object.__getattribute__(self, "_alc_name")
        engine.note(
            f"stream() on {name!r} was materialized and faulted once, then re-yielded "
            "as a single chunk (v0.1 limitation: no per-chunk faults)"
        )

        def materialize(payload: Any, *inner: Any, **kw: Any) -> Any:
            return _join_chunks(list(model.stream(payload, *inner, **kw)))

        yield self._route("stream", messages, args, kwargs, call=materialize)

    async def astream(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Async twin of `stream`, with the same materializing limitation.

        Args:
            messages: The conversation.
            *args: Forwarded.
            **kwargs: Forwarded.

        Yields:
            One chunk carrying the whole faulted response.
        """
        engine = object.__getattribute__(self, "_alc_engine")
        model = object.__getattribute__(self, "_alc_model")
        name = object.__getattribute__(self, "_alc_name")
        engine.note(
            f"astream() on {name!r} was materialized and faulted once, then re-yielded "
            "as a single chunk (v0.1 limitation: no per-chunk faults)"
        )

        async def materialize(payload: Any, *inner: Any, **kw: Any) -> Any:
            return _join_chunks([chunk async for chunk in model.astream(payload, *inner, **kw)])

        yield await self._aroute("astream", messages, args, kwargs, call=materialize)

    def _route(
        self,
        method: str,
        messages: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        call: Callable[..., Any] | None = None,
    ) -> Any:
        """Route one synchronous model call.

        Args:
            method: Which method was called.
            messages: The conversation.
            args: Extra positional arguments.
            kwargs: Extra keyword arguments.
            call: Override the underlying callable, used by `stream`.

        Returns:
            The response.
        """
        engine = object.__getattribute__(self, "_alc_engine")
        model = object.__getattribute__(self, "_alc_model")
        name = object.__getattribute__(self, "_alc_name")
        target = call or getattr(model, method)
        return engine.route_sync(
            _with_lc_messages(target, messages),
            layer="llm",
            name=name,
            args=(from_langchain(messages), *args),
            kwargs=kwargs,
        )

    async def _aroute(
        self,
        method: str,
        messages: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        call: Callable[..., Any] | None = None,
    ) -> Any:
        """Route one asynchronous model call.

        Args:
            method: Which method was called.
            messages: The conversation.
            args: Extra positional arguments.
            kwargs: Extra keyword arguments.
            call: Override the underlying callable, used by `astream`.

        Returns:
            The response.
        """
        engine = object.__getattribute__(self, "_alc_engine")
        model = object.__getattribute__(self, "_alc_model")
        name = object.__getattribute__(self, "_alc_name")
        target = call or getattr(model, method)
        return await engine.route_async(
            _with_lc_messages(target, messages),
            layer="llm",
            name=name,
            args=(from_langchain(messages), *args),
            kwargs=kwargs,
        )


def _with_lc_messages(target: Callable[..., Any], original: Any) -> Callable[..., Any]:
    """Adapt a model call to receive normalized messages and pass LangChain ones.

    Faults operate on the normalized form -- that is what makes one fault catalog
    serve both adapters -- but the model wants its own message classes back. The
    conversion happens here rather than in the engine, so the core never learns what
    a `HumanMessage` is.

    Args:
        target: The underlying model method.
        original: The messages as the caller passed them, used to decide whether
            conversion is wanted at all.

    Returns:
        A callable taking normalized messages.
    """
    wants_objects = not isinstance(original, str) and not (
        isinstance(original, list) and all(isinstance(m, dict) for m in original)
    )

    def call(messages: Any, *args: Any, **kwargs: Any) -> Any:
        payload = to_langchain(messages) if wants_objects and messages else messages
        return target(payload, *args, **kwargs)

    return call


def _join_chunks(chunks: list[Any]) -> Any:
    """Concatenate streamed chunks into a single response.

    Args:
        chunks: What the stream yielded.

    Returns:
        One chunk carrying the joined content, or `None` for an empty stream.
    """
    if not chunks:
        return None
    first = chunks[0]
    if not hasattr(first, "content"):
        return "".join(str(chunk) for chunk in chunks)
    joined = "".join(str(getattr(chunk, "content", "")) for chunk in chunks)
    try:
        return type(first)(content=joined)
    except Exception:
        return joined


def instrument_model(model: Any, engine: ChaosEngine, *, name: str = "default") -> Any:
    """Wrap a chat model so its calls become `(llm, …)` crossings.

    Args:
        model: The chat model.
        engine: The engine to route through.
        name: The LLM alias targets use and the trace records.

    Returns:
        A proxy forwarding everything it does not intercept.
    """
    if isinstance(model, _InstrumentedModel):
        return model
    return _InstrumentedModel(model, engine, name)


def wrap_langchain_tools(tools: Sequence[Any], engine: ChaosEngine) -> list[Any]:
    """Instrument LangChain tools in place.

    LangChain dispatches on a tool's `name`, and rebuilding the object would lose the
    schema it derived from the function's signature. So the tool's underlying
    callables are replaced and the tool object is handed back -- the same reasoning as
    `_wrap_node`.

    Args:
        tools: The tools to instrument.
        engine: The engine to route through.

    Returns:
        The same tool objects, instrumented.
    """
    out: list[Any] = []
    for tool in tools:
        if getattr(tool, _MARKER, False):
            out.append(tool)
            continue
        name = str(getattr(tool, "name", getattr(tool, "__name__", "tool")))
        # Exactly one sync and one async slot. A LangChain tool exposes `func` *and*
        # `_run`, both routing to the same callable, so wrapping every match counts
        # one invocation twice and every call-count probe reads double.
        for group in (("func", "_run"), ("coroutine", "_arun")):
            for attribute in group:
                inner = getattr(tool, attribute, None)
                if inner is None or not callable(inner):
                    continue
                with contextlib.suppress(AttributeError, TypeError, ValueError):
                    setattr(tool, attribute, engine.wrap_callable(inner, layer="tool", name=name))
                    break
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            setattr(tool, _MARKER, True)
        out.append(tool)
    return out
