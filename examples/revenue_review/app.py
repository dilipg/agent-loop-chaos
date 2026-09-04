"""Wires the buggy revenue-risk review into something the engine can run.

A graph, not a chain: five nodes, a dict dispatch, and a `reviewer -> writer`
back-edge. No LangGraph — the adapter is M5 and the core must stay
framework-independent.

`build_app(engine, llm)` is the instrumented entrypoint: it declares every tool's
`side_effecting` flag (so `--preset full` may start, `SAFETY.md` §1 item 3) and
wraps the model, which is what makes `metrics.llm_calls` and `metrics.tool_calls`
real. `graph` is the same app wired to nothing, so
``examples.revenue_review.app:graph`` resolves and the file runs on its own; a
scenario that wants instrumentation must go through `build_app`.
"""

from __future__ import annotations

from typing import Any

from .. import fake_model
from . import tools as tools_module
from .edges import END, route
from .nodes import NODES

__all__ = ["build_app", "graph"]

#: A pass costs 9 LLM calls, so the back-edge has room for exactly one retry inside
#: the engine's default `max_steps=25` before the limit stops the run.
_MAX_NODE_VISITS = 24


def _instrument(engine: Any) -> dict[str, Any]:
    """Register every tool with its side-effect declaration.

    Args:
        engine: The `ChaosEngine`, or `None` for the uninstrumented app.

    Returns:
        The tool table, wrapped when an engine was supplied.
    """
    if engine is None:
        return dict(tools_module.TOOLS)
    return {
        name: engine.tool(fn, name=name, side_effecting=name in tools_module.SIDE_EFFECTING)
        for name, fn in tools_module.TOOLS.items()
    }


def build_app(engine: Any = None, llm: Any = None) -> Any:
    """Build the runnable review workflow.

    Args:
        engine: A `ChaosEngine` to instrument against, or `None` to run bare.
        llm: The model callable. Defaults to `examples.fake_model.respond`.

    Returns:
        ``app(objective, state=None) -> str``, the shape `ChaosEngine.run` resolves
        under D-19.
    """
    tools = _instrument(engine)
    model = llm if llm is not None else fake_model.respond
    if engine is not None:
        model = engine.llm(model, name="reviewer")

    def app(objective: str, state: dict[str, Any] | None = None) -> str:
        """Walk the graph until the reviewer approves.

        Args:
            objective: The review objective.
            state: Optional starting state.

        Returns:
            The approved paragraph.
        """
        working: dict[str, Any] = dict(state or {})
        working.setdefault("messages", [{"role": "user", "content": objective}])
        working.setdefault("attempts", 0)
        node = "planner"
        for _ in range(_MAX_NODE_VISITS):
            working = NODES[node](working, tools, model, engine)
            node = route(node, working)
            if node == END:
                break
        # No guard that the reviewer ever approved: an empty string is a perfectly
        # possible final answer here, which is what `empty_final_answer` is for.
        return str(working.get("report", ""))

    return app


#: Module-level entrypoint, so `examples.revenue_review.app:graph` resolves. It is
#: **uninstrumented**: decorating tools binds them to one engine, so a scenario must
#: call `build_app(engine, llm)` to get a wired app.
graph = build_app()
