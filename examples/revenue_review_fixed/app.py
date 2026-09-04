"""Wires the fixed revenue-risk review. Same graph, same tools, no weaknesses.

This tree is the *demo suite's* negative control and the M8 gate: it must pass
every scenario. `tests/fakes/good_agent` is the fake suite's control, and the two
are not interchangeable (`docs/07-TESTING.md` §8 job 4).

The runner never lets a node's failure become the run's failure: whatever went
wrong, the caller gets a paragraph that names what was unavailable.
"""

from __future__ import annotations

from typing import Any

from .. import fake_model
from ..revenue_review import tools as tools_module
from .edges import END, route
from .nodes import NODES

__all__ = ["build_app", "graph"]

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
        ``app(objective, state=None) -> str``.
    """
    tools = _instrument(engine)
    model = llm if llm is not None else fake_model.respond
    if engine is not None:
        model = engine.llm(model, name="reviewer")

    def app(objective: str, state: dict[str, Any] | None = None) -> str:
        """Walk the graph, then return a report or an explicit degradation notice.

        Args:
            objective: The review objective.
            state: Optional starting state.

        Returns:
            The approved paragraph, or a sentence naming what was unavailable.
            Never an empty string.
        """
        working: dict[str, Any] = dict(state or {})
        working.setdefault("objective", objective)
        working.setdefault("messages", [{"role": "user", "content": objective}])
        working.setdefault("attempts", 0)
        working.setdefault("degraded", [])
        node = "planner"
        for _ in range(_MAX_NODE_VISITS):
            working = NODES[node](working, tools, model, engine)
            node = route(node, working)
            if node == END:
                break

        report = str(working.get("report") or working.get("draft") or "").strip()
        degraded = working.get("degraded") or []
        if not report:
            return (
                "I could not complete the revenue-risk review. Unavailable: "
                + ("; ".join(degraded) or "the report paragraph was never produced")
                + "."
            )
        if degraded:
            return f"{report}\n\nDegraded: {'; '.join(degraded)}."
        return report

    return app


#: Module-level entrypoint, so `examples.revenue_review_fixed.app:graph` resolves.
#: Uninstrumented, exactly as in the buggy tree; use `build_app(engine, llm)`.
graph = build_app()
