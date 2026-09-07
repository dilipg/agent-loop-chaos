"""Instrument a graph the caller never hands over.

`instrument_graph(app, engine)` needs the graph object, and most services do not have
one to give. The common shape is a function that builds, compiles, invokes and returns
a result -- `run_user_pipeline(entity_id, ..., pascal_db, astra_db)` -- with the graph
never leaving the call. Node, edge, state and checkpoint faults then had nothing to
attach to unless somebody wrote a harness whose only job was to expose the graph.

So `StateGraph.compile` is patched for the duration of the run, exactly as
`BaseChatModel.generate` is: any graph compiled while the run is active comes back
instrumented, wherever it was built. A graph already instrumented is left alone, since
two layers of wrapping would double every crossing.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

from ..context import Layer
from .base import Attachment

__all__ = ["LangGraphStrategy"]

logger = logging.getLogger("agent_loop_chaos")

#: Set on a compiled graph this strategy has already wrapped.
_MARKER = "_alc_instrumented"


class LangGraphStrategy:
    """Patch `StateGraph.compile` so an internally built graph is still instrumented."""

    name = "langgraph"

    def __init__(self) -> None:
        """Build the strategy. Nothing is patched until `attach`."""
        self._originals: list[tuple[Any, str, Any]] = []

    def available(self) -> str | None:
        """Report whether `langgraph` is importable.

        Returns:
            `None` when it is, else the reason it is not.
        """
        return None if find_spec("langgraph") is not None else "not installed"

    def attach(self, engine: Any, seen: dict[str, int]) -> list[Attachment]:
        """Patch `StateGraph.compile`.

        Args:
            engine: The engine to route crossings through.
            seen: Shared call tally, keyed by the attachment target.

        Returns:
            One `Attachment` per layer a wrapped graph produces, since a single patch
            supplies all of them and which ones a run uses is not knowable in advance.
        """
        from langgraph.graph import StateGraph

        original = StateGraph.compile
        self._originals.append((StateGraph, "compile", original))
        target = "langgraph.StateGraph.compile"

        def compile_(graph: Any, *args: Any, **kwargs: Any) -> Any:
            compiled = original(graph, *args, **kwargs)
            if not engine.is_active() or getattr(compiled, _MARKER, False):
                return compiled
            try:
                from ..adapters.langgraph import instrument_graph

                instrumented = instrument_graph(
                    compiled,
                    engine,
                    intercept_checkpoints=kwargs.get("checkpointer") is not None,
                )
            except Exception as exc:
                # Never break the run being observed: an un-instrumented graph is a
                # scenario that proves nothing, which the report says plainly.
                logger.warning("could not instrument a compiled graph: %s", exc)
                return compiled
            seen[target] = seen.get(target, 0) + 1
            try:
                object.__setattr__(instrumented, _MARKER, True)
            except Exception:  # pragma: no cover - a graph with no writable attrs
                logger.debug("could not mark a compiled graph as instrumented")
            return instrumented

        # A patch is deliberately looser than the generic signature it replaces.
        StateGraph.compile = compile_  # type: ignore[assignment]
        layers: tuple[Layer, ...] = ("node", "edge", "state", "checkpoint")
        return [Attachment(layer=layer, strategy=self.name, target=target) for layer in layers]

    def detach(self) -> None:
        """Restore `StateGraph.compile`. Safe to call twice."""
        while self._originals:
            owner, attribute, original = self._originals.pop()
            setattr(owner, attribute, original)
