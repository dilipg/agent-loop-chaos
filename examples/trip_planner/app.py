"""Builds and compiles the trip planner graph.

`build_app(engine)` is the instrumented entrypoint: it wraps every tool with its
`side_effecting` declaration and wraps the model, which is what makes
`metrics.tool_calls` and `metrics.llm_calls` real. `graph` is the same app wired to
nothing, so `examples.trip_planner.app:graph` resolves and the module runs on its own.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from .. import fake_model
from . import tools as tools_module
from .edges import route_after_plan, route_after_summarize
from .nodes import TripState, build_nodes

__all__ = ["build_app", "build_graph", "graph", "main"]

QUERY = "3-day packing list for Paris, and the cheapest flight from Delhi"


def _instrument(engine: Any) -> dict[str, Any]:
    """Register every tool with its side-effect declaration.

    Args:
        engine: The `ChaosEngine`, or `None` for the uninstrumented app.

    Returns:
        The tool table, wrapped when an engine was supplied.
    """
    tools_module.reset()
    raw = {
        "get_weather_data": tools_module.get_weather_data,
        "search_flights": tools_module.search_flights,
        "get_account": tools_module.get_account,
        "hold_booking": tools_module.hold_booking,
    }
    if engine is None:
        return raw
    # `hold_booking` places a real hold; the D-23 gate refuses a destructive fault
    # against it unless the scenario opts in by name.
    effects = {"hold_booking": True}
    return {
        name: engine.tool(fn, name=name, side_effecting=effects.get(name, False))
        for name, fn in raw.items()
    }


def build_graph(engine: Any = None, model: Any = None) -> Any:
    """Build the uncompiled graph.

    Args:
        engine: The `ChaosEngine`, or `None`.
        model: A model callable, defaulting to the scripted fake.

    Returns:
        The `StateGraph`.
    """
    chat = model or fake_model.respond
    if engine is not None:
        # The engine's default model name. One model, one name, and a scenario
        # that says `llm: default` -- the conventional target -- reaches it.
        chat = engine.llm(chat, name="default")
    nodes = build_nodes(chat, _instrument(engine))

    builder = StateGraph(TripState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.set_entry_point("plan")
    builder.add_conditional_edges(
        "plan", route_after_plan, {"fetch_weather": "fetch_weather", "ask_clarify": "ask_clarify"}
    )
    builder.add_edge("ask_clarify", END)
    builder.add_edge("fetch_weather", "fetch_flights")
    builder.add_edge("fetch_flights", "summarize")
    builder.add_conditional_edges(
        "summarize",
        route_after_summarize,
        {"fetch_weather": "fetch_weather", "respond": "respond"},
    )
    builder.add_edge("respond", END)
    return builder


def build_app(engine: Any = None, model: Any = None) -> Any:
    """Build the compiled, instrumented app.

    The D-61 builder convention: a callable whose first parameter is named `engine`
    is handed the engine and returns the real agent, which is what lets a suite say
    `entrypoint: examples.trip_planner.app:build_app`.

    Args:
        engine: The `ChaosEngine`, or `None`.
        model: A model callable, defaulting to the scripted fake.

    Returns:
        The compiled graph, instrumented when an engine was supplied.
    """
    builder = build_graph(engine, model)
    if engine is None:
        return builder.compile()
    from langgraph.checkpoint.memory import MemorySaver

    from agent_loop_chaos.adapters.langgraph import instrument_graph

    # A checkpointer, so `resume.checkpoint_rollback` has something to roll back to.
    # Replaying a committed node is how a real resume double-fires a booking, and
    # without persistence that scenario cannot fire at all.
    compiled = builder.compile(checkpointer=MemorySaver())
    instrumented = instrument_graph(compiled, engine, intercept_checkpoints=True)

    def agent(query: Any = None, **state: Any) -> Any:
        """Answer one request.

        An agent returns its answer, not its internal state. Handing back the whole
        state dict would make every output assertion trivially true -- `answer` could
        be empty and `output_non_empty` would still pass on the other keys.

        Args:
            query: The traveller's request.
            **state: Any other starting state.

        Returns:
            The final answer.
        """
        payload = {"query": query or QUERY, "messages": [], "attempts": 0, **state}
        final = instrumented.invoke(payload)
        return final.get("answer")

    return agent


#: The uninstrumented app, so `python -m examples.trip_planner.app` works.
graph = build_graph().compile()


def main() -> None:
    """Run the happy path and print the answer."""
    final = graph.invoke({"query": QUERY, "messages": [], "attempts": 0})
    print(final.get("answer") or final.get("packing_list"))


if __name__ == "__main__":
    main()
