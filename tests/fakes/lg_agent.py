"""A real LangGraph fixture: four nodes, a conditional back-edge, a reducer.

Deliberately small. This is the graph the adapter tests instrument; the demo agent
lives in `examples/`. Both a sync and an async variant, because the adapter must
wrap each with a matching wrapper.
"""

# No `from __future__ import annotations` here, deliberately. It turns every
# annotation into a string, and LangGraph introspects a node's `config` annotation
# against the real `RunnableConfig` type object -- so a module with postponed
# evaluation makes LangGraph warn that a correctly typed node is wrongly typed. A
# fixture that lied about its annotations would not represent a real graph.
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

MAX_ATTEMPTS = 2


class TripState(TypedDict, total=False):
    """The graph's state. `messages` carries a reducer, which is what makes a
    type fault interesting: handing `add_messages` a bare string is a classic break.
    """

    query: str
    location: str
    weather: list[dict[str, Any]]
    summary: str
    attempts: int
    messages: Annotated[list[Any], add_messages]


def get_weather(location: str) -> list[dict[str, Any]]:
    """Return a fixed forecast.

    Args:
        location: Where to look up.

    Returns:
        One record per day.
    """
    return [{"date": "2026-08-26", "condition": "sunny", "temp_c": 31}]


def search_flights(origin: str, dest: str) -> dict[str, Any]:
    """Return a fixed flight quote.

    Args:
        origin: Departure city.
        dest: Arrival city.

    Returns:
        The quote.
    """
    return {"price_usd": 240, "carrier": "Contoso Air"}


def build(model: Any, *, tools: dict[str, Any] | None = None, is_async: bool = False) -> Any:
    """Build the graph.

    Args:
        model: A callable standing in for a chat model.
        tools: Instrumented tools, defaulting to the module-level ones.
        is_async: Build async node functions instead of sync ones.

    Returns:
        An uncompiled `StateGraph`, so a test can instrument before or after compile.
    """
    weather = (tools or {}).get("get_weather", get_weather)
    flights = (tools or {}).get("search_flights", search_flights)

    def plan(state: TripState) -> dict[str, Any]:
        return {"location": "Paris", "attempts": 0, "messages": [f"plan: {state.get('query')}"]}

    def fetch(state: TripState) -> dict[str, Any]:
        return {"weather": weather(state["location"]), "messages": ["fetched"]}

    def summarize(state: TripState, config: Optional[RunnableConfig] = None) -> dict:  # noqa: UP045
        rows = state.get("weather") or []
        text = str(model(f"summarise {rows}"))
        return {"summary": text, "attempts": int(state.get("attempts", 0)) + 1}

    def respond(state: TripState) -> dict[str, Any]:
        flights("LHR", state.get("location", "?"))
        return {"messages": [state.get("summary", "")]}

    async def aplan(state: TripState) -> dict[str, Any]:
        return plan(state)

    async def afetch(state: TripState) -> dict[str, Any]:
        return fetch(state)

    async def asummarize(state: TripState, config: RunnableConfig | None = None) -> dict[str, Any]:
        return summarize(state, config)

    async def arespond(state: TripState) -> dict[str, Any]:
        return respond(state)

    def route(state: TripState) -> str:
        if state.get("summary") or int(state.get("attempts", 0)) >= MAX_ATTEMPTS:
            return "respond"
        return "summarize"

    graph = StateGraph(TripState)
    nodes = (
        {"plan": aplan, "fetch": afetch, "summarize": asummarize, "respond": arespond}
        if is_async
        else {"plan": plan, "fetch": fetch, "summarize": summarize, "respond": respond}
    )
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "fetch")
    graph.add_edge("fetch", "summarize")
    graph.add_conditional_edges(
        "summarize", route, {"summarize": "summarize", "respond": "respond"}
    )
    graph.add_edge("respond", END)
    return graph
