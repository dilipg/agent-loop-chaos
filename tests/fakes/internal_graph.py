"""A service that compiles and invokes its graph inside a function.

This is the shape most services have -- `run_user_pipeline(...)` builds the graph,
compiles it, invokes it and returns a result -- and it is the shape that defeats
`instrument_graph`, because the caller never gets to hold the graph object. Nothing
here imports this library.
"""

from __future__ import annotations

from typing import Any, TypedDict


class State(TypedDict, total=False):
    """The channels the little graph carries."""

    query: str
    facts: dict[str, Any]
    answer: str


def _gather(state: State) -> dict[str, Any]:
    return {"facts": {"temp_c": 21, "city": "Paris"}}


def _answer(state: State) -> dict[str, Any]:
    facts = state.get("facts") or {}
    if "temp_c" not in facts:
        return {"answer": "I could not read the weather."}
    return {"answer": f"It is {facts['temp_c']}C in {facts.get('city', 'unknown')}."}


def run(query: str = "what should I pack?") -> str:
    """Build, compile and invoke the graph, returning only the answer.

    Args:
        query: The user's question.

    Returns:
        The answer. The graph itself never leaves this function.
    """
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(State)
    builder.add_node("gather", _gather)
    builder.add_node("answer", _answer)
    builder.add_edge(START, "gather")
    builder.add_edge("gather", "answer")
    builder.add_edge("answer", END)
    compiled = builder.compile()
    return str(compiled.invoke({"query": query})["answer"])


def _build_compiled() -> Any:
    """Compile the same graph once, the way a warm-start singleton is built."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(State)
    builder.add_node("gather", _gather)
    builder.add_node("answer", _answer)
    builder.add_edge(START, "gather")
    builder.add_edge("gather", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


#: Compiled at import, as a service reusing one graph across invocations does. This is
#: the shape `StateGraph.compile` patching cannot reach: by the time a run starts, the
#: compile has already happened.
COMPILED = _build_compiled()


def run_singleton(query: str = "what should I pack?") -> str:
    """Invoke the module-level graph.

    Args:
        query: The user's question.

    Returns:
        The answer.
    """
    return str(COMPILED.invoke({"query": query})["answer"])
