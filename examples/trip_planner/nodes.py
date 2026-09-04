"""Graph nodes for the trip planner.

Reads like a first working version: the shape is right, the names are sensible, and
every step does what its name says. What it does not do is check anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

__all__ = ["NODES", "TripState", "build_nodes"]

_PROMPTS = Path(__file__).parent / "prompts"
MAX_ATTEMPTS = 3


class TripState(TypedDict, total=False):
    """What flows between nodes."""

    query: str
    location: str | None
    origin: str | None
    weather: list[dict[str, Any]] | None
    flights: dict[str, Any] | None
    packing_list: list[str] | None
    note: str
    answer: str
    messages: Annotated[list[Any], add_messages]
    attempts: int


def _history(state: TripState) -> str:
    """Render the conversation so far, for the prompt.

    The objective lives here and nowhere else: no node re-asserts it from
    `state["query"]`. That is the weakness -- anything that erodes the conversation
    erodes the goal, and the agent has no durable copy to fall back on.

    Args:
        state: The graph state.

    Returns:
        The messages, newest last.
    """
    return " | ".join(str(m) for m in (state.get("messages") or []))


def _content(reply: Any) -> str:
    """Pull the text out of whatever the client returned.

    Args:
        reply: An envelope mapping, or a bare string.

    Returns:
        The message text.
    """
    return str(reply["content"]) if isinstance(reply, dict) else str(reply)


def _prompt(name: str, **values: Any) -> str:
    """Render a prompt asset.

    Args:
        name: File stem under `prompts/`.
        **values: `{{placeholder}}` values.

    Returns:
        The rendered prompt.
    """
    text = (_PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    return text


def build_nodes(model: Any, tools: dict[str, Any]) -> dict[str, Any]:
    """Build the node functions against a model and a tool table.

    Args:
        model: A callable returning `{"content": str, "finish_reason": str}`.
        tools: The tool table, instrumented or raw.

    Returns:
        Node name to function.
    """
    weather_tool = tools["get_weather_data"]
    flights_tool = tools["search_flights"]
    account_tool = tools["get_account"]
    booking_tool = tools["hold_booking"]

    def plan(state: TripState) -> dict[str, Any]:
        """Work out where the traveller is going."""
        reply = model(_prompt("plan", query=state.get("query", "")))
        parsed = json.loads(_content(reply))
        return {
            "location": parsed.get("location"),
            "origin": parsed.get("origin"),
            "attempts": 0,
            "messages": [f"planning trip: {state.get('query', '')}"],
        }

    def ask_clarify(state: TripState) -> dict[str, Any]:
        """Ask for a destination when the request did not carry one."""
        return {"answer": "Which city are you travelling to?", "messages": ["asked for a city"]}

    def fetch_weather(state: TripState) -> dict[str, Any]:
        """Pull the forecast for the destination."""
        rows = weather_tool(state["location"])
        return {"weather": rows, "messages": [f"forecast: {rows[0]['condition']} on day one"]}

    def fetch_flights(state: TripState) -> dict[str, Any]:
        """Pull flight options for the route."""
        quote = flights_tool(state.get("origin") or "DEL", state.get("location") or "Paris")
        return {"flights": quote, "messages": [f"cheapest fare {quote['cheapest']['price_usd']}"]}

    def summarize(state: TripState) -> dict[str, Any]:
        """Turn the forecast into a packing list."""
        quote = state.get("flights") or {}
        reply = model(
            _prompt(
                "summarize",
                history=_history(state),
                weather=state.get("weather"),
                flight=quote.get("cheapest"),
                notes=quote.get("notes", ""),
                account=account_tool(),
            )
        )
        content = _content(reply)
        parsed = json.loads(content)
        return {
            "packing_list": parsed.get("packing_list"),
            "note": parsed.get("note", ""),
            "attempts": int(state.get("attempts", 0)) + 1,
            "messages": [content],
        }

    def respond(state: TripState) -> dict[str, Any]:
        """Write the final answer and hold the seat."""
        quote = state.get("flights") or {}
        cheapest = quote.get("cheapest") or {}
        booking_tool(cheapest.get("flight_id", "?"), "R. Mehta")
        answer = _content(
            model(
                _prompt(
                    "respond",
                    history=_history(state),
                    packing_list=state.get("packing_list"),
                    note=state.get("note", ""),
                    flight=cheapest,
                )
            )
        )
        return {"answer": answer, "messages": [answer]}

    return {
        "plan": plan,
        "ask_clarify": ask_clarify,
        "fetch_weather": fetch_weather,
        "fetch_flights": fetch_flights,
        "summarize": summarize,
        "respond": respond,
    }


#: Uninstrumented nodes, so the module is importable on its own.
NODES: dict[str, Any] = {}
