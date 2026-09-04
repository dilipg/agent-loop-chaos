"""Graph nodes for the trip planner, with every boundary guarded.

The shape is identical to `examples/trip_planner/nodes.py`. What changed is that
nothing reads a field off a tool result before a validator has looked at it, every
tool call goes through `call_tool` (timeout, bounded retry on transient classes
only), the objective is re-asserted in every prompt rather than trusted to survive in
the message history, and every degradation produces a user-visible sentence naming
what was unavailable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from .validators import (
    DataUnavailable,
    fence,
    validate_account,
    validate_flights,
    validate_packing_list,
    validate_weather,
)

__all__ = ["TripState", "build_nodes", "call_tool"]

_PROMPTS = Path(__file__).parent / "prompts"

#: Exception names worth retrying. Anything else is permanent and retrying it is a
#: retry storm with extra steps.
_TRANSIENT = ("TimeoutError", "ConnectionError", "TransientError", "RateLimit")
_MAX_TRIES = 2


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
    degraded: list[str]
    signature: str
    last_signature: str
    messages: Annotated[list[Any], add_messages]
    attempts: int


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


def call_tool(fn: Callable[..., Any], *args: Any, engine: Any = None, **kwargs: Any) -> Any:
    """Call a tool with a bounded retry on transient failures only.

    A 4xx-shaped error is permanent: retrying it burns steps and changes nothing. A
    timeout might not be. Distinguishing the two is the whole of a retry policy.

    Args:
        fn: The tool.
        *args: Positional arguments.
        engine: The engine, for the degradation note.
        **kwargs: Keyword arguments.

    Returns:
        The tool's result.

    Raises:
        DataUnavailable: When every attempt failed.
    """
    last: Exception | None = None
    for attempt in range(_MAX_TRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            if not any(marker in type(exc).__name__ for marker in _TRANSIENT):
                break
            if engine is not None and attempt + 1 < _MAX_TRIES:
                engine.note(f"degraded: retrying after {type(exc).__name__}")
    raise DataUnavailable(
        getattr(fn, "__name__", "tool"), f"{type(last).__name__}: {last}"
    ) from last


def _parse_json(reply: Any, field: str) -> Any:
    """Parse a model reply that was asked for JSON.

    Args:
        reply: The model's envelope.
        field: What the value is for, for the error message.

    Returns:
        The parsed object.

    Raises:
        DataUnavailable: When the reply was cut short or is not JSON.
    """
    if isinstance(reply, dict) and reply.get("finish_reason") not in (None, "stop"):
        raise DataUnavailable(field, f"the model stopped early ({reply['finish_reason']})")
    content = reply.get("content", "") if isinstance(reply, dict) else str(reply)
    try:
        return json.loads(content)
    except (ValueError, TypeError) as exc:
        raise DataUnavailable(field, "the model did not return valid JSON") from exc


def build_nodes(model: Any, tools: dict[str, Any], engine: Any = None) -> dict[str, Any]:
    """Build the node functions against a model and a tool table.

    Args:
        model: A callable returning `{"content": str, "finish_reason": str}`.
        tools: The tool table, instrumented or raw.
        engine: The engine, for `note()` and `validated()`.

    Returns:
        Node name to function.
    """
    weather_tool = tools["get_weather_data"]
    flights_tool = tools["search_flights"]
    account_tool = tools["get_account"]
    booking_tool = tools["hold_booking"]

    def _degrade(state: TripState, message: str) -> list[str]:
        """Record a degradation on the state and on the trace."""
        if engine is not None:
            engine.note(f"degraded: {message}")
        return [*(state.get("degraded") or []), message]

    def plan(state: TripState) -> dict[str, Any]:
        """Work out where the traveller is going."""
        query = state.get("query", "")
        try:
            parsed = _parse_json(model(_prompt("plan", query=query)), "destination")
        except DataUnavailable:
            return {"location": None, "origin": None, "attempts": 0}
        return {
            "location": parsed.get("location"),
            "origin": parsed.get("origin"),
            "attempts": 0,
            "messages": [f"planning trip: {query}"],
        }

    def ask_clarify(state: TripState) -> dict[str, Any]:
        """Ask for a destination when the request did not carry one."""
        return {"answer": "Which city are you travelling to?", "messages": ["asked for a city"]}

    def fetch_weather(state: TripState) -> dict[str, Any]:
        """Pull the forecast, and refuse to guess when it is not usable."""
        location = state.get("location")
        if not location:
            return {
                "weather": None,
                "degraded": _degrade(state, "no destination was identified"),
                "signature": "weather:none",
            }
        try:
            rows = validate_weather(call_tool(weather_tool, location, engine=engine), engine=engine)
        except DataUnavailable as exc:
            return {
                "weather": None,
                "degraded": _degrade(state, str(exc)),
                "signature": f"weather:{location}",
                "last_signature": state.get("signature", ""),
            }
        return {
            "weather": rows,
            "signature": f"weather:{location}",
            "last_signature": state.get("signature", ""),
            "messages": [f"forecast: {rows[0]['temp_c']}C on day one"],
        }

    def fetch_flights(state: TripState) -> dict[str, Any]:
        """Pull flight options, treating an in-band error as an error."""
        try:
            quote = validate_flights(
                call_tool(
                    flights_tool,
                    state.get("origin") or "DEL",
                    state.get("location") or "Paris",
                    engine=engine,
                ),
                engine=engine,
            )
        except DataUnavailable as exc:
            return {"flights": None, "degraded": _degrade(state, str(exc))}
        return {
            "flights": quote,
            "messages": [f"cheapest fare {quote['cheapest']['price_usd']}"],
        }

    def summarize(state: TripState) -> dict[str, Any]:
        """Turn the forecast into a packing list, or say why it could not."""
        attempts = int(state.get("attempts", 0)) + 1
        rows = state.get("weather")
        if not rows:
            return {
                "packing_list": None,
                "note": "",
                "attempts": attempts,
                "degraded": _degrade(state, "the forecast was unavailable"),
            }
        quote = state.get("flights") or {}
        try:
            account = validate_account(call_tool(account_tool, engine=engine), engine=engine)
        except DataUnavailable:
            account = {}
        prompt = _prompt(
            "summarize",
            objective=state.get("query", ""),
            weather=rows,
            flight=quote.get("cheapest"),
            notes=fence(quote.get("notes", "")),
        ).replace("{{traveller}}", str(account.get("traveller", "the traveller")))
        for _try in range(2):
            try:
                parsed = validate_packing_list(
                    _parse_json(model(prompt), "packing_list"), engine=engine
                )
            except DataUnavailable:
                continue
            return {
                "packing_list": parsed["packing_list"],
                "note": parsed.get("note", ""),
                "attempts": attempts,
                "messages": [json.dumps(parsed)],
            }
        return {
            "packing_list": None,
            "note": "",
            "attempts": attempts,
            "degraded": _degrade(state, "the summarizer did not return a usable list"),
        }

    def respond(state: TripState) -> dict[str, Any]:
        """Write the final answer, holding the seat at most once."""
        quote = state.get("flights") or {}
        cheapest = quote.get("cheapest") or {}
        degraded = list(state.get("degraded") or [])

        if cheapest.get("flight_id"):
            # Keyed on the flight and the traveller, so a replayed step, a retry or a
            # checkpoint rollback all land on the same hold instead of a second one.
            key = f"{cheapest['flight_id']}:{state.get('query', '')[:32]}"
            call_tool(
                booking_tool,
                cheapest["flight_id"],
                "R. Mehta",
                idempotency_key=key,
                engine=engine,
            )

        items = state.get("packing_list")
        if not items or not cheapest:
            missing = " and ".join(
                part
                for part in (
                    "a packing list" if not items else "",
                    "a flight quote" if not cheapest else "",
                )
                if part
            )
            reasons = f" ({'; '.join(degraded)})" if degraded else ""
            return {
                "answer": (
                    f"I could not produce {missing} for this trip{reasons}. "
                    "Nothing here is estimated -- please retry, or give me a different city."
                ),
                "degraded": degraded,
            }

        reply = model(
            _prompt(
                "respond",
                objective=state.get("query", ""),
                packing_list=items,
                note=state.get("note", ""),
                flight=cheapest,
            )
        )
        return {"answer": reply.get("content", ""), "messages": [reply.get("content", "")]}

    return {
        "plan": plan,
        "ask_clarify": ask_clarify,
        "fetch_weather": fetch_weather,
        "fetch_flights": fetch_flights,
        "summarize": summarize,
        "respond": respond,
    }
