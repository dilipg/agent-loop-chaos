"""Routing for the trip planner."""

from __future__ import annotations

from typing import Any

__all__ = ["route_after_plan", "route_after_summarize"]


def route_after_plan(state: dict[str, Any]) -> str:
    """Send the traveller to clarification when no destination was found.

    Args:
        state: The graph state.

    Returns:
        The next node's name.
    """
    return "fetch_weather" if state.get("location") else "ask_clarify"


def route_after_summarize(state: dict[str, Any]) -> str:
    """Go back for the forecast when the packing list came out empty.

    Args:
        state: The graph state.

    Returns:
        The next node's name.
    """
    if not state.get("packing_list"):
        return "fetch_weather"
    return "respond"
