"""Routing for the trip planner.

The back-edge is capped and checks for progress. A loop that retries the same call
with the same arguments is not retrying, it is spinning, and the cap is what turns
that from a step-limit timeout into an explicit answer.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MAX_ATTEMPTS", "route_after_plan", "route_after_summarize"]

#: Two attempts at the forecast, then give the traveller what we have.
MAX_ATTEMPTS = 2


def route_after_plan(state: dict[str, Any]) -> str:
    """Send the traveller to clarification when no destination was found.

    Args:
        state: The graph state.

    Returns:
        The next node's name.
    """
    return "fetch_weather" if state.get("location") else "ask_clarify"


def route_after_summarize(state: dict[str, Any]) -> str:
    """Go back for the forecast only while a retry could still change the answer.

    Args:
        state: The graph state.

    Returns:
        The next node's name.
    """
    if state.get("packing_list"):
        return "respond"
    if int(state.get("attempts", 0)) >= MAX_ATTEMPTS:
        return "respond"
    if state.get("last_signature") and state["last_signature"] == state.get("signature"):
        return "respond"
    return "fetch_weather"
