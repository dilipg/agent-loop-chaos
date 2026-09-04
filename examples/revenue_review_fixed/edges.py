"""Routing for the fixed review graph. Same shape, capped back-edge.

`docs/09-DEMO-AGENT.md` §5: cap `attempts` at 2, and compare the new draft to the
previous one so a loop that is not making progress ends instead of spinning until
the harness's `max_steps` limit stops it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["END", "MAX_ATTEMPTS", "NEXT", "route", "route_after_reviewer"]

END = "__end__"

#: Two review passes. A third has never once produced a different draft.
MAX_ATTEMPTS = 2

NEXT = {
    "planner": "data_analyst",
    "data_analyst": "risk_scorer",
    "risk_scorer": "writer",
    "writer": "reviewer",
}


def route_after_reviewer(state: dict[str, Any]) -> str:
    """Decide whether the draft is done, or goes back to the writer once more.

    Args:
        state: The working state.

    Returns:
        `END` on approval, on the attempt cap, or when the last pass changed
        nothing. Otherwise ``"writer"``.
    """
    if "APPROVE" in str(state.get("verdict", "")):
        return END
    if int(state.get("attempts", 0)) >= MAX_ATTEMPTS:
        return END
    if state.get("previous_draft") is not None and state.get("draft") == state["previous_draft"]:
        return END
    return "writer"


def route(node: str, state: dict[str, Any]) -> str:
    """Pick the next node.

    Args:
        node: The node that just ran.
        state: The working state.

    Returns:
        The next node name, or `END`.
    """
    if node == "reviewer":
        return route_after_reviewer(state)
    return NEXT[node]
