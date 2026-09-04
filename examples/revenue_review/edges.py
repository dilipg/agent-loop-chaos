"""Routing for the buggy review graph, including the back-edge.

```
planner -> data_analyst -> risk_scorer -> writer -> reviewer -> END
                                            ^          |
                                            +----------+  (REVISE)
```

The `reviewer -> writer` back-edge is what gives `LoopTrapFault` somewhere to trap
and the retry probes something to observe.
"""

from __future__ import annotations

from typing import Any

__all__ = ["END", "NEXT", "route", "route_after_reviewer"]

#: Terminal node label. Plain string, because there is no framework here.
END = "__end__"

NEXT = {
    "planner": "data_analyst",
    "data_analyst": "risk_scorer",
    "risk_scorer": "writer",
    "writer": "reviewer",
}


def route_after_reviewer(state: dict[str, Any]) -> str:
    """Decide whether the draft is done or goes back to the writer.

    Args:
        state: The working state.

    Returns:
        `END` on approval, otherwise ``"writer"``.
    """
    # Weakness 6, second half: `state["attempts"]` is incremented by the reviewer on
    # every pass and never compared to anything here, so a model that never says
    # APPROVE loops until the harness's `max_steps` limit stops it.
    if "APPROVE" in str(state.get("verdict", "")):
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
