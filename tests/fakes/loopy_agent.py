"""A while-loop over a condition the tool controls.

**Planted weakness:** no cycle detection and no progress check, so a pinned tool
result keeps it going until an external budget intervenes.
"""

from __future__ import annotations


def run(*, status: str, budget: int) -> int:
    """Poll until the tool reports done.

    Args:
        status: What the tool keeps returning.
        budget: An external cap, standing in for the engine's limits.

    Returns:
        How many polls were made.
    """
    calls = 0
    while calls < budget:
        calls += 1
        if status == "done":
            return calls
    return calls
