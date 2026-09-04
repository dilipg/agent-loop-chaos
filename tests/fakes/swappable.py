"""A fake agent whose implementation can be swapped mid-loop.

`RefinementLoop` hands failing results to `on_findings`, the caller "fixes" the
agent, and the loop re-runs. A test needs an agent that can actually be fixed
between rounds without editing a file on disk, which is what the module-level switch
here provides.
"""

from __future__ import annotations

from typing import Any

# Flipped by a test's `on_findings` to simulate a coding agent applying a patch.
FIXED = False
BROKEN_SIBLING = False


def build(engine: Any) -> Any:
    """Build an agent that reads a key a fault removes.

    Args:
        engine: The engine to register the tool on.

    Returns:
        The agent callable.
    """

    @engine.tool(name="weather", side_effecting=False)
    def weather(city: str = "Paris") -> dict[str, Any]:
        return {"city": city, "temp_c": 21}

    def agent(question: Any = None) -> str:
        data = weather()
        if FIXED:
            # The graceful branch: detect the gap, name it, invent nothing.
            if "temp_c" not in data:
                return f"Temperature for {data.get('city', 'the city')} is unavailable."
            return f"It is {data['temp_c']}C in {data['city']}."
        return f"It is {data['temp_c']}C in {data['city']}."

    return agent


def build_sibling(engine: Any) -> Any:
    """Build a second, initially-healthy agent that a bad 'fix' can break.

    Args:
        engine: The engine to register the tool on.

    Returns:
        The agent callable.
    """

    @engine.tool(name="clock", side_effecting=False)
    def clock() -> dict[str, Any]:
        return {"hour": 9}

    def agent(question: Any = None) -> str:
        data = clock()
        if BROKEN_SIBLING:
            raise KeyError("minute")
        return f"It is {data['hour']} o'clock."

    return agent


def reset() -> None:
    """Restore both switches, so one test cannot leak into the next."""
    global FIXED, BROKEN_SIBLING
    FIXED = False
    BROKEN_SIBLING = False


def build_appending(engine: Any) -> Any:
    """Build an agent that appends to a nested value in its initial state.

    A scratchpad an agent writes into is ordinary. It is also the shape that exposes
    a shallow copy of `initial_state`: the list is shared, so a second run sees the
    first one's entries.

    Args:
        engine: The engine, unused beyond the builder convention.

    Returns:
        The agent callable.
    """

    def agent(question: Any = None, claims: Any = None, **_state: Any) -> str:
        entries = claims if claims is not None else []
        entries.append("did some work")
        return f"claims={len(entries)}"

    return agent
