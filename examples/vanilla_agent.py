"""The blueprint's plain-Python agent: no framework, no graph, five lines to wire.

This is `docs/02-API.md` §11's integration promise, runnable. It shares
`naive_tool_agent`'s weakness -- it indexes a tool result without checking it -- so
`alc run` finds something real rather than proving the wiring works on a toy.

    python -m examples.vanilla_agent
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
for entry in (str(REPO / "src"), str(REPO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

__all__ = ["build", "main"]


def build(engine: Any = None) -> Any:
    """Build the agent, wiring its tools to the engine when there is one.

    The D-61 builder convention: a callable whose first parameter is named `engine`
    is handed the engine and returns the real agent.

    Args:
        engine: A `ChaosEngine`, or `None` to run uninstrumented.

    Returns:
        The agent callable.
    """
    from examples.trip_planner import tools as tools_module

    weather = tools_module.get_weather_data
    if engine is not None:
        weather = engine.tool(weather, name="get_weather_data", side_effecting=False)

    def agent(question: Any = None) -> str:
        """Answer a weather question."""
        rows = weather("Paris")
        first = rows[0]
        return f"It is {first['temp_c']}C and {first['condition']} in Paris."

    return agent


def main() -> None:
    """Run the agent once, unfaulted, and print the answer."""
    print(build()("weather in Paris?"))


if __name__ == "__main__":
    main()
