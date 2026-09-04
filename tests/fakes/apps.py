"""Engine-wired agents built from the fakes.

A scenario's `entrypoint` names a callable, but a vanilla agent's tools have to be
wrapped by *this run's* engine before any tool fault can fire. The convention is a
builder: a callable whose first parameter is named `engine` is handed the engine and
returns the real agent. `examples/*/app.py` uses the same shape.
"""

from __future__ import annotations

from typing import Any

from agent_loop_chaos import ChaosEngine

from . import good_agent, naive_tool_agent, strict_json_agent
from .fake_llm import FakeLLM


def build_naive(engine: ChaosEngine) -> Any:
    """Wire `naive_tool_agent` to an instrumented tool.

    Args:
        engine: The run's engine.

    Returns:
        The agent callable.
    """

    @engine.tool(side_effecting=False)
    def fetch_weather(payload: Any = None) -> Any:
        return {"data": [{"temp_c": 21, "city": "Paris", "condition": "sunny"}]}

    def agent(inputs: Any = None) -> str:
        return naive_tool_agent.run(fetch_weather())

    return agent


def build_good(engine: ChaosEngine) -> Any:
    """Wire `good_agent` to the same tool.

    Args:
        engine: The run's engine.

    Returns:
        The agent callable.
    """

    @engine.tool(side_effecting=False)
    def fetch_weather(payload: Any = None) -> Any:
        return {"data": [{"temp_c": 21, "city": "Paris", "condition": "sunny"}]}

    def agent(inputs: Any = None, state: dict[str, Any] | None = None) -> str:
        raw: Any = None
        for _ in range(3):
            try:
                raw = fetch_weather()
                break
            except Exception:
                continue
        if raw is None:
            return "The weather service was unavailable, so I could not complete this."
        return good_agent.summarize(raw)

    return agent


def build_strict_json(engine: ChaosEngine) -> Any:
    """Wire `strict_json_agent` to an instrumented model call.

    Args:
        engine: The run's engine.

    Returns:
        The agent callable.
    """
    llm = FakeLLM(['{"ok": true}'])

    @engine.llm
    def chat(prompt: Any) -> Any:
        return llm(prompt)

    def agent(inputs: Any = None) -> Any:
        return strict_json_agent.run(str(chat("produce json")))

    return agent
