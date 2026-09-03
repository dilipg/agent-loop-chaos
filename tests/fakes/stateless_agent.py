"""Keeps the objective only in the message list.

**Planted weakness:** nothing durable holds the goal, so a context shrink loses it
and there is no location for `goal_token_loss` to compare against.
"""

from __future__ import annotations

from typing import Any


def run(objective: str, state: dict[str, Any]) -> list[dict[str, str]]:
    """Start a conversation from the objective, storing nothing.

    Args:
        objective: The task.
        state: The working state, deliberately left untouched.

    Returns:
        The message list, which is the only place the objective lives.
    """
    return [{"role": "user", "content": objective}]
