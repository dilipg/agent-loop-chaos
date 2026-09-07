"""A hand-rolled client and repository, reachable by neither built-in strategy.

No HTTP, no LangChain base class -- the shape `seams:` exists for. Plenty of real
services look exactly like this: an in-house client class and a module of functions
that read a database through a driver of their own.
"""

from __future__ import annotations

from typing import Any

CANNED = "Pack light layers."


class LLMClient:
    """An in-house model client with its own transport."""

    def chat(self, prompt: str) -> str:
        """Answer a prompt.

        Args:
            prompt: What to answer.

        Returns:
            A canned reply, so the fake needs no network.
        """
        return CANNED


def fetch_weather(city: str) -> dict[str, Any]:
    """Read the weather for a city.

    Args:
        city: Which city.

    Returns:
        A record with the shape a tool fault expects to find.
    """
    return {"temp_c": 21, "city": city}


def fetch_tickets(entity_id: str) -> list[dict[str, Any]]:
    """Read the open tickets for an entity.

    Args:
        entity_id: Whose tickets.

    Returns:
        One ticket, enough to prove a glob matched.
    """
    return [{"id": "T-1", "entity_id": entity_id}]
