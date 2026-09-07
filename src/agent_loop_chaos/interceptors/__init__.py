"""Attach to an agent without asking it to change.

Faults need a call to pass through the engine. `engine.tool` and `engine.llm` do that
by wrapping, which requires editing the agent -- and the agent's author is often not
the person running the chaos suite. These strategies patch the places a call already
goes through instead: the HTTP client every model SDK rides on, and the framework base
classes an in-process model inherits.

See `base.Registry`.
"""

from __future__ import annotations

from .base import Attachment, AttachReport, Registry, Strategy, default_strategies

__all__ = ["AttachReport", "Attachment", "Registry", "Strategy", "default_strategies"]
