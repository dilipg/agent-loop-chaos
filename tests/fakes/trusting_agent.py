"""Echoes tool text straight into the prompt.

**Planted weakness:** untrusted content is concatenated rather than delimited, so an
instruction hidden in a tool result reads to the model as an instruction.
"""

from __future__ import annotations

from typing import Any


def run(tool_text: str, llm: Any) -> Any:
    """Summarise a tool result by pasting it into the prompt.

    Args:
        tool_text: The tool's text, which may carry an injected instruction.
        llm: The model callable.

    Returns:
        The model's response.
    """
    return llm([{"role": "user", "content": f"Summarise this ticket: {tool_text}"}])
