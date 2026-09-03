"""Helpers over the normalized message form.

Section-B faults operate only on a list of
``{"role", "content", "tool_calls"?, "_extra"?}`` dicts, so the same fault behaves
identically under the vanilla and LangGraph adapters. Nothing in this module knows
about a provider's payload class, and nothing in `faults/` may import a framework.

`_extra` is what makes the round trip lossless: any key the normalized form does not
name is parked there and restored on the way out, so a fault never silently discards
a field the provider needed.
"""

from __future__ import annotations

import json
from typing import Any, Literal

__all__ = [
    "KNOWN_KEYS",
    "TokenEstimator",
    "denormalize",
    "estimate_tokens",
    "flatten",
    "insert_at",
    "normalize",
    "split_roles",
    "total_tokens",
]

TokenEstimator = Literal["chars4", "tiktoken"]

KNOWN_KEYS: frozenset[str] = frozenset({"role", "content", "tool_calls"})

_CHARS_PER_TOKEN = 4


def normalize(messages: Any) -> list[dict[str, Any]]:
    """Convert an outbound payload into the normalized message list.

    Args:
        messages: A string, or a sequence of message mappings.

    Returns:
        A new list of normalized messages. Unknown keys are parked in `_extra`; a
        missing role defaults to ``"user"`` so a malformed message cannot crash a
        fault.
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    out: list[dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            out.append({"role": "user", "content": str(message)})
            continue
        normalized: dict[str, Any] = {
            "role": str(message.get("role", "user")),
            "content": message.get("content", ""),
        }
        if message.get("tool_calls") is not None:
            normalized["tool_calls"] = message["tool_calls"]
        extra = {k: v for k, v in message.items() if k not in KNOWN_KEYS}
        if extra:
            normalized["_extra"] = extra
        out.append(normalized)
    return out


def denormalize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore normalized messages to the shape they arrived in.

    Args:
        messages: Normalized messages.

    Returns:
        A new list with `_extra` flattened back into each message.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        restored = {k: v for k, v in message.items() if k != "_extra"}
        restored.update(message.get("_extra") or {})
        out.append(restored)
    return out


def estimate_tokens(text: str, method: TokenEstimator = "chars4") -> int:
    """Estimate the token count of a string.

    Args:
        text: The text to measure.
        method: ``"chars4"`` is ``len(text) // 4``. ``"tiktoken"`` uses the optional
            package when it is installed and otherwise falls back silently — the
            fault echoes which estimator actually ran, so the report never implies a
            precision it did not have (D-43).

    Returns:
        An estimated token count.
    """
    if method == "tiktoken":
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    return len(text) // _CHARS_PER_TOKEN


def _message_text(message: dict[str, Any]) -> str:
    """Render one message as the text a model would see.

    Args:
        message: A normalized message.

    Returns:
        ``role: content``, with any tool calls appended.
    """
    parts = [f"{message.get('role', '?')}: {message.get('content', '')}"]
    for call in message.get("tool_calls") or []:
        name = call.get("name") if isinstance(call, dict) else str(call)
        arguments = call.get("arguments") if isinstance(call, dict) else None
        rendered = json.dumps(arguments, sort_keys=True) if arguments is not None else ""
        parts.append(f"  -> {name}({rendered})")
    return "\n".join(parts)


def flatten(messages: list[dict[str, Any]]) -> str:
    """Render a conversation as the `exact_prompt` a work order quotes.

    Tool calls are included: a rendering that omitted them would misrepresent what
    the model actually saw.

    Args:
        messages: Normalized messages.

    Returns:
        One `role: content` line per message.
    """
    return "\n".join(_message_text(message) for message in messages)


def total_tokens(messages: list[dict[str, Any]], method: TokenEstimator = "chars4") -> int:
    """Estimate the token count of a whole conversation.

    Args:
        messages: Normalized messages.
        method: Which estimator to use.

    Returns:
        The summed estimate. Budgets are measured over the conversation, not one
        message.
    """
    return sum(estimate_tokens(str(m.get("content", "")), method) for m in messages)


def split_roles(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate system messages from everything else.

    `ContextShrinkFault` has to keep the system message whatever else it drops, so
    the split is a primitive rather than an inline filter.

    Args:
        messages: Normalized messages.

    Returns:
        ``(system_messages, other_messages)``, order preserved within each.
    """
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    return system, rest


def insert_at(
    messages: list[dict[str, Any]],
    text: str,
    position: str,
    role: str = "user",
) -> list[dict[str, Any]]:
    """Insert a message at a named position.

    Args:
        messages: Normalized messages. Not mutated.
        text: The content to insert.
        position: ``"start"``, ``"middle"``, ``"end"`` or ``"before_last_user"``.
        role: The role the inserted message carries.

    Returns:
        A new list with the message inserted. `before_last_user` is the position
        that most reliably dilutes attention on the live question; with no user turn
        it degrades to `end`.
    """
    out = [dict(m) for m in messages]
    inserted = {"role": role, "content": text}

    if position == "start":
        index = 0
    elif position == "middle":
        index = len(out) // 2 if out else 0
    elif position == "before_last_user":
        index = len(out)
        for offset in range(len(out) - 1, -1, -1):
            if out[offset].get("role") == "user":
                index = offset
                break
    else:
        index = len(out)

    out.insert(index, inserted)
    return out
