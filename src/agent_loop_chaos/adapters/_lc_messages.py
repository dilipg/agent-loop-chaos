"""Conversion between LangChain messages and the normalized form.

LLM faults operate only on the normalized form, which is what makes them
framework-agnostic and reusable by the vanilla adapter. The round trip is lossless:
any field this module does not model is carried in `_extra` and restored, so a fault
never silently discards something the provider needed.

`langchain_core` is imported locally, so the core never pulls it in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..errors import MissingExtraError

__all__ = ["from_langchain", "to_langchain"]

# `docs/06` §1.5's table.
_ROLE_BY_TYPE: dict[str, str] = {
    "SystemMessage": "system",
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "AIMessageChunk": "assistant",
    "ToolMessage": "tool",
    "FunctionMessage": "tool",
    "ChatMessage": "user",
}

# Modelled explicitly; everything else on the message goes to `_extra`.
_MODELLED = frozenset({"content", "type", "tool_calls", "tool_call_id", "name"})


def _messages_module() -> Any:
    """Import `langchain_core.messages`, or explain the extra.

    Returns:
        The module.

    Raises:
        MissingExtraError: When the extra is not installed.
    """
    try:
        from langchain_core import messages
    except ModuleNotFoundError as exc:
        raise MissingExtraError(
            "LangChain message conversion needs langchain-core: install agent-loop-chaos[langgraph]"
        ) from exc
    return messages


def from_langchain(messages: Any) -> list[dict[str, Any]]:
    """Convert LangChain messages into the normalized form.

    Args:
        messages: A string, a sequence of LangChain messages, or an
            already-normalized list of dicts.

    Returns:
        Normalized messages. A string becomes one user message, matching the vanilla
        adapter; a list of dicts passes through unchanged.
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    out: list[dict[str, Any]] = []
    for message in messages or []:
        if isinstance(message, dict):
            out.append(dict(message))
            continue

        kind = type(message).__name__
        normalized: dict[str, Any] = {
            "role": _ROLE_BY_TYPE.get(kind, "user"),
            "content": getattr(message, "content", ""),
        }
        calls = getattr(message, "tool_calls", None)
        if calls:
            normalized["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "arguments": call.get("args", call.get("arguments")),
                }
                for call in calls
            ]

        extra: dict[str, Any] = {"__lc_type__": kind}
        for field in ("tool_call_id", "name", "id"):
            value = getattr(message, field, None)
            if value is not None:
                extra[field] = value
        kwargs = getattr(message, "additional_kwargs", None)
        if kwargs:
            extra["additional_kwargs"] = dict(kwargs)
        metadata = getattr(message, "response_metadata", None)
        if metadata:
            extra["response_metadata"] = dict(metadata)
        normalized["_extra"] = extra
        out.append(normalized)
    return out


def to_langchain(messages: Sequence[dict[str, Any]]) -> list[Any]:
    """Convert normalized messages back into LangChain messages.

    Args:
        messages: Normalized messages, possibly rewritten by a fault.

    Returns:
        LangChain message objects. A message with no recorded type -- one a fault
        synthesized -- becomes a `HumanMessage`, which is the safe default: it is
        content, not instruction, and no provider rejects it.

    Raises:
        MissingExtraError: When `langchain-core` is not installed.
    """
    module = _messages_module()
    by_name = {
        "SystemMessage": module.SystemMessage,
        "HumanMessage": module.HumanMessage,
        "AIMessage": module.AIMessage,
        "ToolMessage": module.ToolMessage,
    }
    by_role = {
        "system": module.SystemMessage,
        "user": module.HumanMessage,
        "assistant": module.AIMessage,
        "tool": module.ToolMessage,
    }

    out: list[Any] = []
    for message in messages:
        extra = dict(message.get("_extra") or {})
        kind = str(extra.pop("__lc_type__", ""))
        cls = by_name.get(kind) or by_role.get(str(message.get("role")), module.HumanMessage)

        kwargs: dict[str, Any] = {"content": message.get("content", "")}
        for field in ("id", "name", "tool_call_id", "additional_kwargs", "response_metadata"):
            if field in extra:
                kwargs[field] = extra[field]
        if cls is module.ToolMessage:
            kwargs.setdefault("tool_call_id", "unknown")
        if cls is module.AIMessage and message.get("tool_calls"):
            kwargs["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "args": call.get("arguments", call.get("args", {})),
                }
                for call in message["tool_calls"]
            ]
        out.append(cls(**kwargs))
    return out
