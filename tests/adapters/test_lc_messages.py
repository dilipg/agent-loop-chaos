"""LangChain message normalization, both directions.

LLM faults operate only on the normalized form, which is what makes them
framework-agnostic and reusable by the vanilla adapter. The round trip has to be
lossless, or a fault silently discards a field the provider needed and the agent
fails for a reason the report cannot explain.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain_core", reason="the langgraph extra is not installed")

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent_loop_chaos.adapters._lc_messages import (
    from_langchain,
    to_langchain,
)

CASES: list[Any] = [
    SystemMessage(content="be terse"),
    HumanMessage(content="what is the weather"),
    AIMessage(content="it is sunny"),
    AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "get_weather", "args": {"city": "Paris"}}],
    ),
    ToolMessage(content='{"temp_c": 21}', tool_call_id="c1", name="get_weather"),
]


@pytest.mark.parametrize("message", CASES, ids=lambda m: type(m).__name__)
def test_the_round_trip_preserves_the_message(message: Any) -> None:
    """`to_langchain(from_langchain(x)) == x` for every documented type."""
    restored = to_langchain(from_langchain([message]))[0]
    assert type(restored) is type(message)
    assert restored.content == message.content


def test_roles_map_as_documented() -> None:
    """`docs/06` §1.5's table, exactly."""
    roles = [m["role"] for m in from_langchain(CASES)]
    assert roles == ["system", "user", "assistant", "assistant", "tool"]


def test_tool_calls_survive_the_round_trip() -> None:
    """A prompt that lost its tool calls would misrepresent what the model saw."""
    normalized = from_langchain([CASES[3]])
    assert normalized[0]["tool_calls"][0]["name"] == "get_weather"
    restored = to_langchain(normalized)[0]
    assert restored.tool_calls[0]["name"] == "get_weather"


def test_a_tool_message_keeps_its_call_id_and_name() -> None:
    """Both are required to match a result back to the call that asked for it."""
    normalized = from_langchain([CASES[4]])
    restored = to_langchain(normalized)[0]
    assert restored.tool_call_id == "c1"
    assert restored.name == "get_weather"


def test_a_bare_string_becomes_a_user_message() -> None:
    """The documented convenience, matching the vanilla adapter."""
    assert from_langchain("hello") == [{"role": "user", "content": "hello"}]


def test_unknown_fields_are_carried_in_extra_and_restored() -> None:
    """Lossless means lossless: a provider field we do not model still survives."""
    message = HumanMessage(content="hi", id="msg-42", additional_kwargs={"trace": "x"})
    normalized = from_langchain([message])
    assert normalized[0]["_extra"]
    restored = to_langchain(normalized)[0]
    assert restored.id == "msg-42"
    assert restored.additional_kwargs["trace"] == "x"


def test_a_normalized_message_with_no_type_hint_becomes_a_human_message() -> None:
    """A fault may synthesize a message; it must still convert back."""
    restored = to_langchain([{"role": "user", "content": "injected"}])[0]
    assert isinstance(restored, HumanMessage)
    assert restored.content == "injected"


def test_an_already_normalized_list_passes_through() -> None:
    """The vanilla adapter hands us dicts; both shapes must work."""
    plain = [{"role": "user", "content": "hi"}]
    assert from_langchain(plain) == plain
