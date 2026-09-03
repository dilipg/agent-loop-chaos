"""Normalized message helpers.

Section-B faults operate only on the normalized form — a list of
`{"role", "content", "tool_calls"?, "_extra"?}` dicts — so the same fault behaves
identically under the vanilla and LangGraph adapters. Nothing here may know about a
provider's payload class.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_loop_chaos.faults._messages import (
    denormalize,
    estimate_tokens,
    flatten,
    insert_at,
    normalize,
    split_roles,
    total_tokens,
)

ROUND_TRIP_CASES: list[list[dict[str, Any]]] = [
    [{"role": "user", "content": "hello"}],
    [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}],
    [{"role": "assistant", "content": "sure"}],
    [{"role": "tool", "content": '{"temp_c": 21}'}],
    [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "get_weather", "arguments": {"city": "Paris"}}],
        }
    ],
    [{"role": "user", "content": "hi", "name": "dilip", "metadata": {"trace": "x"}}],
    [],
]


@pytest.mark.parametrize("messages", ROUND_TRIP_CASES)
def test_denormalize_round_trips_normalize(messages: list[dict[str, Any]]) -> None:
    """Every role, tool calls, and unknown fields via `_extra`.

    A lossy round trip would mean a fault silently discards a field the provider
    needed, and the agent would fail for a reason the report could not explain.
    """
    assert denormalize(normalize(messages)) == messages


def test_normalize_parks_unknown_fields_in_extra() -> None:
    """Unknown keys survive in `_extra` rather than being dropped."""
    normalized = normalize([{"role": "user", "content": "hi", "name": "dilip"}])
    assert normalized[0]["_extra"] == {"name": "dilip"}
    assert set(normalized[0]) <= {"role", "content", "tool_calls", "_extra"}


def test_normalize_accepts_a_bare_string() -> None:
    """A string prompt becomes one user message (`docs/06` §2.1)."""
    assert normalize("hello") == [{"role": "user", "content": "hello"}]


def test_normalize_defaults_a_missing_role_to_user() -> None:
    """A malformed message must not crash a fault; the report says what it saw."""
    assert normalize([{"content": "hi"}])[0]["role"] == "user"


# ------------------------------------------------------------------ token counting


def test_chars4_is_length_over_four() -> None:
    """The documented estimator: `len(text) // 4`."""
    assert estimate_tokens("a" * 400, method="chars4") == 100


def test_chars4_handles_the_empty_string() -> None:
    """No division surprises on a degenerate input."""
    assert estimate_tokens("", method="chars4") == 0


def test_tiktoken_falls_back_silently_when_the_package_is_absent() -> None:
    """`tiktoken` is optional, so its absence must degrade rather than raise.

    Which estimator actually ran is echoed in the fault's params, so the report
    never implies a precision it did not have.
    """
    assert estimate_tokens("a" * 400, method="tiktoken") > 0


def test_total_tokens_sums_the_whole_conversation() -> None:
    """Budgets are measured over the conversation, not one message."""
    messages = [
        {"role": "system", "content": "a" * 40},
        {"role": "user", "content": "b" * 40},
    ]
    assert total_tokens(messages) == 20


# ---------------------------------------------------------------------- flatten


def test_flatten_renders_role_prefixed_lines() -> None:
    """This is the `exact_prompt` a work order quotes, so it must be faithful."""
    rendered = flatten(
        [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]
    )
    assert rendered == "system: be terse\nuser: hi"


def test_flatten_includes_tool_calls() -> None:
    """A prompt that omitted the tool calls would misrepresent what the model saw."""
    rendered = flatten(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "get_weather", "arguments": {"city": "Paris"}}],
            }
        ]
    )
    assert "get_weather" in rendered


# ------------------------------------------------------------------- split_roles


def test_split_roles_separates_system_from_the_rest() -> None:
    """`ContextShrinkFault` has to keep the system message whatever else it drops."""
    system, rest = split_roles(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert [m["role"] for m in system] == ["system"]
    assert [m["role"] for m in rest] == ["user", "assistant"]


def test_split_roles_handles_a_conversation_with_no_system_message() -> None:
    """Common in hand-rolled loops."""
    system, rest = split_roles([{"role": "user", "content": "u"}])
    assert system == []
    assert len(rest) == 1


# --------------------------------------------------------------------- insert_at


@pytest.mark.parametrize(
    ("position", "expected_index"),
    [("start", 0), ("end", 3), ("middle", 1)],
)
def test_insert_at_places_the_text_where_asked(position: str, expected_index: int) -> None:
    """`ContextNoiseFault` chooses where the distraction lands."""
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    out = insert_at(messages, "NOISE", position)
    assert out[expected_index]["content"] == "NOISE"
    assert len(out) == 4


def test_insert_before_last_user_lands_before_the_final_user_turn() -> None:
    """The position that most reliably dilutes attention on the live question."""
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "last"},
    ]
    out = insert_at(messages, "NOISE", "before_last_user")
    assert out[-2]["content"] == "NOISE"
    assert out[-1]["content"] == "last"


def test_insert_at_does_not_touch_the_input() -> None:
    """The shared purity rule: never mutate the caller's messages."""
    messages = [{"role": "user", "content": "u"}]
    insert_at(messages, "NOISE", "end")
    assert messages == [{"role": "user", "content": "u"}]


def test_insert_at_into_an_empty_conversation() -> None:
    """A degenerate conversation must not raise."""
    assert insert_at([], "NOISE", "before_last_user")[0]["content"] == "NOISE"
