"""Secret redaction.

The property that matters: whatever goes in, no known secret shape comes out — and
the caller's object is never touched (`docs/04-SCHEMAS.md` §4, `docs/07` §7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_loop_chaos.redact import DEFAULT_VALUE_PATTERNS, redact

SECRET_VALUES = [
    "sk-" + "A" * 24,
    "ghp_" + "b" * 24,
    "AKIA" + "C" * 16,
    "xoxb-123-abc-XYZ",
    "-----BEGIN RSA PRIVATE KEY-----",
    "eyJhbGciOiJIUzI1NiJ9.payload",
]


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "API-KEY",
        "access_key",
        "secret",
        "client_secret",
        "token",
        "Bearer",
        "password",
        "passwd",
        "Authorization",
        "auth",
        "Cookie",
        "session",
        "credential",
        "private_key",
    ],
)
def test_deny_listed_keys_are_redacted(key: str) -> None:
    """The key deny-list is case-insensitive."""
    out = redact({key: "whatever-this-is"})
    assert out[key].startswith("<redacted:")


@pytest.mark.parametrize("value", SECRET_VALUES)
def test_known_value_shapes_are_redacted_under_any_key(value: str) -> None:
    """A secret is a secret wherever it appears, however the field is named."""
    assert redact({"harmless_name": value})["harmless_name"].startswith("<redacted:")


def test_the_replacement_names_the_reason_not_the_value() -> None:
    """`<redacted:api_key>` keeps a report legible without leaking anything."""
    assert redact({"api_key": "hunter2"})["api_key"] == "<redacted:api_key>"


def test_redaction_recurses_through_containers() -> None:
    """Nothing reaches a trace event unfiltered because it was nested."""
    out = redact({"a": [{"token": "t"}, ("x", {"secret": "s"})], "b": {"ok": 1}})
    assert out["a"][0]["token"].startswith("<redacted:")
    assert out["a"][1][1]["secret"].startswith("<redacted:")
    assert out["b"] == {"ok": 1}


def test_redaction_handles_dataclasses() -> None:
    """Faults pass dataclasses around, so they must be covered too."""

    @dataclass
    class Creds:
        api_key: str
        region: str

    out = redact(Creds(api_key="sk-" + "A" * 20, region="eu"))
    assert out == {"api_key": "<redacted:api_key>", "region": "eu"}


def test_sets_are_rendered_order_stably() -> None:
    """A set's iteration order is not stable; a report must be."""
    assert redact({"s": {"b", "a", "c"}})["s"] == ["a", "b", "c"]


def test_the_input_is_never_mutated() -> None:
    """Faults and the recorder both depend on this."""
    original = {"api_key": "hunter2", "nested": {"token": "t"}}
    redact(original)
    assert original == {"api_key": "hunter2", "nested": {"token": "t"}}


def test_user_supplied_keys_are_honoured() -> None:
    """`ChaosEngine(redact_keys=…)` extends the deny-list."""
    assert redact({"employee_id": "E1"}, ["employee_id"])["employee_id"] == "<redacted:user_key>"


def test_only_the_canary_is_exempt() -> None:
    """The `secret_in_output` probe must be able to see a canary reach the output.

    Real credentials are never exempted, even alongside a canary (`docs/04` §4).
    """
    canary = "ALC-CANARY-run-3f9a12c4"
    out = redact({"answer": canary, "api_key": "sk-" + "A" * 20}, allow=(canary,))
    assert out["answer"] == canary
    assert out["api_key"].startswith("<redacted:")


def test_a_canary_survives_even_under_a_deny_listed_key() -> None:
    """An exemption that a key name could defeat would be no exemption at all."""
    canary = "ALC-CANARY-run-deadbeef"
    assert redact({"secret": canary}, allow=(canary,))["secret"] == canary


def test_harmless_values_pass_through_unchanged() -> None:
    """Over-redaction would make a report useless."""
    payload = {"city": "Paris", "temp_c": 21, "ok": True, "none": None, "f": 1.5}
    assert redact(payload) == payload


@settings(max_examples=200, deadline=None)
@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(max_size=12),
        lambda children: (
            st.lists(children, max_size=3)
            | st.dictionaries(st.text(min_size=1, max_size=8), children, max_size=3)
        ),
        max_leaves=10,
    ),
    st.sampled_from(SECRET_VALUES),
)
def test_no_known_secret_shape_survives_redaction(payload: Any, secret: str) -> None:
    """Property: planting a secret anywhere never leaves it in the output."""
    wrapped = {"outer": payload, "planted": secret}
    rendered = repr(redact(wrapped))
    for _reason, pattern in DEFAULT_VALUE_PATTERNS:
        assert not re.search(pattern, rendered), f"{pattern} survived redaction"
