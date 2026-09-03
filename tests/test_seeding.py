"""Determinism primitives.

Determinism is a feature, so these are contract tests rather than sanity checks.
"""

from __future__ import annotations

import json
import math

import pytest

from agent_loop_chaos.seeding import (
    canonical_json,
    derive_id,
    rng,
    sanitize_floats,
    sha256_of,
)


def test_same_seed_and_purpose_give_the_same_stream() -> None:
    """The factory is deterministic."""
    assert [rng(7, "a").random() for _ in range(5)] == [rng(7, "a").random() for _ in range(5)]


def test_rng_is_a_pure_factory_with_no_module_level_memo() -> None:
    """D-02: two calls must not share state, or two runs in one process would.

    A module-level cache is exactly what would break the same-seed determinism test,
    baseline-plus-chaos in one process, and `jobs > 1`.
    """
    first, second = rng(7, "a"), rng(7, "a")
    assert first is not second
    first.random()
    # If a memo existed, `second` would have been advanced by the draw above.
    assert second.random() == rng(7, "a").random()


def test_purposes_are_independent() -> None:
    """Drawing from one stream never advances another."""
    a, b = rng(7, "x:trigger"), rng(7, "x:key_choice")
    [a.random() for _ in range(10)]
    assert b.random() == rng(7, "x:key_choice").random()


def test_different_seeds_diverge() -> None:
    """A different seed tells a different story."""
    assert rng(1, "p").random() != rng(2, "p").random()


def test_derive_id_is_deterministic_and_short() -> None:
    """Ids are derived, never `uuid4`, so a report is byte-reproducible."""
    assert derive_id(1337, "span", 3) == derive_id(1337, "span", 3)
    assert derive_id(1337, "span", 3) != derive_id(1337, "span", 4)
    assert len(derive_id(1337, "span", 3)) == 8


def test_canonical_json_is_key_order_independent() -> None:
    """No whitespace or ordering drift can change a hash."""
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_sha256_of_is_stable_and_prefixed() -> None:
    """`plan_hash` and truncation stubs both rely on this shape."""
    digest = sha256_of({"a": 1})
    assert digest.startswith("sha256:")
    assert digest == sha256_of({"a": 1})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), {"__float__": "NaN"}),
        (float("inf"), {"__float__": "Infinity"}),
        (float("-inf"), {"__float__": "-Infinity"}),
    ],
)
def test_non_finite_floats_get_a_canonical_encoding(value: float, expected: dict) -> None:
    """D-09: a report must be valid JSON for a `jq` or Go consumer."""
    assert sanitize_floats({"n": value}) == {"n": expected}


def test_sanitized_output_survives_allow_nan_false() -> None:
    """The whole point of D-09: every dump can forbid non-finite floats."""
    payload = {"a": float("nan"), "b": [float("inf"), 1.5]}
    assert json.dumps(sanitize_floats(payload), allow_nan=False)


def test_finite_floats_are_rounded_for_stable_goldens() -> None:
    """Six decimals, so a golden comparison is not defeated by float noise."""
    assert sanitize_floats(1 / 3) == pytest.approx(0.333333)


def test_sanitize_floats_does_not_mutate_its_input() -> None:
    """Nothing in this library mutates the caller's objects."""
    original = {"n": float("nan")}
    sanitize_floats(original)
    assert math.isnan(original["n"])


def test_canonical_json_refuses_unserializable_values() -> None:
    """Failing loudly beats writing a report nothing can read."""
    with pytest.raises(TypeError):
        canonical_json({"s": {1, 2}})
