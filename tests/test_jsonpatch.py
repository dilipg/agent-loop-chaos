"""The RFC 6902 subset.

`apply(diff(a, b), a) == b` is the property that makes a recorded payload diff
trustworthy, so it is tested both by example and by hypothesis.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_loop_chaos.jsonpatch import apply, diff, escape_token, is_jsonable

CASES: list[tuple[Any, Any]] = [
    ({"a": 1}, {"a": 1}),
    ({"a": 1}, {"a": 2}),
    ({"a": 1}, {}),
    ({}, {"a": 1}),
    ({"a": {"b": [1, 2, 3]}}, {"a": {"b": [1, 9]}}),
    ([1, 2, 3], [1, 2, 3, 4]),
    ([1, 2, 3], [1]),
    ([], [1]),
    ("x", "y"),
    (1, "one"),
    (None, {"a": 1}),
    ({"a/b": 1}, {"a/b": 2}),
    ({"t~x": 1}, {"t~x": 2}),
    ({"a": [{"b": 1}]}, {"a": [{"b": 2}]}),
    ({"a": 1, "b": 2}, {"b": 2, "c": 3}),
    ({"deep": {"deeper": {"deepest": [1, {"k": "v"}]}}}, {"deep": {"deeper": {"deepest": []}}}),
]


@pytest.mark.parametrize(("before", "after"), CASES)
def test_apply_diff_round_trips(before: Any, after: Any) -> None:
    """The core guarantee, by example."""
    assert apply(before, diff(before, after)) == after


def test_equal_values_produce_an_empty_patch() -> None:
    """A fault that changed nothing must not look like it changed something."""
    assert diff({"a": [1, 2]}, {"a": [1, 2]}) == []


def test_only_the_documented_ops_are_emitted() -> None:
    """No `move`, `copy` or `test`: predictability over compactness."""
    ops = {op["op"] for before, after in CASES for op in diff(before, after)}
    assert ops <= {"add", "remove", "replace"}


def test_pointer_tokens_are_escaped() -> None:
    """RFC 6901 §3, in the order the spec requires."""
    assert escape_token("a/b") == "a~1b"
    assert escape_token("t~x") == "t~0x"
    assert escape_token("~/") == "~0~1"


def test_list_removals_are_emitted_tail_first() -> None:
    """Otherwise applying them in order would shift the indices underneath."""
    ops = diff([1, 2, 3, 4], [1])
    removals = [op["path"] for op in ops if op["op"] == "remove"]
    assert removals == ["/3", "/2", "/1"]


def test_unrepresentable_input_yields_an_empty_patch() -> None:
    """The caller distinguishes this from "no change" via `is_jsonable`."""
    before, after = {"s": {1, 2}}, {"s": {1}}
    assert diff(before, after) == []
    assert not is_jsonable(before)


def test_is_jsonable_accepts_non_finite_floats() -> None:
    """`sanitize_floats` encodes them, so they are representable (D-09)."""
    assert is_jsonable({"n": float("nan")})


def test_apply_does_not_mutate_its_input() -> None:
    """A patch application must not disturb the recorded `payload_before`."""
    before = {"a": [1, 2]}
    apply(before, [{"op": "replace", "path": "/a/0", "value": 9}])
    assert before == {"a": [1, 2]}


def test_apply_rejects_an_unknown_op() -> None:
    """Failing loudly beats silently ignoring an operation."""
    with pytest.raises(ValueError, match="unsupported patch op"):
        apply({}, [{"op": "move", "path": "/a", "from": "/b"}])


def test_apply_rejects_removing_the_document_root() -> None:
    """There is nothing to remove it from."""
    with pytest.raises(ValueError, match="document root"):
        apply({"a": 1}, [{"op": "remove", "path": ""}])


_json = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=8),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(min_size=1, max_size=6), children, max_size=4)
    ),
    max_leaves=12,
)


@settings(max_examples=200, deadline=None)
@given(before=_json, after=_json)
def test_apply_diff_round_trips_for_arbitrary_json(before: Any, after: Any) -> None:
    """The property that makes every recorded `json_patch` trustworthy."""
    assert apply(before, diff(before, after)) == after
