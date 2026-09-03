"""The mutation library.

Every mutation is a pure function `(value, rng, **params) -> new_value`. Purity is
not a style note: `ArgumentTamperFault` and the phase-05 state faults share this
registry, and a mutation that touched its input would corrupt the agent's own data
and make `payload_before` a lie.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_loop_chaos.mutations import MUTATIONS

# Catalog A1's list, verbatim. `docs/03-FAULT-CATALOG.md` is normative.
CATALOG_NAMES = [
    "empty_json",
    "empty_list",
    "null_result",
    "drop_key",
    "drop_required_key",
    "rename_key",
    "type_flip",
    "stringify_numbers",
    "null_fields",
    "truncate_string",
    "truncate_list",
    "duplicate_items",
    "reorder_list",
    "nan_numbers",
    "negative_numbers",
    "unit_swap",
    "unicode_noise",
    "deep_nest",
    "wrong_schema",
    "json_as_string",
    "malformed_json_string",
    "whitespace_only",
]


def rng() -> random.Random:
    """A fixed generator, so every assertion below is reproducible.

    Returns:
        A seeded generator.
    """
    return random.Random(1337)


# NOTE: the registry-completeness assertion lives in
# `test_every_catalog_name_is_implemented_and_covered`, at the bottom of this file.
# It is added last on purpose: asserting all 22 names up front would have left the
# suite red for the whole phase, which defeats red-green-refactor.


def test_empty_json_replaces_a_mapping_with_an_empty_one() -> None:
    """Mimics a tool that returns `{}` on a cache miss."""
    assert MUTATIONS["empty_json"]({"a": 1, "b": 2}, rng()) == {}


def test_empty_list_replaces_a_sequence_with_an_empty_one() -> None:
    """Mimics a search that matched nothing but signalled success."""
    assert MUTATIONS["empty_list"]([1, 2, 3], rng()) == []


def test_null_result_replaces_anything_with_none() -> None:
    """Mimics a tool returning `null` where the schema promised an object."""
    assert MUTATIONS["null_result"]({"a": 1}, rng()) is None


# ------------------------------------------------------------------ key mutations


def test_drop_key_removes_a_named_key() -> None:
    """The canonical corruption: a field the agent indexes straight into."""
    out = MUTATIONS["drop_key"]({"temp_c": 21, "city": "Paris"}, rng(), keys=["temp_c"])
    assert out == {"city": "Paris"}


def test_drop_key_applies_to_every_record_in_a_list() -> None:
    """A tool returning three records must lose the key in all three.

    Corrupting only the first would let an agent that checks `data[0]` pass while a
    real outage would have broken it everywhere.
    """
    records = [{"temp_c": 1, "d": "a"}, {"temp_c": 2, "d": "b"}, {"temp_c": 3, "d": "c"}]
    out = MUTATIONS["drop_key"](records, rng(), keys=["temp_c"])
    assert out == [{"d": "a"}, {"d": "b"}, {"d": "c"}]


def test_drop_key_reaches_a_nested_path() -> None:
    """Dotted paths, because real payloads nest."""
    out = MUTATIONS["drop_key"](
        {"outer": {"inner": {"gone": 1, "kept": 2}}}, rng(), keys=["outer.inner.gone"]
    )
    assert out == {"outer": {"inner": {"kept": 2}}}


def test_drop_key_picks_deterministically_when_no_key_is_named() -> None:
    """`rng.sample(sorted(paths), count)` — sorted, so dict order cannot leak in."""
    payload = {"a": 1, "b": 2, "c": 3, "d": 4}
    first = MUTATIONS["drop_key"](payload, rng())
    second = MUTATIONS["drop_key"](dict(reversed(list(payload.items()))), rng())
    assert first == second
    assert len(first) == 3


def test_drop_key_honours_count() -> None:
    """`count` is how a scenario asks for more than one field to go."""
    out = MUTATIONS["drop_key"]({"a": 1, "b": 2, "c": 3}, rng(), count=2)
    assert len(out) == 1


def test_drop_key_leaves_a_missing_path_alone() -> None:
    """A scenario naming a field the tool stopped returning must not crash."""
    payload = {"a": 1}
    assert MUTATIONS["drop_key"](payload, rng(), keys=["nope.deeper"]) == {"a": 1}


def test_rename_key_moves_a_value_to_a_neighbouring_name() -> None:
    """Mimics an API version bump that renamed a field."""
    out = MUTATIONS["rename_key"]({"temp_c": 21}, rng(), keys=["temp_c"])
    assert "temp_c" not in out
    assert 21 in out.values()


def test_null_fields_keeps_the_key_and_nulls_the_value() -> None:
    """Nastier than dropping: the shape check passes and the value is useless."""
    out = MUTATIONS["null_fields"]({"temp_c": 21, "city": "Paris"}, rng(), keys=["temp_c"])
    assert out == {"temp_c": None, "city": "Paris"}


def test_null_fields_applies_across_records() -> None:
    """Same all-records rule as `drop_key`."""
    records = [{"t": 1}, {"t": 2}]
    assert MUTATIONS["null_fields"](records, rng(), keys=["t"]) == [{"t": None}, {"t": None}]


@pytest.mark.parametrize("name", ["drop_key", "rename_key", "null_fields"])
def test_key_mutations_do_not_touch_their_input(name: str) -> None:
    """The shared-registry rule: never mutate the caller's object."""
    payload: dict[str, Any] = {"temp_c": 21, "nested": {"x": 1}}
    MUTATIONS[name](payload, rng(), keys=["temp_c"])
    assert payload == {"temp_c": 21, "nested": {"x": 1}}


# --------------------------------------------------------- number and type mutations


def test_type_flip_turns_an_int_into_a_string() -> None:
    """Mimics a JSON encoder change that quoted every number."""
    out = MUTATIONS["type_flip"]({"n": 21}, rng(), keys=["n"])
    assert out["n"] == "21"


def test_type_flip_turns_a_string_digit_into_an_int() -> None:
    """The reverse direction, per the catalog's `int<->str`."""
    assert MUTATIONS["type_flip"]({"n": "21"}, rng(), keys=["n"])["n"] == 21


def test_type_flip_turns_a_bool_into_a_string() -> None:
    """Catalog A1 lists `bool<->str`."""
    assert MUTATIONS["type_flip"]({"ok": True}, rng(), keys=["ok"])["ok"] == "True"


def test_type_flip_can_render_a_number_as_not_available() -> None:
    """Catalog A1: `number -> "N/A"`, the shape a spreadsheet export produces."""
    out = MUTATIONS["type_flip"]({"n": 21}, rng(), keys=["n"], to="na")
    assert out["n"] == "N/A"


def test_stringify_numbers_quotes_every_number_it_reaches() -> None:
    """Mimics a gateway that serializes all scalars as strings."""
    out = MUTATIONS["stringify_numbers"]({"a": 1, "b": 2.5, "c": "x"}, rng(), keys=["a", "b"])
    assert out == {"a": "1", "b": "2.5", "c": "x"}


def test_negative_numbers_flips_the_sign() -> None:
    """A sign error upstream: the magnitude looks plausible, the meaning inverts."""
    assert MUTATIONS["negative_numbers"]({"amount": 250}, rng(), keys=["amount"])["amount"] == -250


def test_nan_numbers_uses_the_canonical_encoding() -> None:
    """D-09: never a bare `NaN`, which would make report.json invalid JSON."""
    out = MUTATIONS["nan_numbers"]({"n": 1.0}, rng(), keys=["n"])
    assert out["n"] == {"__float__": "NaN"}


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("nan", "NaN"), ("inf", "Infinity"), ("-inf", "-Infinity")],
)
def test_nan_numbers_covers_all_three_non_finite_values(kind: str, expected: str) -> None:
    """All three have a canonical encoding, and none may be emitted bare."""
    out = MUTATIONS["nan_numbers"]({"n": 1.0}, rng(), keys=["n"], kind=kind)
    assert out["n"] == {"__float__": expected}


def test_unit_swap_changes_the_value_and_keeps_the_label() -> None:
    """The nastiest mutation in the catalog.

    `temp_c: 21` becomes the Fahrenheit *number* while the key still says celsius,
    so nothing about the shape is wrong. Only an agent that sanity-checks magnitudes
    notices, which is exactly the weakness being probed.
    """
    out = MUTATIONS["unit_swap"]({"temp_c": 21}, rng(), keys=["temp_c"])
    assert "temp_c" in out, "the label must survive"
    assert out["temp_c"] == pytest.approx(69.8)


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("temp_c", 100, 212.0),
        ("temp_f", 212, 100.0),
        ("distance_km", 10, pytest.approx(6.21371, rel=1e-4)),
        ("distance_mi", 10, pytest.approx(16.0934, rel=1e-4)),
        ("weight_kg", 10, pytest.approx(22.0462, rel=1e-4)),
        ("weight_lb", 10, pytest.approx(4.53592, rel=1e-4)),
        ("price_usd", 100, pytest.approx(92.0)),
        ("price_eur", 92, pytest.approx(100.0)),
    ],
)
def test_the_unit_table_covers_the_documented_pairs(key: str, value: float, expected: Any) -> None:
    """Catalog A1 names temp_c/temp_f, km/mi, kg/lb and usd/eur at a fixed rate."""
    assert MUTATIONS["unit_swap"]({key: value}, rng(), keys=[key])[key] == expected


def test_unit_swap_leaves_a_key_it_has_no_unit_for_alone() -> None:
    """Silently inventing a conversion would be a fault the report could not explain."""
    payload = {"humidity": 60}
    assert MUTATIONS["unit_swap"](payload, rng(), keys=["humidity"])["humidity"] == 60


@pytest.mark.parametrize(
    "name",
    ["type_flip", "stringify_numbers", "negative_numbers", "nan_numbers", "unit_swap"],
)
def test_number_mutations_do_not_touch_their_input(name: str) -> None:
    """The shared-registry purity rule again."""
    payload: dict[str, Any] = {"temp_c": 21, "n": 1}
    MUTATIONS[name](payload, rng(), keys=["temp_c", "n"])
    assert payload == {"temp_c": 21, "n": 1}


# --------------------------------------------------------------- string mutations


def test_truncate_string_cuts_to_about_thirty_percent_mid_word() -> None:
    """Catalog A1: 30% of the length, deliberately mid-word.

    A clean cut at a word boundary reads as a short answer; a mid-word cut is
    recognizably broken, which is what makes `truncated_output_used` detectable.
    """
    text = "a" * 100
    out = MUTATIONS["truncate_string"]({"s": text}, rng(), keys=["s"])
    assert len(out["s"]) == 30
    assert text.startswith(out["s"])


def test_truncate_string_leaves_non_strings_alone() -> None:
    """A number is not a string, and pretending otherwise hides the real mutation."""
    assert MUTATIONS["truncate_string"]({"n": 12345}, rng(), keys=["n"])["n"] == 12345


def test_unicode_noise_inserts_invisible_marks() -> None:
    """Zero-width and RTL marks: text that looks identical and compares unequal."""
    out = MUTATIONS["unicode_noise"]({"s": "Paris"}, rng(), keys=["s"])
    assert out["s"] != "Paris"
    assert any(ch in out["s"] for ch in ("​", "‎", "‮"))
    assert "".join(ch for ch in out["s"] if ch.isprintable() and ch not in "​‎‮")


def test_json_as_string_returns_the_payload_dumped_into_a_string() -> None:
    """A tool that double-encodes its response, so the agent gets a string."""
    out = MUTATIONS["json_as_string"]({"a": 1}, rng())
    assert isinstance(out, str)
    import json

    assert json.loads(out) == {"a": 1}


def test_malformed_json_string_returns_a_string_not_a_dict() -> None:
    """Catalog A1 is explicit: a `str` containing broken JSON, never a dict."""
    out = MUTATIONS["malformed_json_string"]({"a": 1}, rng())
    assert isinstance(out, str)
    import json

    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_whitespace_only_returns_blank_text() -> None:
    """A tool that returns a body of spaces and a 200."""
    out = MUTATIONS["whitespace_only"]({"a": 1}, rng())
    assert isinstance(out, str)
    assert out.strip() == ""
    assert out != ""


# ----------------------------------------------------------------- list mutations


def test_truncate_list_keeps_only_the_first_element() -> None:
    """Catalog A1: keep the first element only.

    This is the fault that catches an agent summarizing three days of forecast from
    one record without noticing.
    """
    assert MUTATIONS["truncate_list"]([1, 2, 3], rng()) == [1]


def test_truncate_list_on_an_empty_list_is_still_empty() -> None:
    """No index errors on a degenerate payload."""
    assert MUTATIONS["truncate_list"]([], rng()) == []


def test_duplicate_items_repeats_the_elements() -> None:
    """A paging bug that returns the same page twice."""
    out = MUTATIONS["duplicate_items"]([1, 2], rng())
    assert len(out) == 4
    assert out.count(1) == 2


def test_reorder_list_changes_the_order_but_not_the_contents() -> None:
    """Proves comparison logic that assumes a stable order."""
    payload = [1, 2, 3, 4, 5]
    out = MUTATIONS["reorder_list"](payload, rng())
    assert sorted(out) == payload
    assert out != payload


def test_reorder_list_is_reproducible_under_a_fixed_seed() -> None:
    """Chaotic on request, deterministic by default."""
    payload = [1, 2, 3, 4, 5]
    assert MUTATIONS["reorder_list"](payload, rng()) == MUTATIONS["reorder_list"](payload, rng())


def test_reorder_list_differs_across_seeds() -> None:
    """Otherwise "seeded" would be indistinguishable from "fixed"."""
    payload = list(range(12))
    outs = {tuple(MUTATIONS["reorder_list"](payload, random.Random(seed))) for seed in range(8)}
    assert len(outs) > 1


def test_reorder_list_reorders_records_inside_a_mapping() -> None:
    """Real payloads wrap their list in an envelope."""
    payload = {"items": [1, 2, 3, 4, 5]}
    out = MUTATIONS["reorder_list"](payload, rng())
    assert sorted(out["items"]) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "name",
    ["truncate_string", "unicode_noise", "truncate_list", "duplicate_items", "reorder_list"],
)
def test_string_and_list_mutations_do_not_touch_their_input(name: str) -> None:
    """The shared-registry purity rule."""
    payload: dict[str, Any] = {"s": "hello world", "items": [1, 2, 3]}
    snapshot = {"s": "hello world", "items": [1, 2, 3]}
    MUTATIONS[name](payload, rng(), keys=["s"])
    assert payload == snapshot


# ------------------------------------------------------------ structural mutations


def test_deep_nest_wraps_the_payload_in_extra_levels() -> None:
    """Catalog A1: five extra levels, so a fixed-depth walker gives up."""
    out = MUTATIONS["deep_nest"]({"a": 1}, rng())
    depth = 0
    node: Any = out
    while isinstance(node, dict) and len(node) == 1 and "nested" in node:
        node = node["nested"]
        depth += 1
    assert depth == 5
    assert node == {"a": 1}


def test_wrong_schema_returns_a_plausible_but_different_object() -> None:
    """The realistic failure: a neighbouring endpoint's response shape.

    An obviously-empty payload trips a shape check. A plausible one does not, which
    is what makes `schema_violation_downstream` worth probing for.
    """
    out = MUTATIONS["wrong_schema"]({"temp_c": 21, "city": "Paris"}, rng())
    assert isinstance(out, dict)
    assert set(out) != {"temp_c", "city"}
    assert out, "an empty object would be `empty_json`, not a wrong schema"


# ---------------------------------------------------------- drop_required_key (D-17)


def test_drop_required_key_targets_a_value_the_baseline_sent_to_the_model() -> None:
    """D-17: "keys the agent used" is not observable, so this is the proxy.

    A key whose scalar value later appears in a baseline `llm_request` is one the
    agent demonstrably carried into its prompt.
    """
    baseline_prompts = ["The temperature in Paris is 21 degrees"]
    out = MUTATIONS["drop_required_key"](
        {"temp_c": 21, "station_id": 99812, "city": "Paris"},
        rng(),
        baseline_prompts=baseline_prompts,
    )
    assert "station_id" in out, "a value the baseline never used must survive"
    assert "temp_c" not in out or "city" not in out


def test_drop_required_key_prefers_the_more_distinctive_value() -> None:
    """D-17 says "ranked by distinctiveness".

    A long, unusual value is stronger evidence the agent used the field than a bare
    digit that could appear in a prompt by coincidence.
    """
    out = MUTATIONS["drop_required_key"](
        {"code": 1, "reference": "QX-88213-ALPHA"},
        rng(),
        baseline_prompts=["ref QX-88213-ALPHA and code 1"],
    )
    assert "reference" not in out
    assert "code" in out


def test_drop_required_key_falls_back_to_drop_key_without_a_baseline() -> None:
    """D-17: degrade rather than fire blind, and let the fault record why."""
    out = MUTATIONS["drop_required_key"]({"a": 1, "b": 2}, rng(), keys=["a"])
    assert out == {"b": 2}


def test_drop_required_key_falls_back_when_the_baseline_matches_nothing() -> None:
    """A baseline that proves nothing is the same as no baseline."""
    out = MUTATIONS["drop_required_key"](
        {"a": 1, "b": 2}, rng(), baseline_prompts=["entirely unrelated text"]
    )
    assert len(out) == 1, "it must still drop something, deterministically"


@pytest.mark.parametrize("name", ["deep_nest", "wrong_schema", "drop_required_key"])
def test_structural_mutations_do_not_touch_their_input(name: str) -> None:
    """The shared-registry purity rule."""
    payload: dict[str, Any] = {"temp_c": 21, "city": "Paris"}
    MUTATIONS[name](payload, rng())
    assert payload == {"temp_c": 21, "city": "Paris"}


# ------------------------------------------------------------------- phase gates


def test_every_catalog_name_is_implemented_and_covered() -> None:
    """The registry matches catalog A1 exactly, and every name is exercised here.

    Deferred to the end of the file on purpose: asserting all 22 names up front
    would have left the suite red for the whole phase, which defeats
    red-green-refactor. A name in the catalog with no implementation is a scenario
    file that fails at load; a name with no test is an untested injection path.
    """
    from pathlib import Path

    assert sorted(MUTATIONS) == sorted(CATALOG_NAMES), "registry vs docs/03 section A1"

    source = Path(__file__).read_text(encoding="utf-8")
    # Count references outside the CATALOG_NAMES literal itself.
    body = source.split("]", 1)[1] if "CATALOG_NAMES = [" in source else source
    uncovered = [name for name in MUTATIONS if f'"{name}"' not in body]
    assert uncovered == [], f"mutations with no test reference: {uncovered}"


@settings(max_examples=150, deadline=None)
@given(
    payload=st.recursive(
        st.none()
        | st.booleans()
        | st.integers()
        | st.floats(allow_nan=False)
        | st.text(max_size=8),
        lambda children: (
            st.lists(children, max_size=3)
            | st.dictionaries(st.text(min_size=1, max_size=6), children, max_size=3)
        ),
        max_leaves=8,
    ),
    name=st.sampled_from(sorted(set(CATALOG_NAMES))),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_no_mutation_ever_touches_its_input(payload: Any, name: str, seed: int) -> None:
    """The property the whole registry rests on.

    `ArgumentTamperFault` mutates arguments *the agent is about to send*, and the
    phase-05 state faults mutate the agent's live state. A mutation that worked in
    place would corrupt the agent's own data and make `payload_before` a lie about
    what happened.
    """
    before = copy.deepcopy(payload)
    MUTATIONS[name](payload, random.Random(seed))
    assert payload == before, f"{name} mutated its input"
