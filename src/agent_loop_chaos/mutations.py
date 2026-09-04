"""Pure payload mutators.

Every mutation is `(value, rng, **params) -> new_value`. Three rules hold for all of
them:

- **Deep-copy first; never touch the input.** `ArgumentTamperFault` and the phase-05
  state faults share this registry, so a mutation that mutated in place would corrupt
  the agent's own data and make `payload_before` a lie. A hypothesis test asserts it.
- **Deterministic selection.** When no keys are named, candidates are chosen with
  `rng.sample(sorted(paths), count)` — sorted, so dict ordering cannot leak into a
  seeded run.
- **No tool-specific assumptions.** This registry is shared.

Each docstring names the real-world case the mutation mimics.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

__all__ = ["MUTATIONS", "Mutation"]

Mutation = Callable[..., Any]

MUTATIONS: dict[str, Mutation] = {}


def _register(name: str) -> Callable[[Mutation], Mutation]:
    """Register a mutation under its catalog name.

    Args:
        name: The name used in `mutation_type` and in scenario files.

    Returns:
        A decorator that registers and returns the function.
    """

    def decorate(fn: Mutation) -> Mutation:
        MUTATIONS[name] = fn
        return fn

    return decorate


@_register("empty_json")
def empty_json(value: Any, rng: Any, **params: Any) -> Any:
    """A tool that returns `{}` on a cache miss but still reports success.

    Args:
        value: The payload, unused beyond shape.
        rng: The seeded generator, unused.
        **params: Unused.

    Returns:
        An empty mapping.
    """
    return {}


@_register("empty_list")
def empty_list(value: Any, rng: Any, **params: Any) -> Any:
    """A search that matched nothing yet signalled success.

    Args:
        value: The payload, unused beyond shape.
        rng: The seeded generator, unused.
        **params: Unused.

    Returns:
        An empty list.
    """
    return []


@_register("null_result")
def null_result(value: Any, rng: Any, **params: Any) -> Any:
    """A tool returning `null` where its schema promised an object.

    Args:
        value: The payload, unused.
        rng: The seeded generator, unused.
        **params: Unused.

    Returns:
        `None`.
    """
    return None


def _copy(value: Any) -> tuple[Any, bool]:
    """Deep-copy a value, reporting failure rather than raising.

    Args:
        value: The value to copy.

    Returns:
        ``(copy, copied)``. On failure the original is returned with `copied` False,
        so the caller can record `mutation_skipped_uncopyable` and pass it through
        untouched.
    """
    try:
        return copy.deepcopy(value), True
    # An uncopyable payload is data, not a bug: the caller records
    # `mutation_skipped_uncopyable` and passes the original through.
    except Exception:
        return value, False


# ------------------------------------------------------------------ path helpers


def _records_of(value: Any) -> list[Any]:
    """Return the mappings a keyed mutation should apply to.

    A tool returning a list of records must lose a key in *every* record. Corrupting
    only the first would let an agent that checks `data[0]` pass a fault a real
    outage would have broken it everywhere.

    Args:
        value: The payload.

    Returns:
        The list of record containers to mutate, or `[value]` for a bare mapping.
    """
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    # Real payloads usually wrap their records in an envelope. Unwrap exactly one
    # list-valued key, and only when there is exactly one: guessing which of several
    # lists holds the records would be worse than doing nothing.
    lists = [v for v in value.values() if isinstance(v, list)]
    if len(lists) == 1 and any(isinstance(item, dict) for item in lists[0]):
        return [item for item in lists[0] if isinstance(item, dict)]
    return [value]


def _candidate_paths(record: Any, depth: int, prefix: str = "") -> list[str]:
    """Collect dotted paths to every key within `depth` levels.

    Args:
        record: A mapping to walk.
        depth: Remaining recursion budget.
        prefix: The path accumulated so far.

    Returns:
        Dotted paths, in insertion order; the caller sorts before sampling.
    """
    if depth <= 0 or not isinstance(record, dict):
        return []
    paths: list[str] = []
    for key, child in record.items():
        path = f"{prefix}{key}"
        paths.append(path)
        paths.extend(_candidate_paths(child, depth - 1, f"{path}."))
    return paths


def _choose_paths(
    records: list[Any], keys: list[str] | None, count: int, depth: int, rng: Any
) -> list[str]:
    """Decide which dotted paths to act on.

    Args:
        records: The record containers.
        keys: Explicit paths, or `None` to choose.
        count: How many to choose when `keys` is `None`.
        depth: Recursion limit for candidate discovery.
        rng: The seeded generator.

    Returns:
        The chosen paths. Selection is `rng.sample(sorted(candidates), count)`, so
        the choice is reproducible and independent of dict ordering.
    """
    if keys is not None:
        return list(keys)
    candidates: set[str] = set()
    for record in records:
        candidates.update(_candidate_paths(record, depth))
    ordered = sorted(candidates)
    if not ordered:
        return []
    chosen: list[str] = rng.sample(ordered, min(count, len(ordered)))
    return chosen


def _parent_and_leaf(record: Any, path: str) -> tuple[Any, str] | None:
    """Walk to the mapping holding the final segment of `path`.

    Args:
        record: The record to walk.
        path: A dotted path.

    Returns:
        ``(parent, leaf)``, or `None` when the path is unreachable.
    """
    parts = path.split(".")
    node = record
    for part in parts[:-1]:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    if isinstance(node, dict) and parts[-1] in node:
        return node, parts[-1]
    return None


def _apply_to_paths(
    value: Any,
    rng: Any,
    params: dict[str, Any],
    action: Callable[[Any, str], None],
) -> Any:
    """Deep-copy `value` and run `action` at every selected path.

    Args:
        value: The payload.
        rng: The seeded generator.
        params: Mutation params; `keys`, `count` and `depth` are consumed here.
        action: Called as ``action(parent_mapping, leaf_key)`` for each hit.

    Returns:
        The mutated copy, or the original when it could not be copied.
    """
    working, copied = _copy(value)
    if not copied:
        return working
    records = _records_of(working)
    paths = _choose_paths(
        records, params.get("keys"), int(params.get("count", 1)), int(params.get("depth", 3)), rng
    )
    for record in records:
        for path in paths:
            found = _parent_and_leaf(record, path)
            if found is not None:
                action(found[0], found[1])
    return working


# ------------------------------------------------------------------ key mutations


@_register("drop_key")
def drop_key(value: Any, rng: Any, **params: Any) -> Any:
    """A field the agent indexes straight into simply stops being returned.

    Args:
        value: The payload.
        rng: The seeded generator, used only when `keys` is not given.
        **params: `keys`, `count`, `depth`.

    Returns:
        A copy with the selected keys removed from every record.
    """

    def remove(parent: Any, leaf: str) -> None:
        parent.pop(leaf, None)

    return _apply_to_paths(value, rng, params, remove)


@_register("rename_key")
def rename_key(value: Any, rng: Any, **params: Any) -> Any:
    """An API version bump renamed a field and nobody told the agent.

    Args:
        value: The payload.
        rng: The seeded generator.
        **params: `keys`, `count`, `depth`, `suffix` (default ``"_v2"``).

    Returns:
        A copy with the selected keys renamed, values preserved.
    """
    suffix = str(params.get("suffix", "_v2"))

    def rename(parent: Any, leaf: str) -> None:
        parent[f"{leaf}{suffix}"] = parent.pop(leaf)

    return _apply_to_paths(value, rng, params, rename)


@_register("null_fields")
def null_fields(value: Any, rng: Any, **params: Any) -> Any:
    """The key survives and the value does not, so a shape check still passes.

    Nastier than dropping the key: an agent that validates presence rather than
    content sails straight through.

    Args:
        value: The payload.
        rng: The seeded generator.
        **params: `keys`, `count`, `depth`.

    Returns:
        A copy with the selected values replaced by `None`.
    """

    def nullify(parent: Any, leaf: str) -> None:
        parent[leaf] = None

    return _apply_to_paths(value, rng, params, nullify)


# --------------------------------------------------------- number and type mutations

# Catalog A1 names these four pairs. `usd`/`eur` uses a fixed rate on purpose: a
# live rate would make a seeded run non-reproducible.
_USD_TO_EUR = 0.92

_UNIT_TABLE: dict[str, Callable[[float], float]] = {
    "temp_c": lambda v: v * 9 / 5 + 32,
    "temp_f": lambda v: (v - 32) * 5 / 9,
    "km": lambda v: v * 0.621371,
    "mi": lambda v: v * 1.609344,
    "kg": lambda v: v * 2.204623,
    "lb": lambda v: v * 0.453592,
    "usd": lambda v: v * _USD_TO_EUR,
    "eur": lambda v: v / _USD_TO_EUR,
}


def _unit_for(key: str) -> Callable[[float], float] | None:
    """Find the conversion for a field name.

    Matches on a trailing unit token so `temp_c`, `distance_km` and `price_usd` all
    resolve. Returns `None` for a field with no known unit rather than inventing a
    conversion the report could not explain.

    Args:
        key: The leaf field name.

    Returns:
        The conversion, or `None`.
    """
    lowered = key.lower()
    for unit, convert in _UNIT_TABLE.items():
        if lowered == unit or lowered.endswith(f"_{unit}"):
            return convert
    return None


@_register("type_flip")
def type_flip(value: Any, rng: Any, **params: Any) -> Any:
    """A JSON encoder change quoted every number, or a field became "N/A".

    Args:
        value: The payload.
        rng: The seeded generator.
        **params: `keys`, `count`, `depth`, and `to` (``"auto"`` or ``"na"``).

    Returns:
        A copy with the selected scalars retyped: int/float to string, a digit
        string back to int, bool to string, or anything to ``"N/A"`` when
        ``to="na"``.
    """
    to = str(params.get("to", "auto"))

    def flip(parent: Any, leaf: str) -> None:
        current = parent[leaf]
        if to == "na":
            parent[leaf] = "N/A"
        elif isinstance(current, (bool, int, float)):
            parent[leaf] = str(current)
        elif isinstance(current, str):
            stripped = current.strip()
            if stripped.lstrip("-").isdigit():
                parent[leaf] = int(stripped)
            else:
                parent[leaf] = "N/A"

    return _apply_to_paths(value, rng, params, flip)


@_register("stringify_numbers")
def stringify_numbers(value: Any, rng: Any, **params: Any) -> Any:
    """A gateway that serializes every scalar as a string.

    Args:
        value: The payload.
        rng: The seeded generator.
        **params: `keys`, `count`, `depth`.

    Returns:
        A copy with the selected numbers rendered as strings.
    """

    def stringify(parent: Any, leaf: str) -> None:
        current = parent[leaf]
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            parent[leaf] = str(current)

    return _apply_to_paths(value, rng, params, stringify)


@_register("negative_numbers")
def negative_numbers(value: Any, rng: Any, **params: Any) -> Any:
    """A sign error upstream: the magnitude stays plausible, the meaning inverts.

    Args:
        value: The payload.
        rng: The seeded generator.
        **params: `keys`, `count`, `depth`.

    Returns:
        A copy with the selected numbers' signs flipped.
    """

    def flip_sign(parent: Any, leaf: str) -> None:
        current = parent[leaf]
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            parent[leaf] = -current

    return _apply_to_paths(value, rng, params, flip_sign)


@_register("nan_numbers")
def nan_numbers(value: Any, rng: Any, **params: Any) -> Any:
    """A division by zero upstream produced a non-finite number.

    Emitted in the canonical D-09 encoding, never bare: a raw `NaN` would make
    `report.json` invalid JSON, break `apply(json_patch, before) == after`, and make
    golden equality unstable because `nan != nan`.

    Args:
        value: The payload.
        rng: The seeded generator.
        **params: `keys`, `count`, `depth`, and `kind`
            (``"nan"`` | ``"inf"`` | ``"-inf"``).

    Returns:
        A copy with the selected numbers replaced by
        ``{"__float__": "NaN" | "Infinity" | "-Infinity"}``.
    """
    encodings = {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}
    encoded = {"__float__": encodings.get(str(params.get("kind", "nan")), "NaN")}

    def to_non_finite(parent: Any, leaf: str) -> None:
        if isinstance(parent[leaf], (int, float)) and not isinstance(parent[leaf], bool):
            parent[leaf] = dict(encoded)

    return _apply_to_paths(value, rng, params, to_non_finite)


@_register("unit_swap")
def unit_swap(value: Any, rng: Any, **params: Any) -> Any:
    """Celsius reported as Fahrenheit, kilometres as miles -- label unchanged.

    The nastiest mutation in the catalog, because nothing about the payload's *shape*
    is wrong. `temp_c: 21` becomes `69.8` while the key still says celsius. Only an
    agent that sanity-checks magnitudes notices, which is precisely the weakness
    being probed.

    Args:
        value: The payload.
        rng: The seeded generator.
        **params: `keys`, `count`, `depth`.

    Returns:
        A copy with the selected numbers converted and their labels intact. A field
        with no known unit is left alone.
    """

    def swap(parent: Any, leaf: str) -> None:
        current = parent[leaf]
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            return
        convert = _unit_for(leaf)
        if convert is not None:
            parent[leaf] = round(convert(float(current)), 6)

    return _apply_to_paths(value, rng, params, swap)


# --------------------------------------------------------------- string mutations

# Invisible characters: text that renders identically and compares unequal.
_ZERO_WIDTH_SPACE = "​"
_LEFT_TO_RIGHT_MARK = "‎"
_RIGHT_TO_LEFT_OVERRIDE = "‮"
_NOISE_MARKS = (_ZERO_WIDTH_SPACE, _LEFT_TO_RIGHT_MARK, _RIGHT_TO_LEFT_OVERRIDE)

_TRUNCATE_FRACTION = 0.3


@_register("truncate_string")
def truncate_string(value: Any, rng: Any, **params: Any) -> Any:
    """A response cut off by a proxy, mid-word.

    Cut to 30% of the length and deliberately not at a word boundary: a clean cut
    reads as a short answer, while a mid-word cut is recognizably broken, which is
    what makes `truncated_output_used` detectable at all.

    Args:
        value: The payload.
        rng: The seeded generator.
        **params: `keys`, `count`, `depth`, `fraction` (default 0.3).

    Returns:
        A copy with the selected strings shortened.
    """
    fraction = float(params.get("fraction", _TRUNCATE_FRACTION))

    def cut(parent: Any, leaf: str) -> None:
        current = parent[leaf]
        if isinstance(current, str) and current:
            parent[leaf] = current[: max(1, int(len(current) * fraction))]

    return _apply_to_paths(value, rng, params, cut)


@_register("unicode_noise")
def unicode_noise(value: Any, rng: Any, **params: Any) -> Any:
    """Invisible zero-width and bidirectional marks smuggled into text.

    Args:
        value: The payload.
        rng: The seeded generator, which picks the insertion points.
        **params: `keys`, `count`, `depth`.

    Returns:
        A copy with marks interleaved into the selected strings.
    """

    def add_noise(parent: Any, leaf: str) -> None:
        current = parent[leaf]
        if not isinstance(current, str) or not current:
            return
        position = rng.randrange(len(current) + 1) if len(current) else 0
        mark = _NOISE_MARKS[rng.randrange(len(_NOISE_MARKS))]
        parent[leaf] = current[:position] + mark + current[position:]

    return _apply_to_paths(value, rng, params, add_noise)


@_register("json_as_string")
def json_as_string(value: Any, rng: Any, **params: Any) -> Any:
    """A tool that double-encodes, so the agent receives a string of JSON.

    Args:
        value: The payload.
        rng: The seeded generator, unused.
        **params: Unused.

    Returns:
        The payload serialized into a string, or its `repr` when it is not
        JSON-serializable.
    """
    import json

    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


@_register("malformed_json_string")
def malformed_json_string(value: Any, rng: Any, **params: Any) -> Any:
    """A truncated JSON body: a string that will not parse.

    Returns a `str`, never a dict -- an agent calling `json.loads` on this is the
    whole point, and handing back a dict would test nothing.

    Args:
        value: The payload.
        rng: The seeded generator, unused.
        **params: Unused.

    Returns:
        A deliberately broken JSON string.
    """
    encoded = json_as_string(value, rng)
    text = encoded if isinstance(encoded, str) else repr(encoded)
    # Cut mid-structure and leave the braces unbalanced.
    return text[: max(1, len(text) - 1)] if len(text) > 2 else '{"truncated": '


@_register("whitespace_only")
def whitespace_only(value: Any, rng: Any, **params: Any) -> Any:
    """A 200 response whose body is nothing but spaces.

    Args:
        value: The payload, unused.
        rng: The seeded generator, unused.
        **params: Unused.

    Returns:
        A short run of whitespace -- not the empty string, so an agent's
        `if not response` check may still pass and the answer still be useless.
    """
    return "   \n  "


# ----------------------------------------------------------------- list mutations


def _first_list_in(value: Any) -> tuple[Any, str | int] | None:
    """Locate the list a list mutation should act on.

    Real payloads wrap their records in an envelope, so a bare list and
    ``{"items": [...]}`` both have to work.

    Args:
        value: The payload.

    Returns:
        ``(container, key)`` addressing the list, or `None` when there is none.
    """
    if isinstance(value, list):
        return None
    if isinstance(value, dict):
        for key in sorted(value):
            if isinstance(value[key], list):
                return value, key
    return None


def _on_list(value: Any, transform: Callable[[list[Any]], list[Any]]) -> Any:
    """Apply `transform` to the payload's list, wherever it is.

    Args:
        value: The payload.
        transform: A pure list-to-list function.

    Returns:
        The mutated copy, or the original when it could not be copied or holds no
        list.
    """
    working, copied = _copy(value)
    if not copied:
        return working
    if isinstance(working, list):
        return transform(working)
    found = _first_list_in(working)
    if found is not None:
        container, key = found
        container[key] = transform(container[key])
    return working


@_register("truncate_list")
def truncate_list(value: Any, rng: Any, **params: Any) -> Any:
    """Only the first record comes back, so a three-day forecast becomes one.

    Args:
        value: The payload.
        rng: The seeded generator, unused.
        **params: `keep` (default 1).

    Returns:
        A copy whose list keeps only the first `keep` elements.
    """
    keep = int(params.get("keep", 1))
    return _on_list(value, lambda items: items[:keep])


@_register("duplicate_items")
def duplicate_items(value: Any, rng: Any, **params: Any) -> Any:
    """A paging bug that serves the same page twice.

    Args:
        value: The payload.
        rng: The seeded generator, unused.
        **params: `times` (default 2).

    Returns:
        A copy whose list is repeated `times` times.
    """
    times = max(1, int(params.get("times", 2)))
    return _on_list(value, lambda items: items * times)


@_register("reorder_list")
def reorder_list(value: Any, rng: Any, **params: Any) -> Any:
    """The same records in a different order, run to run.

    Proves comparison and dedup logic that quietly assumes a stable order.

    Args:
        value: The payload.
        rng: The seeded generator, which chooses the permutation.
        **params: Unused.

    Returns:
        A copy whose list is shuffled. A list of two or more distinct orderings is
        guaranteed to differ from the input.
    """

    def shuffle(items: list[Any]) -> list[Any]:
        if len(items) < 2:
            return items
        shuffled = list(items)
        for _ in range(8):
            rng.shuffle(shuffled)
            if shuffled != items:
                return shuffled
        # Degenerate case: every element is equal, so any order is the same order.
        return shuffled

    return _on_list(value, shuffle)


# ------------------------------------------------------------ structural mutations

_NEST_LEVELS = 5

# A plausible neighbouring response rather than an obviously-empty one: an empty
# object trips a shape check, and a plausible object does not.
_WRONG_SCHEMA_SHAPE: dict[str, Any] = {
    "status": "ok",
    "data": {"id": "rec_000", "attributes": {}},
    "meta": {"page": 1, "total": 0},
}


@_register("deep_nest")
def deep_nest(value: Any, rng: Any, **params: Any) -> Any:
    """A gateway that wrapped the response in envelope after envelope.

    Five extra levels, so a walker with a fixed depth budget gives up before it
    reaches the data.

    Args:
        value: The payload.
        rng: The seeded generator, unused.
        **params: `levels` (default 5).

    Returns:
        A copy wrapped in `levels` additional mappings.
    """
    working, copied = _copy(value)
    if not copied:
        return working
    levels = int(params.get("levels", _NEST_LEVELS))
    for _ in range(levels):
        working = {"nested": working}
    return working


@_register("wrong_schema")
def wrong_schema(value: Any, rng: Any, **params: Any) -> Any:
    """A neighbouring endpoint's response, returned by mistake.

    Plausible but different: it parses, it has fields, and none of them are the ones
    the agent wanted.

    Args:
        value: The payload, unused beyond being replaced.
        rng: The seeded generator, unused.
        **params: `shape` to override the substituted object.

    Returns:
        A different, well-formed object.
    """
    shape = params.get("shape", _WRONG_SCHEMA_SHAPE)
    copied, ok = _copy(shape)
    return copied if ok else dict(_WRONG_SCHEMA_SHAPE)


# ---------------------------------------------------------- drop_required_key (D-17)


def _distinctiveness(value: Any) -> int:
    """Score how strongly a value's presence in a prompt implies the agent used it.

    A long, unusual string is strong evidence. A bare digit could appear in a prompt
    by coincidence, so it scores low.

    Args:
        value: A scalar payload value.

    Returns:
        A score; higher means more distinctive.
    """
    text = str(value)
    score = len(text)
    if any(ch.isalpha() for ch in text):
        score += 10
    if any(ch in "-_/:" for ch in text):
        score += 5
    return score


@_register("drop_required_key")
def drop_required_key(value: Any, rng: Any, **params: Any) -> Any:
    """Remove a field the baseline run demonstrably carried into its prompt.

    "Keys the agent used" is not observable -- the engine sees tool boundaries, not
    attribute access. D-17 redefines it as keys whose scalar value later appears in
    a baseline `llm_request`, ranked by distinctiveness. With no usable baseline it
    degrades to `drop_key`, and the fault records
    ``"no baseline; fell back to drop_key"``.

    Args:
        value: The payload.
        rng: The seeded generator, used for the tie-break and the fallback.
        **params: `baseline_prompts` (the baseline's prompt texts), plus `drop_key`'s
            `keys`, `count` and `depth`.

    Returns:
        A copy with the chosen key removed from every record.
    """
    prompts = params.get("baseline_prompts") or []
    if not prompts or params.get("keys"):
        return drop_key(value, rng, **params)

    haystack = "\n".join(str(p) for p in prompts)
    records = _records_of(value)
    scored: list[tuple[int, str]] = []
    for path in sorted(
        {p for record in records for p in _candidate_paths(record, int(params.get("depth", 3)))}
    ):
        for record in records:
            found = _parent_and_leaf(record, path)
            if found is None:
                continue
            leaf_value = found[0][found[1]]
            is_scalar = isinstance(leaf_value, (str, int, float)) and not isinstance(
                leaf_value, bool
            )
            if is_scalar and str(leaf_value) in haystack:
                scored.append((_distinctiveness(leaf_value), path))
            break

    if not scored:
        return drop_key(value, rng, **params)

    # Sorted by (-score, path) so the choice is deterministic on ties.
    best = sorted(scored, key=lambda item: (-item[0], item[1]))[0][1]
    return drop_key(value, rng, **{**params, "keys": [best]})
