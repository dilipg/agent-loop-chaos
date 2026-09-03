"""Determinism primitives: seeded RNG, derived ids, canonical JSON, float sanitizing.

All randomness in the library goes through `rng`. Nothing calls `random.*` directly,
nothing reads the clock inside decision logic, and no id that lands in a report comes
from `uuid4` — ids are derived from the seed and a counter.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

__all__ = [
    "canonical_json",
    "derive_id",
    "rng",
    "sanitize_floats",
    "sha256_of",
]

_FLOAT_PLACES = 6


def _digest(text: str) -> str:
    """Hex SHA-256 of a string.

    Args:
        text: The string to hash.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rng(seed: int, purpose: str) -> random.Random:
    """Build a `random.Random` for one purpose.

    A **pure factory**: there is deliberately no module-level memo (D-02). A global
    cache would make two runs in one process share a stream, which would break the
    same-seed determinism test, baseline-plus-chaos in one process, every
    `RefinementLoop` round, and would race under `jobs > 1`. The only cache is
    `RunContext.rng_registry`.

    Purposes are independent because the state is seeded from
    ``sha256(f"{seed}|{purpose}")``, so drawing from one stream never advances
    another. Purpose keys are canonically ``f"{fault_key}:{purpose}"`` where
    `fault_key` is the hash of the fault spec rather than its ordinal (D-03) — which
    is what stops inserting or reordering a fault from re-keying every other fault's
    stream.

    Args:
        seed: The run's root seed.
        purpose: A stream label, e.g. ``"a1f39c:trigger"``.

    Returns:
        A fresh generator, at draw zero, for that (seed, purpose) pair.
    """
    return random.Random(int(_digest(f"{seed}|{purpose}"), 16))


def derive_id(seed: int, kind: str, n: int) -> str:
    """Derive a short deterministic id.

    Args:
        seed: The run's root seed.
        kind: What is being identified, e.g. ``"span"``.
        n: A counter, so ids within a kind are distinct and ordered.

    Returns:
        An 8-character hex id. Never a UUID: a report must be byte-reproducible.
    """
    return _digest(f"{seed}|{kind}|{n}")[:8]


def sanitize_floats(obj: Any) -> Any:
    """Replace non-finite floats with their canonical JSON encoding.

    ``NaN``, ``Infinity`` and ``-Infinity`` are not valid JSON, so every dump uses
    ``allow_nan=False`` and this pass first rewrites them as
    ``{"__float__": "NaN" | "Infinity" | "-Infinity"}`` (D-09). Finite floats are
    rounded to 6 decimals so golden comparisons are stable. The result is that
    ``apply(json_patch, before) == after`` still holds and a `jq` or Go consumer can
    always read a report.

    Args:
        obj: Any value bound for serialization.

    Returns:
        A new value with non-finite floats encoded and finite floats rounded. The
        input is never mutated.
    """
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return {"__float__": "NaN"}
        if math.isinf(obj):
            return {"__float__": "Infinity" if obj > 0 else "-Infinity"}
        return round(obj, _FLOAT_PLACES)
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_floats(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Serialize deterministically, for hashing.

    Keys are sorted and separators are fixed, so no whitespace drift can change a
    hash. `sanitize_floats` runs first, so the dump can safely forbid non-finite
    floats (`docs/04-SCHEMAS.md` §3).

    Args:
        obj: The value to serialize.

    Returns:
        The canonical JSON string.

    Raises:
        TypeError: If the value contains something not JSON-serializable.
    """
    return json.dumps(
        sanitize_floats(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ": "),
        allow_nan=False,
    )


def sha256_of(obj: Any) -> str:
    """Hash a value by its canonical JSON form.

    Args:
        obj: The value to hash.

    Returns:
        ``"sha256:<hex>"``, the form used for `plan_hash` and truncation stubs.

    Raises:
        TypeError: If the value is not JSON-serializable.
    """
    return "sha256:" + _digest(canonical_json(obj))
