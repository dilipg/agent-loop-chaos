"""A minimal RFC 6902 subset: `add`, `remove`, `replace`.

Every fault records what it changed as a patch, because "the exact payload diff" is
half of what makes a finding actionable. The subset is deliberate: no `move`, no
`copy`, no `test`, and no move detection in lists — an index-based diff is
predictable, and predictability matters more here than compactness.

`apply` exists so a property test can assert ``apply(diff(a, b), a) == b``.
"""

from __future__ import annotations

import copy
import json
from typing import Any

__all__ = ["apply", "diff", "escape_token", "is_jsonable"]

Op = dict[str, Any]


def escape_token(token: str) -> str:
    """Escape one JSON Pointer reference token (RFC 6901 §3).

    Args:
        token: A raw key.

    Returns:
        The token with ``~`` and ``/`` escaped, in that order.
    """
    return token.replace("~", "~0").replace("/", "~1")


def _unescape_token(token: str) -> str:
    """Reverse `escape_token`.

    Args:
        token: An escaped reference token.

    Returns:
        The raw key.
    """
    return token.replace("~1", "/").replace("~0", "~")


def _join(pointer: str, token: str) -> str:
    """Append a token to a JSON Pointer.

    Args:
        pointer: The parent pointer, ``""`` for the document root.
        token: The child key or index.

    Returns:
        The child pointer.
    """
    return f"{pointer}/{escape_token(token)}"


def is_jsonable(obj: Any) -> bool:
    """Report whether a value can be represented as JSON.

    Args:
        obj: The value to test.

    Returns:
        True when `json.dumps` accepts it. Non-finite floats count as
        representable, because `seeding.sanitize_floats` encodes them (D-09).
    """
    try:
        json.dumps(obj, allow_nan=True)
    except (TypeError, ValueError):
        return False
    return True


def diff(before: Any, after: Any) -> list[Op]:
    """Compute the patch that turns `before` into `after`.

    Args:
        before: The original value.
        after: The mutated value.

    Returns:
        A list of `add`/`remove`/`replace` operations, in an order safe to apply
        sequentially. Empty when the values are equal, or when either side is not
        JSON-representable — in which case the caller records
        `MutationLog.unrepresentable` instead. Use `is_jsonable` to tell the two
        empty cases apart.
    """
    if not is_jsonable(before) or not is_jsonable(after):
        return []
    return _diff(before, after, "")


def _diff(before: Any, after: Any, pointer: str) -> list[Op]:
    """Recursive worker for `diff`.

    Args:
        before: The original value at `pointer`.
        after: The mutated value at `pointer`.
        pointer: The JSON Pointer to this position.

    Returns:
        The operations for this subtree.
    """
    if type(before) is type(after) or (
        isinstance(before, (int, float))
        and isinstance(after, (int, float))
        and not isinstance(before, bool)
        and not isinstance(after, bool)
    ):
        if isinstance(before, dict) and isinstance(after, dict):
            return _diff_dict(before, after, pointer)
        if isinstance(before, list) and isinstance(after, list):
            return _diff_list(before, after, pointer)

    if before == after and type(before) is type(after):
        return []
    return [{"op": "replace", "path": pointer, "value": after}]


def _diff_dict(before: dict[Any, Any], after: dict[Any, Any], pointer: str) -> list[Op]:
    """Diff two mappings.

    Keys are visited in sorted order so the patch is deterministic regardless of
    insertion order.

    Args:
        before: The original mapping.
        after: The mutated mapping.
        pointer: Pointer to this mapping.

    Returns:
        The operations for this mapping.
    """
    ops: list[Op] = []
    before_keys = {str(k) for k in before}
    after_keys = {str(k) for k in after}

    for key in sorted(before_keys - after_keys):
        ops.append({"op": "remove", "path": _join(pointer, key)})
    for key in sorted(after_keys - before_keys):
        ops.append({"op": "add", "path": _join(pointer, key), "value": after[key]})
    for key in sorted(before_keys & after_keys):
        ops.extend(_diff(before[key], after[key], _join(pointer, key)))
    return ops


def _diff_list(before: list[Any], after: list[Any], pointer: str) -> list[Op]:
    """Diff two sequences by index.

    No move detection: an insertion at the head re-describes the tail. That is
    verbose but predictable, and a fault's mutation is almost always local.

    Args:
        before: The original list.
        after: The mutated list.
        pointer: Pointer to this list.

    Returns:
        The operations for this list. Removals are emitted from the tail backwards
        so applying them in order stays correct.
    """
    ops: list[Op] = []
    common = min(len(before), len(after))
    for index in range(common):
        ops.extend(_diff(before[index], after[index], _join(pointer, str(index))))
    for index in range(len(before), len(after)):
        ops.append({"op": "add", "path": _join(pointer, str(index)), "value": after[index]})
    for index in range(len(before) - 1, len(after) - 1, -1):
        ops.append({"op": "remove", "path": _join(pointer, str(index))})
    return ops


def _resolve(doc: Any, tokens: list[str]) -> Any:
    """Walk to the container holding the final token.

    Args:
        doc: The document root.
        tokens: Unescaped reference tokens, excluding the last.

    Returns:
        The container at that path.

    Raises:
        KeyError: If the path does not exist.
    """
    node = doc
    for token in tokens:
        node = node[int(token)] if isinstance(node, list) else node[token]
    return node


def apply(before: Any, patch: list[Op]) -> Any:
    """Apply a patch, returning a new value.

    Used by the property test that pins `diff`'s correctness, and by `replay` when
    reconstructing a payload. The input is never mutated.

    Args:
        before: The document to patch.
        patch: Operations from `diff`.

    Returns:
        A new document with the operations applied.

    Raises:
        KeyError: If an operation's path does not exist.
        ValueError: If an operation has an unknown `op`.
    """
    doc = copy.deepcopy(before)
    for op in patch:
        kind = op["op"]
        path = op["path"]
        if path == "":
            if kind != "replace":
                raise ValueError(f"cannot {kind} the document root")
            doc = copy.deepcopy(op["value"])
            continue

        tokens = [_unescape_token(t) for t in path.split("/")[1:]]
        container = _resolve(doc, tokens[:-1])
        last = tokens[-1]

        if kind == "remove":
            if isinstance(container, list):
                del container[int(last)]
            else:
                del container[last]
        elif kind in {"add", "replace"}:
            value = copy.deepcopy(op["value"])
            if isinstance(container, list):
                index = len(container) if last == "-" else int(last)
                if kind == "add" and index >= len(container):
                    container.append(value)
                elif kind == "add":
                    container.insert(index, value)
                else:
                    container[index] = value
            else:
                container[last] = value
        else:
            raise ValueError(f"unsupported patch op: {kind!r}")
    return doc
