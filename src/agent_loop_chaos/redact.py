"""Secret redaction.

Every value passes through here before it enters a trace event or a report field.
Redaction runs **before** truncation and before hashing, so a hash never carries a
secret (`docs/04-SCHEMAS.md` §4).

The replacement names the reason rather than the value, e.g. ``<redacted:api_key>``,
which keeps a report legible without leaking anything.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = ["DEFAULT_KEY_PATTERNS", "DEFAULT_VALUE_PATTERNS", "redact", "redaction_reason"]

# Key deny-list, grouped so the replacement can name a reason. Case-insensitive.
DEFAULT_KEY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("api_key", r"api[_-]?key|access[_-]?key"),
    ("private_key", r"private[_-]?key"),
    ("secret", r"secret|credential"),
    ("token", r"token|bearer"),
    ("password", r"password|passwd"),
    ("authorization", r"authorization|auth"),
    ("cookie", r"cookie|session"),
)

# Value shapes that are secrets wherever they appear, whatever the key is called.
DEFAULT_VALUE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("openai_key", r"sk-[A-Za-z0-9]{16,}"),
    ("github_token", r"ghp_[A-Za-z0-9]{20,}"),
    ("aws_access_key", r"AKIA[0-9A-Z]{16}"),
    ("slack_token", r"xox[bap]-[A-Za-z0-9-]+"),
    ("private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("jwt", r"eyJ[A-Za-z0-9_-]{10,}\."),
)

_KEY_RES = tuple((reason, re.compile(pat, re.IGNORECASE)) for reason, pat in DEFAULT_KEY_PATTERNS)
_VALUE_RES = tuple((reason, re.compile(pat)) for reason, pat in DEFAULT_VALUE_PATTERNS)

_PLACEHOLDER = "<redacted:{reason}>"


def _key_reason(key: str) -> str | None:
    """Return the deny-list reason a key matches, if any.

    Args:
        key: A mapping key.

    Returns:
        The reason label, or `None` when the key is not sensitive.
    """
    for reason, pattern in _KEY_RES:
        if pattern.search(key):
            return reason
    return None


def _value_reason(value: str) -> str | None:
    """Return the value-pattern reason a string matches, if any.

    Args:
        value: The string to inspect.

    Returns:
        The reason label, or `None` when the string looks harmless.
    """
    for reason, pattern in _VALUE_RES:
        if pattern.search(value):
            return reason
    return None


def redaction_reason(key: str | None, value: Any) -> str | None:
    """Report why a value would be redacted.

    Args:
        key: The mapping key the value sits under, if any.
        value: The value itself.

    Returns:
        A reason label, or `None` when nothing matches.
    """
    if key is not None:
        reason = _key_reason(key)
        if reason is not None:
            return reason
    if isinstance(value, str):
        return _value_reason(value)
    return None


def _compile_extra(extra_keys: Sequence[str]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compile caller-supplied key patterns.

    Args:
        extra_keys: Patterns from `ChaosEngine(redact_keys=…)`.

    Returns:
        Reason/pattern pairs, all reported as ``user_key``.
    """
    return tuple(("user_key", re.compile(pat, re.IGNORECASE)) for pat in extra_keys)


def redact(
    obj: Any,
    extra_keys: Sequence[str] = (),
    *,
    allow: Iterable[str] = (),
) -> Any:
    """Return a redacted deep copy of `obj`.

    Recurses through mappings, sequences, sets and dataclasses, and inspects strings
    for known secret shapes. **The input is never mutated** — faults and the trace
    recorder both rely on that.

    Args:
        obj: The value to redact.
        extra_keys: Additional key regexes, from `ChaosEngine(redact_keys=…)`.
        allow: Literal strings exempt from redaction. This exists for planted
            canaries: the `secret_in_output` probe has to be able to see whether a
            canary reached the output, so the canary — and only the canary — passes
            through. Real credentials are never exempted (`docs/04` §4).

    Returns:
        A new value with every secret replaced by ``<redacted:<reason>>``.
    """
    allowed = frozenset(allow)
    key_res = _KEY_RES + _compile_extra(extra_keys)
    return _walk(obj, None, allowed, key_res)


def _redacted_string(value: str, allowed: frozenset[str]) -> Any:
    """Redact a bare string by its own shape.

    Args:
        value: The string to inspect.
        allowed: Exempt literals.

    Returns:
        The placeholder when the string looks like a secret, else the string.
    """
    if value in allowed:
        return value
    reason = _value_reason(value)
    return _PLACEHOLDER.format(reason=reason) if reason else value


def _walk(
    obj: Any,
    key: str | None,
    allowed: frozenset[str],
    key_res: tuple[tuple[str, re.Pattern[str]], ...],
) -> Any:
    """Recursively redact a value.

    Args:
        obj: The value under inspection.
        key: The mapping key it sits under, if any.
        allowed: Exempt literals.
        key_res: Compiled key patterns, including any user additions.

    Returns:
        The redacted copy.
    """
    if key is not None and not (isinstance(obj, str) and obj in allowed):
        for reason, pattern in key_res:
            if pattern.search(key):
                return _PLACEHOLDER.format(reason=reason)

    if isinstance(obj, str):
        return _redacted_string(obj, allowed)
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, Mapping):
        return {k: _walk(v, str(k), allowed, key_res) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        # Sorted by repr so the output is order-stable; a set's iteration order is not.
        return sorted((_walk(v, None, allowed, key_res) for v in obj), key=repr)
    if isinstance(obj, tuple):
        return tuple(_walk(v, None, allowed, key_res) for v in obj)
    if isinstance(obj, (list, Sequence)) and not isinstance(obj, (str, bytes)):
        return [_walk(v, None, allowed, key_res) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _walk(getattr(obj, f.name), f.name, allowed, key_res)
            for f in dataclasses.fields(obj)
        }
    return obj
