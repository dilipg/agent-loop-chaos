"""Offline JSON Schema registry and validation.

The four schemas are packaged as data files so a consumer can validate a report
without cloning the repo. `judge_verdict.schema.json` and
`chaos_report.schema.json` reference each other, so validation needs a
`referencing.Registry` built from all four and keyed by `$id`.

**Nothing here ever touches the network.** The registry is given a retrieve function
that raises, so an unresolvable `$id` fails loudly instead of being fetched
(`docs/04-SCHEMAS.md` §1).
"""

from __future__ import annotations

import json
from functools import cache, lru_cache
from typing import Any, Literal, cast

from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from .errors import SchemaError

__all__ = ["SCHEMA_FILES", "SchemaName", "load_schema", "registry", "validate_obj", "validator_for"]

SchemaName = Literal["report", "trace", "verdict", "scenario"]

SCHEMA_FILES: dict[SchemaName, str] = {
    "report": "chaos_report.schema.json",
    "trace": "trace_event.schema.json",
    "verdict": "judge_verdict.schema.json",
    "scenario": "scenario.schema.json",
}

_MAX_VALUE_CHARS = 200
_MAX_MESSAGE_CHARS = 200


def _read_packaged(filename: str) -> dict[str, Any]:
    """Read one packaged schema file.

    Args:
        filename: Bare filename inside `agent_loop_chaos/schemas/`.

    Returns:
        The parsed schema document.

        Raises:
        SchemaError: If the file is missing or is not valid JSON.
    """
    from importlib.resources import files

    try:
        text = (files("agent_loop_chaos") / "schemas" / filename).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise SchemaError(f"packaged schema not found: {filename}") from exc
    try:
        return cast("dict[str, Any]", json.loads(text))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"packaged schema is not valid JSON: {filename}: {exc}") from exc


@cache
def load_schema(name: SchemaName) -> dict[str, Any]:
    """Load one schema by short name.

    Args:
        name: One of ``"report"``, ``"trace"``, ``"verdict"``, ``"scenario"``.

    Returns:
        The parsed schema document.

    Raises:
        SchemaError: If `name` is unknown, or the file is missing or malformed.
    """
    try:
        filename = SCHEMA_FILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(SCHEMA_FILES))
        raise SchemaError(f"unknown schema {name!r}; expected one of: {known}") from exc
    return _read_packaged(filename)


def _refuse_to_fetch(uri: str) -> Resource[Any]:
    """Retrieve hook that always refuses.

    Args:
        uri: The `$id` the validator tried to resolve.

    Returns:
        Never returns.

    Raises:
        SchemaError: Always.
    """
    raise SchemaError(
        f"refusing to resolve schema over the network: {uri!r}. "
        "Only the four packaged schemas are resolvable."
    )


@lru_cache(maxsize=1)
def registry() -> Registry[Any]:
    """Build the schema registry once per process.

    Every schema is registered under both its `$id` and its bare filename, so
    relative and absolute `$ref`s both resolve locally.

    Returns:
        A `referencing.Registry` over the four packaged schemas, whose retrieve hook
        raises rather than fetching.

    Raises:
        SchemaError: If a packaged schema is missing or malformed.
    """
    # `retrieve` is the attrs alias for the private field `_retrieve`; mypy's attrs
    # plugin does not honour the alias, so it reports a spurious call-arg error.
    # Verified working at runtime against referencing's public constructor.
    reg: Registry[Any] = Registry(retrieve=_refuse_to_fetch)  # type: ignore[call-arg]
    for name, filename in SCHEMA_FILES.items():
        doc = load_schema(name)
        resource = Resource.from_contents(doc, default_specification=DRAFT202012)
        reg = reg.with_resource(uri=filename, resource=resource)
        schema_id = doc.get("$id")
        if isinstance(schema_id, str) and schema_id:
            reg = reg.with_resource(uri=schema_id, resource=resource)
    return reg


@cache
def validator_for(name: SchemaName) -> Validator:
    """Return a compiled validator for one schema.

    Args:
        name: One of ``"report"``, ``"trace"``, ``"verdict"``, ``"scenario"``.

    Returns:
        A Draft 2020-12 validator wired to the offline registry.

    Raises:
        SchemaError: If `name` is unknown, or the schema itself is invalid.
    """
    schema = load_schema(name)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise SchemaError(
            f"{SCHEMA_FILES[name]} is not a valid Draft 2020-12 schema: {exc}"
        ) from exc
    return cast("Validator", Draft202012Validator(schema, registry=registry()))


def _clip(text: str, limit: int) -> str:
    """Truncate `text` to `limit` characters, marking the cut.

    Args:
        text: The text to clip.
        limit: Maximum length before the ellipsis marker.

    Returns:
        `text` unchanged when short enough, otherwise its first `limit`
        characters followed by an ellipsis.
    """
    return text if len(text) <= limit else text[:limit] + "…"


def _truncate(value: object) -> str:
    """Render a failing value compactly for an error string.

    Args:
        value: The value that failed validation.

    Returns:
        Its repr, truncated to 200 characters with an ellipsis marker.
    """
    return _clip(repr(value), _MAX_VALUE_CHARS)


def validate_obj(obj: Any, name: SchemaName) -> list[str]:
    """Validate an object against one schema.

    Error strings carry the JSON pointer and the failing value, because they surface
    in the report's `schema_errors[]` and in CI logs.

    Args:
        obj: The instance to validate.
        name: Which schema to validate against.

    Returns:
        Human-readable error strings, sorted by JSON pointer. Empty when valid.

    Raises:
        SchemaError: If `name` is unknown or the schema cannot be compiled.
    """
    validator = validator_for(name)
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path))
    out: list[str] = []
    for err in errors:
        pointer = "/" + "/".join(str(part) for part in err.absolute_path)
        # jsonschema embeds the offending instance in `message`, so capping only the
        # value would still let a multi-megabyte payload reach a CI log. Cap both.
        message = _clip(err.message, _MAX_MESSAGE_CHARS)
        out.append(f"{pointer}: {message} (value={_truncate(err.instance)})")
    return out
