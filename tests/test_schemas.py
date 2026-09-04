"""The packaged schemas, and the examples they govern.

The schemas are the product, so these are the tests that matter most in M0: the four
documents compile, they resolve each other's `$ref`s from local resources only, and
every shipped example validates. `docs/04-SCHEMAS.md` §1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import enums
from agent_loop_chaos.errors import SchemaError
from agent_loop_chaos.schema import (
    SCHEMA_FILES,
    load_schema,
    registry,
    validate_obj,
    validator_for,
)

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
DATA = TESTS_DIR / "data"
NAMES = sorted(SCHEMA_FILES)


@pytest.mark.parametrize("name", NAMES)
def test_every_schema_loads_and_compiles(name: str) -> None:
    """Each schema is valid Draft 2020-12 and yields a working validator."""
    schema = load_schema(name)  # type: ignore[arg-type]
    assert schema["$schema"].endswith("2020-12/schema")
    assert validator_for(name) is not None  # type: ignore[arg-type]


@pytest.mark.parametrize("name", NAMES)
def test_every_schema_id_is_substituted(name: str) -> None:
    """No `$id` still carries the `OWNER` placeholder (D-39, D-50)."""
    schema_id = load_schema(name)["$id"]  # type: ignore[arg-type]
    assert "OWNER" not in schema_id
    assert schema_id.startswith("https://github.com/dilipg/agent-loop-chaos/schemas/")


def test_registry_is_built_once_and_covers_both_keys() -> None:
    """Every schema resolves by `$id` and by bare filename."""
    reg = registry()
    assert registry() is reg, "the registry must be cached per process"
    for name, filename in SCHEMA_FILES.items():
        assert reg.get_or_retrieve(filename).value is not None
        assert reg.get_or_retrieve(load_schema(name)["$id"]).value is not None


def test_resolving_an_unknown_id_raises_instead_of_fetching() -> None:
    """The retrieve hook refuses the network, so an unknown `$id` fails loudly.

    `referencing` wraps whatever the hook raises in `Unretrievable`, so the guarantee
    is checked on the `__cause__`: no fetch was attempted, and the reason is ours.
    """
    from referencing.exceptions import Unretrievable

    with pytest.raises(Unretrievable) as exc:
        registry().get_or_retrieve("https://example.invalid/not-a-schema.json")
    cause = exc.value.__cause__
    assert isinstance(cause, SchemaError)
    assert "refusing to resolve schema over the network" in str(cause)


def test_unknown_schema_name_is_a_schema_error() -> None:
    """A bad short name is a `SchemaError`, not a `KeyError`."""
    with pytest.raises(SchemaError, match="unknown schema"):
        load_schema("nope")  # type: ignore[arg-type]


def test_cross_file_refs_resolve_locally() -> None:
    """`judge_verdict` and `chaos_report` reference each other and still validate."""
    verdict_schema = json.dumps(load_schema("verdict"))
    report_schema = json.dumps(load_schema("report"))
    assert "chaos_report.schema.json" in verdict_schema or "judge_verdict" in report_schema
    # Compiling both against the shared registry is what proves the refs resolve.
    assert validator_for("verdict") is not None
    assert validator_for("report") is not None


def test_report_example_validates() -> None:
    """`report_failing.json` validates against the report schema."""
    obj = json.loads((DATA / "report_failing.json").read_text(encoding="utf-8"))
    assert validate_obj(obj, "report") == []


def test_trace_excerpt_validates_line_by_line() -> None:
    """Every non-blank line of `trace_excerpt.jsonl` is a valid trace event."""
    lines = (DATA / "trace_excerpt.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    assert events, "the trace excerpt fixture is empty"
    failures = {i: validate_obj(ev, "trace") for i, ev in enumerate(events, 1)}
    assert {i: errs for i, errs in failures.items() if errs} == {}


def test_suite_example_validates() -> None:
    """`suite_demo.yaml` validates against the scenario schema."""
    yaml = pytest.importorskip("yaml", reason="pyyaml is in [dev]; skip if absent")
    suite = yaml.safe_load((DATA / "suite_demo.yaml").read_text(encoding="utf-8"))
    assert validate_obj(suite, "scenario") == []


def test_validation_errors_carry_a_pointer_and_a_truncated_value() -> None:
    """Error strings surface in `schema_errors[]` and in CI logs, so they must be legible."""
    errors = validate_obj({"schema_version": 1}, "report")
    assert errors, "an obviously invalid report must produce errors"
    assert all(err.startswith("/") for err in errors)
    assert any("value=" in err for err in errors)


def test_a_long_failing_value_is_truncated() -> None:
    """A huge offending value does not flood `schema_errors[]` or a CI log.

    Both halves are capped at 200 chars: jsonschema's `message`, which embeds the
    instance, and the appended `value=` repr. So the worst case is the JSON pointer
    plus two 201-char segments plus the literals — comfortably under 500, against
    5000 characters of input.
    """
    errors = validate_obj({"schema_version": "x" * 5000}, "report")
    offending = [e for e in errors if e.startswith("/schema_version")]
    assert offending
    assert len(offending[0]) < 500
    assert "x" * 300 not in offending[0], "the value was not actually truncated"
    assert offending[0].count("…") == 2, "both the message and the value are clipped"


@pytest.mark.parametrize(
    ("alias", "prop", "schema_name"),
    [
        (enums.FAILURE_MODES, "failure_mode", "report"),
        (enums.SEVERITIES, "severity", "report"),
        (enums.OBSERVED_BEHAVIORS, "observed_behavior", "verdict"),
    ],
)
def test_literal_aliases_match_their_schema_enum(
    alias: tuple[str, ...], prop: str, schema_name: str
) -> None:
    """The `Literal` aliases cannot drift from the schemas (D-51)."""
    schema: dict[str, Any] = load_schema(schema_name)  # type: ignore[arg-type]
    assert list(alias) == schema["properties"][prop]["enum"]


def test_expected_behavior_alias_matches_the_scenario_schema() -> None:
    """`ExpectedBehavior` mirrors `scenario.schema.json`, not the report schema."""
    body = load_schema("scenario")["$defs"]["scenarioBody"]["properties"]
    assert list(enums.EXPECTED_BEHAVIORS) == body["expected_behavior"]["enum"]


def test_latency_budget_exceeded_is_retained_but_unused() -> None:
    """D-46: the enum value stays for compatibility; nothing emits it."""
    assert "latency_budget_exceeded" in enums.FAILURE_MODES


def test_vendored_schemas_are_byte_identical_to_the_reference_copies() -> None:
    """`src/agent_loop_chaos/schemas/` is copied verbatim from `schemas/`."""
    reference = REPO_ROOT / "schemas"
    if not reference.exists():  # pragma: no cover - not shipped in the wheel
        pytest.skip("reference schemas/ is not present in an installed distribution")
    vendored = REPO_ROOT / "src" / "agent_loop_chaos" / "schemas"
    for filename in SCHEMA_FILES.values():
        assert (reference / filename).read_bytes() == (vendored / filename).read_bytes(), filename
