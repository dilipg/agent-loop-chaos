"""Scenarios, the matrix, presets, and suite loading.

Presets are literal `FaultSpec` lists rather than prose, because they feed
`plan_hash` and the golden tests (D-14). Matrix ids follow the single canonical rule
in D-15. `must_not` is validated against the registered probe codes, because a typo
that silently never matches is worse than a loud failure (D-33).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.scenarios import (
    PRESETS,
    FaultSpec,
    Scenario,
    load_suite,
    resolve_preset,
    slug,
)


def scenario(**kw: Any) -> Scenario:
    """Build a scenario with defaults.

    Args:
        **kw: Field overrides.

    Returns:
        The scenario.
    """
    kw.setdefault("id", "s1")
    kw.setdefault("entrypoint", "tests.fakes.naive_tool_agent:run")
    return Scenario(**kw)


# ------------------------------------------------------------------ the slug rule


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("drop_key", "drop_key"),
        ("Unit Swap", "unit-swap"),
        ("http_500", "http_500"),
        ("a//b", "a-b"),
        ("Ünïcode!", "-n-code-"),
        (1337, "1337"),
        (True, "true"),
    ],
)
def test_slug_maps_anything_outside_the_id_alphabet(value: Any, expected: str) -> None:
    """D-15: lowercase, map anything outside `[a-z0-9._-]` to `-`, collapse runs."""
    assert slug(value) == expected


# -------------------------------------------------------------- matrix expansion


def test_matrix_expands_the_cartesian_product() -> None:
    """The phase's own example: two mutation types by two seeds is four scenarios."""
    expanded = scenario(
        faults=[{"type": "ToolCorruptionFault", "params": {"mutation_type": "drop_key"}}],
        matrix={"faults.0.params.mutation_type": ["drop_key", "unit_swap"], "seed": [1, 2]},
    ).expand()
    assert len(expanded) == 4


def test_matrix_ids_follow_the_canonical_rule() -> None:
    """`<base_id>-<last_key_segment>-<slug(value)>`, and no brackets.

    Brackets violate the id pattern in `scenario.schema.json`, which is why the old
    `id[value]` form had to go.
    """
    expanded = scenario(
        id="tool.contract",
        faults=[{"type": "ToolCorruptionFault", "params": {"mutation_type": "x"}}],
        matrix={"faults.0.params.mutation_type": ["drop_key"]},
    ).expand()
    assert expanded[0].id == "tool.contract-mutation_type-drop_key"
    assert "[" not in expanded[0].id


def test_matrix_products_are_ordered_by_sorted_key_then_declared_value_order() -> None:
    """D-15 pins the order, because the ids land in `suite.json` and in goldens."""
    expanded = scenario(
        faults=[{"type": "NoopFault", "params": {"x": "seed"}}],
        matrix={"seed": [2, 1], "faults.0.params.x": ["b", "a"]},
    ).expand()
    assert [s.id for s in expanded] == [
        "s1-x-b-seed-2",
        "s1-x-b-seed-1",
        "s1-x-a-seed-2",
        "s1-x-a-seed-1",
    ]


def test_matrix_substitutes_the_value_into_the_scenario() -> None:
    """An id that says `unit_swap` while the fault still says `drop_key` is a lie."""
    base = scenario(
        faults=[{"type": "ToolCorruptionFault", "params": {"mutation_type": "drop_key"}}],
        matrix={"faults.0.params.mutation_type": ["unit_swap"]},
    )
    expanded = base.expand()
    assert expanded[0].faults[0]["params"]["mutation_type"] == "unit_swap"


def test_matrix_substitutes_a_top_level_field() -> None:
    """`seed` is the common case."""
    assert scenario(seed=1, matrix={"seed": [99]}).expand()[0].seed == 99


def test_a_scenario_without_a_matrix_expands_to_itself() -> None:
    """Expansion is always safe to call."""
    expanded = scenario().expand()
    assert len(expanded) == 1
    assert expanded[0].id == "s1"


def test_expansion_does_not_mutate_the_base_scenario() -> None:
    """The base is reused across products, so sharing state would corrupt them."""
    base = scenario(
        faults=[{"type": "ToolCorruptionFault", "params": {"mutation_type": "drop_key"}}],
        matrix={"faults.0.params.mutation_type": ["unit_swap", "type_flip"]},
    )
    base.expand()
    assert base.faults[0]["params"]["mutation_type"] == "drop_key"


# --------------------------------------------------------------------- presets


def test_all_eleven_presets_ship() -> None:
    """`docs/03` §D names them, and `--preset full` must be able to reference each."""
    assert set(PRESETS) == {
        "smoke",
        "tool_contract",
        "transient_faults",
        "long_horizon",
        "structured_output",
        "loop_safety",
        "adversarial",
        "state_integrity",
        "resume_safety",
        "hallucination",
        "full",
    }


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_is_a_literal_fault_spec_list(name: str) -> None:
    """D-14: prose presets cannot feed `plan_hash`."""
    specs = PRESETS[name]
    assert specs, f"{name} is empty"
    for spec in specs:
        assert isinstance(spec, FaultSpec)
        assert spec.type


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_declares_explicit_triggers(name: str) -> None:
    """ "explicit `FaultSpec` lists with explicit triggers" (D-14).

    An implicit trigger would make the plan hash depend on a default that could
    change, which is exactly what the golden tests cannot tolerate.
    """
    for spec in PRESETS[name]:
        assert spec.trigger, f"{name}/{spec.type} has no explicit trigger"


def test_full_contains_every_other_preset() -> None:
    """ "every preset above" -- the release gate."""
    others = {s.type for name, specs in PRESETS.items() if name != "full" for s in specs}
    assert others <= {s.type for s in PRESETS["full"]}


def test_a_preset_resolves_to_registered_faults() -> None:
    """`tool_contract` is entirely phase-02 faults, so it resolves now."""
    resolved, skipped = resolve_preset("tool_contract")
    assert resolved
    assert skipped == []


def test_the_state_presets_resolve_completely_as_of_m5() -> None:
    """D-14 said `state_integrity` would resolve lazily until phase 05 landed.

    Phase 05 landed, so it must now resolve with nothing skipped. The degradation
    mechanism itself is covered by its own test below.
    """
    for preset in ("state_integrity", "resume_safety"):
        _resolved, skipped = resolve_preset(preset)
        assert skipped == [], f"{preset} still has unregistered faults: {skipped}"


def test_an_unknown_preset_is_a_config_error() -> None:
    """A typo in a scenario file fails at load."""
    with pytest.raises(ConfigError, match="smoek"):
        resolve_preset("smoek")


# ------------------------------------------------------------ must_not (D-33)


def test_must_not_accepts_a_real_probe_code() -> None:
    """The happy path."""
    assert scenario(must_not=["secret_in_output"]).must_not == ["secret_in_output"]


def test_a_typo_in_must_not_is_rejected_with_the_nearest_match() -> None:
    """D-33: a typo loads clean, never matches, and the scenario passes while the
    refinement loop hashes the list as if it were protecting something."""
    with pytest.raises(ConfigError) as exc:
        scenario(must_not=["secret_in_ouput"])
    message = str(exc.value)
    assert "secret_in_ouput" in message
    assert "secret_in_output" in message, "the nearest match must be offered"


# ---------------------------------------------------------------- suite loading


def write_suite(tmp_path: Path, body: dict[str, Any], *, name: str = "suite.json") -> Path:
    """Write a suite file.

    Args:
        tmp_path: pytest's temp directory.
        body: The suite document.
        name: The filename, which selects the loader.

    Returns:
        The path.
    """
    path = tmp_path / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


MINIMAL = {
    "scenarios": [
        {"id": "s1", "entrypoint": "tests.fakes.naive_tool_agent:run", "faults": []},
    ]
}


def test_a_json_suite_loads_with_no_extras(tmp_path: Path) -> None:
    """ "JSON suites work with no extras" -- `pyyaml` is optional at runtime."""
    suite = load_suite(write_suite(tmp_path, MINIMAL))
    assert [s.id for s in suite.scenarios] == ["s1"]


def test_a_yaml_suite_loads_when_pyyaml_is_available(tmp_path: Path) -> None:
    """`pyyaml` is in `[dev]`, so every gate command can run a `.yaml` suite (D-27)."""
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(MINIMAL), encoding="utf-8")
    assert [s.id for s in load_suite(path).scenarios] == ["s1"]


def test_a_loaded_suite_is_validated_against_the_schema(tmp_path: Path) -> None:
    """ "validate the loaded document before use and raise with the JSON pointer"."""
    bad = {"scenarios": [{"entrypoint": "m:a"}]}  # missing the required `id`
    with pytest.raises(ConfigError) as exc:
        load_suite(write_suite(tmp_path, bad))
    assert "/scenarios/0" in str(exc.value), "the pointer must locate the problem"


def test_defaults_concatenate_fault_lists_and_replace_everything_else(
    tmp_path: Path,
) -> None:
    """The documented merge rule, and the only one that is not "last wins"."""
    body = {
        "defaults": {
            "seed": 99,
            "faults": [{"type": "NoopFault", "params": {}}],
            "expected_behavior": "ignore_and_continue",
        },
        "scenarios": [
            {
                "id": "s1",
                "entrypoint": "m:a",
                "seed": 7,
                "faults": [{"type": "ToolCorruptionFault", "params": {}}],
            }
        ],
    }
    loaded = load_suite(write_suite(tmp_path, body)).scenarios[0]
    assert loaded.seed == 7, "a scalar is replaced"
    assert [f["type"] for f in loaded.faults] == ["NoopFault", "ToolCorruptionFault"]
    assert loaded.expected_behavior == "ignore_and_continue", "defaults still apply"


def test_a_preset_reference_expands_into_faults(tmp_path: Path) -> None:
    """ "A preset must be expressible in YAML with a single line"."""
    body = {"scenarios": [{"id": "s1", "entrypoint": "m:a", "preset": "smoke"}]}
    loaded = load_suite(write_suite(tmp_path, body)).scenarios[0]
    assert len(loaded.faults) == len(PRESETS["smoke"])


def test_a_matrix_in_a_file_expands_on_load(tmp_path: Path) -> None:
    """A suite file's scenario count is what the runner will actually execute."""
    body = {
        "scenarios": [
            {"id": "s1", "entrypoint": "m:a", "faults": [], "matrix": {"seed": [1, 2, 3]}}
        ]
    }
    assert len(load_suite(write_suite(tmp_path, body)).scenarios) == 3


def test_a_missing_file_is_a_config_error(tmp_path: Path) -> None:
    """A wrong path is the commonest mistake and deserves a clear message."""
    with pytest.raises(ConfigError, match=r"nope\.yaml"):
        load_suite(tmp_path / "nope.yaml")


def test_a_matrix_key_that_addresses_nothing_is_a_config_error() -> None:
    """A bare `IndexError` mid-expansion tells an author nothing.

    The commonest cause is a matrix key naming a fault index the scenario never
    declared, which is easy to get wrong and impossible to diagnose from a traceback.
    """
    with pytest.raises(ConfigError, match=r"faults\.0\.params\.mutation_type"):
        scenario(faults=[], matrix={"faults.0.params.mutation_type": ["drop_key"]}).expand()


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_fault_is_actually_constructible(name: str) -> None:
    """A preset that cannot be built is worse than one that does not exist.

    `state_integrity` shipped with an `EdgeMisrouteFault` carrying no `force_to`,
    so the preset raised the moment M5 registered the fault. Nothing checked that
    presets could be constructed, only that they were listed.
    """
    from agent_loop_chaos.faults.base import fault_from_dict

    resolved, _skipped = resolve_preset(name)
    for spec in resolved:
        fault_from_dict(spec.to_dict())


def test_the_skip_mechanism_still_works_for_an_unregistered_fault() -> None:
    """Every shipped preset now resolves fully, so the mechanism needs its own test.

    A preset naming a fault a later phase will add must degrade, not raise (D-14).
    """
    registry = PRESETS
    registry["_probe_only"] = [FaultSpec("NotYetBuiltFault", {}, {}, {"on_call": 1})]
    try:
        resolved, skipped = resolve_preset("_probe_only")
    finally:
        del registry["_probe_only"]
    assert resolved == []
    assert skipped == [{"type": "NotYetBuiltFault", "reason": "fault_not_registered"}]
