"""Plain-English text for every code the report can emit.

The dashboard's report view is for someone who did not write the agent and does not
read Python. `failure_mode: unverified_claim_emitted` means nothing to them; "the
agent stated something it had no source for" does.

The point of testing it is completeness. A glossary with a hole shows a raw enum name
to exactly the reader who cannot decode one, so every value of every enum must have
an entry, and a test fails when someone adds a code without adding its sentence.
"""

from __future__ import annotations

import re
from typing import get_args

import pytest

from agent_loop_chaos import glossary
from agent_loop_chaos.enums import ExpectedBehavior, FailureMode, ObservedBehavior, Severity
from agent_loop_chaos.probes import PROBE_PRECEDENCE


class TestCompleteness:
    @pytest.mark.parametrize(
        ("alias", "table"),
        [
            (FailureMode, "FAILURE_MODES"),
            (ObservedBehavior, "OBSERVED"),
            (ExpectedBehavior, "EXPECTED"),
            (Severity, "SEVERITIES"),
        ],
    )
    def test_every_enum_value_is_explained(self, alias: object, table: str) -> None:
        entries = getattr(glossary, table)
        missing = [value for value in get_args(alias) if value not in entries]
        assert not missing, f"{table} has no plain English for {missing}"

    def test_no_table_explains_a_code_that_does_not_exist(self) -> None:
        assert set(glossary.FAILURE_MODES) == set(get_args(FailureMode))
        assert set(glossary.OBSERVED) == set(get_args(ObservedBehavior))

    def test_every_probe_is_explained(self) -> None:
        missing = [code for code in PROBE_PRECEDENCE if code not in glossary.PROBES]
        assert not missing, f"PROBES has no plain English for {missing}"

    def test_every_fault_is_explained(self) -> None:
        """Faults come from the registry, so the catalog cannot drift from the page."""
        entries = glossary.faults()
        from agent_loop_chaos.faults.base import FAULT_REGISTRY

        assert set(entries) == set(FAULT_REGISTRY)
        assert all(entries[kind]["summary"] for kind in entries)


class TestItReadsLikeEnglish:
    @pytest.mark.parametrize("table", ["FAILURE_MODES", "OBSERVED", "EXPECTED", "SEVERITIES"])
    def test_no_entry_leaks_a_snake_case_code(self, table: str) -> None:
        for code, text in getattr(glossary, table).items():
            assert not re.search(r"\b[a-z]+_[a-z_]+\b", text), f"{code} still reads as code: {text}"

    @pytest.mark.parametrize("table", ["FAILURE_MODES", "OBSERVED", "PROBES"])
    def test_every_entry_is_a_sentence(self, table: str) -> None:
        for code, text in getattr(glossary, table).items():
            assert text[0].isupper() or text.startswith("The"), f"{code}: {text!r}"
            assert len(text.split()) >= 4, f"{code} is too terse to help: {text!r}"


class TestTheBundle:
    def test_it_serializes_whole(self) -> None:
        bundle = glossary.bundle()
        assert set(bundle) >= {"failure_modes", "observed", "severities", "probes", "faults"}
        assert bundle["failure_modes"]["secret_leak"]

    def test_it_carries_the_intensity_dial(self) -> None:
        """The report view says how hard the run pushed, in words."""
        levels = glossary.bundle()["intensity"]
        assert len(levels) == 10
        assert levels[0]["label"] == "strict"
        assert levels[-1]["label"] == "creative"


class TestAssertionChecks:
    def test_every_expect_field_is_explained(self) -> None:
        """`assertions_failed` alone tells a reader nothing: *which* rule is the finding."""
        import dataclasses

        from agent_loop_chaos.assertions import Expect

        declared = {f.name for f in dataclasses.fields(Expect)}
        missing = declared - set(glossary.ASSERTIONS)
        assert not missing, f"ASSERTIONS has no plain English for {sorted(missing)}"

    def test_it_explains_no_check_that_does_not_exist(self) -> None:
        import dataclasses

        from agent_loop_chaos.assertions import Expect

        declared = {f.name for f in dataclasses.fields(Expect)}
        assert set(glossary.ASSERTIONS) <= declared

    def test_the_bundle_carries_them(self) -> None:
        assert glossary.bundle()["assertions"]["no_invented_tools"]
