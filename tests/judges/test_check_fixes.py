"""A fix that names the check, not the layer.

`assertions_failed` is the dominant symptom for most findings, and its fix entry said
the same thing every time: "make the agent meet the scenario's declared expectation
with the fault still injected". True, and useless — it tells a reader to pass the test
without saying what to change.

Which check failed *is* the finding, and each one has a different remedy: an unsourced
number needs a validation branch at the call site, an invented tool needs the answer
constrained to work actually done, a duplicated effect needs an idempotency key. The
fix table is keyed by `Expect` field, and a test asserts it covers every one.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from agent_loop_chaos.assertions import Expect
from agent_loop_chaos.judges.rules import CHECK_FIX_TABLE, FIX_TABLE, RuleJudge


def _evidence(**kw: Any) -> Any:
    from agent_loop_chaos.judges.base import JudgeEvidence

    defaults: dict[str, Any] = {
        "scenario_id": "tool.drop_key",
        "expected_behavior": "graceful_degradation",
        "observed_behavior": "answered_confidently_wrong",
        "passed": False,
        "seed": 1337,
        "symptoms": [{"code": "assertions_failed", "severity": "high"}],
        "assertions": [],
    }
    defaults.update(kw)
    return JudgeEvidence(**defaults)


class TestTheTableIsComplete:
    def test_every_expect_field_has_a_fix(self) -> None:
        declared = {f.name for f in dataclasses.fields(Expect)} - {"auto"}
        missing = declared - set(CHECK_FIX_TABLE)
        assert not missing, f"CHECK_FIX_TABLE has no fix for {sorted(missing)}"

    def test_it_names_no_check_that_does_not_exist(self) -> None:
        declared = {f.name for f in dataclasses.fields(Expect)}
        assert set(CHECK_FIX_TABLE) <= declared

    def test_every_kind_is_in_the_report_schema(self) -> None:
        """`suggested_fixes[].kind` is a closed vocabulary; a new one fails validation."""
        import json
        from pathlib import Path as _Path

        schema = json.loads(
            (
                _Path(__file__).resolve().parents[2] / "schemas" / "chaos_report.schema.json"
            ).read_text()
        )
        allowed = set(
            schema["properties"]["suggested_fixes"]["items"]["properties"]["kind"]["enum"]
        )
        used = {entry["kind"] for entry in CHECK_FIX_TABLE.values()}
        assert used <= allowed, f"not in the schema enum: {sorted(used - allowed)}"

    @pytest.mark.parametrize("check", sorted(CHECK_FIX_TABLE))
    def test_each_entry_is_actionable(self, check: str) -> None:
        entry = CHECK_FIX_TABLE[check]
        assert entry["kind"], check
        assert len(entry["description"].split()) >= 8, f"{check} is too terse to act on"

    @pytest.mark.parametrize("check", sorted(CHECK_FIX_TABLE))
    def test_no_entry_is_the_old_generic_one(self, check: str) -> None:
        generic = FIX_TABLE["assertions_failed"]["description"]
        assert CHECK_FIX_TABLE[check]["description"] != generic


class TestThePickerPrefersTheCheck:
    def _fix(self, failed: list[str]) -> dict[str, Any]:
        evidence = _evidence(
            assertions=[{"check": name, "ok": False} for name in failed]
            + [{"check": "output_non_empty", "ok": True}]
        )
        fixes = RuleJudge._fixes(evidence, "assertions_failed")
        assert fixes, "no fix was produced"
        return fixes[0]

    def test_an_unsourced_number_gets_its_own_fix(self) -> None:
        fix = self._fix(["no_unsourced_numbers"])
        assert fix["description"] == CHECK_FIX_TABLE["no_unsourced_numbers"]["description"]

    def test_an_invented_tool_gets_its_own_fix(self) -> None:
        fix = self._fix(["no_invented_tools"])
        assert "did not" in fix["description"] or "actually" in fix["description"]

    def test_a_duplicated_effect_gets_the_idempotency_fix(self) -> None:
        assert self._fix(["idempotent_effects"])["kind"] == "idempotency"

    def test_an_injection_fix_is_about_untrusted_content(self) -> None:
        assert self._fix(["must_not_call_tools"])["kind"] == "untrusted_content_handling"

    def test_the_first_failed_check_wins(self) -> None:
        """Deterministic: assertion order is the scenario's, and the report's."""
        fix = self._fix(["no_unsourced_numbers", "output_non_empty"])
        assert fix["description"] == CHECK_FIX_TABLE["no_unsourced_numbers"]["description"]

    def test_an_unknown_check_falls_back_rather_than_vanishing(self) -> None:
        fix = self._fix(["some_future_check"])
        assert fix["description"] == FIX_TABLE["assertions_failed"]["description"]

    def test_a_structural_probe_still_gets_its_own_fix(self) -> None:
        """Only `assertions_failed` defers to the check table."""
        fixes = RuleJudge._fixes(_evidence(), "unhandled_exception")
        assert fixes[0]["description"] == FIX_TABLE["unhandled_exception"]["description"]

    def test_a_clean_run_produces_no_fix(self) -> None:
        assert RuleJudge._fixes(_evidence(), None) == []
