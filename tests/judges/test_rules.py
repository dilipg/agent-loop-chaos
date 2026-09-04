"""`RuleJudge` — the deterministic path (`docs/05` §6).

Not a stub. CI runs this judge, and `--judge rules` is the only mode guaranteed to
work with no network at all, so its narration and hints have to be genuinely useful
on their own.
"""

from __future__ import annotations

import pytest

from agent_loop_chaos.judges.base import JudgeEvidence
from agent_loop_chaos.judges.rules import FIX_TABLE, HINT_TABLE, RuleJudge
from agent_loop_chaos.probes import PROBES


def _evidence(**over: object) -> JudgeEvidence:
    base: dict[str, object] = {
        "scenario_id": "s1",
        "expected_behavior": "graceful_degradation",
        "observed_behavior": "crashed",
        "passed": False,
        "seed": 42,
        "injected": [
            {
                "fault_id": "f1",
                "type": "ToolSchemaDriftFault",
                "target": {"tool": "get_weather"},
                "fire_count": 1,
                "note": "dropped key 'temp_c' from get_weather's result",
            }
        ],
        "symptoms": [
            {"code": "unhandled_exception", "severity": "high", "detail": "KeyError: 'temp_c'"}
        ],
        "metrics": {"steps": 3, "tool_calls": 2, "llm_calls": 3, "retries": 0},
    }
    base.update(over)
    return JudgeEvidence(**base)  # type: ignore[arg-type]


class TestCoverage:
    """Adding a probe later must not silently produce an empty hint."""

    @pytest.mark.parametrize("code", sorted(p.code for p in PROBES))
    def test_every_probe_code_has_a_hint(self, code: str) -> None:
        assert HINT_TABLE.get(code), f"probe {code!r} has no refinement hint"

    @pytest.mark.parametrize("code", sorted(p.code for p in PROBES))
    def test_every_probe_code_has_a_fix(self, code: str) -> None:
        fix = FIX_TABLE.get(code)
        assert fix, f"probe {code!r} has no suggested fix"
        assert fix["kind"], f"probe {code!r} has no fix kind"

    def test_tables_have_no_entries_for_codes_that_do_not_exist(self) -> None:
        codes = {p.code for p in PROBES}
        assert set(HINT_TABLE) == codes
        assert set(FIX_TABLE) == codes

    def test_hints_are_one_imperative_sentence(self) -> None:
        for code, hint in HINT_TABLE.items():
            assert hint == hint.strip(), code
            assert hint.endswith("."), code
            assert len(hint) <= 400, code
            assert hint.count(".") <= 2, f"{code}: hint should be one sentence"


class TestVerdict:
    def test_kind_and_confidence_are_fixed(self) -> None:
        verdict = RuleJudge().judge(_evidence())
        assert verdict.judge_meta.kind == "rules"
        # A lookup is not an estimate.
        assert verdict.confidence == 1.0
        assert verdict.judge_meta.transport == "none"

    def test_carries_passed_without_deciding_it(self) -> None:
        assert RuleJudge().judge(_evidence(passed=True)).passed is True
        assert RuleJudge().judge(_evidence(passed=False)).passed is False

    def test_narrative_names_the_seed_fault_and_symptom(self) -> None:
        narrative = RuleJudge().judge(_evidence()).narrative
        assert "42" in narrative
        assert "get_weather" in narrative or "temp_c" in narrative
        assert narrative.endswith(".")
        assert len(narrative) <= 1200

    def test_hint_comes_from_the_dominant_symptom(self) -> None:
        verdict = RuleJudge().judge(_evidence())
        assert verdict.refinement_hint == HINT_TABLE["unhandled_exception"]

    def test_fix_is_ranked_and_grounded(self) -> None:
        fixes = RuleJudge().judge(_evidence()).suggested_fixes
        assert fixes
        assert fixes[0]["kind"] == FIX_TABLE["unhandled_exception"]["kind"]
        assert fixes[0]["confidence"] == 1.0

    def test_assertions_failed_hint_names_the_failed_checks(self) -> None:
        verdict = RuleJudge().judge(
            _evidence(
                symptoms=[{"code": "assertions_failed", "severity": "high", "detail": "2 failed"}],
                assertions=[
                    {"check": "output_mentions_any", "ok": False, "detail": "no mention"},
                    {"check": "output_non_empty", "ok": True, "detail": ""},
                ],
            )
        )
        assert verdict.refinement_hint is not None
        assert "output_mentions_any" in verdict.refinement_hint
        assert "output_non_empty" not in verdict.refinement_hint

    def test_clean_run_says_so_rather_than_inventing_a_problem(self) -> None:
        verdict = RuleJudge().judge(
            _evidence(passed=True, observed_behavior="graceful_degradation", symptoms=[])
        )
        assert verdict.failure_mode == "none"
        assert verdict.severity == "info"
        assert verdict.suggested_fixes == []
        assert "no symptom" in verdict.narrative.lower() or "handled" in verdict.narrative.lower()

    def test_no_fault_fired_is_narrated_honestly(self) -> None:
        narrative = RuleJudge().judge(_evidence(injected=[])).narrative
        assert "no fault" in narrative.lower()

    def test_never_raises_on_sparse_evidence(self) -> None:
        sparse = JudgeEvidence(
            scenario_id=None,
            expected_behavior="graceful_degradation",
            observed_behavior="indeterminate",
            passed=False,
            seed=0,
        )
        assert RuleJudge().judge(sparse).narrative

    def test_is_deterministic(self) -> None:
        a = RuleJudge().judge(_evidence()).to_dict()
        b = RuleJudge().judge(_evidence()).to_dict()
        assert a == b

    def test_makes_no_network_call(self, no_network: None) -> None:
        # The autouse socket fixture is already active; this asserts the guarantee
        # explicitly so `--judge rules` offline stays a tested property.
        assert RuleJudge().judge(_evidence()).narrative


class TestHintPrefersTheMechanism:
    """`assertions_failed` says *that* the agent failed, never *why*.

    It outranks most structural probes in `PROBE_PRECEDENCE`, which is right for
    classification -- the report's `failure_mode` is what the run failed at. It is
    wrong for the hint: "satisfy the declared expectation" is not something a coding
    agent can act on without asking a question, and `AGENT_TASK.md` promises it can
    (`docs/05` §6).
    """

    @staticmethod
    def _both() -> JudgeEvidence:
        return _evidence(
            symptoms=[
                {"code": "assertions_failed", "severity": "high", "detail": "1 failed"},
                {"code": "unhandled_exception", "severity": "high", "detail": "KeyError"},
            ],
            assertions=[{"check": "output_non_empty", "ok": False, "detail": "empty"}],
        )

    def test_hint_names_the_mechanism_not_the_assertion(self) -> None:
        assert RuleJudge().judge(self._both()).refinement_hint == HINT_TABLE["unhandled_exception"]

    def test_fix_follows_the_mechanism_too(self) -> None:
        fixes = RuleJudge().judge(self._both()).suggested_fixes
        assert fixes[0]["kind"] == FIX_TABLE["unhandled_exception"]["kind"]

    def test_failure_mode_still_follows_the_precedence_table(self) -> None:
        # The classification is unchanged; only the hint's source moves.
        assert RuleJudge().judge(self._both()).failure_mode == "crash_unhandled_exception"

    def test_assertions_failed_alone_still_names_the_checks(self) -> None:
        only = _evidence(
            symptoms=[{"code": "assertions_failed", "severity": "high", "detail": "1 failed"}],
            assertions=[{"check": "no_unsourced_numbers", "ok": False, "detail": "24C"}],
        )
        hint = RuleJudge().judge(only).refinement_hint
        assert hint is not None and "no_unsourced_numbers" in hint

    def test_the_narrative_still_reports_the_dominant_symptom(self) -> None:
        # The report and the narrative must agree on what dominated.
        assert "assertions_failed" in RuleJudge().judge(self._both()).narrative
