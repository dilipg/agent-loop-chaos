"""`EnsembleJudge` — rules decide, the model explains (`docs/05` §7).

This is the eighth fake-SLM case from `docs/07` §5: a model whose `failure_mode`
contradicts the probes. The probes win, and the contradiction is recorded rather
than discarded -- a high disagreement rate is the single most useful signal for
improving the probes or the prompt.
"""

from __future__ import annotations

import json

import pytest

from agent_loop_chaos.judges.base import JudgeEvidence, JudgeMeta, Verdict
from agent_loop_chaos.judges.ensemble import EnsembleJudge
from agent_loop_chaos.judges.rules import RuleJudge
from agent_loop_chaos.judges.slm import SLMJudge
from tests.fakes.fake_slm import VALID_OUTPUT, FakeSLM, Reply


def _evidence(**over: object) -> JudgeEvidence:
    base: dict[str, object] = {
        "scenario_id": "s1",
        "expected_behavior": "graceful_degradation",
        "observed_behavior": "crashed",
        "passed": False,
        "seed": 42,
        "symptoms": [{"code": "unhandled_exception", "severity": "high", "detail": "KeyError"}],
        "injected": [{"fault_id": "f1", "type": "ToolSchemaDriftFault", "note": "dropped temp_c"}],
        "metrics": {"steps": 3, "tool_calls": 2, "llm_calls": 3, "retries": 0},
    }
    base.update(over)
    return JudgeEvidence(**base)  # type: ignore[arg-type]


class _StubModel:
    """A model judge that returns exactly what a test wants, with no network."""

    name = "stub"

    def __init__(self, **over: object) -> None:
        self.over = over
        self.last_call: dict[str, object] = {"request": {}, "response": {}}

    def judge(self, ev: JudgeEvidence) -> Verdict:
        defaults: dict[str, object] = {
            "passed": True,  # a model asserting the opposite of the probes
            "expected_behavior": ev.expected_behavior,
            "observed_behavior": "graceful_degradation",
            "failure_mode": "graceful_degradation",
            "severity": "low",
            "confidence": 0.44,
            "narrative": "A model-written narrative.",
            "root_cause_hypothesis": "A model-written hypothesis.",
            "refinement_hint": "A model-written hint.",
            "suggested_fixes": [{"kind": "prompt_change", "description": "model fix"}],
            "judge_meta": JudgeMeta(kind="slm", model="stub", transport="ollama"),
        }
        defaults.update(self.over)
        return Verdict(**defaults)  # type: ignore[arg-type]


class TestAuthority:
    """Rules own the classification; the model owns the prose."""

    @staticmethod
    def _judged() -> Verdict:
        return EnsembleJudge(model_judge=_StubModel()).judge(_evidence())

    def test_rules_own_passed(self) -> None:
        assert self._judged().passed is False

    def test_rules_own_observed_behavior(self) -> None:
        assert self._judged().observed_behavior == "crashed"

    def test_rules_own_failure_mode(self) -> None:
        assert self._judged().failure_mode == RuleJudge().judge(_evidence()).failure_mode

    def test_rules_own_severity(self) -> None:
        assert self._judged().severity == RuleJudge().judge(_evidence()).severity

    def test_model_owns_the_narrative(self) -> None:
        assert self._judged().narrative == "A model-written narrative."

    def test_model_owns_hypothesis_hint_and_fixes(self) -> None:
        verdict = self._judged()
        assert verdict.root_cause_hypothesis == "A model-written hypothesis."
        assert verdict.refinement_hint == "A model-written hint."
        assert verdict.suggested_fixes[0]["description"] == "model fix"

    def test_model_owns_confidence(self) -> None:
        assert self._judged().confidence == 0.44

    def test_kind_is_ensemble(self) -> None:
        assert self._judged().judge_meta.kind == "ensemble"


class TestDisagreement:
    """Case 8 of `docs/07` §5."""

    def test_contradicting_failure_mode_is_recorded_not_applied(self) -> None:
        verdict = EnsembleJudge(model_judge=_StubModel()).judge(_evidence())
        assert verdict.judge_disagreement is not None
        assert "model said graceful_degradation" in verdict.judge_disagreement
        assert "probes said crash_unhandled_exception" in verdict.judge_disagreement
        assert verdict.failure_mode == "crash_unhandled_exception"

    def test_agreement_leaves_the_field_null(self) -> None:
        agreeing = _StubModel(failure_mode="crash_unhandled_exception")
        assert EnsembleJudge(model_judge=agreeing).judge(_evidence()).judge_disagreement is None

    def test_disagreement_survives_the_real_wire(self, loopback: None) -> None:
        contradicting = {**VALID_OUTPUT, "failure_mode": "graceful_degradation"}
        with FakeSLM([Reply(json.dumps(contradicting))]) as server:
            model = SLMJudge(model="fake", base_url=server.base_url, timeout_s=5.0)
            verdict = EnsembleJudge(model_judge=model).judge(_evidence())
        assert verdict.failure_mode == "crash_unhandled_exception"
        assert verdict.judge_disagreement is not None

    def test_disagreement_is_not_reported_when_the_model_fell_back(self) -> None:
        # A rules verdict cannot disagree with itself; recording one would inflate
        # the rate that tells us whether the probes or the prompt need work.
        fell_back = _StubModel(
            judge_meta=JudgeMeta(kind="slm", fell_back_to_rules=True),
            failure_mode="graceful_degradation",
        )
        assert EnsembleJudge(model_judge=fell_back).judge(_evidence()).judge_disagreement is None


class TestResilience:
    def test_a_raising_model_judge_does_not_break_the_run(self) -> None:
        class Exploding:
            name = "boom"

            def judge(self, ev: JudgeEvidence) -> Verdict:
                raise RuntimeError("model exploded")

        verdict = EnsembleJudge(model_judge=Exploding()).judge(_evidence())
        assert verdict.passed is False
        assert verdict.narrative  # the rules narrative
        assert verdict.judge_meta.fell_back_to_rules is True

    def test_a_model_returning_an_empty_narrative_keeps_the_rules_one(self) -> None:
        verdict = EnsembleJudge(model_judge=_StubModel(narrative="")).judge(_evidence())
        assert verdict.narrative == RuleJudge().judge(_evidence()).narrative

    def test_no_model_judge_configured_is_just_rules(self) -> None:
        verdict = EnsembleJudge(model_judge=None).judge(_evidence())
        assert verdict.narrative == RuleJudge().judge(_evidence()).narrative
        assert verdict.judge_meta.fell_back_to_rules is True


class TestSelection:
    """`judge=None` picks by reachability, and records that it did."""

    def test_unreachable_endpoint_selects_rules(self) -> None:
        from agent_loop_chaos.judges import select_judge

        judge = select_judge(None, base_url="http://127.0.0.1:1", transport="ollama")
        assert isinstance(judge, RuleJudge)

    def test_reachable_endpoint_selects_ensemble(self, loopback: None) -> None:
        from agent_loop_chaos.judges import select_judge

        with FakeSLM() as server:
            judge = select_judge(None, base_url=server.base_url, transport="ollama", model="fake")
        assert isinstance(judge, EnsembleJudge)

    def test_auto_selection_is_recorded_in_judge_meta(self) -> None:
        from agent_loop_chaos.judges import select_judge

        judge = select_judge(None, base_url="http://127.0.0.1:1", transport="ollama")
        assert judge.judge(_evidence()).judge_meta.auto_selected is True

    def test_named_judge_is_not_marked_auto(self) -> None:
        from agent_loop_chaos.judges import select_judge

        judge = select_judge("rules")
        assert judge.judge(_evidence()).judge_meta.auto_selected is False

    @pytest.mark.parametrize("name", ["rules", "slm", "ensemble"])
    def test_every_name_resolves(self, name: str) -> None:
        from agent_loop_chaos.judges import select_judge

        assert select_judge(name, base_url="http://localhost:11434") is not None

    def test_an_instance_passes_through(self) -> None:
        from agent_loop_chaos.judges import select_judge

        rules = RuleJudge()
        assert select_judge(rules) is rules

    def test_an_unknown_name_is_a_config_error(self) -> None:
        from agent_loop_chaos.errors import ConfigError
        from agent_loop_chaos.judges import select_judge

        with pytest.raises(ConfigError):
            select_judge("psychic")
