"""`EnsembleJudge` — rules decide, the model explains (`docs/05` §7).

The split is the whole point of the design:

    passed, observed_behavior, failure_mode, severity   <- rules (authority)
    narrative, root_cause_hypothesis, refinement_hint,
    suggested_fixes, confidence                         <- model (advisory)

When the model's classification contradicts the probes, the probes win and the
contradiction is recorded in `judge_disagreement` rather than discarded. A high rate
means the probes or the prompt need work, which makes it the most useful signal the
library produces about itself.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import JudgeEvidence, Verdict
from .rules import RuleJudge

__all__ = ["EnsembleJudge"]

log = logging.getLogger("agent_loop_chaos")


class EnsembleJudge:
    """The default judge when a model endpoint is reachable.

    Proves the weakness: nothing -- it observes. What it guarantees is that a model
    cannot change a pass into a fail or the reverse, however confidently it argues.
    """

    name = "ensemble"

    def __init__(self, *, rules: RuleJudge | None = None, model_judge: Any | None = None) -> None:
        """Initialise.

        Args:
            rules: The authoritative rule judge. Defaults to `RuleJudge()`.
            model_judge: The narrating judge. `None` degrades to rules-only, which is
                what a suite run does when the endpoint disappears mid-run.
        """
        self.rules = rules or RuleJudge()
        self.model_judge = model_judge

    @property
    def last_call(self) -> dict[str, Any]:
        """The model's redacted request and response, for `judge.json`.

        Returns:
            The model judge's record, or an empty dict when no model ran.
        """
        return dict(getattr(self.model_judge, "last_call", {}) or {})

    def judge(self, ev: JudgeEvidence) -> Verdict:
        """Combine an authoritative rules verdict with a model's narration.

        Args:
            ev: The evidence assembled by the engine.

        Returns:
            The rules verdict, with the model-owned fields overlaid when the model
            produced usable ones. A model that raises, returns nothing, or is absent
            leaves the rules verdict untouched.
        """
        verdict = self.rules.judge(ev)
        meta = verdict.judge_meta

        if self.model_judge is None:
            verdict.judge_meta = _replace(meta, kind="ensemble", fell_back_to_rules=True)
            return verdict

        try:
            model = self.model_judge.judge(ev)
        except Exception as exc:
            # An engine bug must never be reported as an agent failure, and neither
            # must a judge bug.
            log.warning("model judge raised, keeping the rules verdict: %s", exc)
            verdict.judge_meta = _replace(meta, kind="ensemble", fell_back_to_rules=True)
            return verdict

        fell_back = bool(model.judge_meta.fell_back_to_rules)
        if not fell_back and model.failure_mode != verdict.failure_mode:
            verdict.judge_disagreement = (
                f"model said {model.failure_mode}, probes said {verdict.failure_mode}"
            )

        if model.narrative:
            verdict.narrative = model.narrative
        if not fell_back:
            verdict.confidence = model.confidence
            if model.root_cause_hypothesis:
                verdict.root_cause_hypothesis = model.root_cause_hypothesis
            if model.refinement_hint:
                verdict.refinement_hint = model.refinement_hint
            if model.suggested_fixes:
                verdict.suggested_fixes = list(model.suggested_fixes)

        verdict.judge_meta = _replace(model.judge_meta, kind="ensemble")
        return verdict


def _replace(meta: Any, **changes: Any) -> Any:
    """Copy a frozen `JudgeMeta` with fields changed.

    Args:
        meta: The metadata block.
        changes: Fields to override.

    Returns:
        A new `JudgeMeta`.
    """
    from dataclasses import replace

    return replace(meta, **changes)
