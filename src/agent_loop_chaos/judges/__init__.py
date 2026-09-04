"""Judges.

Rules decide; models explain. A judge is **advisory only**: it may author
`failure_mode`, `root_cause_hypothesis`, `refinement_hint`, narration and ranked
fixes, and it never touches `success`. When a judge contradicts the probes, the
probes win and the disagreement is recorded in `verdict.judge_disagreement`
(`docs/11-OUTCOMES-AND-ASSERTIONS.md` §1).
"""

from __future__ import annotations

import logging
from typing import Any

from ..errors import ConfigError
from .base import (
    HARD_CAP_BYTES,
    TARGET_BYTES,
    Judge,
    JudgeEvidence,
    JudgeMeta,
    Verdict,
    build_evidence,
    escape_fences,
    render,
)
from .ensemble import EnsembleJudge
from .rules import FIX_TABLE, HINT_TABLE, RuleJudge
from .slm import SLMJudge, judge_output_schema, normalize_base_url

__all__ = [
    "FIX_TABLE",
    "HARD_CAP_BYTES",
    "HINT_TABLE",
    "TARGET_BYTES",
    "EnsembleJudge",
    "Judge",
    "JudgeEvidence",
    "JudgeMeta",
    "RuleJudge",
    "SLMJudge",
    "Verdict",
    "build_evidence",
    "escape_fences",
    "judge_output_schema",
    "normalize_base_url",
    "render",
    "select_judge",
]

log = logging.getLogger("agent_loop_chaos")


def select_judge(judge: Judge | str | None, **model_kwargs: Any) -> Judge:
    """Resolve `ChaosEngine(judge=…)` to a judge instance.

    `None` means "use a model if one is there": probe the endpoint once, take the
    ensemble when it answers and rules when it does not. The choice is recorded in
    `judge_meta.auto_selected`, because "the judge was rules" and "the judge was
    rules because nothing was listening" are different facts about a report.

    Args:
        judge: `"rules"`, `"slm"`, `"ensemble"`, `None`, or an instance.
        model_kwargs: Passed to `SLMJudge` when one is constructed.

    Returns:
        A judge.

    Raises:
        ConfigError: When `judge` is a string that names nothing.
    """
    if judge is None:
        model: SLMJudge | None = None
        try:
            model = SLMJudge(**model_kwargs)
            reachable = model.reachable()
        except ConfigError as exc:
            # A remote endpoint without consent is not an error when nobody asked
            # for a model judge -- it just means the answer is rules.
            log.info("no model judge available: %s", exc)
            reachable = False
        chosen: Judge = EnsembleJudge(model_judge=model) if reachable else RuleJudge()
        return _mark_auto(chosen)

    if not isinstance(judge, str):
        return judge
    if judge == "rules":
        return RuleJudge()
    if judge == "slm":
        return SLMJudge(**model_kwargs)
    if judge == "ensemble":
        return EnsembleJudge(model_judge=SLMJudge(**model_kwargs))
    raise ConfigError(
        f"unknown judge {judge!r}; expected one of rules, slm, ensemble, or an instance"
    )


def _mark_auto(judge: Judge) -> Judge:
    """Tag a judge so its verdicts record that the library chose it.

    Args:
        judge: The judge selected by reachability.

    Returns:
        The same judge, wrapped so `judge_meta.auto_selected` is set.
    """
    from dataclasses import replace

    inner = judge.judge

    def judged(ev: JudgeEvidence) -> Verdict:
        verdict = inner(ev)
        verdict.judge_meta = replace(verdict.judge_meta, auto_selected=True)
        return verdict

    judge.judge = judged  # type: ignore[method-assign]
    return judge
