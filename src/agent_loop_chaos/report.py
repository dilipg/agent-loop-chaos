"""`ChaosResult` — the report object.

The schema is the product; this class exists to fill
`schemas/chaos_report.schema.json`, which sets ``additionalProperties: false``. Every
required property is emitted on every run, so a consumer never has to guess whether
a field is missing or absent-by-meaning.

M1 populates identity, target, metrics, loop, artifacts, reproduce, randomness and
`injected_faults`, and pins `success=True` with `failure_mode="none"`. Computing
those two for real is M4's job: `success` comes from the probes, the assertions layer
and the scenario's `expected_behavior`, never from a language model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import ExpectedBehavior, FailureMode, Severity
from .seeding import sanitize_floats
from .version import __version__

__all__ = ["SCHEMA_VERSION", "ChaosResult", "empty_verdict"]

SCHEMA_VERSION = "1.1"


def empty_verdict(expected_behavior: ExpectedBehavior = "graceful_degradation") -> dict[str, Any]:
    """Build the placeholder verdict M1 emits.

    `passed` is carried, not decided, by a judge — it comes from the probes and the
    scenario's `expected_behavior` (`docs/11-OUTCOMES-AND-ASSERTIONS.md` §1). Until
    M4 computes it, this is a schema-valid stub that claims nothing.

    Args:
        expected_behavior: What good behaviour would have looked like.

    Returns:
        A dict satisfying `judge_verdict.schema.json`'s required fields.
    """
    return {
        "passed": True,
        "expected_behavior": expected_behavior,
        "observed_behavior": "completed_unaffected",
        "failure_mode": "none",
        "severity": "info",
        "confidence": 0.0,
        "narrative": "",
    }


@dataclass(slots=True)
class ChaosResult:
    """One run's full finding, schema-valid on serialization.

    Field names match the schema exactly. Anything the schema does not describe must
    not appear here, and anything here must appear in the schema — M4 adds a test
    asserting that parity in both directions.
    """

    run_id: str
    seed: int
    plan_hash: str
    started_at: str
    finished_at: str
    duration_ms: int
    target: dict[str, Any]
    metrics: dict[str, Any]
    loop: dict[str, Any]
    randomness: dict[str, Any]
    reproduce: dict[str, Any]
    artifacts: dict[str, Any]
    verdict: dict[str, Any]

    schema_version: str = SCHEMA_VERSION
    library_version: str = __version__
    scenario_id: str | None = None
    attempt: int = 1
    dry_run: bool = False

    success: bool = True
    failure_mode: FailureMode = "none"
    severity: Severity = "info"
    symptoms: list[dict[str, Any]] = field(default_factory=list)
    assertions: list[dict[str, Any]] = field(default_factory=list)

    injected_faults: list[dict[str, Any]] = field(default_factory=list)
    chaos_narrative: str = ""

    agent_state_pre_fault: Any = None
    agent_state_post_fault: Any = None
    llm_exchanges: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_output: Any = None
    error: dict[str, Any] | None = None

    baseline: dict[str, Any] | None = None
    delta_vs_baseline: dict[str, Any] | None = None

    root_cause_hypothesis: str | None = None
    refinement_hint: str | None = None
    suggested_fixes: list[dict[str, Any]] = field(default_factory=list)
    code_pointers: list[dict[str, Any]] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the `chaos_report.schema.json` shape.

        Non-finite floats are encoded and finite floats rounded by `sanitize_floats`,
        so the result is always valid JSON for a non-Python consumer (D-09).

        Returns:
            A plain dict with every required property present. `tags` is omitted
            when empty, since it is the one optional property.
        """
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "library_version": self.library_version,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "plan_hash": self.plan_hash,
            "attempt": self.attempt,
            "dry_run": self.dry_run,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "target": self.target,
            "success": self.success,
            "verdict": self.verdict,
            "failure_mode": self.failure_mode,
            "severity": self.severity,
            "symptoms": self.symptoms,
            "assertions": self.assertions,
            "injected_faults": self.injected_faults,
            "chaos_narrative": self.chaos_narrative,
            "randomness": self.randomness,
            "agent_state_pre_fault": self.agent_state_pre_fault,
            "agent_state_post_fault": self.agent_state_post_fault,
            "llm_exchanges": self.llm_exchanges,
            "tool_calls": self.tool_calls,
            "final_output": self.final_output,
            "error": self.error,
            "baseline": self.baseline,
            "delta_vs_baseline": self.delta_vs_baseline,
            "metrics": self.metrics,
            "loop": self.loop,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "refinement_hint": self.refinement_hint,
            "suggested_fixes": self.suggested_fixes,
            "code_pointers": self.code_pointers,
            "reproduce": self.reproduce,
            "artifacts": self.artifacts,
            "schema_errors": self.schema_errors,
        }
        if self.tags:
            out["tags"] = self.tags
        return dict(sanitize_floats(out))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON.

        Args:
            indent: Indentation for the on-disk form. `None` gives the compact form.

        Returns:
            The report as a JSON string, with sorted keys and `allow_nan=False`.
        """
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ChaosResult:
        """Rebuild a result from its serialized form.

        Args:
            d: A dict matching `chaos_report.schema.json`.

        Returns:
            The reconstructed `ChaosResult`, ignoring properties this version does
            not know — a consumer must tolerate a MINOR addition (`docs/04` §2).
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> list[str]:
        """Validate against the packaged report schema.

        Returns:
            Human-readable error strings, each carrying a JSON pointer and the
            failing value truncated to 200 chars. Empty when valid.
        """
        # Imported here, not at module scope, so `import agent_loop_chaos` pulls in
        # nothing third-party at all -- asserted by tests/test_no_hard_deps.py.
        from .schema import validate_obj

        return validate_obj(self.to_dict(), "report")

    def agent_task_markdown(self) -> str:
        """Render the `AGENT_TASK.md` work order.

        Returns:
            The markdown body.

        Raises:
            NotImplementedError: Until M4 (`prompts/04-report-and-probes.md`).
        """
        raise NotImplementedError(
            "ChaosResult.agent_task_markdown arrives in M4 (prompts/04-report-and-probes.md)"
        )

    def summary_line(self) -> str:
        """Render the one-line CLI summary.

        Returns:
            The scenario or run id, pass/fail, failure mode and severity, in the
            column shape `alc run` prints.
        """
        label = self.scenario_id or self.run_id
        status = "pass" if self.success else "FAIL"
        return f"{label:<32}{status:<6}{self.failure_mode:<32}{self.severity}"
