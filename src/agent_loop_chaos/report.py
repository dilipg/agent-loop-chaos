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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .enums import ExpectedBehavior, FailureMode, Severity
from .seeding import sanitize_floats
from .version import __version__

__all__ = [
    "SCHEMA_VERSION",
    "ChaosResult",
    "assemble",
    "empty_verdict",
    "extract_code_context",
    "extract_code_pointers",
]

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
        # `default=str` because `final_output` and `llm_exchanges` hold whatever the
        # agent produced -- a LangChain message, a dataclass, a client object. A
        # report that refused to serialize one of those would lose the whole finding
        # over a field that is only ever read as evidence.
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
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


# --------------------------------------------------------------- code pointers

# `docs/DECISIONS.md` D-30: code pointers are advisory. A pointer derived from a
# faulted crossing rather than a raised frame is a hint about where to look, not a
# claim about where the bug is, and its confidence says so.
_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<symbol>\S+)')
_LIBRARY_MARKERS = ("agent_loop_chaos", "site-packages", "/lib/python", "<frozen", "<string>")
_CONTEXT_LINES = 4


def _is_user_frame(path: str) -> bool:
    """Report whether a traceback frame belongs to the user's own code.

    Args:
        path: The file path from the frame.

    Returns:
        False for the library's own source, the standard library, and installed
        packages -- a pointer into any of those sends a coding agent to fix nothing.
    """
    return not any(marker in path for marker in _LIBRARY_MARKERS)


def extract_code_pointers(
    error: Mapping[str, Any] | None, trace: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Work out where a reader should look first.

    Args:
        error: The captured exception, if the run ended by raising.
        trace: The loaded trace, used when there is no exception at all.

    Returns:
        Pointers, innermost frame first. When nothing was raised -- which is the case
        in the reference example, and in every silent-wrong-answer finding -- a
        pointer is derived from the function that produced the faulted crossing, so
        the work order always names somewhere to open.
    """
    pointers: list[dict[str, Any]] = []

    if error and error.get("traceback"):
        frames: list[dict[str, Any]] = []
        for line in str(error["traceback"]).splitlines():
            match = _FRAME_RE.match(line)
            if match and _is_user_frame(match.group("file")):
                frames.append(
                    {
                        "file": match.group("file"),
                        "line": int(match.group("line")),
                        "symbol": match.group("symbol"),
                        "why": (
                            f"{error.get('type', 'the exception')} was raised here while "
                            "consuming the faulted value"
                        ),
                        "confidence": 0.9,
                    }
                )
        pointers.extend(reversed(frames))

    if not pointers:
        for event in trace:
            if event.get("kind") != "fault_fired":
                continue
            name = event.get("name")
            if not name:
                continue
            pointers.append(
                {
                    "file": f"<caller of {name}>",
                    "line": None,
                    "symbol": str(name),
                    "why": (
                        f"the fault changed what {name!r} returned; the caller that consumes "
                        "it is where the missing validation belongs"
                    ),
                    # Advisory, not derived from a frame the interpreter reported.
                    "confidence": 0.4,
                }
            )
            break
    return pointers


def extract_code_context(pointers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Read the source around each pointer.

    Used by the judge in phase 06. Redacted before it leaves the machine, and a
    non-loopback endpoint needs explicit consent first (D-22).

    Args:
        pointers: The code pointers.

    Returns:
        One entry per readable pointer. A missing file is skipped rather than raised:
        a stored run replayed on another machine must not break the pipeline.
    """
    out: list[dict[str, Any]] = []
    for pointer in pointers:
        line = pointer.get("line")
        path = Path(str(pointer.get("file", "")))
        if line is None or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:  # pragma: no cover - unreadable but present
            continue
        start = max(0, int(line) - 1 - _CONTEXT_LINES)
        end = min(len(lines), int(line) + _CONTEXT_LINES)
        out.append(
            {
                "file": str(path),
                "line": int(line),
                "source": "\n".join(lines[start:end]),
                "start_line": start + 1,
            }
        )
    return out


def _tools_called(trace: Sequence[Mapping[str, Any]]) -> list[str]:
    """Tool names the run invoked, in call order.

    Args:
        trace: The loaded trace.

    Returns:
        The names, with duplicates preserved.
    """
    return [
        str(e.get("name"))
        for e in trace
        if e.get("kind") == "tool_call_requested" and e.get("name")
    ]


def assemble(
    *,
    trace: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    plan_hash: str,
    run_id: str,
    scenario_id: str | None,
    started_at: str,
    finished_at: str,
    wall_ms: int,
    final_output: Any,
    error: dict[str, Any] | None,
    symptoms: Sequence[Any],
    assertions: Sequence[Any],
    expected_behavior: ExpectedBehavior = "graceful_degradation",
    must_not: Sequence[str] = (),
    injected_faults: Sequence[Mapping[str, Any]] = (),
    limit_hit: str | None = None,
    baseline: dict[str, Any] | None = None,
    delta: dict[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    reproduce: Mapping[str, Any] | None = None,
    randomness: Mapping[str, Any] | None = None,
    target: Mapping[str, Any] | None = None,
    attempt: int = 1,
    dry_run: bool = False,
    tags: Mapping[str, str] | None = None,
    destructive_mutation: bool = False,
    internal_error: bool = False,
    expected_errors: Sequence[str] = (),
    agent_broke_cycle: bool = False,
    acknowledged_failure: bool = False,
    retry_succeeded: bool = False,
    goal_fault_fired: bool = False,
    fault_severity_hints: Sequence[str] = (),
    strict_schema: bool = False,
    metrics: Mapping[str, Any] | None = None,
) -> ChaosResult:
    """Build a complete `ChaosResult` from a trace and a plan.

    Pure over `(trace, plan)`, which is what lets `alc judge` and `alc report`
    rebuild a result from disk and re-judge it with a better model without re-running
    the agent (`docs/01` §5).

    Args:
        trace: The loaded trace.
        plan: The frozen plan.
        plan_hash: Its hash.
        run_id: The run's id.
        scenario_id: The scenario's id.
        started_at: ISO-8601 start.
        finished_at: ISO-8601 end.
        wall_ms: Wall time, a timing field only.
        final_output: The agent's answer.
        error: The captured exception, if any.
        symptoms: What the probes found.
        assertions: What the assertions layer decided.
        expected_behavior: The scenario's expectation.
        must_not: Probe codes that always fail the run.
        injected_faults: The fault records.
        limit_hit: Which limit stopped the run.
        baseline: A baseline reference.
        delta: The baseline delta.
        artifacts: Paths written for this run.
        reproduce: How to re-run it.
        randomness: The RNG log.
        target: What was instrumented.
        attempt: Which attempt this is.
        dry_run: Whether faults were armed but never applied.
        tags: Free-form labels.
        destructive_mutation: Whether a data-destroying mutation fired.
        internal_error: Whether the failure was the library's own.
        expected_errors: Exception names the scenario treats as explicit.
        agent_broke_cycle: Whether the agent stopped a repeat cycle itself.
        acknowledged_failure: Whether the output acknowledges the failure.
        retry_succeeded: Whether a faulted call was followed by a successful one.
        goal_fault_fired: Whether a goal or context fault fired.
        fault_severity_hints: `severity_hint` of every fault that fired.
        strict_schema: Raise `SchemaError` rather than recording `schema_errors[]`.
        metrics: Metrics already computed by the caller. Passed in rather than
            recomputed here, because the live counters know things the trace does
            not -- the vanilla adapter increments steps without emitting
            `step_started`. Omit it and the trace is the only source.

    Returns:
        The assembled result.

    Raises:
        SchemaError: When `strict_schema` and the report does not validate.
    """
    from .errors import SchemaError
    from .metrics import compute_metrics
    from .outcomes import (
        ClassificationInput,
        classify_behavior,
        classify_failure_mode,
        compute_severity,
        compute_success,
    )

    spec_gaps: list[dict[str, Any]] = []
    computed = dict(metrics) if metrics else compute_metrics(trace, wall_ms=wall_ms)

    classification = ClassificationInput(
        symptoms=list(symptoms),
        assertions=list(assertions),
        final_output=final_output,
        error=error,
        limit_hit=limit_hit,
        internal_error=internal_error,
        fault_had_data_effect=destructive_mutation or any(f.get("fired") for f in injected_faults),
        expected_errors=list(expected_errors),
        agent_broke_cycle=agent_broke_cycle,
        acknowledged_failure=acknowledged_failure,
        retry_succeeded=retry_succeeded,
    )
    observed = classify_behavior(classification, on_spec_gap=spec_gaps.append)
    success = compute_success(
        observed, expected_behavior, list(assertions), list(symptoms), must_not
    )
    failure_mode = classify_failure_mode(
        observed=observed,
        symptoms=list(symptoms),
        success=success,
        destructive_mutation=destructive_mutation,
        goal_fault_fired=goal_fault_fired,
        objective_assertion_failed=any(
            not a.ok and a.check in {"output_mentions_any", "no_claim_about"} for a in assertions
        ),
        on_spec_gap=spec_gaps.append,
    )
    severity = compute_severity(list(symptoms), list(fault_severity_hints), success=success)

    pointers = extract_code_pointers(error, trace)
    verdict = {
        **empty_verdict(expected_behavior),
        "passed": success,
        "observed_behavior": observed,
        "failure_mode": failure_mode,
        "severity": severity,
    }

    result = ChaosResult(
        run_id=run_id,
        scenario_id=scenario_id,
        seed=int(plan.get("seed", 0)),
        plan_hash=plan_hash,
        attempt=attempt,
        dry_run=dry_run,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=wall_ms,
        target=dict(target or {"framework": "vanilla", "adapter_version": "unknown"}),
        metrics=computed,
        loop={
            "steps": computed["steps"],
            "max_steps": int((plan.get("limits") or {}).get("max_steps", 25)),
            "limit_hit": limit_hit,
        },
        randomness=dict(randomness or {"seed": int(plan.get("seed", 0)), "streams": {}}),
        reproduce=dict(
            reproduce or {"cmd": f"alc replay {run_id}", "seed": int(plan.get("seed", 0))}
        ),
        artifacts=dict(artifacts or {"report": "report.json", "trace": "trace.jsonl"}),
        verdict=verdict,
        success=success,
        failure_mode=failure_mode,
        severity=severity,
        symptoms=[s.to_dict() for s in symptoms],
        assertions=[a.to_dict() for a in assertions],
        injected_faults=[dict(f) for f in injected_faults],
        final_output=final_output,
        error=error,
        baseline=baseline,
        delta_vs_baseline=delta,
        code_pointers=pointers,
        tags=dict(tags or {}),
        tool_calls=[],
        llm_exchanges=[],
    )

    errors = result.validate()
    if errors:
        if strict_schema:
            raise SchemaError(f"assembled report is not schema-valid: {errors[:5]}")
        result.schema_errors = errors
    return result
