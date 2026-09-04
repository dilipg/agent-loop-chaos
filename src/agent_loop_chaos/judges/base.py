"""Judge protocol, verdict objects, and the evidence projection.

Nothing here talks to a model. This module owns the three things every judge shares:
the `Verdict` it returns, the bounded `JudgeEvidence` it reads, and the ten-line
renderer that turns a prompt asset into a string.

The evidence projection is the security boundary. A trace holds whatever the agent
under test saw, including the adversarial text this library injects on purpose, and
that text is about to be handed to a model whose output a human reads and a coding
agent acts on. So every untrusted span is fenced, fence markers inside the data are
escaped (D-21), `code_context` is redacted like any other payload, and the whole
projection is capped (D-22, `docs/05` §2).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from ..enums import ExpectedBehavior, FailureMode, ObservedBehavior, Severity
from ..redact import DEFAULT_KEY_PATTERNS, redact

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..report import ChaosResult

__all__ = [
    "HARD_CAP_BYTES",
    "TARGET_BYTES",
    "Judge",
    "JudgeEvidence",
    "JudgeMeta",
    "Verdict",
    "build_evidence",
    "build_prompt_values",
    "escape_fences",
    "redact_source",
    "render",
]

# Caps from `docs/05` §2. A 3B model's usable context is the constraint, not its
# advertised one: an oversized prompt degrades a small model long before it errors.
TARGET_BYTES = 6 * 1024
HARD_CAP_BYTES = 12 * 1024
_FINAL_OUTPUT_CAP = 900
_BASELINE_CAP = 600
_EXCHANGE_CAP = 700
_EXCHANGES_KEPT = 2
_PAYLOAD_CAP = 800
_PATCH_OPS_CAP = 20
_ERROR_FRAMES = 5

_DROP_ORDER = ("code_context", "baseline_output", "older_exchanges", "tool_summary")

# Untrusted sections per D-21, plus `baseline_output`, which `judge_user.md` also
# fences: it is the agent's own text from a clean run, and an agent that emits an
# injection payload does so with or without a fault.
_UNTRUSTED = (
    "injected",
    "last_exchanges",
    "final_output",
    "tool_summary",
    "code_context",
    "baseline_output",
)

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def render(template: str, values: Mapping[str, Any]) -> str:
    """Substitute `{{name}}` placeholders from a flat mapping.

    Deliberately not Jinja. The prompt assets embed JSON schema examples, and a real
    template engine would try to evaluate them; double braces are the delimiter
    precisely so single-brace JSON needs no escaping. Substitution is single-pass, so
    a value that itself looks like a placeholder is not expanded again -- untrusted
    evidence reaches this function.

    Args:
        template: The prompt asset's text.
        values: Placeholder values. Non-strings are stringified.

    Returns:
        The rendered string. A missing key becomes the empty string, so a prompt
        never leaks a raw `{{name}}` to a model.
    """
    return _PLACEHOLDER.sub(lambda m: str(values.get(m.group(1), "")), template)


def escape_fences(text: str) -> str:
    """Neutralize untrusted-data fence markers occurring inside the data (D-21).

    Captured text that contains a closing fence would otherwise end the quarantine
    early and promote the rest of the payload to instructions.

    Args:
        text: The captured text.

    Returns:
        The text with both fence markers broken by a zero-width-free marker that
        still reads correctly to a human.
    """
    return text.replace("<<<END_UNTRUSTED_DATA", "<!!<END_UNTRUSTED_DATA").replace(
        "<<<UNTRUSTED_DATA", "<!!<UNTRUSTED_DATA"
    )


# `NAME = "value"`, `name: T = "value"`, `"name": "value"` -- the three ways a
# credential is bound to a name in the source files this library reads. The value
# patterns in `redact.py` catch known secret *shapes*; a hardcoded password has no
# shape, and only the name it is bound to gives it away (D-22).
_ASSIGNMENT = re.compile(
    r"""(?P<lhs>["']?(?P<name>[A-Za-z_][A-Za-z0-9_]*)["']?\s*(?::[^=\n]*)?\s*[:=]\s*)
        (?P<quote>["'])(?P<value>(?:[^"'\\\n]|\\.)*)(?P=quote)""",
    re.VERBOSE,
)
_SENSITIVE_NAME = re.compile(
    "|".join(pattern for _reason, pattern in DEFAULT_KEY_PATTERNS), re.IGNORECASE
)


def redact_source(text: str) -> str:
    """Redact string literals bound to a sensitive-looking name.

    `redact()` matches known secret shapes and sensitive mapping keys. Source code is
    neither: a hardcoded password is an ordinary string literal under an ordinary
    identifier, and it is precisely what D-22 says user source "routinely contains".
    This closes that gap without touching code that is merely near a secret -- the
    surrounding lines stay readable, because a judge may only name symbols it saw.

    Args:
        text: A source excerpt.

    Returns:
        The excerpt with sensitive-looking literals replaced.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group("name")
        reason = next(
            (r for r, pat in DEFAULT_KEY_PATTERNS if re.search(pat, name, re.IGNORECASE)), None
        )
        if reason is None:
            return match.group(0)
        quote = match.group("quote")
        return f"{match.group('lhs')}{quote}<redacted:{reason}>{quote}"

    return _ASSIGNMENT.sub(_replace, text)


def _clip(value: Any, limit: int, *, tail: bool = False) -> str:
    """Render a value as a string of at most `limit` characters.

    Args:
        value: Anything.
        limit: Maximum length, including the elision marker.
        tail: Keep the end rather than the start. Prompts are more informative at
            the end than the beginning.

    Returns:
        The clipped string.
    """
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return text
    return "…" + text[-(limit - 1) :] if tail else text[: limit - 1] + "…"


@dataclass(frozen=True, slots=True)
class JudgeMeta:
    """How a verdict was produced, so it is attributable to an exact prompt version."""

    kind: Literal["rules", "slm", "ensemble", "anthropic", "custom"] = "rules"
    model: str | None = None
    endpoint: str | None = None
    transport: Literal["openai", "ollama", "anthropic", "none"] | None = None
    temperature: float | None = None
    prompt_hash: str | None = None
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    attempts: int | None = None
    fell_back_to_rules: bool = False
    raw_response_path: str | None = None
    evidence_dropped: tuple[str, ...] = ()
    auto_selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the schema's `judge_meta` shape.

        Returns:
            A plain dict. `evidence_dropped` becomes a list.
        """
        return {
            "kind": self.kind,
            "model": self.model,
            "endpoint": self.endpoint,
            "transport": self.transport,
            "temperature": self.temperature,
            "prompt_hash": self.prompt_hash,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "attempts": self.attempts,
            "fell_back_to_rules": self.fell_back_to_rules,
            "raw_response_path": self.raw_response_path,
            "evidence_dropped": list(self.evidence_dropped),
            "auto_selected": self.auto_selected,
        }


@dataclass(slots=True)
class Verdict:
    """A classified outcome plus its narration.

    `passed` is authoritative but computed: it comes from the probes, the assertions
    layer and the scenario's `expected_behavior` (`docs/11` §1). A judge carries it,
    never decides it -- which is why it is a constructor argument here and never
    read back from a model's response.
    """

    passed: bool
    expected_behavior: ExpectedBehavior
    observed_behavior: ObservedBehavior
    failure_mode: FailureMode
    severity: Severity
    confidence: float
    narrative: str
    root_cause_hypothesis: str | None = None
    refinement_hint: str | None = None
    suggested_fixes: list[dict[str, Any]] = field(default_factory=list)
    judge_disagreement: str | None = None
    judge_meta: JudgeMeta = field(default_factory=JudgeMeta)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the `judge_verdict.schema.json` shape.

        Returns:
            The verdict as a plain dict, with every required property present.
        """
        return {
            "passed": self.passed,
            "expected_behavior": self.expected_behavior,
            "observed_behavior": self.observed_behavior,
            "failure_mode": self.failure_mode,
            "severity": self.severity,
            "confidence": self.confidence,
            "narrative": self.narrative,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "refinement_hint": self.refinement_hint,
            "suggested_fixes": list(self.suggested_fixes),
            "judge_disagreement": self.judge_disagreement,
            "judge_meta": self.judge_meta.to_dict(),
        }


@runtime_checkable
class Judge(Protocol):
    """What every judge implements."""

    name: str

    def judge(self, ev: JudgeEvidence) -> Verdict:
        """Turn probe output and trace evidence into a verdict.

        Args:
            ev: The evidence assembled by the engine. Untrusted spans inside it are
                fenced before they reach a model (D-21).

        Returns:
            The verdict. `passed` is supplied by the engine, never by a model.
        """
        ...


@dataclass(slots=True)
class JudgeEvidence:
    """The compact, redacted projection of a run (`docs/05` §2).

    Never hand the raw trace to a model. Everything here is capped at construction,
    and `fit` shrinks it further along a fixed path when it is still too large.
    """

    scenario_id: str | None
    expected_behavior: ExpectedBehavior
    observed_behavior: ObservedBehavior
    passed: bool
    seed: int
    must_not: list[str] = field(default_factory=list)
    injected: list[dict[str, Any]] = field(default_factory=list)
    symptoms: list[dict[str, Any]] = field(default_factory=list)
    assertions: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    delta: dict[str, Any] | None = None
    baseline_output: str | None = None
    final_output: str | None = None
    last_exchanges: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    tool_summary: list[dict[str, Any]] = field(default_factory=list)
    code_context: list[dict[str, Any]] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, Any]:
        """Flatten to the values `judge_user.md`, `narrator.md` and `refiner.md` read.

        Untrusted sections are fence-escaped here rather than at construction, so the
        stored evidence stays faithful to what the run produced and only the rendered
        prompt is defanged.

        Returns:
            A flat mapping of placeholder name to value.
        """
        metrics = self.metrics
        out: dict[str, Any] = {
            "scenario_id": self.scenario_id or "",
            "seed": self.seed,
            "expected_behavior": self.expected_behavior,
            "observed_behavior": self.observed_behavior,
            "passed": self.passed,
            "must_not": ", ".join(self.must_not) or "(none)",
            "injected": _block(self.injected),
            "symptoms": _block(self.symptoms),
            "assertions": _block(self.assertions),
            "steps": metrics.get("steps", 0),
            "tool_calls": metrics.get("tool_calls", 0),
            "llm_calls": metrics.get("llm_calls", 0),
            "retries": metrics.get("retries", 0),
            "limit_hit": metrics.get("limit_hit") or "none",
            "tool_summary": _block(self.tool_summary),
            "last_exchanges": _block(self.last_exchanges),
            "final_output": self.final_output or "(no output)",
            "baseline_output": self.baseline_output or "(no baseline run)",
            "delta": _block(self.delta) if self.delta else "(no baseline run)",
            "error": _block(self.error) if self.error else "(none)",
            "code_context": _block(self.code_context) if self.code_context else "(none provided)",
        }
        for key in _UNTRUSTED:
            out[key] = escape_fences(str(out[key]))
        return out

    def size_bytes(self) -> int:
        """Measure what would actually reach the model.

        Returns:
            The UTF-8 length of the serialized prompt values -- not of the dataclass,
            which carries structure the prompt never renders.
        """
        return len(json.dumps(self.to_prompt_dict(), default=str).encode("utf-8"))

    def fit(
        self, *, target: int = TARGET_BYTES, hard_cap: int = HARD_CAP_BYTES
    ) -> tuple[JudgeEvidence, list[str]]:
        """Shrink to the budget along a fixed path, without mutating self.

        Drop order is `code_context` → `baseline_output` → older exchanges →
        `tool_summary` (`docs/05` §2). Sections are dropped whole, so the result is
        never truncated mid-JSON. If everything has gone and the projection is still
        over the hard cap, the remaining free text is clipped -- the alternative is
        sending a prompt that a small model silently truncates at an arbitrary point.

        Args:
            target: Size to get under by dropping sections.
            hard_cap: Size that must not be exceeded under any circumstances.

        Returns:
            ``(fitted_copy, dropped)`` where `dropped` names the sections removed, in
            the order they were removed.
        """
        current = replace(self)
        dropped: list[str] = []
        if current.size_bytes() <= target:
            return current, dropped

        for section in _DROP_ORDER:
            if section == "code_context":
                current = replace(current, code_context=[])
            elif section == "baseline_output":
                current = replace(current, baseline_output=None, delta=None)
            elif section == "older_exchanges":
                current = replace(current, last_exchanges=current.last_exchanges[-1:])
            else:
                current = replace(current, tool_summary=[])
            dropped.append(section)
            if current.size_bytes() <= target:
                return current, dropped

        if current.size_bytes() > hard_cap:
            # Everything droppable is gone; what is left is the injected faults and
            # the agent's own output, none of which can be removed without making the
            # verdict unattributable. Clip instead, hardest-hitting field first.
            current = replace(
                current,
                injected=[_clip_fires(f) for f in current.injected],
                final_output=_clip(current.final_output or "", 400),
                last_exchanges=[
                    {
                        "prompt": _clip(e.get("prompt", ""), 200, tail=True),
                        "response": _clip(e.get("response", ""), 200),
                    }
                    for e in current.last_exchanges
                ],
            )
            dropped.append("clipped_free_text")
        return current, dropped


def _clip_fires(fault: Mapping[str, Any]) -> dict[str, Any]:
    """Shrink one injected-fault record to its identity plus a one-line note.

    Args:
        fault: A projected fault record.

    Returns:
        A copy without payload bodies or patch ops.
    """
    return {
        "fault_id": fault.get("fault_id"),
        "type": fault.get("type"),
        "target": fault.get("target"),
        "note": _clip(fault.get("note") or "", 120),
    }


def _block(value: Any) -> str:
    """Render a structure for a prompt.

    Args:
        value: Anything JSON-serializable.

    Returns:
        Compact JSON, or ``"(none)"`` for an empty collection. `default=str` because
        an evidence value may still hold a non-JSON object the redactor passed
        through (D-67).
    """
    if not value:
        return "(none)"
    return json.dumps(value, default=str, sort_keys=False)


def _tool_summary(tool_calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate tool calls into per-tool counts.

    Repeated identical signatures are the tell for a retry storm or a loop, so they
    are counted rather than listed.

    Args:
        tool_calls: The report's `tool_calls` records.

    Returns:
        One entry per tool name, in first-call order.
    """
    order: list[str] = []
    stats: dict[str, dict[str, Any]] = {}
    signatures: dict[str, dict[str, int]] = {}
    for call in tool_calls:
        name = str(call.get("name") or call.get("tool") or "?")
        if name not in stats:
            order.append(name)
            stats[name] = {"tool": name, "calls": 0, "ok": 0, "failed": 0}
            signatures[name] = {}
        entry = stats[name]
        entry["calls"] += 1
        failed = call.get("error") or call.get("ok") is False
        entry["failed" if failed else "ok"] += 1
        sig = _clip(call.get("args") or call.get("kwargs") or {}, 120)
        signatures[name][sig] = signatures[name].get(sig, 0) + 1
    for name in order:
        repeated = max(signatures[name].values(), default=0)
        if repeated > 1:
            stats[name]["repeated_signature"] = repeated
    return [stats[name] for name in order]


def _project_faults(injected_faults: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reduce fault records to what a model needs to narrate the injection.

    Args:
        injected_faults: The report's `injected_faults`.

    Returns:
        One entry per fault that fired, capped per `docs/05` §2.
    """
    out: list[dict[str, Any]] = []
    for fault in injected_faults:
        if not fault.get("fired"):
            continue
        fires = list(fault.get("fires") or [])
        first = fires[0] if fires else {}
        entry: dict[str, Any] = {
            "fault_id": fault.get("fault_id"),
            "type": fault.get("type"),
            "target": fault.get("target"),
            "fire_count": len(fires),
            "note": _clip(first.get("note") or "", 200),
        }
        patch = first.get("json_patch")
        if patch:
            entry["json_patch"] = list(patch)[:_PATCH_OPS_CAP]
        for side in ("payload_before", "payload_after"):
            if first.get(side) is not None:
                entry[side] = _clip(first[side], _PAYLOAD_CAP)
        out.append(entry)
    return out


def _project_error(error: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Reduce a captured exception to its type, message and top user frames.

    Args:
        error: The report's `error` block.

    Returns:
        The projection, or `None`.
    """
    if not error:
        return None
    frames = list(error.get("traceback") or error.get("frames") or [])[:_ERROR_FRAMES]
    return {
        "type": error.get("type"),
        "message": _clip(error.get("message") or "", 400),
        "frames": frames,
    }


def _redact_context(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Redact one code-context excerpt two ways (D-22).

    Args:
        entry: A `{file, line, source}` excerpt.

    Returns:
        A redacted copy: known secret shapes via `redact()`, plus string literals
        bound to a sensitive-looking identifier via `redact_source()`.
    """
    out: dict[str, Any] = dict(redact(dict(entry)))
    source = out.get("source")
    if isinstance(source, str):
        out["source"] = redact_source(source)
    return out


def build_evidence(
    result: ChaosResult, *, code_context: Sequence[Mapping[str, Any]] | None = None
) -> JudgeEvidence:
    """Project a finished result into the bounded evidence a judge reads.

    Pure over a `ChaosResult`, which is what lets `alc judge` rebuild evidence from a
    stored `report.json` and re-judge it with a better model without re-running the
    agent.

    Args:
        result: The assembled report object.
        code_context: Source excerpts around the error frames. Redacted here, since
            user source routinely carries hardcoded credentials (D-22).

    Returns:
        The projection, already capped but not yet fitted to the byte budget.
    """
    verdict = result.verdict or {}
    exchanges = [
        {
            "prompt": _clip(e.get("prompt") or e.get("messages") or "", _EXCHANGE_CAP, tail=True),
            "response": _clip(e.get("response") or e.get("content") or "", _EXCHANGE_CAP),
        }
        for e in list(result.llm_exchanges)[-_EXCHANGES_KEPT:]
    ]
    baseline_output = (result.baseline or {}).get("final_output") if result.baseline else None
    return JudgeEvidence(
        scenario_id=result.scenario_id,
        expected_behavior=verdict.get("expected_behavior", "graceful_degradation"),
        observed_behavior=verdict.get("observed_behavior", "indeterminate"),
        passed=result.success,
        seed=result.seed,
        must_not=[],
        injected=_project_faults(result.injected_faults),
        symptoms=[
            {"code": s.get("code"), "severity": s.get("severity"), "detail": s.get("detail")}
            for s in result.symptoms
        ],
        assertions=[
            {"check": a.get("check"), "ok": a.get("ok"), "detail": a.get("detail")}
            for a in result.assertions
        ],
        metrics={
            "steps": result.metrics.get("steps", 0),
            "tool_calls": result.metrics.get("tool_calls", 0),
            "llm_calls": result.metrics.get("llm_calls", 0),
            "retries": result.metrics.get("retries", 0),
            "tokens": result.metrics.get("tokens_total"),
            "limit_hit": result.loop.get("limit_hit"),
        },
        delta=dict(result.delta_vs_baseline) if result.delta_vs_baseline else None,
        baseline_output=_clip(baseline_output, _BASELINE_CAP) if baseline_output else None,
        final_output=(
            _clip(result.final_output, _FINAL_OUTPUT_CAP)
            if result.final_output is not None
            else None
        ),
        last_exchanges=exchanges,
        error=_project_error(result.error),
        tool_summary=_tool_summary(result.tool_calls),
        code_context=[_redact_context(c) for c in code_context or ()],
    )


def build_prompt_values(ev: JudgeEvidence, verdict: Verdict | None = None) -> dict[str, Any]:
    """Prompt values for `refiner.md`, which needs the classification too.

    Args:
        ev: The fitted evidence.
        verdict: The verdict so far, if one exists.

    Returns:
        `to_prompt_dict()` plus `failure_mode`, `severity` and
        `root_cause_hypothesis`, which the refiner template reads and the evidence
        alone does not carry.
    """
    values = ev.to_prompt_dict()
    values.update(
        {
            "failure_mode": verdict.failure_mode if verdict else "unknown",
            "severity": verdict.severity if verdict else "medium",
            "root_cause_hypothesis": (verdict.root_cause_hypothesis if verdict else None)
            or "(none recorded)",
        }
    )
    return values
