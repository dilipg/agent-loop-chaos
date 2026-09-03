"""Deterministic probes over the trace, and the symptoms they emit.

Three binding rules, each with its own test (`docs/07-TESTING.md` §3):

1. **Harness attribution.** A probe must not fire on an event or a value the harness
   caused. `docs/11` §2 rules R1-R4 are normative. This is what makes the
   `good_agent` control passable at all.
2. **No clock in decision logic.** No probe reads wall-clock time, `ts`, or
   `ts_mono_ms`. Timing-derived findings were removed for exactly this reason.
3. **Every symptom cites evidence.** A symptom with an empty `evidence` list fails a
   test.

Prefer a false negative to a false positive. A missed finding costs one scenario; a
false finding sends a coding agent to fix nothing and teaches the user to distrust
the tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import Severity

__all__ = ["PROBE_PRECEDENCE", "SEVERITY_ORDER", "Symptom"]

# Tie-break order for `dominant_symptom` (`docs/11` §8): egress and safety first,
# then crashes, then correctness, then budgets. A test asserts this list and the
# `docs/07` §3 table stay equal as sets.
PROBE_PRECEDENCE: tuple[str, ...] = (
    "secret_in_output",
    "redacted_value_in_output",
    "injection_followed",
    "duplicate_side_effect",
    "assertions_failed",
    "unhandled_exception",
    "schema_violation",
    "state_key_read_after_drop",
    "loop_repeat_cycle",
    "max_steps_exhausted",
    "progress_stalled",
    "retry_storm",
    "no_retry_on_transient",
    "truncated_output_used",
    "empty_final_answer",
    "no_output_validation",
    "instruction_precedence_violation",
    "goal_token_loss",
    "pre_existing_invalid_args",
    "token_blowup",
)

SEVERITY_ORDER: dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "info": 0,
}


@dataclass(frozen=True, slots=True)
class Symptom:
    """One structural finding, with its citations.

    Attributes:
        code: The probe code. Must appear in `PROBE_PRECEDENCE`.
        severity: How serious this finding is on its own.
        detail: One clause of plain English, quoted by the narrator and the work
            order.
        evidence: At least one citation, each carrying a real trace `seq`. A symptom
            with no evidence is one a coding agent cannot verify, and a test forbids
            it.
    """

    code: str
    severity: Severity
    detail: str
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the `symptoms[]` shape.

        Returns:
            A plain dict matching `chaos_report.schema.json`.
        """
        return {
            "code": self.code,
            "severity": self.severity,
            "detail": self.detail,
            "evidence": self.evidence,
        }

    def first_seq(self) -> int:
        """The earliest trace sequence this symptom cites.

        Returns:
            The lowest `seq` among the evidence, or a large sentinel when the
            evidence carries none, so an uncited symptom sorts last rather than
            first.
        """
        seqs = [int(e["seq"]) for e in self.evidence if isinstance(e.get("seq"), int)]
        return min(seqs) if seqs else 10**9
