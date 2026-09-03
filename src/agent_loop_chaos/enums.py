"""String enums mirrored from the JSON Schemas.

`Literal` aliases rather than `enum.Enum`: `docs/02-API.md` assigns them bare
strings (``expected_behavior: ExpectedBehavior = "graceful_degradation"``) and every
value is serialized straight into JSON. `tests/test_schemas.py` asserts each alias
still matches its schema enum, so these cannot drift from
`chaos_report.schema.json` (D-51).
"""

from __future__ import annotations

from typing import Literal, get_args

__all__ = [
    "EXPECTED_BEHAVIORS",
    "FAILURE_MODES",
    "OBSERVED_BEHAVIORS",
    "SEVERITIES",
    "ExpectedBehavior",
    "FailureMode",
    "ObservedBehavior",
    "Severity",
]

Severity = Literal["critical", "high", "medium", "low", "info"]

ExpectedBehavior = Literal[
    "graceful_degradation",
    "explicit_error",
    "retry_then_succeed",
    "abort_with_message",
    "ignore_and_continue",
]

ObservedBehavior = Literal[
    "graceful_degradation",
    "explicit_error",
    "retried_then_succeeded",
    "aborted_with_message",
    "completed_unaffected",
    "crashed",
    "hallucinated",
    "answered_confidently_wrong",
    "looped",
    "hit_step_limit",
    "stalled",
    "followed_injected_instruction",
    "leaked_secret",
    "emitted_empty",
    "timed_out",
    "harness_error",
    "indeterminate",
]

FailureMode = Literal[
    "none",
    "graceful_degradation",
    "recovered_after_retry",
    "explicit_error_surfaced",
    "crash_unhandled_exception",
    "hallucination_on_corrupt_data",
    "silent_wrong_answer",
    "unverified_claim_emitted",
    "goal_drift",
    "context_loss",
    "instruction_precedence_violation",
    "infinite_loop",
    "max_iterations_exhausted",
    "no_retry_on_transient_error",
    "retry_storm",
    "schema_violation_downstream",
    "truncated_output_used",
    "tool_dispatch_error",
    "duplicate_side_effect",
    "state_corruption_propagated",
    "prompt_injection_followed",
    "secret_leak",
    "empty_final_answer",
    "timeout",
    # Retained in the enum but never emitted: the probe was removed because it made
    # `success` wall-clock dependent (D-46).
    "latency_budget_exceeded",
    "harness_error",
    "unknown",
]

SEVERITIES: tuple[str, ...] = get_args(Severity)
EXPECTED_BEHAVIORS: tuple[str, ...] = get_args(ExpectedBehavior)
OBSERVED_BEHAVIORS: tuple[str, ...] = get_args(ObservedBehavior)
FAILURE_MODES: tuple[str, ...] = get_args(FailureMode)
