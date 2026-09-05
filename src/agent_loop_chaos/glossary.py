"""Plain English for every code a report can emit.

`report.json` is written for machines, and every field in it is a stable identifier
that a consumer can switch on. That makes it unreadable to the person who most needs
to read it: someone who did not write the agent, is deciding whether to ship it, and
has never seen the string `unverified_claim_emitted`.

This module is the one place that translates. The dashboard's report view and the HTML
export both read it, so there is one wording to fix rather than two, and a test
asserts every value of every enum has an entry -- a glossary with a hole shows a raw
code to exactly the reader who cannot decode one.

Fault descriptions are **not** listed here. They come from the fault registry, which
already carries a summary per kind, so the catalog and the page cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from .intensity import MAX_LEVEL, MIN_LEVEL, profile

__all__ = [
    "ASSERTIONS",
    "EXPECTED",
    "FAILURE_MODES",
    "OBSERVED",
    "PROBES",
    "SEVERITIES",
    "bundle",
    "faults",
]

#: What went wrong, as a consequence rather than a category.
FAILURE_MODES: dict[str, str] = {
    "none": "Nothing went wrong. The agent handled what was thrown at it.",
    "graceful_degradation": "The agent noticed the problem and still did something sensible.",
    "recovered_after_retry": "The agent hit an error, tried again, and got there.",
    "explicit_error_surfaced": "The agent stopped and said clearly that it could not continue.",
    "crash_unhandled_exception": (
        "The agent crashed. The error reached the top without anyone catching it, so a "
        "user would have seen a failure page rather than an answer."
    ),
    "hallucination_on_corrupt_data": (
        "The data the agent read had been corrupted, and instead of noticing, it "
        "answered confidently using the corrupted values."
    ),
    "silent_wrong_answer": (
        "The agent produced a confident, well-formed, wrong answer. Nothing in the "
        "output signals that anything went wrong, which is why this one is dangerous."
    ),
    "unverified_claim_emitted": (
        "The agent stated something it had no source for -- an invented number, "
        "reference, or a claim about work it never did."
    ),
    "goal_drift": "The agent lost track of what it was asked to do and answered something else.",
    "context_loss": "Information the agent needed fell out of its memory and it carried on anyway.",
    "instruction_precedence_violation": (
        "The agent followed an instruction that arrived in data instead of the one its "
        "operator gave it."
    ),
    "infinite_loop": "The agent went round in circles doing the same thing and never finished.",
    "max_iterations_exhausted": (
        "The agent used up its whole step budget without producing an answer."
    ),
    "no_retry_on_transient_error": (
        "A temporary error came back and the agent gave up instead of trying again."
    ),
    "retry_storm": (
        "The agent retried far too many times, which in production means hammering a "
        "service that is already struggling."
    ),
    "schema_violation_downstream": (
        "The agent passed on data in the wrong shape, so whatever consumes its output would break."
    ),
    "truncated_output_used": (
        "A response was cut off part way through and the agent used the fragment as if "
        "it were complete."
    ),
    "tool_dispatch_error": "The agent tried to call a tool in a way the tool could not accept.",
    "duplicate_side_effect": (
        "Something that should have happened once happened twice -- a double booking, a "
        "double charge, a duplicate record."
    ),
    "state_corruption_propagated": (
        "Bad data entered the agent's working memory and spread, so later steps were wrong too."
    ),
    "prompt_injection_followed": (
        "Text hidden in retrieved content told the agent to do something, and it "
        "obeyed. This is the attack shape that matters most."
    ),
    "secret_leak": (
        "A credential or other secret ended up somewhere it should never appear, such "
        "as the answer sent back to the user."
    ),
    "empty_final_answer": "The agent returned nothing at all. The user gets a blank response.",
    "timeout": "The agent ran out of time before finishing.",
    "latency_budget_exceeded": "The agent finished, but took far longer than it is allowed to.",
    "harness_error": (
        "Something went wrong inside the testing tool itself, not in the agent. This is "
        "our bug, not yours."
    ),
    "unknown": "The run could not be classified. Treat this as a gap in the tooling.",
}

#: What the agent was seen to do, before any judgement about whether it was right.
OBSERVED: dict[str, str] = {
    "graceful_degradation": "Noticed the problem and worked around it.",
    "explicit_error": "Stopped and reported the error to the caller.",
    "retried_then_succeeded": "Failed once, retried, and succeeded.",
    "aborted_with_message": "Gave up and explained why.",
    "completed_unaffected": "Finished as if nothing had happened.",
    "crashed": "Threw an error that nobody caught anywhere.",
    "hallucinated": "Stated something it had no source for.",
    "answered_confidently_wrong": "Gave a confident answer that was wrong.",
    "looped": "Repeated the same work over and over again.",
    "hit_step_limit": "Ran until the step budget cut it off.",
    "stalled": "Kept going but stopped making progress.",
    "followed_injected_instruction": "Obeyed an instruction hidden in its input.",
    "leaked_secret": "Put a secret somewhere it should not be.",
    "emitted_empty": "Returned nothing at all.",
    "timed_out": "Ran out of time before finishing.",
    "harness_error": "The testing tool failed, not the agent.",
    "indeterminate": "Could not be determined from the run.",
}

#: What the scenario author said good behaviour would look like here.
EXPECTED: dict[str, str] = {
    "graceful_degradation": "Notice the problem and still do something useful.",
    "explicit_error": "Stop and report the error rather than guessing.",
    "retry_then_succeed": "Try again and get there.",
    "abort_with_message": "Give up, but explain why.",
    "ignore_and_continue": "Carry on unaffected -- nothing here should change the answer.",
}

#: How much this matters.
SEVERITIES: dict[str, str] = {
    "critical": "Would cause real harm in production. Fix before shipping.",
    "high": "A user would notice and be misled or blocked. Fix soon.",
    "medium": "Worth fixing, but the damage is contained.",
    "low": "Minor. Worth knowing about.",
    "info": "Not a problem. Recorded so the run is complete.",
}

#: The structural failures the trace can prove, in the reader's terms.
PROBES: dict[str, str] = {
    "secret_in_output": "A secret appeared in the answer the user would have seen.",
    "redacted_value_in_output": "A value we had already masked still reached the output.",
    "injection_followed": "The agent obeyed an instruction that was hidden in its input.",
    "duplicate_side_effect": "A real-world action happened more than once.",
    "assertions_failed": "The run broke a rule the scenario said it had to keep.",
    "unhandled_exception": "An error escaped without being caught.",
    "schema_violation": "The output was not the shape it was supposed to be.",
    "state_key_read_after_drop": (
        "The agent read a value that had been removed, without checking it was there."
    ),
    "loop_repeat_cycle": "The agent repeated the same sequence of steps.",
    "max_steps_exhausted": "The agent used its entire step budget.",
    "progress_stalled": "The agent kept working but stopped getting anywhere.",
    "retry_storm": "The agent retried far more than it should have.",
    "no_retry_on_transient": "A temporary error was treated as permanent, with no retry.",
    "truncated_output_used": "A cut-off response was used as if complete.",
    "empty_final_answer": "The final answer was empty.",
    "no_output_validation": "The agent used a tool result without checking it first.",
    "instruction_precedence_violation": (
        "An instruction from data outranked the one from the operator."
    ),
    "goal_token_loss": "The original request faded out of the agent's context.",
    "pre_existing_invalid_args": "The agent called a tool with arguments it could not accept.",
    "token_blowup": "The context grew far beyond what the work needed.",
}


#: What each declared check was asking for. `assertions_failed` on its own tells a
#: reader nothing -- *which* rule, and what it was protecting, is the finding.
ASSERTIONS: dict[str, str] = {
    "auto": "Checks the tool chose automatically, based on what was broken.",
    "final_state_has": "The agent's memory should still hold certain values at the end.",
    "final_state_lacks": "Certain values should no longer be in the agent's memory.",
    "idempotent_effects": "Doing the same thing twice should have had one effect, not two.",
    "max_steps": "The agent should have finished within a step budget.",
    "max_tool_calls": "The agent should not have called tools more than a set number of times.",
    "must_call_tools": "The agent should have used certain tools and did not.",
    "must_not_call_tools": "The agent should not have used certain tools, and it did.",
    "no_claim_about": "The agent should not state a value for something that was removed.",
    "no_fabricated_citations": (
        "Every reference in the answer should appear in something the agent actually retrieved."
    ),
    "no_invented_tools": "The agent should not claim to have done work it did not do.",
    "no_unsourced_numbers": "Every figure in the answer should trace back to something real.",
    "output_is_json": "The answer should have been valid JSON.",
    "output_json_schema": "The answer should have matched an agreed shape.",
    "output_matches": "The answer should have contained something specific and did not.",
    "output_mentions_any": "The answer should still be about what was asked.",
    "output_non_empty": "The agent should have produced an answer at all.",
    "output_not_matches": "The answer should not have contained something, and it did.",
    "tool_call_count": "A tool should have been called a certain number of times.",
}


def faults() -> dict[str, dict[str, Any]]:
    """What each fault does, from the registry rather than a second copy.

    Returns:
        Fault kind to `{summary, severity_hint, layers}`.
    """
    from .faults import list_faults

    return {
        info.kind: {
            "summary": info.summary,
            "severity_hint": info.severity_hint,
            "layers": sorted({pair.split(":")[0] for pair in info.layer_phases}),
        }
        for info in list_faults()
    }


def bundle() -> dict[str, Any]:
    """Everything a reader-facing page needs, in one document.

    Returns:
        The glossary, ready to serve or to inline in an export.
    """
    return {
        "assertions": dict(ASSERTIONS),
        "failure_modes": dict(FAILURE_MODES),
        "observed": dict(OBSERVED),
        "expected": dict(EXPECTED),
        "severities": dict(SEVERITIES),
        "probes": dict(PROBES),
        "faults": faults(),
        "intensity": [
            {
                "level": profile(level).level,
                "label": profile(level).label,
                "summary": profile(level).summary,
            }
            for level in range(MIN_LEVEL, MAX_LEVEL + 1)
        ],
    }
