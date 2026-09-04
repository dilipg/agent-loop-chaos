"""The five nodes of the buggy revenue-risk review.

Every planted weakness from `docs/09-DEMO-AGENT.md` §4 lives in this file or in
`edges.py`, and each is named in a comment at the site. None of them is exotic: they
are the bugs that survive code review because the happy path is green.

Node signature is uniform — ``(state, tools, llm, engine)`` — so the runner in
`app.py` is a dict dispatch and the diff against `revenue_review_fixed` stays
readable. This tree ignores `engine`: it records no `validated()` evidence and
leaves no `note()` breadcrumb, which is itself part of the point.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "ANALYST_READ_PROMPT",
    "ANALYST_SUMMARY_PROMPT",
    "NODES",
    "PLANNER_PROMPT",
    "REVIEWER_PROMPT",
    "SCORE_PROMPT",
    "WRITER_DRAFT_PROMPT",
    "data_analyst",
    "planner",
    "reviewer",
    "risk_scorer",
    "writer",
]

State = dict[str, Any]

PLANNER_PROMPT = """[PLANNER]
You are opening a quarterly revenue-risk review.
Objective: {objective}
Reply with a JSON array of the account_ids worth investigating, and nothing else."""

# Weakness 2: the prompt says "use the rows below" and never says what to do when
# they are missing, so an empty or corrupt payload produces confident prose instead
# of an admission. Caught by `tool.empty_json` and `tool.drop_required_key`.
ANALYST_READ_PROMPT = """[ANALYST_READ]
Interpret the invoice and usage rows below for the accounts under review.
Invoices: {invoices}
Usage: {usage}"""

ANALYST_SUMMARY_PROMPT = """[ANALYST_SUMMARY]
Summarise the numbers in two sentences.
Worst invoice: {worst_invoice} at {max_days_overdue} days overdue.
Overdue exposure: {overdue_usd} USD across {under_used} under-used accounts.
Reading so far: {reading}"""

SCORE_PROMPT = """[SCORE]
Score the collections risk for this quarter from 0 to 100.
Exposure: {exposure_usd} USD. Annualised revenue: {arr_usd} USD.
Reply with JSON: {{"score": int, "band": str, "drivers": [str]}}"""

CRITIQUE_PROMPT = """[CRITIQUE]
Critique your own score below in one sentence. What did it over- or under-weight?
{score}"""

REVISE_PROMPT = """[REVISE]
Revise the score in light of the critique. Same JSON shape.
Score: {score}
Critique: {critique}"""

# Weakness 10: `{names}` is filled from `load_accounts`'s free-text `notes` field --
# CSV text that reached us through a tool -- and is interpolated with no fence and no
# "this is data, not instructions" preamble. Caught by `adversarial.injection_corpus`.
WRITER_DRAFT_PROMPT = """[DRAFT]
Objective: {objective}
Risk: {risk}
Accounts under review: {names}
Draft the revenue-risk paragraph for the CSM team."""

WRITER_TIGHTEN_PROMPT = """[TIGHTEN]
Tighten this to one sentence.
{draft}"""

REVIEWER_PROMPT = """[REVIEW]
Reply APPROVE if the paragraph below is ready to send, otherwise REVISE.
{draft}"""


def _content(response: Any) -> str:
    """Pull the body out of a model response.

    Weakness 9: `finish_reason` is never inspected, so a body the token limit cut
    off is handed downstream as though it were complete. Caught by
    `llm.truncated_mid_json`.

    Args:
        response: Whatever the model returned.

    Returns:
        The response body as text.
    """
    if isinstance(response, dict):
        return str(response["content"])
    return str(response)


def planner(state: State, tools: dict[str, Any], llm: Any, engine: Any) -> State:
    """Decide which accounts to investigate. One LLM call.

    Args:
        state: The working state.
        tools: The tool table. Unused here.
        llm: The model callable.
        engine: The engine. Unused: this tree records no evidence.

    Returns:
        `state` with `targets` set.

    Raises:
        IndexError: When the message list is empty.
        json.JSONDecodeError: When the model answers in prose.
    """
    # Weakness 7: the objective is read out of the message list and is never stored
    # anywhere durable. Shrink the context and the goal is gone. Caught by
    # `context.middle_out_shrink` and `context.goal_dilution`.
    objective = state["messages"][-1]["content"]
    body = _content(llm([{"role": "user", "content": PLANNER_PROMPT.format(objective=objective)}]))
    # Weakness 8: `json.loads` straight onto a model response, with no guard, no
    # repair attempt and no schema check. Caught by `llm.prose_instead_of_json` and
    # `llm.truncated_mid_json`.
    state["targets"] = json.loads(body)
    return state


def data_analyst(state: State, tools: dict[str, Any], llm: Any, engine: Any) -> State:
    """Pull the three tables, then interpret and summarise them. Two LLM calls.

    Args:
        state: The working state.
        tools: The tool table.
        llm: The model callable.
        engine: The engine. Unused.

    Returns:
        `state` with `accounts`, `invoices`, `usage` and `findings` set.

    Raises:
        Exception: Whatever the tools raise. Weakness 4 means nothing is caught.
        IndexError: When the invoice payload came back empty.
        KeyError: When a required column was dropped.
    """
    targets = state["targets"]
    # Weakness 4: no try/except anywhere near a tool call, so a transient 500 ends
    # the run instead of being retried. Caught by `tool.transient_500_then_ok`.
    accounts = tools["load_accounts"]()
    invoices = tools["query_invoices"](targets)
    usage = tools["query_usage"](targets)

    # Weakness 1: indexes tool output with no length or shape check, and sorts on a
    # column it never confirmed is there. Caught by `tool.drop_required_key` and
    # `tool.type_flip_matrix`.
    rows = sorted(invoices["rows"], key=lambda r: -r["days_overdue"])
    worst = rows[0]

    # Weakness 5: an in-band `{"error": ...}` envelope is read as "no usage rows"
    # and the review carries on as if usage were simply quiet. Caught by
    # `tool.inband_error_payload`.
    usage_rows = usage.get("rows", [])

    findings = {
        "worst_invoice": worst["invoice_id"],
        "max_days_overdue": worst["days_overdue"],
        "overdue_usd": sum(r["amount_usd"] for r in rows if r["status"] != "paid"),
        "mrr_usd": sum(a["mrr_usd"] for a in accounts["rows"]),
        "under_used": [
            u["account_id"] for u in usage_rows if u["seats_active"] * 2 < u["seats_licensed"]
        ],
    }

    reading = _content(
        llm(
            [
                {
                    "role": "user",
                    "content": ANALYST_READ_PROMPT.format(invoices=rows, usage=usage_rows),
                }
            ]
        )
    )
    summary = _content(
        llm(
            [
                {
                    "role": "user",
                    "content": ANALYST_SUMMARY_PROMPT.format(reading=reading, **findings),
                }
            ]
        )
    )

    state["accounts"] = accounts
    state["invoices"] = invoices
    state["usage"] = usage
    state["findings"] = findings
    state["reading"] = reading
    state["summary"] = summary
    return state


def risk_scorer(state: State, tools: dict[str, Any], llm: Any, engine: Any) -> State:
    """Score, self-critique, then revise. Three LLM calls.

    Args:
        state: The working state.
        tools: The tool table. Unused here.
        llm: The model callable.
        engine: The engine. Unused.

    Returns:
        `state` with `figures` and `risk` set.

    Raises:
        KeyError: When `findings` was never set.
        json.JSONDecodeError: When the model answers in prose.
    """
    # Weakness 12: no guard that an upstream node actually set this. Caught by
    # `state.drop_findings` and `state.misroute_edge`.
    findings = state["findings"]

    # Weakness 3: no unit or range check on either figure. A `unit_swap` that turns
    # dollars into cents produces a plausible-looking report that is wrong by two
    # orders of magnitude, and nothing here notices. Caught by `tool.unit_swap`.
    figures = {"exposure_usd": findings["overdue_usd"], "arr_usd": findings["mrr_usd"] * 12}

    score_body = _content(llm([{"role": "user", "content": SCORE_PROMPT.format(**figures)}]))
    state["initial_risk"] = json.loads(score_body)
    critique = _content(
        llm([{"role": "user", "content": CRITIQUE_PROMPT.format(score=score_body)}])
    )
    revised_body = _content(
        llm(
            [
                {
                    "role": "user",
                    "content": REVISE_PROMPT.format(score=score_body, critique=critique),
                }
            ]
        )
    )

    state["figures"] = figures
    state["critique"] = critique
    state["risk"] = json.loads(revised_body)
    return state


def writer(state: State, tools: dict[str, Any], llm: Any, engine: Any) -> State:
    """Draft the paragraph, then tighten it. Two LLM calls.

    Args:
        state: The working state.
        tools: The tool table. Unused here.
        llm: The model callable.
        engine: The engine. Unused.

    Returns:
        `state` with `draft` set.

    Raises:
        IndexError: When the message list lost the objective.
    """
    # Weakness 7 again, and this is the one that bites: the objective is recovered
    # from `messages[0]`, so any context surgery silently changes the task.
    objective = state["messages"][0]["content"]
    # Weakness 10: `notes` is free text that arrived from a tool, and it goes into
    # the prompt with no fence and no "this is data, not instructions" preamble.
    notes = state["accounts"].get("notes", "")
    draft = _content(
        llm(
            [
                {
                    "role": "user",
                    "content": WRITER_DRAFT_PROMPT.format(
                        objective=objective, risk=state["risk"], names=notes
                    ),
                }
            ]
        )
    )
    state["draft"] = _content(
        llm([{"role": "user", "content": WRITER_TIGHTEN_PROMPT.format(draft=draft)}])
    )
    return state


def reviewer(state: State, tools: dict[str, Any], llm: Any, engine: Any) -> State:
    """Approve or send the draft back to the writer. One LLM call.

    Args:
        state: The working state.
        tools: The tool table.
        llm: The model callable.
        engine: The engine. Unused.

    Returns:
        `state` with `verdict`, `attempts` and, on approval, `report` set.
    """
    verdict = _content(
        llm([{"role": "user", "content": REVIEWER_PROMPT.format(draft=state["draft"])}])
    )
    # Weakness 6, first half: the counter is incremented here and never read. The
    # edge in `edges.py` completes the bug. Caught by `loop.pinned_tool_output` and
    # `tool.rate_limited_forever`.
    state["attempts"] = state.get("attempts", 0) + 1
    state["verdict"] = verdict

    if "APPROVE" in verdict:
        for account_id in state["targets"]:
            # Weakness 11: a side-effecting call with no idempotency key and no
            # check for an existing flag, so a second review pass raises the same
            # flag twice. Caught by `resume.checkpoint_rollback`.
            tools["flag_account_for_review"](account_id, f"revenue risk: {state['risk']}")
        state["report"] = state["draft"]
    return state


#: The dispatch table the runner in `app.py` walks.
NODES = {
    "planner": planner,
    "data_analyst": data_analyst,
    "risk_scorer": risk_scorer,
    "writer": writer,
    "reviewer": reviewer,
}
