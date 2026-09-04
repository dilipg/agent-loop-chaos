"""The five nodes again, with every weakness in `docs/09-DEMO-AGENT.md` §4 fixed.

Same graph, same tools, same nine LLM calls, same node signature. What changed:

- shapes are validated before anything indexes into them, and a successful check
  calls `engine.validated()` so the harness has positive evidence (§5);
- tool calls go through `call_tool`, which retries transients twice and never
  retries a 4xx;
- an in-band `{"error": ...}` envelope is detected before use;
- the objective lives in `state["objective"]` and is re-asserted in every prompt;
- `json.loads` is guarded with one repair attempt, then fails explicitly;
- `finish_reason` is checked before the body is used;
- untrusted CSV text is fenced;
- the back-edge is capped in `edges.py`;
- the side-effecting tool gets an idempotency key.

No node raises: each records what was unavailable in `state["degraded"]`, calls
`engine.note("degraded: …")`, and lets the writer say so in the report. That
sentence is the checkable meaning of `graceful_degradation` here, and it is what
the scenarios' `output_matches` assertions look for.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..revenue_review import tools as tools_module
from .validators import (
    DataUnavailable,
    ask,
    call_tool,
    fence,
    note,
    parse_json,
    validate_money,
    validate_rows,
    validated,
)

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

TOOLS = tools_module.TOOLS

# What a degraded run tells the *user*. Deliberately domain language, never the
# payload's own key names: the harness's auto `no_claim_about` assertion reads any
# mention of a field the fault removed as a claim about it (`docs/11` §4.4), so a
# report that echoes "rows" or "days_overdue" fails for describing its own gap.
# The technical reason still reaches the trace, through `engine.note()`.
_TABLE_LABEL = {
    "load_accounts": "the customer table",
    "query_invoices": "the billing table",
    "query_usage": "the product-usage table",
}

_ACCOUNT_COLUMNS = ("account_id", "name", "tier", "mrr_usd", "csm")
_INVOICE_COLUMNS = ("invoice_id", "account_id", "amount_usd", "status", "days_overdue")
_USAGE_COLUMNS = ("account_id", "seats_licensed", "seats_active")

# Plausible ceilings for this book of business. A `unit_swap` that turns dollars
# into cents lands outside them, which is the only way to catch it.
_MAX_EXPOSURE_USD = 5e8
_MAX_ARR_USD = 5e9

PLANNER_PROMPT = """[PLANNER]
You are opening a quarterly revenue-risk review.
Objective: {objective}
Reply with a JSON array of the account_ids worth investigating, and nothing else.
If you cannot decide, reply with an empty JSON array rather than prose."""

ANALYST_READ_PROMPT = """[ANALYST_READ]
Objective: {objective}
Interpret the invoice and usage rows below.
Some inputs may be unavailable. Where a figure is unavailable, say it is
unavailable and do not estimate it.
Invoices: {invoices}
Usage: {usage}
Unavailable: {degraded}"""

ANALYST_SUMMARY_PROMPT = """[ANALYST_SUMMARY]
Objective: {objective}
Summarise in two sentences, quoting only figures given here. If a figure is
unavailable, say so instead of estimating.
Worst invoice: {worst_invoice} at {max_days_overdue} days overdue.
Overdue exposure: {overdue_usd} USD across {under_used} under-used accounts.
Reading so far: {reading}
Unavailable: {degraded}"""

SCORE_PROMPT = """[SCORE]
Objective: {objective}
Score the collections risk for this quarter from 0 to 100.
Exposure: {exposure_usd} USD. Annualised revenue: {arr_usd} USD.
Reply with JSON only: {{"score": int, "band": str, "drivers": [str]}}"""

CRITIQUE_PROMPT = """[CRITIQUE]
Objective: {objective}
Critique your own score below in one sentence. What did it over- or under-weight?
{score}"""

REVISE_PROMPT = """[REVISE]
Objective: {objective}
Revise the score in light of the critique. Same JSON shape, JSON only.
Score: {score}
Critique: {critique}"""

WRITER_DRAFT_PROMPT = """[DRAFT]
Objective: {objective}
Risk: {risk}
Anything unavailable, which you must name in the paragraph: {degraded}
Draft the revenue-risk paragraph for the CSM team. Quote no figure that is not
given above.
The account names follow.
{names}"""

WRITER_TIGHTEN_PROMPT = """[TIGHTEN]
Objective: {objective}
Tighten this to one sentence, keeping any statement about unavailable data.
{draft}"""

REVIEWER_PROMPT = """[REVIEW]
Objective: {objective}
Reply APPROVE if the paragraph below is ready to send, otherwise REVISE.
{draft}"""


def _degraded(state: State) -> list[str]:
    """Get the run's degradation log, creating it on first use.

    Args:
        state: The working state.

    Returns:
        The mutable list of degradation reasons.
    """
    log: list[str] = state.setdefault("degraded", [])
    return log


def _objective(state: State) -> str:
    """Recover the objective from durable state, pinning it if it is not there yet.

    Weakness 7: never read from `messages`, which context surgery can rewrite.

    Args:
        state: The working state.

    Returns:
        The objective.
    """
    objective = state.get("objective")
    if not objective:
        messages = state.get("messages") or []
        objective = str(messages[-1].get("content", "")) if messages else ""
        state["objective"] = objective
    return str(objective)


def _fetch(
    tools: dict[str, Any],
    name: str,
    args: tuple[Any, ...],
    columns: tuple[str, ...],
    numeric: tuple[str, ...],
    state: State,
    engine: Any,
) -> dict[str, Any] | None:
    """Call one tool and validate what it returned.

    Args:
        tools: The tool table.
        name: Which tool.
        args: Positional arguments for it.
        columns: Columns every row must carry.
        numeric: Columns that must be non-negative numbers.
        state: The working state, for the degradation log.
        engine: The engine, for evidence and breadcrumbs.

    Returns:
        The whole validated payload -- rows plus any free-text `notes` the writer
        will need -- or `None` when the data could not be trusted.
    """
    try:
        payload = call_tool(tools[name], *args)
        rows = validate_rows(payload, name=name, required=columns, numeric=numeric)
    except DataUnavailable as exc:
        _degraded(state).append(f"{_TABLE_LABEL.get(name, name)} was unavailable")
        note(engine, f"degraded: {exc}")
        return None
    validated(engine, {"tool": name, "rows": len(rows)}, f"{name}_rows")
    return dict(payload)


def planner(state: State, tools: dict[str, Any], llm: Any, engine: Any) -> State:
    """Decide which accounts to investigate. One LLM call.

    Args:
        state: The working state.
        tools: The tool table. Unused here.
        llm: The model callable.
        engine: The engine, for evidence and breadcrumbs.

    Returns:
        `state` with `targets` set. An empty list means "every account".
    """
    objective = _objective(state)
    body = ask(llm, PLANNER_PROMPT.format(objective=objective))
    targets = parse_json(body) if body is not None else None
    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        _degraded(state).append(
            "the shortlist was unavailable, so every customer was reviewed instead"
        )
        note(engine, "degraded: planner returned no usable account list")
        targets = []
    else:
        validated(engine, targets, "planner_targets")
    state["targets"] = targets
    return state


def data_analyst(state: State, tools: dict[str, Any], llm: Any, engine: Any) -> State:
    """Pull the three tables, validate them, then interpret and summarise. Two calls.

    Args:
        state: The working state.
        tools: The tool table.
        llm: The model callable.
        engine: The engine.

    Returns:
        `state` with `findings` set, or with the reason it could not be.
    """
    objective = _objective(state)
    targets = state.get("targets") or []
    log = _degraded(state)

    accounts = _fetch(tools, "load_accounts", (), _ACCOUNT_COLUMNS, ("mrr_usd",), state, engine)
    invoices = _fetch(
        tools,
        "query_invoices",
        (targets,),
        _INVOICE_COLUMNS,
        ("amount_usd", "days_overdue"),
        state,
        engine,
    )
    usage = _fetch(
        tools,
        "query_usage",
        (targets,),
        _USAGE_COLUMNS,
        ("seats_licensed", "seats_active"),
        state,
        engine,
    )

    findings: dict[str, Any] | None = None
    if invoices is not None and accounts is not None:
        ranked = sorted(invoices["rows"], key=lambda r: -r["days_overdue"])
        findings = {
            "worst_invoice": ranked[0]["invoice_id"],
            "max_days_overdue": ranked[0]["days_overdue"],
            "overdue_usd": sum(r["amount_usd"] for r in ranked if r["status"] != "paid"),
            "mrr_usd": sum(a["mrr_usd"] for a in accounts["rows"]),
            "under_used": [
                u["account_id"]
                for u in (usage or {}).get("rows", [])
                if u["seats_active"] * 2 < u["seats_licensed"]
            ],
        }

    reading = ask(
        llm,
        ANALYST_READ_PROMPT.format(
            objective=objective, invoices=invoices, usage=usage, degraded=log or "nothing"
        ),
    )
    summary = ask(
        llm,
        ANALYST_SUMMARY_PROMPT.format(
            objective=objective,
            reading=reading,
            degraded=log or "nothing",
            **(
                findings
                or {
                    "worst_invoice": "unavailable",
                    "max_days_overdue": "unavailable",
                    "overdue_usd": "unavailable",
                    "under_used": "unavailable",
                }
            ),
        ),
    )

    state["accounts"] = accounts or {"rows": []}
    state["invoices"] = invoices or {"rows": []}
    state["usage"] = usage or {"rows": []}
    state["findings"] = findings
    state["reading"] = reading
    state["summary"] = summary
    return state


def risk_scorer(state: State, tools: dict[str, Any], llm: Any, engine: Any) -> State:
    """Score, self-critique, then revise. Three LLM calls when the inputs hold up.

    Args:
        state: The working state.
        tools: The tool table. Unused here.
        llm: The model callable.
        engine: The engine.

    Returns:
        `state` with `risk` set, or with an explicitly unknown score.
    """
    objective = _objective(state)
    unknown = {"score": None, "band": "unknown", "drivers": []}
    findings = state.get("findings")

    if not isinstance(findings, dict):
        _degraded(state).append("the risk rating was unavailable")
        note(engine, "degraded: risk_scorer had no findings to score")
        state["risk"] = unknown
        return state

    try:
        figures = {
            "exposure_usd": validate_money(
                findings.get("overdue_usd"), "overdue exposure", high=_MAX_EXPOSURE_USD
            ),
            "arr_usd": validate_money(
                validate_money(findings.get("mrr_usd"), "MRR", high=_MAX_ARR_USD / 12) * 12,
                "annualised revenue",
                high=_MAX_ARR_USD,
            ),
        }
    except DataUnavailable as exc:
        _degraded(state).append(
            "the risk rating was unavailable because a figure was outside its plausible range"
        )
        note(engine, f"degraded: {exc}")
        state["risk"] = unknown
        return state
    validated(engine, figures, "risk_figures")
    state["figures"] = figures

    score_body = ask(llm, SCORE_PROMPT.format(objective=objective, **figures))
    critique = ask(llm, CRITIQUE_PROMPT.format(objective=objective, score=score_body))
    revised_body = ask(
        llm, REVISE_PROMPT.format(objective=objective, score=score_body, critique=critique)
    )

    revised = parse_json(revised_body) if revised_body is not None else None
    if not isinstance(revised, dict) or not isinstance(revised.get("score"), (int, float)):
        _degraded(state).append("the risk rating was unavailable")
        note(engine, "degraded: risk_scorer could not parse a score")
        revised = unknown
    else:
        validated(engine, revised, "risk_score")

    state["critique"] = critique
    state["risk"] = revised
    return state


def writer(state: State, tools: dict[str, Any], llm: Any, engine: Any) -> State:
    """Draft the paragraph, then tighten it. Two LLM calls.

    Args:
        state: The working state.
        tools: The tool table. Unused here.
        llm: The model callable.
        engine: The engine.

    Returns:
        `state` with `draft` set, and `previous_draft` kept for the no-progress check.
    """
    objective = _objective(state)
    log = _degraded(state)
    # Weakness 10: the same untrusted `notes` field the buggy tree pastes in raw,
    # length-capped, stripped of imperative lines, and labelled as data.
    names = fence((state.get("accounts") or {}).get("notes") or "none listed")

    draft = ask(
        llm,
        WRITER_DRAFT_PROMPT.format(
            objective=objective,
            risk=state.get("risk"),
            names=names,
            degraded=log or "nothing",
        ),
    )
    tightened = (
        ask(llm, WRITER_TIGHTEN_PROMPT.format(objective=objective, draft=draft))
        if draft is not None
        # Still spend the second call, so the fixed tree's call profile matches the
        # buggy one and a scenario compares like with like.
        else ask(llm, WRITER_TIGHTEN_PROMPT.format(objective=objective, draft=""))
    )

    state["previous_draft"] = state.get("draft")
    if tightened is None:
        _degraded(state).append("the report paragraph was unavailable")
        note(engine, "degraded: writer produced no usable draft")
        state["draft"] = ""
    else:
        state["draft"] = tightened
    return state


def reviewer(state: State, tools: dict[str, Any], llm: Any, engine: Any) -> State:
    """Approve or send the draft back to the writer. One LLM call.

    Args:
        state: The working state.
        tools: The tool table.
        llm: The model callable.
        engine: The engine.

    Returns:
        `state` with `verdict`, `attempts` and, on approval, `report` set.
    """
    objective = _objective(state)
    verdict = ask(llm, REVIEWER_PROMPT.format(objective=objective, draft=state.get("draft", "")))
    state["attempts"] = int(state.get("attempts", 0)) + 1
    state["verdict"] = verdict or "REVISE"

    if verdict is None or "APPROVE" not in verdict:
        return state

    # Weakness 11: the key is derived from the objective and the account, so a
    # second review pass -- or a checkpoint rollback -- re-sends the same flag
    # rather than raising a second one.
    thread = hashlib.sha256(objective.encode("utf-8")).hexdigest()[:8]
    for account_id in state.get("targets") or []:
        outcome = call_tool(
            tools["flag_account_for_review"],
            account_id,
            f"revenue risk: {state.get('risk')}",
            idempotency_key=f"{thread}:{account_id}",
        )
        validated(engine, outcome, "flag_idempotency")
    state["report"] = state.get("draft", "")
    return state


#: The dispatch table the runner in `app.py` walks.
NODES = {
    "planner": planner,
    "data_analyst": data_analyst,
    "risk_scorer": risk_scorer,
    "writer": writer,
    "reviewer": reviewer,
}
