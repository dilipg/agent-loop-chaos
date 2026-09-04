"""CSV-backed tools for the revenue-risk review.

Deterministic fakes over `data/*.csv`, so the demo runs with no API keys and no
network. Three read tools and one side-effecting write tool. Both trees import
these unchanged: the tools are not where the planted weaknesses live, the *callers*
are, and sharing them keeps the buggy/fixed diff to the agent code
(`docs/09-DEMO-AGENT.md` §5, "keep the diff small and legible").

The free-text `name` and `csm` columns are the injection carriers: a
`PromptInjectionFault` targeting `(tool, post)` rides in on one of them.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

__all__ = [
    "DATA_DIR",
    "FLAGGED",
    "TOOLS",
    "flag_account_for_review",
    "load_accounts",
    "query_invoices",
    "query_usage",
]

DATA_DIR = Path(__file__).resolve().parent / "data"

# Columns that are numbers in the domain even though CSV hands them over as text.
# Coercing here is what makes `unit_swap` and `type_flip` mean something downstream.
_NUMERIC = frozenset(
    {
        "mrr_usd",
        "amount_usd",
        "days_overdue",
        "seats_licensed",
        "seats_active",
        "api_calls",
        "support_tickets",
    }
)

#: The fake external system `flag_account_for_review` writes to. Tests clear it.
FLAGGED: list[dict[str, Any]] = []


def _read(table: str) -> list[dict[str, Any]]:
    """Load one CSV table with its numeric columns coerced.

    Args:
        table: The file stem under `data/`.

    Returns:
        One dict per row.
    """
    with (DATA_DIR / f"{table}.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [{k: int(v) if k in _NUMERIC else v for k, v in row.items()} for row in rows]


def load_accounts() -> dict[str, Any]:
    """Load the account book, with the CSM roster as a free-text note.

    `notes` is assembled from the CSV's free-text `name` and `csm` columns. It is
    the injection carrier: `PromptInjectionFault` writes its payload into exactly
    this kind of field, and the buggy tree interpolates it into a prompt with no
    fence (`docs/09-DEMO-AGENT.md` §3 and weakness 10).

    Returns:
        ``{"rows": [{account_id, name, tier, region, mrr_usd, csm, renewal_month}],
        "notes": str}``.
    """
    rows = _read("accounts")
    roster = "; ".join(f"{row['name']} - CSM {row['csm']}" for row in rows)
    return {"rows": rows, "notes": f"CSM roster for this book: {roster}"}


def query_invoices(account_ids: list[str]) -> dict[str, Any]:
    """Fetch the invoices for a set of accounts.

    Args:
        account_ids: Accounts to include. Empty means every account.

    Returns:
        ``{"rows": [{invoice_id, account_id, issued, due, amount_usd, status,
        days_overdue}]}``.
    """
    wanted = set(account_ids or ())
    rows = _read("invoices")
    return {"rows": [r for r in rows if not wanted or r["account_id"] in wanted]}


def query_usage(account_ids: list[str]) -> dict[str, Any]:
    """Fetch last month's product usage for a set of accounts.

    Args:
        account_ids: Accounts to include. Empty means every account.

    Returns:
        ``{"rows": [{account_id, month, seats_licensed, seats_active, api_calls,
        support_tickets}]}``.
    """
    wanted = set(account_ids or ())
    rows = _read("usage")
    return {"rows": [r for r in rows if not wanted or r["account_id"] in wanted]}


def flag_account_for_review(
    account_id: str, reason: str, idempotency_key: str | None = None
) -> dict[str, Any]:
    """Raise a CSM review flag on an account.

    This is the side-effecting tool: it writes to `FLAGGED`, which stands in for a
    real CRM. It is **not** idempotent unless the caller supplies a key, which is
    exactly what makes weakness 11 observable — the buggy tree omits it and a second
    review pass raises the same flag twice.

    Args:
        account_id: The account to flag.
        reason: Free text recorded on the flag.
        idempotency_key: When supplied and already seen, no new flag is written.

    Returns:
        ``{"status": "flagged"|"already_flagged", "flag_id": str}``.
    """
    if idempotency_key is not None:
        for existing in FLAGGED:
            if existing["idempotency_key"] == idempotency_key:
                return {"status": "already_flagged", "flag_id": existing["flag_id"]}
    flag = {
        "flag_id": f"FLG-{len(FLAGGED) + 1:03d}",
        "account_id": account_id,
        "reason": reason,
        "idempotency_key": idempotency_key,
    }
    FLAGGED.append(flag)
    return {"status": "flagged", "flag_id": flag["flag_id"]}


#: Name-to-callable, the shape `ChaosEngine.wrap_tools` takes.
TOOLS: dict[str, Any] = {
    "load_accounts": load_accounts,
    "query_invoices": query_invoices,
    "query_usage": query_usage,
    "flag_account_for_review": flag_account_for_review,
}

#: Tools that perform a real action, for the `SAFETY.md` §1 / D-23 gate.
SIDE_EFFECTING = frozenset({"flag_account_for_review"})
