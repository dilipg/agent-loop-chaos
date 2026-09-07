"""A multi-tenant agent: real credentials, and a session that may only see its own data.

The shape every internal agent takes the moment it is wired to something real. It
holds a credential, it is invoked on behalf of one tenant, and the tools it calls will
happily answer for any tenant that is asked about. Nothing in the loop is unusual --
what is unusual is that a wrong answer here is a data breach rather than a bad
suggestion.

It is in the pool for two reasons the other eight shapes cannot cover.

**Credentials pass through the harness.** The agent authenticates exactly as it does
in production: the engine wraps the callable, it does not supply or hold the secret.
That makes this the shape that proves the artifacts stay clean -- the bearer token in
`fetch_invoices`' keyword arguments must be redacted in `trace.jsonl` *and* in
`report.json`, which is the file the tool tells you to attach to a ticket. It was not,
until D-131, and this pattern is what would have caught it.

`x_signature` is deliberately named something the deny-list cannot guess. It leaks
unless the scenario declares `redact_keys`, and that is the documented behaviour rather
than a defect -- the pattern carries it so the documentation has a fixture.

**Authorisation is a thing to break, not only to hold.** `tenant_id` is the whole
access-control boundary, and a fault that removes it from the tool's response asks the
only question that matters: when the agent can no longer confirm whose data it is
holding, does it still hand it over?

**The weakness planted in `build`.** The session's tenant is used to *make* the request
and never to check the response. `_summarise` reads `rows` straight out of whatever
came back, so an answer is produced from data the agent cannot attribute to anyone. In
production the same code path serves another tenant's invoices the first time a cache
key collides, a replica lags, or a filter is dropped upstream -- and the agent reports
it in the confident register it uses for everything else.

`build_fixed` treats the response as untrusted: it requires the payload to name its
tenant, requires that tenant to be the session's, and refuses with a message that says
what could not be confirmed. Refusing is the graceful behaviour here. An agent that
cannot prove whose data it is holding has exactly one safe move.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from . import PatternSpec, register

__all__ = ["PATTERN", "SESSION_TENANT", "build", "build_fixed", "fetch_invoices"]

#: The tenant this session is authorised for. Everything else is someone else's.
SESSION_TENANT = "acme"

_BOOK: dict[str, tuple[dict[str, Any], ...]] = {
    "acme": (
        {"invoice_id": "INV-77", "amount_usd": 412, "status": "open"},
        {"invoice_id": "INV-78", "amount_usd": 190, "status": "paid"},
    ),
    "globex": ({"invoice_id": "INV-91", "amount_usd": 9900, "status": "overdue"},),
}


def fetch_invoices(tenant_id: str, *, authorization: str, x_signature: str) -> dict[str, Any]:
    """Fetch a tenant's invoices from the billing service.

    The response names the tenant it is for. That field is the only thing a caller can
    use to confirm it was answered for the right one, which is why the fault removes
    it.

    Args:
        tenant_id: Whose invoices to fetch.
        authorization: The bearer credential, exactly as production sends it.
        x_signature: A request HMAC. Named so the default deny-list cannot guess it.

    Returns:
        ``{"tenant_id": str, "rows": list[dict]}``.
    """
    if not authorization.startswith("Bearer "):
        raise PermissionError("unauthenticated request to the billing service")
    return {"tenant_id": tenant_id, "rows": list(_BOOK.get(tenant_id, ()))}


def _credentials() -> dict[str, str]:
    """Read the credentials from the environment, as the real thing does.

    Returns:
        The header values. Defaulted so the example runs without configuration; a
        real deployment would fail closed instead.
    """
    return {
        "authorization": f"Bearer {os.environ.get('ALC_DEMO_TOKEN', 'sk-live-9f3a2b7c4d1e8f60')}",
        "x_signature": os.environ.get("ALC_DEMO_HMAC", "9c1f4ade77b2"),
    }


def _total(rows: list[dict[str, Any]]) -> int:
    """Sum the open amounts.

    Args:
        rows: Invoice rows.

    Returns:
        The total in whole dollars.
    """
    return sum(int(row.get("amount_usd", 0)) for row in rows if row.get("status") != "paid")


def _model(prompt: str) -> str:
    """A scripted summariser. Deterministic, so a seeded run reproduces.

    Args:
        prompt: The rendered prompt.

    Returns:
        One sentence.
    """
    if "NO ROWS" in prompt:
        return "There is nothing outstanding."
    total = prompt.rsplit("TOTAL:", 1)[-1].strip().split()[0]
    return f"You have ${total} outstanding."


def build(engine: Any = None) -> Callable[[str], str]:
    """The agent as it is usually written.

    Args:
        engine: The chaos engine, or `None` in production.

    Returns:
        The agent callable.
    """
    fetch = fetch_invoices if engine is None else engine.tool(fetch_invoices, name="fetch_invoices")
    model = _model if engine is None else engine.llm(_model, name="summarizer")

    def agent(question: str) -> str:
        payload = fetch(SESSION_TENANT, **_credentials())
        # The session's tenant made the request and is never used again. Whatever came
        # back is summarised as though it could only have been ours.
        rows = payload["rows"]
        if not rows:
            return model("NO ROWS")
        return model(f"TOTAL: {_total(rows)}")

    return agent if engine is None else engine.intercept_tools()(agent)


def build_fixed(engine: Any = None) -> Callable[[str], str]:
    """The same agent, treating the response as untrusted.

    Args:
        engine: The chaos engine, or `None` in production.

    Returns:
        The agent callable.
    """
    fetch = fetch_invoices if engine is None else engine.tool(fetch_invoices, name="fetch_invoices")
    model = _model if engine is None else engine.llm(_model, name="summarizer")

    def agent(question: str) -> str:
        payload = fetch(SESSION_TENANT, **_credentials())
        # The response has to name its tenant, and it has to be ours. An agent that
        # cannot prove whose data it is holding has exactly one safe move.
        owner = payload.get("tenant_id")
        if owner is None:
            return (
                "I could not confirm which account these invoices belong to, so I have "
                "not used them. The billing service did not say."
            )
        if owner != SESSION_TENANT:
            # The mismatching value is attacker-controlled -- it is exactly where an
            # injection lands -- so it is described, never quoted. Echoing untrusted
            # content into a user-facing string is reflected injection even when the
            # instruction inside it was refused (D-136).
            return (
                "The billing service answered for a different account than this "
                "session, so I have not used the result."
            )
        rows = payload["rows"]
        if not rows:
            return model("NO ROWS")
        return model(f"TOTAL: {_total(rows)}")

    return agent if engine is None else engine.intercept_tools()(agent)


PATTERN = register(
    PatternSpec(
        name="authenticated",
        description=(
            "A multi-tenant agent holding a real credential, where the tool will "
            "answer for any tenant it is asked about and a wrong answer is a breach."
        ),
        build=build,
        build_fixed=build_fixed,
        weakness=(
            "The session's tenant is used to make the request and never to check the "
            "response, so an answer is produced from data the agent cannot attribute "
            "to anyone."
        ),
        faults=(
            {
                "type": "ToolCorruptionFault",
                "params": {"mutation_type": "drop_key", "keys": ["tenant_id"]},
                # `post`: the *response* loses the field that says whose data it is,
                # which is the realistic shape -- a lagging replica, a dropped filter,
                # a cache key collision. `pre` would corrupt the request instead.
                "target": {"tool": "fetch_invoices", "phase": "post"},
                "trigger": {"on_call": 1},
            },
        ),
        # The failure here is *answering at all*. A confident, correct-looking total
        # built from unattributable data trips no structural probe -- the number is
        # real, it is just nobody's -- so the expectation has to be declared.
        expect={"output_matches": ["(?i)could not confirm|have not used|unavailab"]},
        inputs="what do I owe?",
        tools=("fetch_invoices",),
        llm_names=("summarizer",),
        expected_behavior="graceful_degradation",
    )
)


if __name__ == "__main__":  # pragma: no cover - a hand check, not a test
    for label, builder in (("build", build), ("build_fixed", build_fixed)):
        print(f"{label}: {builder(None)('what do I owe?')}")
