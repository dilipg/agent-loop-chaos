"""An OpenAI-style function-calling loop: `tool_calls` in, `role: "tool"` back.

The shape almost every agent written after native tool calling shipped: the model
returns an assistant message carrying a `tool_calls` list, each entry naming a
function and an `arguments` **string**; the caller parses that string, dispatches,
appends one `{"role": "tool", "tool_call_id": ..., "content": ...}` message per
call, and loops until the model answers with content and no calls.

**The weakness in `build`.** Nothing crossing the tool boundary is checked, in
either direction. The arguments string goes straight into `json.loads` and the
parsed object straight into the call; the returned payload is serialised into a tool
message unread; and the summary step -- which recomputes the total from the payloads
rather than trusting the model's arithmetic, the *cautious*-looking choice -- reads
`amount_usd` out of each one without ever asking whether it is there. Every tool
here has a declared JSON Schema, registered with the engine, and the loop never
validates a single response against it. That is the bug: not a missing schema, an
unused one. A payload that comes back a key short takes the whole run down from
inside the line that was supposed to be the careful part.

`build_fixed` validates each result against the tool's own schema before it is used,
records `note()` and `validated()`, hands the model an explicit error envelope in
place of the unusable payload so it cannot quietly fill the gap, drops that invoice
from the total, and names it in the answer.

**The scripted model.** `scripted_model` is a plain function over the message list,
deterministic and offline. It is a mediocre one on purpose: told to report each
invoice's amount, it reports an amount even for a payload that no longer has one,
because inventing a plausible number is what a model does when a field it was asked
for is absent. It backs off only when it is told in so many words that the value
could not be retrieved -- which is precisely the difference the fix makes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import jsonschema

from . import PatternSpec, register

__all__ = [
    "MAX_STEPS",
    "SCHEMAS",
    "SPEC",
    "TOOLS",
    "build",
    "build_fixed",
    "get_invoice",
    "scripted_model",
]

#: Two model turns is the whole task; the cap stops a fault from hanging the loop.
MAX_STEPS = 6

QUESTION = "How much is outstanding on invoices INV-2041 and INV-2042?"

SYSTEM = (
    "You are a billing assistant. Look up every invoice the user names and report "
    "each one's amount in USD and how overdue it is."
)

_INVOICES: dict[str, dict[str, Any]] = {
    "INV-2041": {
        "invoice_id": "INV-2041",
        "customer": "Northwind Trading",
        "amount_usd": 4820,
        "status": "overdue",
        "days_overdue": 47,
    },
    "INV-2042": {
        "invoice_id": "INV-2042",
        "customer": "Northwind Trading",
        "amount_usd": 1290,
        "status": "overdue",
        "days_overdue": 12,
    },
}


def get_invoice(invoice_id: str) -> dict[str, Any]:
    """Fetch one invoice from the billing system.

    Args:
        invoice_id: The invoice to fetch.

    Returns:
        ``{"invoice_id", "customer", "amount_usd", "status", "days_overdue"}``.

    Raises:
        KeyError: When the invoice does not exist.
    """
    return dict(_INVOICES[invoice_id])


#: Name-to-callable. `get_invoice` only reads, so it registers with an explicit
#: ``side_effecting=False`` (`SAFETY.md` §1 item 3).
TOOLS: dict[str, Callable[..., Any]] = {"get_invoice": get_invoice}

#: The contract each tool's result is meant to honour. Registered with the engine
#: *and* kept here so `build_fixed` can enforce it with no engine attached.
SCHEMAS: dict[str, dict[str, Any]] = {
    "get_invoice": {
        "type": "object",
        "required": ["invoice_id", "amount_usd", "status", "days_overdue"],
        "properties": {
            "invoice_id": {"type": "string"},
            "customer": {"type": "string"},
            "amount_usd": {"type": "number"},
            "status": {"type": "string"},
            "days_overdue": {"type": "number"},
        },
    }
}


def _tool_payloads(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse every tool message's content back into an object.

    Args:
        messages: The conversation so far.

    Returns:
        One object per `role: "tool"` message, in order. A message whose content is
        not JSON becomes ``{"error": <the raw text>}``.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            parsed = json.loads(str(message.get("content", "")))
        except json.JSONDecodeError:
            parsed = {"error": str(message.get("content", ""))}
        out.append(parsed if isinstance(parsed, dict) else {"error": str(parsed)})
    return out


def scripted_model(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Answer as a mediocre function-calling model would, deterministically.

    Keyed on whether any tool result is already in the conversation: ask for the
    invoices, then summarise them. When a payload no longer carries `amount_usd` it
    reports a plausible figure regardless, because it was told to report an amount
    and nothing told it the value was missing. An explicit ``error`` envelope is the
    one thing that stops it.

    Args:
        messages: The conversation so far, OpenAI message shape.

    Returns:
        An assistant message: either `tool_calls`, or `content` with none.
    """
    payloads = _tool_payloads(messages)
    if not payloads:
        asked = " ".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        wanted = [key for key in _INVOICES if key in asked] or sorted(_INVOICES)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": "get_invoice",
                        "arguments": json.dumps({"invoice_id": invoice_id}),
                    },
                }
                for index, invoice_id in enumerate(wanted, start=1)
            ],
        }

    lines: list[str] = []
    for position, payload in enumerate(payloads, start=1):
        request = payload.get("request")
        asked_for = request.get("invoice_id") if isinstance(request, dict) else None
        label = str(payload.get("invoice_id") or asked_for or position)
        if payload.get("error"):
            lines.append(f"{label}: the amount could not be retrieved ({payload['error']}).")
            continue
        # Asked for an amount and handed a payload without one, it supplies one.
        amount = payload.get("amount_usd", 2500)
        overdue = payload.get("days_overdue", 30)
        lines.append(f"{label}: ${float(amount):,.2f}, {overdue} days overdue.")
    return {"role": "assistant", "content": " ".join(lines), "tool_calls": []}


def _wire(
    engine: Any,
) -> tuple[dict[str, Callable[..., Any]], Callable[[list[dict[str, Any]]], Any]]:
    """Bind the tools and the model to an engine, or hand back the raw callables.

    Args:
        engine: A `ChaosEngine`, or `None` to run uninstrumented.

    Returns:
        ``(tools, llm)``.
    """
    if engine is None:
        return dict(TOOLS), scripted_model
    tools = {
        name: engine.tool(fn, name=name, side_effecting=False, schema=SCHEMAS.get(name))
        for name, fn in TOOLS.items()
    }
    llm: Callable[[list[dict[str, Any]]], Any] = engine.llm(scripted_model, name="assistant")
    return tools, llm


def _calls_of(reply: Any) -> list[dict[str, Any]]:
    """Read the tool calls off an assistant message.

    Args:
        reply: Whatever the model returned.

    Returns:
        The calls, or an empty list when the reply is a plain answer.
    """
    if isinstance(reply, Mapping):
        calls = reply.get("tool_calls")
        if isinstance(calls, list):
            return [call for call in calls if isinstance(call, dict)]
    return []


def _answer(reply: Any) -> str:
    """Read the content off an assistant message.

    Args:
        reply: Whatever the model returned.

    Returns:
        The content as text.
    """
    if isinstance(reply, Mapping):
        return str(reply.get("content") or "")
    return str(reply or "")


def build(engine: Any = None) -> Callable[[str | None], str]:
    """Build the naive function-calling loop.

    Args:
        engine: A `ChaosEngine` to instrument against, or `None`.

    Returns:
        ``agent(question=None) -> str``.
    """
    tools, llm = _wire(engine)

    def agent(question: str | None = None) -> str:
        """Loop until the model answers with no tool calls.

        Args:
            question: The user's question. Defaults to `QUESTION`.

        Returns:
            The model's answer with the total appended.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question or QUESTION},
        ]
        for _ in range(MAX_STEPS):
            reply = llm(messages)
            messages.append(dict(reply) if isinstance(reply, Mapping) else {"content": reply})
            calls = _calls_of(reply)
            if not calls:
                total = sum(p["amount_usd"] for p in _tool_payloads(messages))
                return f"{_answer(reply)} Total outstanding: ${float(total):,.2f}."
            for call in calls:
                function = call["function"]
                arguments = json.loads(function["arguments"])
                result = tools[function["name"]](**arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, sort_keys=True),
                    }
                )
        return f"No answer after {MAX_STEPS} model turns."

    return agent


def _validate(name: str, arguments: Mapping[str, Any], result: Any) -> str | None:
    """Check a tool result against the tool's declared schema.

    Args:
        name: The tool that produced it.
        arguments: What it was called with, so the reason names the request.
        result: The payload it returned.

    Returns:
        A one-line reason the payload is unusable, or `None` when it is fine.
    """
    schema = SCHEMAS.get(name)
    if schema is None:
        return None
    try:
        jsonschema.validate(result, schema)
    except jsonschema.ValidationError as exc:
        path = "/".join(str(part) for part in exc.absolute_path) or "(root)"
        call = f"{name}({json.dumps(dict(arguments), sort_keys=True)})"
        return f"{call} returned a payload violating its schema at {path}: {exc.message}"
    return None


def build_fixed(engine: Any = None) -> Callable[[str | None], str]:
    """Build the hardened function-calling loop.

    Every result is checked against the tool's declared schema before it is used or
    shown to the model, and an unusable payload is reported rather than filled in.

    Args:
        engine: A `ChaosEngine` to instrument against, or `None`.

    Returns:
        ``agent(question=None) -> str``.
    """
    tools, llm = _wire(engine)

    def agent(question: str | None = None) -> str:
        """Loop with the tool boundary checked in both directions.

        Args:
            question: The user's question. Defaults to `QUESTION`.

        Returns:
            The model's answer, with the total over the payloads that were usable
            and a line naming anything that was not.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question or QUESTION},
        ]
        unusable: list[str] = []
        for _ in range(MAX_STEPS):
            reply = llm(messages)
            messages.append(dict(reply) if isinstance(reply, Mapping) else {"content": reply})
            calls = _calls_of(reply)
            if not calls:
                amounts = [p["amount_usd"] for p in _tool_payloads(messages) if "amount_usd" in p]
                answer = f"{_answer(reply)} Total outstanding across the invoices that "
                answer += f"returned an amount: ${float(sum(amounts)):,.2f}."
                if unusable:
                    answer += (
                        f" {len(unusable)} lookup(s) returned no usable amount and are excluded"
                        f" from that total: {'; '.join(unusable)}"
                    )
                return answer
            for call in calls:
                function = call.get("function") or {}
                name = str(function.get("name", ""))
                tool = tools.get(name)
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    arguments, tool = {}, None
                    reason: str | None = f"the model's arguments for {name!r} were not JSON: {exc}"
                else:
                    reason = None if tool is not None else f"no such tool: {name!r}"
                result = tool(**arguments) if tool is not None else None
                if reason is None:
                    reason = _validate(name, arguments, result)
                if reason is not None:
                    unusable.append(reason)
                    if engine is not None:
                        engine.note(f"degraded: {reason}")
                    content = json.dumps({"error": reason, "request": arguments}, sort_keys=True)
                else:
                    content = json.dumps(result, sort_keys=True)
                if engine is not None:
                    engine.validated(content, name=f"{name}_result")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", "call_?"),
                        "content": content,
                    }
                )
        return f"No answer after {MAX_STEPS} model turns."

    return agent


SPEC = register(
    PatternSpec(
        name="function_calling",
        description=(
            "An OpenAI-style tool-calling loop: a `tool_calls` list in, one "
            '`role: "tool"` message back per call, until the model answers.'
        ),
        build=build,
        build_fixed=build_fixed,
        weakness=(
            "Nothing crossing the tool boundary is checked: the arguments string is "
            "parsed unguarded and the result is used without validating it against "
            "the JSON Schema the tool was registered with."
        ),
        faults=(
            {
                "type": "ToolCorruptionFault",
                "params": {"mutation_type": "drop_key", "keys": ["amount_usd"]},
                "target": {"tool": "get_invoice"},
                "trigger": {"on_call": 1},
            },
        ),
        inputs=QUESTION,
        tools=("get_invoice",),
        llm_names=("assistant",),
        expected_behavior="graceful_degradation",
    )
)


if __name__ == "__main__":  # pragma: no cover - a hand check, not a test
    for label, builder in (("build", build), ("build_fixed", build_fixed)):
        out = builder(None)(QUESTION)
        assert "$6,110.00" in out, out
        print(f"{label}: {out}")
