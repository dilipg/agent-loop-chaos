"""An object-oriented agent: conversation, tool registry and step budget on `self`.

The shape a service class takes once an agent grows past one function. Everything
the loop needs is instance state -- `self.messages`, `self.tools`, `self.facts`,
`self.steps_left` -- so instrumentation has to attach to *bound methods on an
instance*, which is what `ChaosEngine.instrument_object` is for. Nothing here is a
module-level function the engine could have decorated.

The weakness in `build`: `absorb` merges every tool result into `self.facts` and
nothing ever re-validates that state. `facts` is cumulative and the fields it is
supposed to carry are never checked against what actually arrived, so a field a tool
quietly stopped returning does not raise anywhere -- it is simply absent from the
context `_prompt` hands the model two steps later. The model, told to report a
balance it was not given, invents one, and the invented figure is what reaches the
customer.

`build_fixed` gives `absorb` the set of fields each tool promised, records the gap
on `self.gaps` instead of merging blindly, marks the check with `engine.validated`,
and when a required field never arrived it writes the sentence itself rather than
asking a model to fill the hole.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from . import PatternSpec, register

__all__ = ["PATTERN", "TriageAgent", "build", "build_fixed"]

_DEFAULT_TICKET = "T-2291"
#: Bounds the loop: a fault must never be able to hang the suite.
_STEP_BUDGET = 4
_TOOL_NAMES = ("fetch_ticket", "fetch_billing")
#: The fields `fetch_billing` is contracted to return.
_BILLING_FIELDS = frozenset({"balance_usd", "days_overdue"})

_TICKETS: dict[str, dict[str, Any]] = {
    "T-2291": {"account_id": "ACME-4", "subject": "invoice dispute", "priority": "high"},
    "T-2310": {"account_id": "BOLT-9", "subject": "seat upgrade", "priority": "normal"},
}
_BILLING: dict[str, dict[str, Any]] = {
    "ACME-4": {"balance_usd": 1240, "days_overdue": 12, "plan": "growth"},
    "BOLT-9": {"balance_usd": 310, "days_overdue": 3, "plan": "starter"},
}

_PROMPT = """[TRIAGE]
Ticket: {ticket}
FACTS: {facts}
Reply `CALL <tool>` for one of {tools}, or `ANSWER <one sentence for the
customer, quoting balance_usd and days_overdue>`."""

#: What the scripted model says when the facts do not carry a balance. A real
#: mediocre model does this, and it is the whole reason the grounding assertion
#: exists.
_INVENTED_BALANCE = 1875


class TriageAgent:
    """A billing-triage agent holding its whole loop on `self`.

    Attributes:
        messages: The running conversation.
        tools: Name-to-callable registry, rebuilt at `run` time so it picks up the
            instrumented bound methods.
        facts: What the tools have returned so far, merged.
        gaps: Fields a tool promised and did not deliver. Only the fixed tree fills
            this in.
        steps_left: The remaining step budget.
        turns: How many model calls have been made.
        strict: Whether `absorb` validates what it is given.
        engine: The `ChaosEngine`, or `None`.
    """

    def __init__(self, engine: Any = None, *, strict: bool = False) -> None:
        """Build an agent.

        Args:
            engine: A `ChaosEngine` for `note`/`validated`, or `None`.
            strict: `True` to validate tool results before merging them.
        """
        self.messages: list[dict[str, str]] = []
        self.tools: dict[str, Callable[[str], dict[str, Any]]] = {}
        self.facts: dict[str, Any] = {}
        self.gaps: list[str] = []
        self.steps_left = _STEP_BUDGET
        self.turns = 0
        self.strict = strict
        self.engine = engine

    # ------------------------------------------------------------------- tools

    def fetch_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Load a support ticket. Read-only.

        Args:
            ticket_id: The ticket key.

        Returns:
            ``{"ticket_id", "account_id", "subject", "priority"}``.

        Raises:
            KeyError: When the ticket is unknown.
        """
        return {"ticket_id": ticket_id, **_TICKETS[ticket_id]}

    def fetch_billing(self, account_id: str) -> dict[str, Any]:
        """Load an account's billing position. Read-only.

        Args:
            account_id: The account key.

        Returns:
            ``{"account_id", "balance_usd", "days_overdue", "plan"}``.

        Raises:
            KeyError: When the account is unknown.
        """
        return {"account_id": account_id, **_BILLING[account_id]}

    # ------------------------------------------------------------------- model

    def decide(self, prompt: str) -> str:
        """Answer one prompt, as a mediocre scripted model would.

        Keyed on `self.turns` for the routing decisions and on the prompt's `FACTS`
        line for the answer. No network, no clock, no randomness. Asked for a
        balance the facts do not carry, it invents one rather than admitting it is
        missing.

        Args:
            prompt: The rendered prompt.

        Returns:
            ``CALL <tool>`` or ``ANSWER <sentence>``.
        """
        self.turns += 1
        if self.turns <= len(_TOOL_NAMES):
            return f"CALL {_TOOL_NAMES[self.turns - 1]}"
        facts = json.loads(prompt.split("FACTS: ", 1)[1].splitlines()[0])
        balance = facts.get("balance_usd", _INVENTED_BALANCE)
        return (
            f"ANSWER Ticket {facts.get('ticket_id', '?')} ({facts.get('account_id', '?')}): "
            f"balance_usd: {balance} USD, days_overdue: {facts.get('days_overdue', '?')}. "
            f"Please settle the outstanding invoice."
        )

    # -------------------------------------------------------------------- loop

    def absorb(self, result: Mapping[str, Any], promised: Iterable[str] = ()) -> None:
        """Merge a tool result into the accumulated facts.

        Args:
            result: What the tool returned.
            promised: Fields the tool is contracted to return. Honoured only when
                `strict` is set.
        """
        if self.strict:
            missing = sorted(set(promised) - set(result))
            self.gaps.extend(missing)
            if self.engine is not None:
                self.engine.validated(
                    {"fields": sorted(result), "missing": missing}, name="tool_result_fields"
                )
        self.facts.update(result)

    def _prompt(self, ticket: str) -> str:
        """Render the next prompt from the state on `self`.

        Args:
            ticket: The ticket under triage.

        Returns:
            The rendered prompt.
        """
        return _PROMPT.format(
            ticket=ticket,
            facts=json.dumps(self.facts, sort_keys=True),
            tools=", ".join(_TOOL_NAMES),
        )

    def _degraded(self, ticket: str) -> str:
        """Write the answer the agent can actually support.

        Args:
            ticket: The ticket under triage.

        Returns:
            A sentence naming every field that never arrived.
        """
        missing = ", ".join(sorted(set(self.gaps)))
        if self.engine is not None:
            self.engine.note(f"degraded: {missing} never arrived from fetch_billing")
        overdue = self.facts.get("days_overdue")
        tail = f" days_overdue: {overdue}." if overdue is not None else ""
        return (
            f"Ticket {ticket} ({self.facts.get('account_id', 'unknown account')}): "
            f"{missing} is unavailable -- fetch_billing did not return it -- so I have "
            f"not quoted a figure.{tail}"
        )

    def run(self, ticket_id: str | None = None) -> str:
        """Triage one ticket.

        Args:
            ticket_id: The ticket key. Defaults to `_DEFAULT_TICKET`.

        Returns:
            One sentence for the customer.
        """
        ticket = ticket_id or _DEFAULT_TICKET
        self.tools = {name: getattr(self, name) for name in _TOOL_NAMES}
        self.messages.append({"role": "user", "content": f"triage {ticket}"})
        while self.steps_left > 0:
            self.steps_left -= 1
            if self.strict and self.gaps:
                return self._degraded(ticket)
            reply = self.decide(self._prompt(ticket))
            self.messages.append({"role": "assistant", "content": reply})
            if reply.startswith("ANSWER "):
                return reply[len("ANSWER ") :]
            name = reply[len("CALL ") :].strip()
            if name == "fetch_billing":
                self.absorb(self.tools[name](str(self.facts["account_id"])), _BILLING_FIELDS)
            else:
                self.absorb(self.tools[name](ticket))
        return f"Ticket {ticket}: triage did not finish within {_STEP_BUDGET} steps."


def _wire(engine: Any, *, strict: bool) -> TriageAgent:
    """Build an agent and attach the engine to its methods.

    Args:
        engine: A `ChaosEngine`, or `None` to run uninstrumented.
        strict: Passed to `TriageAgent`.

    Returns:
        The instance, instrumented when an engine was supplied.
    """
    agent = TriageAgent(engine, strict=strict)
    if engine is None:
        return agent
    # `instrument_object` takes no `side_effecting` flag, and a tool must never be
    # left undeclared (SAFETY.md §1 item 3). Declaring first works because the
    # wrapper registration underneath uses `setdefault`.
    for name in _TOOL_NAMES:
        engine.tool(getattr(agent, name), name=name, side_effecting=False)
    engine.instrument_object(agent, tools=_TOOL_NAMES, llm_methods=("decide",))
    return agent


def build(engine: Any = None) -> Callable[..., str]:
    """Build the naive class-based agent.

    Args:
        engine: A `ChaosEngine` to instrument against, or `None`.

    Returns:
        The instance's bound `run`.
    """
    return _wire(engine, strict=False).run


def build_fixed(engine: Any = None) -> Callable[..., str]:
    """Build the hardened class-based agent.

    Args:
        engine: A `ChaosEngine` to instrument against, or `None`.

    Returns:
        The instance's bound `run`, validating every tool result before it becomes
        accumulated state.
    """
    return _wire(engine, strict=True).run


PATTERN = register(
    PatternSpec(
        name="class_based",
        description="An agent class carrying conversation, tool registry and budget on `self`.",
        build=build,
        build_fixed=build_fixed,
        weakness=(
            "tool results are merged into `self.facts` and never re-validated, so a "
            "field a tool stops returning is silently absent from the context two "
            "steps later and the model invents a value for it"
        ),
        faults=(
            {
                "type": "ToolCorruptionFault",
                "params": {"mutation_type": "drop_key", "keys": ["balance_usd"]},
                "target": {"tool": "fetch_billing"},
                "trigger": {"on_call": 1},
            },
        ),
        inputs=_DEFAULT_TICKET,
        tools=_TOOL_NAMES,
        llm_names=("decide",),
        expected_behavior="graceful_degradation",
    )
)


if __name__ == "__main__":
    for builder in (build, build_fixed):
        answer = builder(None)(_DEFAULT_TICKET)
        assert "balance_usd: 1240 USD" in answer, answer
        assert "days_overdue: 12" in answer, answer
    print("ok")
