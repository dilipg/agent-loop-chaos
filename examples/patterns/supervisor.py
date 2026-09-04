"""A supervisor delegating to named specialist sub-agents.

The LangGraph-supervisor / AutoGen / CrewAI shape: a coordinator model picks which
named specialist takes the next turn, each specialist runs with its **own** system
prompt, its own model alias and its own subset of the tools, whatever it produces
accumulates in one shared scratchpad, and the coordinator decides when the work is
done. Each specialist's turn goes through `engine.route_node` and the coordinator's
choice through `engine.route_edge`, so a scenario can target one sub-agent's model
(`researcher`, `analyst`, `writer`) or the delegation edge itself.

**The weakness planted in `build`.** The coordinator's control flow is driven
entirely by free text. Each specialist appends a one-line claim -- "researcher: DONE
on turn 2 ..." -- and the supervisor prompt is rendered from those claims and
nothing else; the scratchpad the specialists are supposedly filling is never
consulted. So "this stage is finished" and "this specialist said it was finished"
are the same proposition, a specialist that ran and produced nothing is
indistinguishable from one that succeeded, and each specialist in turn indexes
straight into its predecessor's slot on the strength of the running order. Deliver a
specialist out of order -- which is all a misrouted delegation edge does -- and the
writer drafts a brief with a seat figure the model invented to fill the empty slot,
then reads a slot that was never written.

`build_fixed` is the same three specialists, but the supervisor renders its prompt
from the scratchpad slots that are actually populated rather than from claims, and
every specialist checks the state it was entered with, calls `engine.note` when the
check fails, records what was unavailable and returns without writing. Nothing is
invented from an empty slot, and the final brief says what was missing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from . import PatternSpec, register

__all__ = ["PATTERN", "build", "build_fixed", "search_tickets", "usage_stats"]

#: Delegation order, and the scratchpad slot each specialist owns.
SPECIALIST_ORDER: tuple[str, ...] = ("researcher", "analyst", "writer")
SLOT: dict[str, str] = {"researcher": "research", "analyst": "analysis", "writer": "brief"}

#: The coordinator's terminal choice.
DONE = "DONE"

#: Hard cap on delegations, so a misroute or a stuck coordinator cannot hang a test
#: run. Four turns is the happy path; the rest is head-room for one recovery.
MAX_DELEGATIONS = 8

_TICKETS: tuple[dict[str, Any], ...] = (
    {
        "ticket_id": "T-9001",
        "account": "Northwind",
        "theme": "onboarding",
        "quote": "nobody on my team knows how to set up a workspace",
    },
    {
        "ticket_id": "T-9002",
        "account": "Northwind",
        "theme": "reporting",
        "quote": "the weekly export still has to be done by hand",
    },
    {
        "ticket_id": "T-9003",
        "account": "Contoso",
        "theme": "onboarding",
        "quote": "our second cohort never got their invites",
    },
)

_SEATS: tuple[dict[str, Any], ...] = (
    {"account": "Northwind", "seats_licensed": 40, "seats_active": 12},
    {"account": "Contoso", "seats_licensed": 25, "seats_active": 21},
)


def search_tickets(objective: str) -> dict[str, Any]:
    """Search the support archive. The researcher's only tool.

    Args:
        objective: The review objective. The fixture is small enough that every
            ticket is relevant, so the objective is recorded rather than filtered on.

    Returns:
        ``{"objective": str, "rows": [{ticket_id, account, theme, quote}]}``.
    """
    return {"objective": objective, "rows": [dict(row) for row in _TICKETS]}


def usage_stats(accounts: list[str]) -> dict[str, Any]:
    """Fetch seat activity. The analyst's only tool.

    Args:
        accounts: Accounts to include. Empty means every account.

    Returns:
        ``{"rows": [{account, seats_licensed, seats_active}]}``.
    """
    wanted = set(accounts or ())
    return {"rows": [dict(r) for r in _SEATS if not wanted or r["account"] in wanted]}


# ------------------------------------------------------------------- the model

_TAG = re.compile(r"\[([A-Z_]+)\]")
_FIELD = re.compile(r"^(\w+)=(.*)$", re.MULTILINE)
_NEXT = re.compile(r"NEXT:\s*(\w+)")

#: What the writer's model says when it is handed an empty `utilisation=` slot. A
#: mediocre model fills the gap from what a number like that usually looks like
#: rather than saying it was never given one.
FABRICATED_UTILISATION = "68% of licensed seats"

_SUPERVISE_PROMPT = (
    "[SUPERVISE]\nobjective={objective}\ncompleted={completed}\n"
    "Reply with `NEXT: <researcher|analyst|writer|DONE>` and nothing else."
)
_RESEARCH_PROMPT = (
    "[RESEARCH]\nYou are the researcher. Read support tickets and report themes only.\n"
    "objective={objective}\ntickets={tickets}\nReport what you found in one line."
)
_ANALYSE_PROMPT = (
    "[ANALYSE]\nYou are the analyst. Turn seat counts into a utilisation picture.\n"
    "objective={objective}\nthemes={themes}\nseats={seats}\nReport it in one line."
)
_WRITE_PROMPT = (
    "[WRITE]\nYou are the writer. Write the churn brief for the CSM team.\n"
    "objective={objective}\nthemes={themes}\nutilisation={utilisation}\n"
    "First line is your status claim; the rest is the brief."
)


def _scripted_model() -> Callable[..., dict[str, Any]]:
    """Build this pattern's stand-in for a mediocre model.

    Routes on the ``[STAGE]`` marker and the ``key=value`` lines each prompt carries,
    with a per-instance turn counter that lands in the specialists' claims -- which
    is exactly the free text the naive supervisor routes on. No clock, no randomness,
    no network.

    Returns:
        ``respond(prompt) -> {"content": str, "finish_reason": "stop"}``.
    """
    turns = [0]

    def respond(prompt: Any, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """Answer one prompt.

        Args:
            prompt: The rendered prompt.
            *_args: Ignored, so the signature matches a real client.
            **_kwargs: Ignored.

        Returns:
            An OpenAI-shaped envelope.
        """
        turns[0] += 1
        text = str(prompt)
        fields = dict(_FIELD.findall(text))
        found = _TAG.search(text)
        stage = found.group(1) if found else ""
        if stage == "SUPERVISE":
            completed = fields.get("completed", "")
            choice = next((n for n in SPECIALIST_ORDER if n not in completed), DONE)
            body = f"NEXT: {choice}"
        elif stage == "RESEARCH":
            body = f"researcher: DONE on turn {turns[0]} - read the ticket archive."
        elif stage == "ANALYSE":
            body = f"analyst: DONE on turn {turns[0]} - summarised seat activity."
        elif stage == "WRITE":
            themes = fields.get("themes", "").strip() or "the open themes"
            utilisation = fields.get("utilisation", "").strip() or FABRICATED_UTILISATION
            body = (
                f"writer: DONE on turn {turns[0]} - brief ready.\n"
                f"Churn brief: {themes} are the live risks across the book, "
                f"and seat activity stands at {utilisation}."
            )
        else:
            body = f"OK (turn {turns[0]})."
        return {"content": body, "finish_reason": "stop"}

    return respond


def _content(response: Any) -> str:
    """Pull the body out of a model response.

    Args:
        response: Whatever the model returned.

    Returns:
        The body as text.
    """
    if isinstance(response, dict):
        return str(response.get("content", ""))
    return str(response)


def _choice(reply: str) -> str:
    """Read the coordinator's pick out of its reply.

    Args:
        reply: The supervisor model's body.

    Returns:
        A specialist name, or `DONE` when the reply names none.
    """
    found = _NEXT.search(reply)
    name = found.group(1) if found else DONE
    return name if name in SPECIALIST_ORDER else DONE


def _split_claim(body: str) -> tuple[str, str]:
    """Separate the writer's status claim from the brief itself.

    Args:
        body: The writer model's reply.

    Returns:
        ``(claim, brief)``.
    """
    lines = body.splitlines()
    return (lines[0] if lines else ""), "\n".join(lines[1:]).strip()


def _wire(engine: Any) -> tuple[dict[str, Callable[..., Any]], dict[str, Callable[..., Any]]]:
    """Register the tools and one model alias per sub-agent.

    Args:
        engine: A `ChaosEngine`, or `None` to run uninstrumented.

    Returns:
        ``(tools, models)`` keyed by tool name and by sub-agent alias.
    """
    model = _scripted_model()
    tools: dict[str, Callable[..., Any]] = {
        "search_tickets": search_tickets,
        "usage_stats": usage_stats,
    }
    aliases = ("supervisor", "researcher", "analyst", "writer")
    if engine is None:
        return tools, dict.fromkeys(aliases, model)
    wrapped = {name: engine.tool(fn, name=name, side_effecting=False) for name, fn in tools.items()}
    return wrapped, {alias: engine.llm(model, name=alias) for alias in aliases}


def _dispatch(
    engine: Any,
    pick: Callable[[dict[str, Any]], str],
    specialists: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]],
    board: dict[str, Any],
) -> dict[str, Any]:
    """Walk the delegation loop until the coordinator says stop.

    Args:
        engine: The engine, or `None`.
        pick: The coordinator's routing function.
        specialists: Sub-agent name to node callable.
        board: The shared scratchpad.

    Returns:
        The scratchpad after the last delegation.
    """
    for _ in range(MAX_DELEGATIONS):
        if engine is None:
            choice = pick(board)
        else:
            choice = str(engine.route_edge(pick, name="supervisor", state=board))
        node = specialists.get(choice)
        if node is None:
            break
        board = node(board) if engine is None else engine.route_node(node, name=choice, state=board)
    return board


# ----------------------------------------------------------- the naive supervisor


def build(engine: Any = None) -> Callable[..., str]:
    """Build the naive supervisor and its three specialists.

    Args:
        engine: A `ChaosEngine` to instrument against, or `None` to run bare.

    Returns:
        ``agent(objective, state=None) -> str``.
    """
    tools, models = _wire(engine)

    def pick(board: dict[str, Any]) -> str:
        """Ask the coordinator who goes next.

        Args:
            board: The shared scratchpad.

        Returns:
            A specialist name, or `DONE`.
        """
        completed = " | ".join(board.get("claims", []))
        reply = _content(
            models["supervisor"](
                _SUPERVISE_PROMPT.format(objective=board["objective"], completed=completed)
            )
        )
        return _choice(reply)

    def researcher(board: dict[str, Any]) -> dict[str, Any]:
        """Read the ticket archive and record the themes.

        Args:
            board: The shared scratchpad.

        Returns:
            The scratchpad with `research` and a new claim.
        """
        rows = tools["search_tickets"](board["objective"])["rows"]
        claim = _content(
            models["researcher"](
                _RESEARCH_PROMPT.format(objective=board["objective"], tickets=rows)
            )
        )
        board["research"] = {
            "themes": sorted({row["theme"] for row in rows}),
            "accounts": sorted({row["account"] for row in rows}),
        }
        board["claims"].append(claim)
        return board

    def analyst(board: dict[str, Any]) -> dict[str, Any]:
        """Turn seat counts into a utilisation line.

        Args:
            board: The shared scratchpad.

        Returns:
            The scratchpad with `analysis` and a new claim.

        Raises:
            KeyError: When the researcher's slot was never written.
        """
        research = board["research"]
        rows = tools["usage_stats"](research["accounts"])["rows"]
        worst = min(rows, key=lambda row: row["seats_active"] / row["seats_licensed"])
        claim = _content(
            models["analyst"](
                _ANALYSE_PROMPT.format(
                    objective=board["objective"], themes=research["themes"], seats=rows
                )
            )
        )
        board["analysis"] = {
            "account": worst["account"],
            "seats_active": worst["seats_active"],
            "seats_licensed": worst["seats_licensed"],
        }
        board["claims"].append(claim)
        return board

    def writer(board: dict[str, Any]) -> dict[str, Any]:
        """Write the churn brief from what the other two left behind.

        Args:
            board: The shared scratchpad.

        Returns:
            The scratchpad with `brief`, `watchlist` and a new claim.

        Raises:
            KeyError: When an upstream specialist's slot was never written.
        """
        research = board.get("research") or {}
        analysis = board.get("analysis") or {}
        utilisation = (
            f"{analysis['seats_active']} of {analysis['seats_licensed']} seats "
            f"on {analysis['account']}"
            if analysis
            else ""
        )
        body = _content(
            models["writer"](
                _WRITE_PROMPT.format(
                    objective=board["objective"],
                    themes=", ".join(research.get("themes", ())),
                    utilisation=utilisation,
                )
            )
        )
        claim, brief = _split_claim(body)
        board["brief"] = brief
        board["watchlist"] = [board["analysis"]["account"]]
        board["claims"].append(claim)
        return board

    specialists = {"researcher": researcher, "analyst": analyst, "writer": writer}

    def agent(objective: str, state: dict[str, Any] | None = None) -> str:
        """Delegate until the coordinator stops, then return the brief.

        Args:
            objective: The review objective.
            state: Optional starting scratchpad.

        Returns:
            The churn brief.
        """
        board: dict[str, Any] = dict(state or {})
        board["objective"] = objective
        board.setdefault("claims", [])
        board = _dispatch(engine, pick, specialists, board)
        return str(board.get("brief", ""))

    return agent


# -------------------------------------------------------------- the hardened twin


def build_fixed(engine: Any = None) -> Callable[..., str]:
    """Build the same supervisor, routing on the scratchpad instead of on claims.

    Args:
        engine: A `ChaosEngine` to instrument against, or `None` to run bare.

    Returns:
        ``agent(objective, state=None) -> str``.
    """
    tools, models = _wire(engine)

    def note(message: str) -> None:
        """Leave a breadcrumb when the engine is present.

        Args:
            message: The note.
        """
        if engine is not None:
            engine.note(message)

    def entered_without(board: dict[str, Any], *required: str) -> list[str]:
        """Check a specialist's preconditions and record the evidence.

        Args:
            board: The shared scratchpad.
            *required: Slots that must already be populated.

        Returns:
            The slots that are missing, empty when the entry was legitimate.
        """
        missing = [slot for slot in required if not board.get(slot)]
        if engine is not None:
            engine.validated({"required": list(required), "missing": missing}, name="preconditions")
        return missing

    def degrade(board: dict[str, Any], who: str, missing: list[str]) -> dict[str, Any]:
        """Record an out-of-order entry instead of writing from an empty slot.

        Args:
            board: The shared scratchpad.
            who: The specialist that was entered.
            missing: The slots it needed and did not have.

        Returns:
            The scratchpad with the gap recorded and nothing invented.
        """
        note(f"degraded: {who} was entered before {', '.join(missing)} existed; nothing written")
        board["unavailable"] = sorted(set(board.get("unavailable", [])) | set(missing))
        return board

    def pick(board: dict[str, Any]) -> str:
        """Ask the coordinator who goes next, from the scratchpad's own contents.

        Args:
            board: The shared scratchpad.

        Returns:
            A specialist name, or `DONE`.
        """
        completed = ", ".join(name for name in SPECIALIST_ORDER if board.get(SLOT[name]))
        reply = _content(
            models["supervisor"](
                _SUPERVISE_PROMPT.format(objective=board["objective"], completed=completed)
            )
        )
        return _choice(reply)

    def researcher(board: dict[str, Any]) -> dict[str, Any]:
        """Read the ticket archive and record the themes.

        Args:
            board: The shared scratchpad.

        Returns:
            The scratchpad with `research` set.
        """
        rows = tools["search_tickets"](board["objective"])["rows"]
        claim = _content(
            models["researcher"](
                _RESEARCH_PROMPT.format(objective=board["objective"], tickets=rows)
            )
        )
        board["research"] = {
            "themes": sorted({row["theme"] for row in rows}),
            "accounts": sorted({row["account"] for row in rows}),
        }
        board["claims"].append(claim)
        return board

    def analyst(board: dict[str, Any]) -> dict[str, Any]:
        """Turn seat counts into a utilisation line, if research ran.

        Args:
            board: The shared scratchpad.

        Returns:
            The scratchpad with `analysis` set, or unchanged when entered early.
        """
        missing = entered_without(board, "research")
        if missing:
            return degrade(board, "analyst", missing)
        research = board["research"]
        rows = tools["usage_stats"](research["accounts"])["rows"]
        worst = min(rows, key=lambda row: row["seats_active"] / row["seats_licensed"])
        claim = _content(
            models["analyst"](
                _ANALYSE_PROMPT.format(
                    objective=board["objective"], themes=research["themes"], seats=rows
                )
            )
        )
        board["analysis"] = {
            "account": worst["account"],
            "seats_active": worst["seats_active"],
            "seats_licensed": worst["seats_licensed"],
        }
        board["claims"].append(claim)
        return board

    def writer(board: dict[str, Any]) -> dict[str, Any]:
        """Write the churn brief, if both upstream slots are populated.

        Args:
            board: The shared scratchpad.

        Returns:
            The scratchpad with `brief` and `watchlist` set, or unchanged when
            entered early.
        """
        missing = entered_without(board, "research", "analysis")
        if missing:
            return degrade(board, "writer", missing)
        analysis = board["analysis"]
        research = board["research"]
        utilisation = (
            f"{analysis['seats_active']} of {analysis['seats_licensed']} seats "
            f"on {analysis['account']}"
        )
        body = _content(
            models["writer"](
                _WRITE_PROMPT.format(
                    objective=board["objective"],
                    themes=", ".join(research["themes"]),
                    utilisation=utilisation,
                )
            )
        )
        claim, brief = _split_claim(body)
        board["brief"] = brief
        board["watchlist"] = [analysis["account"]]
        board["claims"].append(claim)
        return board

    specialists = {"researcher": researcher, "analyst": analyst, "writer": writer}

    def agent(objective: str, state: dict[str, Any] | None = None) -> str:
        """Delegate until the coordinator stops, then return the brief and the gaps.

        Args:
            objective: The review objective.
            state: Optional starting scratchpad.

        Returns:
            The churn brief, or an explicit statement of what was never available.
            Never empty and never a figure the specialists did not supply.
        """
        board: dict[str, Any] = dict(state or {})
        board["objective"] = objective
        board.setdefault("claims", [])
        board = _dispatch(engine, pick, specialists, board)

        brief = str(board.get("brief", "")).strip()
        missing = sorted(board.get("unavailable", []))
        gaps = ", ".join(missing)
        verb = "were" if len(missing) > 1 else "was"
        if not brief:
            return (
                "Churn brief not written: "
                f"{gaps or 'nothing any specialist produced'} {verb} unavailable, so "
                "nothing was written from it."
            )
        if gaps:
            return (
                f"{brief}\n\nNote: {gaps} {verb} unavailable when the writer was first "
                "reached, so the brief was held until the specialists had run."
            )
        return brief

    return agent


PATTERN = register(
    PatternSpec(
        name="supervisor",
        description=(
            "A coordinator delegating each turn to a named specialist sub-agent with "
            "its own prompt, model alias and tool subset -- the LangGraph-supervisor "
            "/ AutoGen / CrewAI shape."
        ),
        build=build,
        build_fixed=build_fixed,
        weakness=(
            "The coordinator routes on each specialist's own free-text claim of "
            "success and never inspects the scratchpad, so a specialist delivered out "
            "of order is indistinguishable from one that succeeded and the next "
            "specialist indexes straight into a slot nobody wrote."
        ),
        faults=(
            {
                "type": "EdgeMisrouteFault",
                "params": {"from_node": "supervisor", "force_to": "writer", "times": 1},
                "target": {"layer": "edge"},
                "trigger": {"on_call": 1},
            },
        ),
        inputs="Which accounts are the biggest churn risk this quarter?",
        initial_state={"claims": []},
        tools=("search_tickets", "usage_stats"),
        llm_names=("supervisor", "researcher", "analyst", "writer"),
        expected_behavior="graceful_degradation",
    )
)
