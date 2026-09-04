"""A fixed pipeline -- extract, transform, validate, render -- with no agent loop.

Four functions called in sequence. No branching, no back-edge, no "decide what to do
next": the LCEL / "chain, not agent" shape that a large share of shipped "agents"
really are. It is in the pool to prove the library does not need a loop to attach to
-- `engine.tool` and `engine.llm` wrap the stages a chain already has, and a tool
fault lands in exactly the same trace it would for a ReAct loop.

**The weakness planted in `build`.** No stage boundary checks the shape of what it
was handed. `_extract` returns a positional tuple, `_transform` unpacks it
positionally and grows it, `_validate` grows it again, and every stage indexes tool
rows by key on the assumption that the row shape seen in development is the row
shape that will arrive. So a field the incident log stops returning is not caught
where it enters; `_transform` never touches it; `_validate` asks the model about it
with an empty slot and the model helpfully invents an owning team rather than
admitting it was not told one; and the change finally surfaces as a `KeyError` in
`_render`, three stages and three model calls downstream of the tool that changed.
That distance is the whole point of the shape: the crash names the wrong file.

`build_fixed` checks the row contract once, at the boundary where the data enters,
records the gap with `engine.note`, carries it forward as a named unavailability,
never asks the model to fill it, and says in the rendered digest what was missing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from . import PatternSpec, register

__all__ = ["PATTERN", "build", "build_fixed", "fetch_incident_log"]

#: Every field a row is contracted to carry. `build_fixed` checks against it; `build`
#: assumes it.
REQUIRED_ROW_FIELDS: tuple[str, ...] = ("incident_id", "service", "minutes_down", "owner")

_LOG_ROWS: tuple[dict[str, Any], ...] = (
    {
        "incident_id": "INC-4471",
        "service": "checkout-api",
        "minutes_down": 42,
        "owner": "payments-guild",
    },
    {
        "incident_id": "INC-4472",
        "service": "search",
        "minutes_down": 11,
        "owner": "discovery-guild",
    },
    {
        "incident_id": "INC-4473",
        "service": "checkout-api",
        "minutes_down": 7,
        "owner": "payments-guild",
    },
)


def fetch_incident_log(window_days: int = 7) -> dict[str, Any]:
    """Return the incident rows for the trailing window.

    Args:
        window_days: How far back to look. The fixture covers one week, so the value
            is echoed in the envelope and does not change the rows.

    Returns:
        ``{"window_days": int, "rows": [{incident_id, service, minutes_down,
        owner}]}``.
    """
    return {"window_days": window_days, "rows": [dict(row) for row in _LOG_ROWS]}


# ------------------------------------------------------------------- the model

_TAG = re.compile(r"\[([A-Z_]+)\]")
_FIELD = re.compile(r"^(\w+)=(.*)$", re.MULTILINE)

#: What the model says when the validate stage hands it an empty `owner=` slot. A
#: mediocre model does not answer "you did not tell me"; it answers with the team it
#: would guess, and the guess is wrong.
FABRICATED_OWNER = "platform-oncall"

_EXTRACT_PROMPT = "[EXTRACT]\nquestion={question}\nrows={rows}\nName the pattern in one sentence."
_TRANSFORM_PROMPT = (
    "[TRANSFORM]\nworst={worst}\ntotals={totals}\nSay where the downtime is concentrated."
)
_VALIDATE_PROMPT = "[VALIDATE]\nworst={worst}\nowner={owner}\nConfirm who owns the worst incident."
_RENDER_PROMPT = (
    "[RENDER]\nworst={worst}\nminutes={minutes}\nreading={reading}\nWrite the one-line digest."
)


def _scripted_model() -> Callable[..., dict[str, Any]]:
    """Build this pipeline's stand-in for a mediocre model.

    Routes on the ``[STAGE]`` marker and the ``key=value`` lines each prompt carries,
    with a per-instance call counter for the fallback reply. No clock, no randomness,
    no network, so the pipeline is byte-stable under a fixed seed.

    Returns:
        ``respond(prompt) -> {"content": str, "finish_reason": "stop"}``.
    """
    calls = [0]

    def respond(prompt: Any, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """Answer one stage prompt.

        Args:
            prompt: The rendered prompt.
            *_args: Ignored, so the signature matches a real client.
            **_kwargs: Ignored.

        Returns:
            An OpenAI-shaped envelope.
        """
        calls[0] += 1
        text = str(prompt)
        fields = dict(_FIELD.findall(text))
        found = _TAG.search(text)
        stage = found.group(1) if found else ""
        worst = fields.get("worst", "").strip() or "one service"
        if stage == "EXTRACT":
            body = "Three incidents landed in the window and one service appears twice."
        elif stage == "TRANSFORM":
            body = f"Downtime is concentrated on {worst}."
        elif stage == "VALIDATE":
            owner = fields.get("owner", "").strip() or FABRICATED_OWNER
            body = f"Checked: the worst incident belongs to {owner}."
        elif stage == "RENDER":
            body = f"Incident digest: {worst} took the worst hit this week."
        else:
            body = f"OK (call {calls[0]})."
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


def _wire(engine: Any) -> tuple[dict[str, Callable[..., Any]], Callable[..., Any]]:
    """Register the tool and the model against the engine, if there is one.

    Args:
        engine: A `ChaosEngine`, or `None` to run uninstrumented.

    Returns:
        ``(tools, model)``, wrapped when an engine was supplied.
    """
    model = _scripted_model()
    if engine is None:
        return {"fetch_incident_log": fetch_incident_log}, model
    tool = engine.tool(fetch_incident_log, name="fetch_incident_log", side_effecting=False)
    return {"fetch_incident_log": tool}, engine.llm(model, name="chain")


def _totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Sum downtime per service.

    Args:
        rows: Incident rows.

    Returns:
        Service name to total minutes.
    """
    out: dict[str, int] = {}
    for row in rows:
        out[row["service"]] = out.get(row["service"], 0) + int(row["minutes_down"])
    return out


def _worst(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the single longest incident.

    Args:
        rows: Incident rows.

    Returns:
        The row with the most minutes down, or an empty mapping when there are none.
    """
    return max(rows, key=lambda row: int(row["minutes_down"]), default={})


# --------------------------------------------------------------- the naive chain


def build(engine: Any = None) -> Callable[[str], str]:
    """Build the naive four-stage pipeline.

    Args:
        engine: A `ChaosEngine` to instrument against, or `None` to run bare.

    Returns:
        ``chain(question) -> str``.
    """
    tools, model = _wire(engine)

    def extract(question: str) -> tuple[Any, ...]:
        """Pull the log and read it. One tool call, one model call.

        Args:
            question: The user's question.

        Returns:
            ``(rows, reading, row_count)``.
        """
        log = tools["fetch_incident_log"](7)
        rows = log["rows"]
        reading = _content(model(_EXTRACT_PROMPT.format(question=question, rows=rows)))
        return rows, reading, len(rows)

    def transform(payload: tuple[Any, ...]) -> tuple[Any, ...]:
        """Aggregate downtime per service. One model call.

        Args:
            payload: Whatever `extract` returned.

        Returns:
            ``(rows, totals, reading, row_count)``.
        """
        rows, reading, row_count = payload
        totals = _totals(rows)
        worst = _worst(rows)
        note = _content(
            model(_TRANSFORM_PROMPT.format(worst=worst.get("service", ""), totals=totals))
        )
        return rows, totals, f"{reading} {note}", row_count

    def validate(payload: tuple[Any, ...]) -> tuple[Any, ...]:
        """Ask the model to confirm the owner. One model call.

        Args:
            payload: Whatever `transform` returned.

        Returns:
            ``(rows, totals, reading, worst_row, check)``.
        """
        rows, totals, reading, _row_count = payload
        worst = _worst(rows)
        check = _content(
            model(
                _VALIDATE_PROMPT.format(
                    worst=worst.get("service", ""), owner=worst.get("owner", "")
                )
            )
        )
        return rows, totals, reading, worst, check

    def render(payload: tuple[Any, ...]) -> str:
        """Write the digest. One model call.

        Args:
            payload: Whatever `validate` returned.

        Returns:
            The digest.

        Raises:
            KeyError: When a row lost a field somewhere upstream.
        """
        rows, _totals_by_service, reading, worst, check = payload
        roster = ", ".join(f"{row['service']} -> {row['owner']}" for row in rows)
        body = _content(
            model(
                _RENDER_PROMPT.format(
                    worst=worst.get("service", ""),
                    minutes=worst.get("minutes_down", 0),
                    reading=reading,
                )
            )
        )
        minutes = worst.get("minutes_down", 0)
        return f"{body}\nWorst single incident: {minutes} minutes.\n{check}\nOwners: {roster}"

    def chain(question: str) -> str:
        """Run the four stages in order.

        Args:
            question: The user's question.

        Returns:
            The rendered digest.
        """
        return render(validate(transform(extract(question))))

    return chain


# -------------------------------------------------------------- the hardened twin


def build_fixed(engine: Any = None) -> Callable[[str], str]:
    """Build the same pipeline with a contract check at the entry boundary.

    Args:
        engine: A `ChaosEngine` to instrument against, or `None` to run bare.

    Returns:
        ``chain(question) -> str``.
    """
    tools, model = _wire(engine)

    def note(message: str) -> None:
        """Leave a breadcrumb when the engine is present.

        Args:
            message: The note.
        """
        if engine is not None:
            engine.note(message)

    def extract(question: str) -> dict[str, Any]:
        """Pull the log, check the row contract, then read it.

        Args:
            question: The user's question.

        Returns:
            A named bundle carrying `rows`, `missing`, `reading` and `question`.
        """
        log = tools["fetch_incident_log"](7)
        rows = [row for row in log.get("rows", []) if isinstance(row, dict)]
        missing = sorted({name for name in REQUIRED_ROW_FIELDS for row in rows if name not in row})
        if engine is not None:
            engine.validated(
                {"rows": len(rows), "missing_fields": missing}, name="incident_row_contract"
            )
        if missing:
            note(
                f"degraded: fetch_incident_log returned rows without "
                f"{', '.join(missing)}; downstream stages will not be told a value for it"
            )
        usable = [{k: v for k, v in row.items() if k in REQUIRED_ROW_FIELDS} for row in rows]
        reading = _content(model(_EXTRACT_PROMPT.format(question=question, rows=usable)))
        return {"question": question, "rows": rows, "missing": missing, "reading": reading}

    def transform(bundle: dict[str, Any]) -> dict[str, Any]:
        """Aggregate downtime per service, on the fields that survived.

        Args:
            bundle: The bundle `extract` produced.

        Returns:
            The bundle with `totals` and `worst` added.
        """
        if "minutes_down" in bundle["missing"] or "service" in bundle["missing"]:
            note("degraded: downtime cannot be aggregated without service and minutes_down")
            return {**bundle, "totals": {}, "worst": {}}
        totals = _totals(bundle["rows"])
        worst = _worst(bundle["rows"])
        reading = _content(
            model(_TRANSFORM_PROMPT.format(worst=worst.get("service", ""), totals=totals))
        )
        return {
            **bundle,
            "totals": totals,
            "worst": worst,
            "reading": f"{bundle['reading']} {reading}",
        }

    def validate(bundle: dict[str, Any]) -> dict[str, Any]:
        """Confirm the owner, but only when there is an owner to confirm.

        Args:
            bundle: The bundle `transform` produced.

        Returns:
            The bundle with `check` added.
        """
        worst = bundle["worst"]
        if "owner" in bundle["missing"] or not worst.get("owner"):
            return {**bundle, "check": ""}
        check = _content(
            model(_VALIDATE_PROMPT.format(worst=worst["service"], owner=worst["owner"]))
        )
        return {**bundle, "check": check}

    def render(bundle: dict[str, Any]) -> str:
        """Write the digest and name anything the log did not supply.

        Args:
            bundle: The bundle `validate` produced.

        Returns:
            The digest, always non-empty.
        """
        worst = bundle["worst"]
        if not worst:
            return (
                "Incident digest unavailable: the incident log did not return "
                f"{', '.join(bundle['missing']) or 'any usable rows'}, so no service "
                "can be ranked."
            )
        body = _content(
            model(
                _RENDER_PROMPT.format(
                    worst=worst["service"],
                    minutes=worst["minutes_down"],
                    reading=bundle["reading"],
                )
            )
        )
        lines = [body, f"Worst single incident: {worst['minutes_down']} minutes."]
        if bundle["check"]:
            lines.append(bundle["check"])
        if bundle["missing"]:
            lines.append(
                f"Unavailable: the incident log did not return "
                f"{', '.join(bundle['missing'])}, so none is reported here."
            )
        else:
            lines.append(
                "Owners: "
                + ", ".join(f"{row['service']} -> {row['owner']}" for row in bundle["rows"])
            )
        return "\n".join(lines)

    def chain(question: str) -> str:
        """Run the four stages in order.

        Args:
            question: The user's question.

        Returns:
            The rendered digest.
        """
        return render(validate(transform(extract(question))))

    return chain


PATTERN = register(
    PatternSpec(
        name="pipeline",
        description=(
            "A fixed extract -> transform -> validate -> render chain with no agent "
            "loop at all -- the LCEL shape a great many shipped 'agents' really are."
        ),
        build=build,
        build_fixed=build_fixed,
        weakness=(
            "No stage boundary checks the shape of its input, so a field the incident "
            "log stops returning is papered over by the model at validate and only "
            "crashes at render, three stages downstream of the change."
        ),
        faults=(
            {
                "type": "ToolCorruptionFault",
                "params": {"mutation_type": "drop_key", "keys": ["owner"]},
                "target": {"tool": "fetch_incident_log"},
                "trigger": {"on_call": 1},
            },
        ),
        inputs="Which service cost us the most downtime this week?",
        tools=("fetch_incident_log",),
        llm_names=("chain",),
        expected_behavior="graceful_degradation",
    )
)
