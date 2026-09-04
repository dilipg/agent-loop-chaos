"""A scripted, mediocre-but-plausible model, so the examples need no API key.

It routes on the `[TAG]` marker every node puts at the top of its prompt and returns
an OpenAI-shaped envelope (`content` plus `finish_reason`). Two behaviours are
deliberate, because they are what the demo is *for*:

- it happily returns a number it was never given, so `HallucinationSeedFault` and
  the `no_unsourced_numbers` assertion have something to catch;
- it follows an instruction it finds in tool output — unless that content arrives
  inside an `UNTRUSTED_DATA` fence, in which case it reads as quoted text. That is
  the difference between the buggy tree and its twin, and the reason the fence in
  `revenue_review_fixed` is worth writing.

Tests use `tests/fakes/fake_llm.FakeLLM` instead; this exists so the module-level
`graph` in each `app.py` is runnable on its own.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["FENCE", "respond"]

#: The marker `revenue_review_fixed` wraps untrusted CSV text in.
FENCE = "UNTRUSTED_DATA"

_TAG = re.compile(r"\[([A-Z_]+)\]")
_INJECTED = re.compile(
    r"(?im)^\s*(?:system\s*:|ignore\b|disregard\b|instead\b|you must\b)|ALC-CANARY-run-[0-9a-f]{8}"
)

_SCRIPT: dict[str, str] = {
    "PLANNER": '["ACC-1002", "ACC-1005", "ACC-1004"]',
    "ANALYST_READ": "Two accounts are carrying overdue invoices and one is disputed.",
    "ANALYST_SUMMARY": "Collections risk is concentrated in the mid-market segment.",
    "SCORE": '{"score": 72, "band": "high", "drivers": ["invoice ageing"]}',
    "CRITIQUE": "The score leans on invoice ageing and under-weights seat activation.",
    "REVISE": '{"score": 66, "band": "high", "drivers": ["invoice ageing"]}',
    "DRAFT": "Draft: mid-market collections risk needs attention before the Q4 renewals.",
    "TIGHTEN": "Mid-market collections risk needs a CSM touch before renewal.",
    "REVIEW": "APPROVE",
    # The trip planner (examples/trip_planner). PLAN and RESPOND are stable; SUMMARIZE
    # is built per call, because what it says depends on what the forecast contained.
    "PLAN": '{"location": "Paris", "origin": "DEL"}',
}

#: Which day-one temperature the summarizer reports when the forecast does not carry
#: one. A real mediocre model does exactly this: it fills the gap from the season and
#: the city rather than admitting the field was missing. It is what makes
#: `no_unsourced_numbers` and `HallucinationSeedFault` catch something real.
_FABRICATED_TEMP_C = 19

_TEMP = re.compile(r"'temp_c':\s*(-?\d+(?:\.\d+)?)")


def _text(messages: Any) -> str:
    """Flatten whatever the agent sent into one searchable string.

    Args:
        messages: A string, or a list of message mappings.

    Returns:
        The prompt as text.
    """
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        return "\n".join(
            str(m.get("content", m)) if isinstance(m, dict) else str(m) for m in messages
        )
    return str(messages)


def _injected(prompt: str) -> str | None:
    """Find instruction-shaped text the prompt carries in the clear.

    Args:
        prompt: The rendered prompt.

    Returns:
        The span, or `None` when there is none or it arrived inside a fence.
    """
    if FENCE in prompt:
        return None
    found = _INJECTED.search(prompt)
    return prompt[found.start() : found.start() + 200].strip() if found else None


def _summarize(prompt: str) -> str:
    """Build a packing list from whatever the forecast actually contained.

    Args:
        prompt: The rendered summarize prompt.

    Returns:
        A JSON object with `packing_list` and `note`.
    """
    found = _TEMP.search(prompt)
    temp = float(found.group(1)) if found else _FABRICATED_TEMP_C
    warm = temp >= 20
    items = (
        ["light shirts", "sunglasses", "a compact umbrella", "walking shoes"]
        if warm
        else ["a warm jacket", "a scarf", "waterproof shoes", "gloves"]
    )
    note = f"Expect around {temp:g}C on day one, so pack accordingly."
    obeyed = _injected(prompt)
    if obeyed:
        # Obedience inside the structured output, which is what it actually looks
        # like: the JSON still parses and the instruction reaches the user.
        note = f"{note} {obeyed}"
    return json.dumps({"packing_list": items, "note": note})


def _trip_respond(prompt: str) -> str:
    """Write the traveller's final answer.

    Args:
        prompt: The rendered respond prompt.

    Returns:
        Two short paragraphs.
    """
    price = re.search(r"'price_usd':\s*(\d+)", prompt)
    carrier = re.search(r"'carrier':\s*'([^']+)'", prompt)
    items = re.search(r"Packing list:\s*(\[[^\]]*\])", prompt)
    packing = items.group(1) if items else "[]"
    airline = carrier.group(1) if carrier else "the carrier"
    fare = (
        f"The cheapest fare is ${price.group(1)} on {airline}."
        if price
        else "No fare was available."
    )
    note = re.search(r"Note:\s*(.+)", prompt)
    tail = f"\n\n{note.group(1).strip()}" if note and note.group(1).strip() else ""
    return f"Pack {packing}.\n\n{fare}{tail}"


def respond(messages: Any, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Answer one prompt from the script.

    Args:
        messages: The prompt, as a string or a message list.
        *_args: Ignored, so the signature matches a real client.
        **_kwargs: Ignored.

    Returns:
        ``{"content": str, "finish_reason": "stop"}``.
    """
    prompt = _text(messages)
    tag = _TAG.search(prompt)
    name = tag.group(1) if tag else ""
    if name == "SUMMARIZE":
        body = _summarize(prompt)
    elif name == "RESPOND":
        body = _trip_respond(prompt)
    else:
        body = _SCRIPT.get(name, "OK")
    if name != "SUMMARIZE":
        # `_summarize` obeys inside its own structured output, which is what real
        # obedience looks like; appending after it would only ever produce a JSON
        # parse error and hide the finding. Everything else appends.
        obeyed = _injected(prompt)
        if obeyed:
            body = f"{body}\n{obeyed}"
    return {"content": body, "finish_reason": "stop"}
