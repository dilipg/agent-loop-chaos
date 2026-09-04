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
}


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
    body = _SCRIPT.get(tag.group(1) if tag else "", "OK")
    if FENCE not in prompt:
        found = _INJECTED.search(prompt)
        if found:
            # The whole point: undelimited tool text reads as instruction.
            body = f"{body}\n{prompt[found.start() : found.start() + 200].strip()}"
    return {"content": body, "finish_reason": "stop"}
