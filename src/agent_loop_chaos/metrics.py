"""Counters computed from the trace, and the delta against a baseline.

Metrics are computed **before** probes (D-11), because a third of the probe rules
need them. Everything here is pure over the trace, so `alc judge` and `alc report`
can rebuild the numbers from a stored run with no re-execution.

Nothing derives a value from `ts` or `ts_mono_ms`. `wall_ms` is supplied by the
engine as a clearly-labelled timing field; deriving it here would smuggle a clock
into the post-run pipeline.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["compute_delta", "compute_metrics", "output_similarity"]

_CHARS_PER_TOKEN = 4
_WHITESPACE = re.compile(r"\s+")


def _estimate(text: str) -> int:
    """Estimate tokens with the `chars4` rule.

    Args:
        text: The text to measure.

    Returns:
        An estimated token count.
    """
    return len(text) // _CHARS_PER_TOKEN


def _render(value: Any) -> str:
    """Render a payload as text for estimation.

    Args:
        value: Any payload.

    Returns:
        The value as a string.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - default=str makes this rare
        return str(value)


def compute_metrics(
    trace: Sequence[Mapping[str, Any]],
    *,
    injected_tokens: int = 0,
    injected_delay_ms: int = 0,
    wall_ms: int = 0,
    retries: int = 0,
) -> dict[str, Any]:
    """Compute the report's `metrics` block from a trace.

    Args:
        trace: The loaded trace.
        injected_tokens: Tokens the harness itself added. Reported separately so
            every budget comparison can subtract them (R3).
        injected_delay_ms: Delay the harness itself added, for the same reason.
        wall_ms: Wall time, supplied by the engine as a timing field.
        retries: Observed retries.

    Returns:
        The metrics mapping, matching `chaos_report.schema.json`.
    """
    kinds = [e.get("kind") for e in trace]
    tokens_in = 0
    tokens_out = 0
    for event in trace:
        payload = event.get("payload") or {}
        if event.get("kind") == "llm_request":
            tokens_in += _estimate(_render(payload.get("messages")))
        elif event.get("kind") == "llm_response":
            tokens_out += _estimate(_render(payload.get("result")))

    return {
        "steps": kinds.count("step_started"),
        "tool_calls": kinds.count("tool_call_requested"),
        "llm_calls": kinds.count("llm_request"),
        "retries": retries,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "wall_ms": wall_ms,
        "injected_delay_ms": injected_delay_ms,
        "injected_tokens": injected_tokens,
        "faults_armed": kinds.count("fault_armed"),
        "faults_fired": kinds.count("fault_fired"),
        # D-43: real counts come from a provider response when present; the `chars4`
        # estimator fills them otherwise, and the report says which was used.
        "tokens_estimated": True,
    }


def output_similarity(left: Any, right: Any) -> float:
    """Compare two final outputs.

    `difflib.SequenceMatcher` on lowercased, whitespace-collapsed text: deterministic
    and dependency-free. Embeddings would be more nuanced and would destroy the
    byte-identical guarantee.

    Args:
        left: One output.
        right: The other.

    Returns:
        A ratio in ``[0, 1]``, rounded to six decimals so goldens stay stable.
    """
    normalized = [_WHITESPACE.sub(" ", _render(value).strip().lower()) for value in (left, right)]
    return round(difflib.SequenceMatcher(None, normalized[0], normalized[1]).ratio(), 6)


def compute_delta(
    chaos: dict[str, Any],
    baseline: dict[str, Any],
    *,
    chaos_output: Any,
    baseline_output: Any,
    chaos_tools: Sequence[str],
    baseline_tools: Sequence[str],
    diff_path: str | None = None,
) -> dict[str, Any]:
    """Compare a chaos run against its baseline.

    Args:
        chaos: The chaos run's metrics.
        baseline: The baseline run's metrics.
        chaos_output: The chaos run's final output.
        baseline_output: The baseline's final output.
        chaos_tools: Tools the chaos run called.
        baseline_tools: Tools the baseline called.
        diff_path: Where `baseline.diff` was written, if anywhere.

    Returns:
        The `delta_vs_baseline` mapping. Which tools appeared or vanished is the most
        legible signal in it, so both sets are named explicitly.
    """

    def difference(key: str) -> int:
        return int(chaos.get(key, 0) or 0) - int(baseline.get(key, 0) or 0)

    chaos_set, baseline_set = set(chaos_tools), set(baseline_tools)
    return {
        "steps": difference("steps"),
        "tool_calls": difference("tool_calls"),
        "llm_calls": difference("llm_calls"),
        "tokens_in": difference("tokens_in"),
        "tokens_out": difference("tokens_out"),
        "output_similarity": output_similarity(chaos_output, baseline_output),
        "new_tool_calls": sorted(chaos_set - baseline_set),
        "missing_tool_calls": sorted(baseline_set - chaos_set),
        "diff_path": diff_path,
    }
