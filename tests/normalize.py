"""The normalizer, and the only list of non-deterministic report fields.

`docs/04-SCHEMAS.md` §3 makes this module the owner of the strip list: nothing else
may add to it without updating a test. Two guarantees are being separated here
(D-07) — harness determinism is unconditional, while a byte-identical report also
needs a deterministic agent, a scripted or temperature-0 model, `--judge rules` and
a single-threaded graph.
"""

from __future__ import annotations

from typing import Any

# Wall-clock, path-dependent, or model-authored. Everything else must be stable.
STRIP_KEYS: frozenset[str] = frozenset(
    {
        # timing
        "started_at",
        "finished_at",
        "duration_ms",
        "wall_ms",
        "ts",
        "ts_mono_ms",
        "latency_ms",
        "cost_usd",
        # identity that is proven by plan_hash instead (D-04)
        "run_id",
        # environment
        "library_version",
        "adapter_version",
        "env",
        # model-authored fields (D-07)
        "chaos_narrative",
        "root_cause_hypothesis",
        "refinement_hint",
        "suggested_fixes",
        "narrative",
        "confidence",
    }
)

# Values that embed an absolute path or a run id.
PATH_KEYS: frozenset[str] = frozenset(
    {
        "report",
        "trace",
        "plan",
        "agent_task",
        "judge",
        "baseline_diff",
        "run_dir",
        "plan_path",
        "cmd",
        "suite_path",
    }
)


def normalize(obj: Any) -> Any:
    """Strip non-deterministic fields, recursively.

    Args:
        obj: A report, trace event, or any nested value.

    Returns:
        A new value with the strip list removed and path-bearing values replaced by
        a placeholder. The input is not mutated.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in STRIP_KEYS:
                continue
            if key in PATH_KEYS:
                out[key] = "<path>" if value is not None else None
                continue
            out[key] = normalize(value)
        return out
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    return obj
