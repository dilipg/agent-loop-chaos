"""Hand-written trace fixtures for the probe tests.

The fixtures are written by hand on purpose. Generating them from the engine would
let a probe bug and an engine bug cancel out, and the probes are the layer that
decides pass/fail.
"""

from __future__ import annotations

from typing import Any

from agent_loop_chaos.assertions import AssertionResult, HarnessFacts
from agent_loop_chaos.context import Limits, ToolInfo

RUN_ID = "run-3f9a12c4"
CANARY = f"ALC-CANARY-{RUN_ID}"


def ev(kind: str, seq: int, **kw: Any) -> dict[str, Any]:
    """Build one schema-shaped trace event.

    Args:
        kind: The event kind.
        seq: Its sequence number.
        **kw: Optional fields.

    Returns:
        The event.
    """
    event: dict[str, Any] = {
        "schema_version": "1.1",
        "run_id": RUN_ID,
        "seq": seq,
        "ts": "1970-01-01T00:00:00Z",
        "kind": kind,
        "level": "standard",
    }
    event.update({k: v for k, v in kw.items() if v is not None})
    return event


def tool_pair(seq: int, name: str, args: Any, result: Any, step: int = 1) -> list[dict[str, Any]]:
    """A requested/returned pair for one successful tool call.

    Args:
        seq: The sequence number of the request; the return takes `seq + 1`.
        name: The tool name.
        args: The arguments sent.
        result: The result received.
        step: The agent iteration.

    Returns:
        Two events.
    """
    return [
        ev(
            "tool_call_requested",
            seq,
            layer="tool",
            phase="pre",
            name=name,
            step=step,
            payload={"args": [], "kwargs": args},
        ),
        ev(
            "tool_call_returned",
            seq + 1,
            layer="tool",
            phase="post",
            name=name,
            step=step,
            payload={"result": result},
        ),
    ]


def probe_ctx(**kw: Any) -> Any:
    """Build a `ProbeContext` with harmless defaults.

    Args:
        **kw: Field overrides.

    Returns:
        The context.
    """
    from agent_loop_chaos.probes import ProbeContext

    defaults: dict[str, Any] = {
        "metrics": {"steps": 1, "tool_calls": 1, "llm_calls": 1, "tokens_in": 100},
        "limits": Limits(),
        "baseline": None,
        "assertions": [],
        "harness": HarnessFacts(canary=CANARY),
        "final_output": "an ordinary answer",
        "error": None,
        "limit_hit": None,
        "expected_behavior": "graceful_degradation",
        "tool_registry": {},
        "objective": None,
        "objective_state_key": "query",
        "final_state": {},
        "injection_payloads": [],
    }
    defaults.update(kw)
    return ProbeContext(**defaults)


def fired(code: str, symptoms: list[Any]) -> bool:
    """Report whether a probe emitted a given code.

    Args:
        code: The probe code.
        symptoms: What the probe returned.

    Returns:
        True when the code is present.
    """
    return any(s.code == code for s in symptoms)


def failing(check: str = "output_matches") -> AssertionResult:
    """A failing assertion carrying evidence.

    Args:
        check: The check name.

    Returns:
        The result.
    """
    return AssertionResult(check=check, ok=False, detail="nope", evidence=[{"seq": 3}])


def side_effecting(name: str, idempotency_arg: str | None = None) -> dict[str, ToolInfo]:
    """A registry holding one declared side-effecting tool.

    Args:
        name: The tool name.
        idempotency_arg: The argument that makes a repeat safe, if any.

    Returns:
        The registry.
    """
    return {name: ToolInfo(name=name, side_effecting=True, idempotency_arg=idempotency_arg)}
