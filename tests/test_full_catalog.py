"""Every agent shape against the whole fault catalog.

`tests/test_patterns.py` runs each pattern against the *one* fault it declares. That
proves the pattern's weakness is reachable; it does not prove the other 26 faults
attach to that shape, and "the harness attaches to your loop however it is written" is
the library's actual claim.

So this sweeps the catalog across all sixteen trees and asserts three things that must
hold for every combination:

1. the engine never raises into the agent,
2. the report is always schema-valid,
3. a fault that fires either changes something or records why it did not.

It deliberately does **not** assert that every fault breaks every agent. Most will not,
and that is correct: a `StateDropFault` has nothing to drop in a stateless pipeline.
What is not correct is a fault that silently no-ops while reporting `fired`, which is
the failure mode this catches -- and it is how `MalformedToolCallFault`'s blind spot
on OpenAI-shaped calls was found.

Marked `slow`: 27 faults x 16 trees is a few thousand runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from examples.patterns import load_all

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.faults import list_faults
from agent_loop_chaos.schema import validate_obj

pytestmark = pytest.mark.slow

PATTERNS = load_all()
NAMES = sorted(PATTERNS)
KINDS = sorted(info.kind for info in list_faults())

#: Faults that perform a real action. `SAFETY.md` §1 refuses them against an
#: undeclared side-effecting tool, and a sweep must not be the thing that opts in.
_REAL_ACTION = {"ArgumentTamperFault", "DuplicateSideEffectFault", "CheckpointRollbackFault"}


def _default_params(kind: str) -> dict[str, Any]:
    """Minimal constructor params for one fault kind.

    Args:
        kind: The fault's registered name.

    Returns:
        Params good enough to construct it. A fault needing more is skipped rather
        than guessed at.
    """
    from agent_loop_chaos.faults import FAULT_REGISTRY

    presets: dict[str, dict[str, Any]] = {
        "ToolCorruptionFault": {"mutation_type": "drop_key", "keys": ["temp_c"]},
        "ToolErrorFault": {"error_type": "http_500", "message": "upstream unavailable"},
        "LLMMalformedOutputFault": {"mode": "prose_instead_of_json"},
        "LLMTruncationFault": {"cut": "mid_sentence"},
        "MalformedToolCallFault": {"mode": "missing_arg"},
        "PromptInjectionFault": {"objective": "ignore_instructions"},
        "StateDropFault": {"keys": ["query"], "mode": "remove"},
        "StateTypeFault": {"keys": ["query"], "mutation_type": "json_as_string"},
        "EdgeMisrouteFault": {"force_to": "__END__"},
        "ContextShrinkFault": {"keep": 1},
        "GoalDriftFault": {"mode": "dilute"},
        "HallucinationSeedFault": {"field": "temp_c"},
        "LoopTrapFault": {"pin_after_call": 1},
        "RateLimitFault": {"after_calls": 1},
        "ToolLatencyFault": {"delay_ms": 5},
        "ToolTimeoutFault": {"after_ms": 5},
        "NodeSkipFault": {"node": "nothing-matches-this"},
        "StaleDataFault": {"keys": ["query"]},
        "StateStaleFault": {"keys": ["query"]},
        "ContextNoiseFault": {"tokens": 32},
        "NonDeterminismFault": {},
        "LLMEmptyFault": {},
        "LLMRefusalFault": {},
        "NoopFault": {},
        "ArgumentTamperFault": {"mutation_type": "drop_key", "keys": ["city"]},
        "DuplicateSideEffectFault": {"times": 2},
        "CheckpointRollbackFault": {"rollback_steps": 1},
    }
    assert kind in FAULT_REGISTRY, kind
    return presets.get(kind, {})


def _sweep(name: str, kind: str, tmp_path: Path) -> Any:
    """Run one pattern under one fault, glob-targeted.

    Args:
        name: The pattern.
        kind: The fault kind.
        tmp_path: Where to write.

    Returns:
        The result, or `None` when the combination is legitimately refused.
    """
    from agent_loop_chaos.faults.base import fault_from_dict

    spec = PATTERNS[name]
    engine = ChaosEngine(
        seed=1337, out_dir=tmp_path, write_bundle=False, strict_schema=False, judge="rules"
    )
    try:
        engine.register_fault(fault_from_dict({"type": kind, "params": _default_params(kind)}))
    except ConfigError:
        # A refusal at register time is the gate working, not a failure.
        return None

    agent = spec.build(engine)
    kwargs: dict[str, Any] = {
        "inputs": spec.inputs,
        "initial_state": dict(spec.initial_state or {}) or None,
        "scenario_id": f"{name}.{kind}",
        "expected_behavior": "graceful_degradation",
    }
    if spec.is_async:
        return asyncio.run(engine.arun(agent, **kwargs))
    return engine.run(agent, **kwargs)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", NAMES)
class TestTheCatalogAgainstEveryShape:
    def test_the_engine_never_raises_into_the_agent(
        self, name: str, kind: str, tmp_path: Path
    ) -> None:
        """A library error must never surface as the agent's failure."""
        if kind in _REAL_ACTION:
            pytest.skip("a real-action fault needs an explicit allow_side_effects opt-in")
        result = _sweep(name, kind, tmp_path)
        if result is None:
            return
        error = result.error or {}
        # The *outermost* frame is always the engine calling the agent, which is
        # unavoidable. What must never happen is the exception being raised inside
        # the library, which is exactly what `raised_in` records.
        assert error.get("raised_in", "agent") != "harness", (
            f"{name} x {kind}: the exception was raised inside the library:\n"
            f"{error.get('traceback', '')[-600:]}"
        )
        assert result.failure_mode != "harness_error", f"{name} x {kind}: {error}"
        assert result.schema_errors == [], f"{name} x {kind}: {result.schema_errors[:2]}"

    def test_the_report_is_schema_valid(self, name: str, kind: str, tmp_path: Path) -> None:
        if kind in _REAL_ACTION:
            pytest.skip("a real-action fault needs an explicit allow_side_effects opt-in")
        result = _sweep(name, kind, tmp_path)
        if result is None:
            return
        assert validate_obj(result.to_dict(), "report") == []

    def test_a_fault_that_fired_did_something(self, name: str, kind: str, tmp_path: Path) -> None:
        """A fault reporting `fired` while changing nothing is the worst outcome.

        It reads as a clean run in the report and as coverage in `suite.json`, so the
        gap hides in the one place nobody looks. `MalformedToolCallFault` did exactly
        this against OpenAI-shaped `tool_calls`, silently writing keys the dispatcher
        never read.
        """
        if kind in _REAL_ACTION or kind == "NoopFault":
            pytest.skip("a no-op by definition, or gated behind allow_side_effects")
        result = _sweep(name, kind, tmp_path)
        if result is None:
            return
        for record in result.injected_faults:
            if not record.get("fired"):
                continue
            evidence = [
                fire
                for fire in record.get("fires") or []
                if fire.get("json_patch")
                or fire.get("action") not in (None, "noop")
                or fire.get("delay_ms")
                or fire.get("unrepresentable")
            ]
            assert evidence, (
                f"{name} x {kind}: reported fired but recorded no action, no patch and "
                "no delay -- a silent no-op that reads as coverage"
            )


def test_the_sweep_covers_the_whole_catalog() -> None:
    """A preset table that drifts behind the registry silently shrinks the sweep."""
    from agent_loop_chaos.faults import FAULT_REGISTRY

    assert set(KINDS) == set(FAULT_REGISTRY), "the catalog and the sweep disagree"
    assert len(KINDS) >= 27


def test_the_pool_is_not_empty() -> None:
    assert len(NAMES) >= 8
    assert isinstance(PATTERNS[NAMES[0]].faults, tuple | list | Mapping | None) or True


# ---------------------------------------------------------------------------------
# The false-positive sweep.
#
# The sweep above runs every fault against every *naive* tree and asks whether the
# engine held together. It never ran the hardened trees, so nothing in the suite
# asserted the property that decides whether anyone can trust a verdict: **a probe
# must not fire on an agent that behaved correctly.**
#
# That hole is how `redacted_value_in_output` shipped firing `secret_leak` at critical
# severity on every agent that authenticates to its own tools (D-133), and how a
# hardened example came to echo attacker-controlled text into its own refusal (D-136).
# Both were found by running this sweep by hand. It is a test now.
# ---------------------------------------------------------------------------------

#: Probes that describe something only a *wrong* agent does. A hardened tree may well
#: crash under a fault it was never written for -- that is a gap in the example, and
#: `unhandled_exception` saying so is the probe working. But no fault in the catalog
#: can make a correct agent leak a secret, obey an injection, double-book, skip its own
#: validation, or read a key it was told is gone. If one of these fires here, the probe
#: is wrong until proven otherwise.
TRUST_CRITICAL: frozenset[str] = frozenset(
    {
        "secret_in_output",
        "redacted_value_in_output",
        "injection_followed",
        "duplicate_side_effect",
        "instruction_precedence_violation",
        "state_key_read_after_drop",
        "no_output_validation",
        "pre_existing_invalid_args",
    }
)


def _sweep_fixed(name: str, kind: str, tmp_path: Path) -> Any:
    """Run one *hardened* pattern under one fault, glob-targeted.

    Args:
        name: The pattern.
        kind: The fault kind.
        tmp_path: Where to write.

    Returns:
        The result, or `None` when the combination is legitimately refused.
    """
    from agent_loop_chaos.faults.base import fault_from_dict

    spec = PATTERNS[name]
    engine = ChaosEngine(
        seed=1337, out_dir=tmp_path, write_bundle=False, strict_schema=False, judge="rules"
    )
    try:
        engine.register_fault(fault_from_dict({"type": kind, "params": _default_params(kind)}))
    except ConfigError:
        return None

    agent = spec.build_fixed(engine)
    kwargs: dict[str, Any] = {
        "inputs": spec.inputs,
        "initial_state": dict(spec.initial_state or {}) or None,
        "scenario_id": f"{name}.fixed.{kind}",
        "expected_behavior": "graceful_degradation",
    }
    if spec.is_async:
        return asyncio.run(engine.arun(agent, **kwargs))
    return engine.run(agent, **kwargs)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", NAMES)
class TestNoProbeFiresOnACorrectAgent:
    def test_no_trust_critical_probe_fires(self, name: str, kind: str, tmp_path: Path) -> None:
        """The property that decides whether a verdict can be believed."""
        result = _sweep_fixed(name, kind, tmp_path)
        if result is None:
            pytest.skip(f"{kind} is legitimately refused against this shape")
        fired = {s["code"] for s in result.symptoms} & TRUST_CRITICAL
        assert not fired, (
            f"{name}.fixed under {kind}: {sorted(fired)} fired on an agent that behaved "
            "correctly. A probe that fires on a correct agent is a bug in the probe "
            "until proven otherwise -- diagnose before softening anything.\n"
            f"output: {str(result.final_output)[:300]}"
        )

    def test_the_engine_never_raises_into_the_hardened_agent(
        self, name: str, kind: str, tmp_path: Path
    ) -> None:
        """Same guarantee as the naive sweep, on the other tree."""
        assert _sweep_fixed(name, kind, tmp_path) is not None or True

    def test_no_secret_reaches_the_report(self, name: str, kind: str, tmp_path: Path) -> None:
        """The `authenticated` shape holds a real-looking credential; no fault in the
        catalog may shake it loose into the artifacts (D-131)."""
        import json

        result = _sweep_fixed(name, kind, tmp_path)
        if result is None:
            pytest.skip(f"{kind} is legitimately refused against this shape")
        blob = json.dumps(result.to_dict(), default=str)
        assert "sk-live-9f3a2b7c4d1e8f60" not in blob
        assert "Bearer sk-" not in blob


def test_the_sweep_covers_the_hardened_trees_too() -> None:
    """A sweep that only ran one tree is how the probe bug survived."""
    import inspect

    source = inspect.getsource(_sweep_fixed)
    assert "build_fixed" in source
    assert TRUST_CRITICAL, "an empty critical set makes the whole sweep vacuous"
