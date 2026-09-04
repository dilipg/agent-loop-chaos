"""The false-positive guard: `good_agent` must pass, `naive_tool_agent` must fail.

`docs/07-TESTING.md` §2: "If a new probe makes either fail, the probe has a false
positive and the probe is wrong until proven otherwise." This is the test that makes
that sentence enforceable, and it exercises all twenty probes at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults.base import fault_from_dict
from agent_loop_chaos.scenarios import PRESETS, resolve_preset
from agent_loop_chaos.targeting import Target, Trigger
from tests.fakes import FakeLLM, good_agent, naive_tool_agent

RAW = {"data": [{"temp_c": 21, "city": "Paris", "condition": "sunny"}]}


def arm(engine: ChaosEngine, preset: str) -> int:
    """Register every registered fault in a preset against the fixture agent.

    Args:
        engine: The engine to arm.
        preset: The preset name.

    Returns:
        How many faults were armed.
    """
    specs, _skipped = resolve_preset(preset)
    armed = 0
    for index, spec in enumerate(specs):
        fault = fault_from_dict(spec.to_dict())
        layers = {layer for layer, _phase in fault.accepts}
        target = Target(tool="fetch") if "tool" in layers else Target(llm="default")
        try:
            engine.register_fault(
                fault,
                target=target,
                trigger=Trigger(**{**spec.trigger, "max_fires": spec.trigger.get("max_fires", 3)}),
                fault_id=f"f{index + 1}",
            )
        except Exception:
            continue
        armed += 1
    return armed


def run_control(tmp_path: Path, preset: str, agent: Any) -> Any:
    """Run one agent under one preset.

    Args:
        tmp_path: pytest's temp directory.
        preset: The preset to arm.
        agent: The agent callable.

    Returns:
        The `ChaosResult`.
    """
    engine = ChaosEngine(seed=1337, out_dir=tmp_path / preset, write_bundle=False)
    arm(engine, preset)
    return engine.run(
        agent, inputs="plan a trip to Paris", initial_state={"query": "plan a trip to Paris"}
    )


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_good_agent_survives_every_preset(tmp_path: Path, preset: str) -> None:
    """The negative control. Any probe that fires here has a false positive.

    `good_agent` validates before indexing, never invents a number, retries with a
    cap and says when it gave up, breaks a repeat cycle at three, fences untrusted
    content, and keeps the objective durable. Nothing the harness does to it should
    produce a finding attributable to the agent.
    """
    engine = ChaosEngine(seed=1337, out_dir=tmp_path / preset, write_bundle=False)
    arm(engine, preset)

    llm = FakeLLM(["The weather data was unavailable, so I cannot give a temperature."])

    @engine.tool
    def fetch() -> dict[str, Any]:
        return RAW

    def agent(query: str, state: dict[str, Any]) -> str:
        good_agent.plan(query, state)
        # The control handles a raising tool the way the catalog prescribes: a
        # bounded retry, then degrade and say so. Calling `fetch()` bare would be
        # `no_retry_agent`, not the control.
        raw: dict[str, Any] | None = None
        for _ in range(3):
            try:
                raw = fetch()
                break
            except Exception:
                continue
        if raw is None:
            return "The weather service was unavailable, so I could not complete this."
        summary = good_agent.summarize(raw)
        good_agent.summarize_ticket(summary, llm)
        return summary

    result = engine.run(
        agent, inputs="plan a trip to Paris", initial_state={"query": "plan a trip to Paris"}
    )
    attributable = [
        s
        for s in result.symptoms
        # `no_output_validation` at `low` is the documented weak-evidence case for an
        # uninstrumented agent, not a finding against this one.
        if not (s["code"] == "no_output_validation" and s["severity"] == "low")
    ]
    assert attributable == [], f"{preset}: probes fired on the control: {attributable}"


def test_naive_tool_agent_fails_tool_contract(tmp_path: Path) -> None:
    """The positive control. A preset that finds nothing is a preset that proves nothing."""
    engine = ChaosEngine(seed=1337, out_dir=tmp_path / "tc", write_bundle=False)
    arm(engine, "tool_contract")

    @engine.tool
    def fetch() -> dict[str, Any]:
        return RAW

    def agent(query: str) -> str:
        return naive_tool_agent.run(fetch())

    result = engine.run(agent, inputs="plan a trip to Paris")
    assert result.success is False, "tool_contract must find the unguarded index"
