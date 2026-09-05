"""A coroutine entrypoint, driven from a suite.

`engine.run` is synchronous and `engine.arun` is not, which is fine when a caller
picks. A YAML suite does not get to pick -- it names `module:attr` and the loop
resolves it -- so `run_suite` has to notice a coroutine agent and drive it correctly.

Before this, it called the coroutine function, never awaited the coroutine, and
reported `failure_mode: unknown` with a `RuntimeWarning` on stderr. That is the exact
shape of failure this library exists to catch: something reporting it did the thing
while doing nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.loop import run_suite
from agent_loop_chaos.scenarios import Scenario


def build_async(engine: ChaosEngine) -> object:
    """An async agent that calls one tool.

    Args:
        engine: The engine to register the tool with.

    Returns:
        A coroutine function.
    """

    async def lookup(sku: str) -> dict[str, int]:
        return {"on_hand": 4}

    wrapped = engine.wrap_callable(lookup, name="lookup", layer="tool")

    async def agent(inputs: str) -> str:
        stock = await wrapped(inputs)
        return f"{inputs}: {stock['on_hand']} in stock"

    return agent


def _scenario() -> Scenario:
    return Scenario(
        id="async.entrypoint",
        entrypoint="tests.test_async_entrypoint:build_async",
        inputs="SKU-77",
        faults=[
            {
                "type": "ToolCorruptionFault",
                "params": {"mutation_type": "drop_key", "keys": ["on_hand"]},
                "trigger": {"on_call": 1},
            }
        ],
        expected_behavior="graceful_degradation",
    )


class TestSuiteDrivesACoroutineAgent:
    def test_it_actually_runs(self, tmp_path: Path) -> None:
        [result] = run_suite([_scenario()], out_dir=tmp_path, judge="rules")
        assert result.failure_mode != "unknown", (
            "the coroutine was never awaited; the suite reported a verdict about nothing having run"
        )
        assert len(result.tool_calls) >= 1

    def test_the_fault_reaches_it(self, tmp_path: Path) -> None:
        [result] = run_suite([_scenario()], out_dir=tmp_path, judge="rules")
        assert [f["type"] for f in result.injected_faults if f.get("fired")] == [
            "ToolCorruptionFault"
        ]

    def test_the_baseline_runs_too(self, tmp_path: Path) -> None:
        [result] = run_suite([_scenario()], out_dir=tmp_path, judge="rules")
        assert result.baseline and result.baseline.get("ran") is not False


class TestRunRefusesACoroutineAgent:
    """`engine.run` cannot await; saying so beats returning a coroutine object."""

    def test_it_names_arun(self, tmp_path: Path) -> None:
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, write_bundle=False)
        with pytest.raises(ConfigError, match="arun"):
            engine.run(build_async(engine), inputs="SKU-77")
