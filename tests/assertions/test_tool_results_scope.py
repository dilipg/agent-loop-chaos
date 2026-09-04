"""`EvidenceContext.tool_results` holds tool results, and only tool results.

The engine records every crossing's result into one history, keyed by name across
every layer -- tool, llm, node, edge, state. `tool_results` was built from all of it,
so a model's own response text was in the list the grounding check searches. Any
number the model invented therefore sourced itself, and `no_unsourced_numbers` could
essentially never fire on the case it exists for: an agent that fabricates a value
because a tool stopped returning it.
"""

from __future__ import annotations

from typing import Any

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import ToolCorruptionFault
from agent_loop_chaos.targeting import Target, Trigger


def _agent(engine: ChaosEngine) -> Any:
    """An agent whose model invents the number the tool stopped returning."""

    @engine.tool(name="billing", side_effecting=False)
    def billing(account: str = "ACME") -> dict[str, Any]:
        return {"account": account, "balance_usd": 240}

    def model(prompt: str) -> str:
        # The number is in the model's own words and nowhere in the tool result.
        return "The outstanding balance is 1875 USD."

    chat = engine.llm(model, name="default")

    def run(question: Any = None) -> str:
        billing()
        return chat("what is the balance?")

    return run


def _run(tmp_path: Any) -> Any:
    engine = ChaosEngine(
        seed=1337, out_dir=tmp_path, write_bundle=False, strict_schema=False, judge="rules"
    )
    engine.register_fault(
        ToolCorruptionFault(mutation_type="drop_key", keys=["balance_usd"]),
        target=Target(tool="billing"),
        trigger=Trigger(on_call=1),
    )
    return engine.run(
        _agent(engine),
        inputs={"question": "balance?"},
        scenario_id="grounding",
        expected_behavior="graceful_degradation",
        expect={"no_unsourced_numbers": {"enabled": True}},
    )


def test_a_model_invented_number_is_not_self_sourcing(tmp_path: Any) -> None:
    result = _run(tmp_path)
    grounding = [a for a in result.assertions if a["check"] == "no_unsourced_numbers"]
    assert grounding, "the scenario asked for the grounding check"
    assert not grounding[0]["ok"], (
        "1875 appears only in the model's own response; sourcing it from that response "
        "makes the check unable to fire on the case it exists for"
    )


def test_the_run_is_classified_as_a_fabrication(tmp_path: Any) -> None:
    result = _run(tmp_path)
    assert result.success is False
    assert result.failure_mode == "hallucination_on_corrupt_data"
