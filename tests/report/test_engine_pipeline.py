"""Lifecycle steps 8-13 wired into `ChaosEngine.run`.

The pieces already exist and are unit-tested; this is the wiring, and the only
tests worth having here are end-to-end ones that prove the seams connect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import ToolCorruptionFault


def engine(tmp_path: Path, **kw: Any) -> ChaosEngine:
    """Build an engine writing into a temp directory.

    Args:
        tmp_path: pytest's temp directory.
        **kw: Constructor overrides.

    Returns:
        The engine.
    """
    kw.setdefault("seed", 1337)
    kw.setdefault("out_dir", tmp_path / ".chaos")
    return ChaosEngine(**kw)


def test_a_clean_run_passes_and_produces_no_work_order(tmp_path: Path) -> None:
    """No fault, no findings, no AGENT_TASK.md."""
    eng = engine(tmp_path)

    @eng.tool
    def get_weather() -> dict[str, Any]:
        return {"data": [{"temp_c": 21}]}

    result = eng.run(
        lambda: f"It is {get_weather()['data'][0]['temp_c']}C.",
        expected_behavior="ignore_and_continue",
    )
    assert result.success is True
    assert result.validate() == []
    assert not (Path(result.artifacts["run_dir"]) / "AGENT_TASK.md").exists()


def test_the_motivating_example_fails(tmp_path: Path) -> None:
    """Tool returns nothing usable, agent invents a number, run must FAIL.

    This is the case the whole project exists for, and the one the old
    `completed_unaffected` hole let pass.
    """
    eng = engine(tmp_path)
    eng.register_fault(
        ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
        target_tool="get_weather",
    )

    @eng.tool
    def get_weather() -> dict[str, Any]:
        return {"data": [{"temp_c": 21, "city": "Paris"}]}

    def agent() -> str:
        get_weather()
        return "It is 24C in Paris."  # invented

    result = eng.run(agent, scenario_id="motivating")
    assert result.success is False
    assert result.verdict["observed_behavior"] == "hallucinated"
    assert result.failure_mode == "hallucination_on_corrupt_data"
    assert result.validate() == []


def test_the_bundle_is_written_for_a_failing_run(tmp_path: Path) -> None:
    """All seven artifacts, per `docs/04` §9."""
    eng = engine(tmp_path)
    eng.register_fault(ToolCorruptionFault(mutation_type="empty_json"), target_tool="t")

    @eng.tool
    def t() -> dict[str, Any]:
        return {"temp_c": 21}

    result = eng.run(lambda: (t(), "It is 24C.")[1], scenario_id="bundled")
    run_dir = Path(result.artifacts["run_dir"])
    for name in ("report.json", "trace.jsonl", "plan.json", "AGENT_TASK.md"):
        assert (run_dir / name).exists(), name
    written = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert written["run_id"] == result.run_id


def test_auto_assertions_are_synthesized_from_what_the_fault_did(tmp_path: Path) -> None:
    """A scenario with no `expect` block is not unchecked (§4.4)."""
    eng = engine(tmp_path)
    eng.register_fault(
        ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]), target_tool="t"
    )

    @eng.tool
    def t() -> dict[str, Any]:
        return {"temp_c": 21}

    result = eng.run(lambda: (t(), "ok")[1])
    checks = {a["check"] for a in result.assertions}
    assert "no_claim_about" in checks
    assert all(a["source"] == "auto" for a in result.assertions)


def test_a_declared_expect_block_is_evaluated(tmp_path: Path) -> None:
    """The author's own checks, alongside the synthesized ones."""
    from agent_loop_chaos.assertions import Expect

    eng = engine(tmp_path)

    @eng.tool
    def t() -> str:
        return "x"

    result = eng.run(
        lambda: (t(), "done")[1],
        expect=Expect(output_matches=["never matches this"]),
        expected_behavior="ignore_and_continue",
    )
    assert result.success is False
    assert any(a["check"] == "output_matches" and not a["ok"] for a in result.assertions)


def test_probes_run_and_land_in_the_report(tmp_path: Path) -> None:
    """Symptoms reach `symptoms[]` with real evidence."""
    eng = engine(tmp_path)

    @eng.tool
    def t() -> str:
        return "x"

    result = eng.run(lambda: (t(), "")[1], expected_behavior="ignore_and_continue")
    codes = {s["code"] for s in result.symptoms}
    assert "empty_final_answer" in codes
    assert all(s["evidence"] for s in result.symptoms)


def test_the_dry_run_control_passes(tmp_path: Path) -> None:
    """D-12: a dry run must classify exactly as an unfaulted baseline."""
    eng = engine(tmp_path, dry_run=True)
    eng.register_fault(ToolCorruptionFault(mutation_type="empty_json"), target_tool="t")

    @eng.tool
    def t() -> dict[str, Any]:
        return {"temp_c": 21}

    result = eng.run(lambda: f"It is {t()['temp_c']}C.", expected_behavior="ignore_and_continue")
    assert result.success is True
    assert result.verdict["observed_behavior"] == "completed_unaffected"


def test_must_not_blocks_a_run_that_would_otherwise_pass(tmp_path: Path) -> None:
    """A scenario can declare codes that always fail it."""
    eng = engine(tmp_path)

    @eng.tool
    def t() -> str:
        return "x"

    result = eng.run(
        lambda: (t(), "")[1],
        must_not=["empty_final_answer"],
        expected_behavior="ignore_and_continue",
    )
    assert result.success is False


def test_unit_swap_is_detected_because_injected_values_are_recorded(tmp_path: Path) -> None:
    """R2 needs `values_injected` populated, or the nastiest fault is undetectable.

    `unit_swap` changes the value and keeps the label, so the swapped number *is* in
    the payload the agent received. Unless the harness records that it planted it,
    `no_unsourced_numbers` finds it sourced and the run passes -- which is exactly
    the silent-wrong-answer case the catalog says this fault exists to prove.
    """
    eng = engine(tmp_path)
    eng.register_fault(
        ToolCorruptionFault(mutation_type="unit_swap", keys=["temp_c"]), target_tool="t"
    )

    @eng.tool
    def t() -> dict[str, Any]:
        return {"temp_c": 21}

    def agent() -> str:
        return f"It is {t()['temp_c']}C."

    result = eng.run(agent)
    assert result.success is False, "an unsourced swapped unit must not pass"
    assert result.verdict["observed_behavior"] == "hallucinated"
