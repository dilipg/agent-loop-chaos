"""`LoopReport.markdown()` against a golden.

This document is the entire output of an unattended loop -- the thing a human reads
the next morning. A golden is the only way to notice it quietly becoming unreadable.

Refresh with `make golden-update` and read the diff: if the table got harder to scan,
that is the change, not a formatting detail.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent_loop_chaos.loop import LoopReport, RoundResult
from agent_loop_chaos.report import ChaosResult, assemble

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "loop_report.md"


def _result(scenario_id: str, *, success: bool, mode: str = "none") -> ChaosResult:
    """Build a result with just the fields `markdown()` reads."""
    result = assemble(
        trace=[],
        plan={"seed": 1337, "limits": {"max_steps": 25}},
        plan_hash="deadbeef",
        run_id=f"run-{scenario_id[:8]}",
        scenario_id=scenario_id,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        wall_ms=1,
        final_output="x",
        error=None,
        symptoms=[],
        assertions=[],
    )
    result.success = success
    result.failure_mode = mode  # type: ignore[assignment]
    return result


def _report() -> LoopReport:
    """A three-round loop covering every outcome the table can show."""
    rows = {
        "tool.drop_required_key": [False, True, True],
        "loop.pinned_tool_output": [False, False, False],
        "state.drop_location": [True, False, False],
    }
    modes = {
        "tool.drop_required_key": "crash_unhandled_exception",
        "loop.pinned_tool_output": "max_iterations_exhausted",
        "state.drop_location": "state_corruption_propagated",
    }
    outcomes = [
        {
            "tool.drop_required_key": "fail",
            "loop.pinned_tool_output": "fail",
            "state.drop_location": "pass",
        },
        {
            "tool.drop_required_key": "flipped_to_pass",
            "loop.pinned_tool_output": "still_failing",
            "state.drop_location": "flipped_to_fail",
        },
        {
            "tool.drop_required_key": "pass",
            "loop.pinned_tool_output": "still_failing",
            "state.drop_location": "still_failing",
        },
    ]
    rounds = [
        RoundResult(
            round=n + 1,
            results=[
                _result(name, success=passes[n], mode="none" if passes[n] else modes[name])
                for name, passes in rows.items()
            ],
            outcomes=outcomes[n],  # type: ignore[arg-type]
            wall_ms=100,
        )
        for n in range(3)
    ]
    return LoopReport(
        rounds=rounds,
        new_failures_per_round=[2, 1, 0],
        fixed_between_rounds=[[], ["tool.drop_required_key"], []],
        regressions=[
            "state.drop_location: regressed in round 2 (state_corruption_propagated); "
            "it passed in round 1"
        ],
        tasks_written=[Path("/out/loop.pinned_tool_output/run-loop.pin/AGENT_TASK.md")],
        out_dir=Path("/out"),
        tamper={"probes_sha256": "0" * 64, "probes_changed": False, "modified": []},
        wall_ms=300,
    )


def test_markdown_matches_its_golden() -> None:
    rendered = _report().markdown()
    if os.environ.get("ALC_UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(rendered, encoding="utf-8")
    assert GOLDEN.is_file(), "run `make golden-update`"
    assert rendered == GOLDEN.read_text(encoding="utf-8"), (
        "loop_report.md drifted. If the change is intended, run `make golden-update` "
        "and review the diff in the commit."
    )


def test_the_golden_shows_all_three_outcomes() -> None:
    """A golden that only covers the happy path is not protecting anything."""
    text = _report().markdown()
    assert "fixed in r2" in text
    assert "still failing (max_iterations_exhausted)" in text
    assert "**REGRESSION in r2**" in text
