"""Determinism, with its boundary stated.

Two guarantees, and it matters which one is under test (D-07):

- **Harness determinism** is unconditional — same seed and `plan_hash` give
  identical fault decisions and RNG draws.
- **Run determinism** additionally needs a deterministic agent, a scripted model,
  `--judge rules` and a single-threaded graph. Every test here supplies those.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults.base import NoopFault
from agent_loop_chaos.targeting import Target, Trigger
from tests.normalize import STRIP_KEYS, normalize


def scripted_run(tmp_path: Path, seed: int, *, probability: float = 1.0) -> Any:
    """Run a fully deterministic agent under one probabilistic fault.

    Args:
        tmp_path: pytest's temp directory.
        seed: The run seed.
        probability: The trigger probability, so a test can choose whether the RNG
            is consulted at all.

    Returns:
        The `ChaosResult`.
    """
    eng = ChaosEngine(seed=seed, out_dir=tmp_path / f".chaos-{seed}", strict_trace=True)
    eng.register_fault(
        NoopFault(),
        target=Target(tool="t", phase="post"),
        trigger=Trigger(probability=probability, max_fires=99),
    )

    @eng.tool
    def t(x: int) -> dict[str, int]:
        return {"x": x}

    @eng.llm
    def gen(prompt: str) -> str:
        return f"scripted:{prompt}"

    def agent(q: int) -> dict[str, Any]:
        return {"tools": [t(q + i) for i in range(4)], "text": gen("summarize")}

    return eng.run(agent, inputs=1, scenario_id="determinism")


def test_the_same_seed_gives_an_identical_normalized_report(tmp_path: Path) -> None:
    """The headline guarantee, under the conditional set D-07 names."""
    first = scripted_run(tmp_path, 1337)
    second = scripted_run(tmp_path, 1337)
    assert normalize(first.to_dict()) == normalize(second.to_dict())


def test_the_same_seed_gives_an_identical_plan_hash(tmp_path: Path) -> None:
    """`plan_hash` proves plan identity, which is why `run_id` may be stripped."""
    assert scripted_run(tmp_path, 1337).plan_hash == scripted_run(tmp_path, 1337).plan_hash


def test_two_different_seeds_diverge_for_a_probabilistic_fault(tmp_path: Path) -> None:
    """Without this, "seeded" would be indistinguishable from "fixed"."""
    fire_counts = set()
    for seed in range(20, 40):
        result = scripted_run(tmp_path, seed, probability=0.5)
        fire_counts.add(result.injected_faults[0]["fire_count"])
    assert len(fire_counts) > 1, "a probabilistic fault must vary across seeds"


def test_a_probabilistic_fault_is_reproducible_within_a_seed(tmp_path: Path) -> None:
    """Chaotic on request, deterministic by default."""
    first = scripted_run(tmp_path, 4242, probability=0.5)
    second = scripted_run(tmp_path, 4242, probability=0.5)
    assert first.injected_faults[0]["fire_count"] == second.injected_faults[0]["fire_count"]
    assert normalize(first.to_dict()) == normalize(second.to_dict())


def test_two_runs_in_one_process_do_not_share_a_stream(tmp_path: Path) -> None:
    """D-02: the reason `rng` has no module-level memo.

    If a global cache existed, the second run would continue the first run's stream
    and produce a different fire pattern from the same seed.
    """
    first = scripted_run(tmp_path, 777, probability=0.5)
    second = scripted_run(tmp_path, 777, probability=0.5)
    assert first.randomness == second.randomness


def test_the_normalizer_strips_every_documented_field(tmp_path: Path) -> None:
    """`tests/normalize.py` owns the strip list, and nothing else may add to it."""
    report = scripted_run(tmp_path, 1337).to_dict()
    normalized = normalize(report)
    assert "started_at" in report
    assert "started_at" not in normalized
    assert "run_id" not in normalized
    assert normalized["artifacts"]["trace"] == "<path>"


def test_the_strip_list_is_the_documented_one() -> None:
    """A silent addition here would hide real non-determinism (`docs/04` §3)."""
    for field in ("started_at", "finished_at", "duration_ms", "wall_ms", "ts", "ts_mono_ms"):
        assert field in STRIP_KEYS
    for field in ("chaos_narrative", "root_cause_hypothesis", "refinement_hint", "narrative"):
        assert field in STRIP_KEYS, "model-authored fields are non-deterministic (D-07)"


def test_the_trace_is_deterministic_after_normalization(tmp_path: Path) -> None:
    """Harness determinism covers the trace, not just the report."""
    from agent_loop_chaos.trace import TraceRecorder

    first = scripted_run(tmp_path, 5150)
    second = scripted_run(tmp_path, 5150)
    left = [normalize(e) for e in TraceRecorder.load(first.artifacts["trace"])]
    right = [normalize(e) for e in TraceRecorder.load(second.artifacts["trace"])]
    assert left == right
