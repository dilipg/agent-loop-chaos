"""Golden files: the regression guard for everything the report pipeline produces.

Normalized per `docs/04-SCHEMAS.md` §3 — `tests/normalize.py` owns the strip list
and nothing else may add to it. A diff here means the report changed; that is
sometimes correct, and `make golden-update` is the only way to bless it.

D-40: the shipped fixtures under `schemas/examples/` must stay mutually consistent
with what the code actually emits, so they are regenerated from a real run and
diffed rather than hand-maintained.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.bundle import render_agent_task
from agent_loop_chaos.faults import ToolCorruptionFault
from agent_loop_chaos.schema import validate_obj
from agent_loop_chaos.targeting import Target, Trigger
from tests.normalize import normalize

GOLDEN = Path(__file__).resolve().parent / "golden"
UPDATE = os.environ.get("ALC_UPDATE_GOLDEN") == "1"


def golden_run(tmp_path: Path, mutation: str) -> Any:
    """Produce a deterministic failing run.

    Args:
        tmp_path: pytest's temp directory.
        mutation: Which mutation to inject.

    Returns:
        The `ChaosResult`.
    """
    # `judge="rules"` is pinned, not left to default. `judge=None` probes for a
    # model endpoint, so a developer with Ollama running would get an ensemble
    # verdict and a golden that never matches. D-07 names `--judge rules` as one of
    # the conditions for a byte-identical report; this is that condition.
    engine = ChaosEngine(seed=1337, out_dir=tmp_path / ".chaos", strict_schema=True, judge="rules")
    engine.register_fault(
        ToolCorruptionFault(mutation_type=mutation, keys=["temp_c"]),
        target=Target(tool="get_weather_data", phase="post"),
        trigger=Trigger(on_call=1),
    )

    @engine.tool(side_effecting=False)
    def get_weather_data(location: str = "Paris") -> dict[str, Any]:
        return {"data": [{"date": "2026-08-26", "condition": "sunny", "temp_c": 31}]}

    def agent(query: str) -> str:
        rows = get_weather_data()["data"]
        return f"It is {rows[0].get('temp_c', 24)}C in Paris."

    return engine.run(
        agent, inputs="Pack list for Paris", scenario_id=f"weather.tool_contract.{mutation}"
    )


def scrub(text: str) -> str:
    """Replace absolute paths with a placeholder.

    The run directory is a pytest temp path, so it differs every run. `run_id` is
    derived from the seed and the plan hash and is deliberately left alone -- it is
    stable, and pinning it is part of what the golden proves.

    Args:
        text: The rendered document.

    Returns:
        The text with absolute paths replaced.
    """
    return re.sub(
        r"/private/var/\S*?/\.chaos", "<run-dir>", re.sub(r"/\S*?/\.chaos", "<run-dir>", text)
    )


def compare(name: str, actual: str) -> None:
    """Compare rendered output against a stored golden.

    Args:
        name: The golden filename.
        actual: What the code produced.

    Raises:
        AssertionError: When they differ and the update flag is not set.
    """
    path = GOLDEN / name
    if UPDATE or not path.exists():
        path.write_text(actual, encoding="utf-8")
        if not UPDATE:
            pytest.skip(f"created missing golden {name}; re-run to compare")
        return
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{name} drifted. If the change is intended, run `make golden-update` and "
        "review the diff in the commit."
    )


@pytest.mark.parametrize("mutation", ["drop_key", "unit_swap"])
def test_report_matches_its_golden(tmp_path: Path, mutation: str) -> None:
    """Two scenarios, normalized, byte-compared."""
    result = golden_run(tmp_path, mutation)
    rendered = json.dumps(normalize(result.to_dict()), indent=2, sort_keys=True) + "\n"
    compare(f"report_{mutation}.json", rendered)


def test_agent_task_matches_its_golden(tmp_path: Path) -> None:
    """The work order is the product, so its rendering is pinned."""
    compare("AGENT_TASK.drop_key.md", scrub(render_agent_task(golden_run(tmp_path, "drop_key"))))


def test_the_same_seed_produces_the_same_report(tmp_path: Path) -> None:
    """Harness determinism, end to end through the real pipeline."""
    first = normalize(golden_run(tmp_path / "a", "drop_key").to_dict())
    second = normalize(golden_run(tmp_path / "b", "drop_key").to_dict())
    assert first == second


# ------------------------------------------------------------- D-40 consistency


def test_the_shipped_report_fixture_still_validates() -> None:
    """`schemas/examples/report_failing.json` is the reference a consumer reads."""
    path = Path(__file__).resolve().parents[1] / "schemas" / "examples" / "report_failing.json"
    assert validate_obj(json.loads(path.read_text(encoding="utf-8")), "report") == []


def test_the_shipped_report_fixture_uses_only_real_probe_codes() -> None:
    """D-40: the fixtures must stay consistent with the code that emits them.

    A fixture citing a probe that does not exist teaches a reader a code they will
    never see, and `tools/verify_pack.py` checks it for exactly that reason.
    """
    from agent_loop_chaos.probes import PROBE_PRECEDENCE

    path = Path(__file__).resolve().parents[1] / "schemas" / "examples" / "report_failing.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    for symptom in report["symptoms"]:
        assert symptom["code"] in set(PROBE_PRECEDENCE), symptom["code"]


def test_the_shipped_report_fixture_uses_only_real_assertion_checks() -> None:
    """The same consistency rule for the assertions block."""
    import dataclasses

    from agent_loop_chaos.assertions import Expect

    path = Path(__file__).resolve().parents[1] / "schemas" / "examples" / "report_failing.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(Expect)}
    for assertion in report.get("assertions", []):
        assert assertion["check"] in known, assertion["check"]


def test_the_shipped_trace_fixture_validates_line_by_line() -> None:
    """Every line of `trace_excerpt.jsonl` is a real event."""
    path = Path(__file__).resolve().parents[1] / "schemas" / "examples" / "trace_excerpt.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert validate_obj(json.loads(line), "trace") == []


def test_a_real_run_emits_only_documented_event_kinds(tmp_path: Path) -> None:
    """D-35: the schema's `kind` enum is the single source of truth.

    A real run that emitted something outside it would mean the fixture and the code
    had drifted in the one direction the schema cannot catch.
    """
    from typing import get_args

    from agent_loop_chaos.trace import EventKind, TraceRecorder

    result = golden_run(tmp_path, "drop_key")
    kinds = {e["kind"] for e in TraceRecorder.load(result.artifacts["trace"])}
    assert kinds <= set(get_args(EventKind))
