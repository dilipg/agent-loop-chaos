"""`suite.schema.json` — the file CI, the loop and the dashboard all consume.

D-25 gives it a real schema in this phase, and versions its field sets once: `1.0`
was written at suite end, `1.1` adds the loop fields *and* the live fields phase 10
needs, together. Anything the loop writes must validate here, which is the point of
having the schema at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loop_chaos.schema import SCHEMA_FILES, load_schema, validate_obj

REPO = Path(__file__).resolve().parents[2]


def _minimal() -> dict[str, object]:
    return {
        "schema_version": "1.1",
        "status": "completed",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:10Z",
        "seed": 1337,
        "planned": ["a", "b"],
        "current": None,
        "round": 1,
        "rounds_planned": 1,
        "passed": 1,
        "failed": 1,
        "skipped": 0,
        "failure_modes": {"crash_unhandled_exception": 1},
        "coverage": {"ToolCorruptionFault": 1},
        "results": [
            {
                "scenario_id": "a",
                "run_id": "run-abc12345",
                "success": True,
                "failure_mode": "none",
                "severity": "info",
                "report": "a/report.json",
                "agent_task": None,
            }
        ],
        "worst": ["b"],
        "tasks_written": ["b/AGENT_TASK.md"],
        "rounds": [],
        "flipped": [],
        "regressions": [],
        "judge_disagreement_rate": 0.0,
        "judge_latency_ms_total": 0,
        "wall_ms": 10,
        "tamper": {"probes_sha256": "0" * 64, "probes_changed": False, "scenarios": {}},
    }


class TestRegistration:
    def test_the_schema_file_exists_in_both_places(self) -> None:
        packaged = REPO / "src" / "agent_loop_chaos" / "schemas" / "suite.schema.json"
        assert (REPO / "schemas" / "suite.schema.json").is_file()
        assert packaged.is_file()

    def test_the_two_copies_are_byte_identical(self) -> None:
        a = (REPO / "schemas" / "suite.schema.json").read_bytes()
        b = (REPO / "src" / "agent_loop_chaos" / "schemas" / "suite.schema.json").read_bytes()
        assert a == b

    def test_it_is_registered_under_a_short_name(self) -> None:
        assert SCHEMA_FILES["suite"] == "suite.schema.json"
        assert load_schema("suite")["$id"].endswith("suite.schema.json")


class TestValidation:
    def test_a_minimal_document_validates(self) -> None:
        assert validate_obj(_minimal(), "suite") == []

    @pytest.mark.parametrize(
        "field",
        ["schema_version", "seed", "passed", "failed", "results"],
    )
    def test_required_fields_are_required(self, field: str) -> None:
        document = _minimal()
        del document[field]
        assert validate_obj(document, "suite") != []

    def test_unknown_top_level_fields_are_rejected(self) -> None:
        # `additionalProperties: false` is what makes "no field outside the schema"
        # enforceable rather than aspirational.
        assert validate_obj({**_minimal(), "surprise": 1}, "suite") != []

    def test_status_enum_is_closed(self) -> None:
        assert validate_obj({**_minimal(), "status": "vibing"}, "suite") != []

    def test_finished_at_is_nullable_for_a_live_run(self) -> None:
        live = {**_minimal(), "status": "running", "finished_at": None, "current": "b"}
        assert validate_obj(live, "suite") == []

    def test_judge_disagreement_rate_is_top_level_not_in_coverage(self) -> None:
        # D-25: `coverage`'s namespace is fault kinds and nothing else.
        schema = load_schema("suite")
        assert "judge_disagreement_rate" in schema["properties"]
        coverage = schema["properties"]["coverage"]
        assert coverage["additionalProperties"] == {"type": "integer"}

    def test_the_1_0_shape_still_validates(self) -> None:
        # Additive evolution: a 1.0 file written by M4-M6 must not become invalid.
        old = {
            k: v
            for k, v in _minimal().items()
            if k
            not in {
                "status",
                "planned",
                "current",
                "round",
                "rounds_planned",
                "rounds",
                "flipped",
                "regressions",
                "wall_ms",
                "tamper",
            }
        }
        old["schema_version"] = "1.0"
        assert validate_obj(old, "suite") == []


def test_a_written_suite_json_validates(tmp_path: Path) -> None:
    from agent_loop_chaos.bundle import write_suite_json

    path = Path(write_suite_json(tmp_path, [], seed=7))
    assert validate_obj(json.loads(path.read_text()), "suite") == []
