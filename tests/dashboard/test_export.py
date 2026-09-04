"""The single-file HTML export.

`docs/10` §7: one file, data inlined, no server, no network. §9 adds the two rules
that make it safe to attach to a ticket -- a second `redact()` pass, and `</`
escaped so a payload cannot end the script element it is sitting inside (D-38).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agent_loop_chaos.dashboard.export import build_blob, export_html


def _run(root: Path, scenario: str, run_id: str, events: list[dict]) -> Path:
    run = root / scenario / run_id
    run.mkdir(parents=True)
    (run / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run / "report.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "scenario_id": scenario,
                "success": False,
                "failure_mode": "silent_wrong_answer",
                "severity": "high",
            }
        )
    )
    (run / "AGENT_TASK.md").write_text("# fix the thing\n")
    return run


@pytest.fixture
def chaos(tmp_path: Path) -> Path:
    root = tmp_path / ".chaos"
    _run(
        root,
        "tool.drop_key",
        "run-a",
        [
            {"seq": 1, "kind": "run_started", "layer": "engine"},
            {
                "seq": 2,
                "kind": "fault_fired",
                "layer": "tool",
                "payload": {"note": "dropped price"},
            },
            {"seq": 3, "kind": "run_finished", "layer": "engine"},
        ],
    )
    (root / "suite.json").write_text(json.dumps({"schema_version": "1.1", "status": "completed"}))
    return root


def _blob_of(page: str) -> dict:
    match = re.search(r"window\.__ALC__ = (.*);\n</script>", page)
    assert match, "no data blob in the export"
    return json.loads(match.group(1).replace("<\\/", "</"))


class TestExport:
    def test_it_inlines_the_data(self, chaos: Path) -> None:
        blob = _blob_of(export_html(chaos))
        assert blob["suite"]["status"] == "completed"
        assert [r["row"]["run_id"] for r in blob["runs"]] == ["run-a"]
        assert [e["seq"] for e in blob["runs"][0]["events"]] == [1, 2, 3]
        assert blob["runs"][0]["task"].startswith("# fix the thing")

    def test_it_references_nothing_external(self, chaos: Path) -> None:
        page = export_html(chaos)
        assert "http" + "://" not in page and "http" + "s://" not in page

    def test_it_notes_truncation_honestly(self, chaos: Path) -> None:
        _run(chaos, "tool.loop", "run-b", [{"seq": i, "kind": "step"} for i in range(1, 40)])
        page = export_html(chaos, max_trace_events=5)
        assert "trace truncated for run-b" in page
        blob = _blob_of(page)
        long_run = next(r for r in blob["runs"] if r["row"]["run_id"] == "run-b")
        assert long_run["truncated"] > 0

    def test_it_is_one_document(self, chaos: Path) -> None:
        page = export_html(chaos)
        assert page.count("<html") == 1 and page.strip().endswith("</html>")


class TestExportSafety:
    """The XSS fixture: a trace that tries to break out of the page."""

    @pytest.fixture
    def hostile(self, tmp_path: Path) -> Path:
        root = tmp_path / ".chaos"
        _run(
            root,
            "llm.injection",
            "run-x",
            [
                {
                    "seq": 1,
                    "kind": "fault_fired",
                    "layer": "llm",
                    "payload": {
                        "note": "<script>alert(1)</script>",
                        "text": "before </script><img src=x onerror=alert(2)> after",
                        "bidi": "total: 100\u202eUSD\u202c",
                    },
                },
            ],
        )
        return root

    def test_a_payload_cannot_close_the_script_element(self, hostile: Path) -> None:
        page = export_html(hostile)
        # Exactly the page's own script elements survive; the payload's do not.
        body = page.split("window.__ALC__ = ", 1)[1].split(";\n</script>", 1)[0]
        assert "</script>" not in body
        assert "</" not in body
        assert "<\\/script>" in body

    def test_the_payload_still_round_trips(self, hostile: Path) -> None:
        blob = _blob_of(export_html(hostile))
        payload = blob["runs"][0]["events"][0]["payload"]
        assert payload["note"] == "<script>alert(1)</script>"
        assert "\u202e" in payload["bidi"]

    def test_a_secret_in_the_trace_is_redacted_again(self, tmp_path: Path) -> None:
        root = tmp_path / ".chaos"
        _run(
            root,
            "tool.leak",
            "run-s",
            [
                {"seq": 1, "kind": "tool_result", "payload": {"api_key": "sk-live-abcdef123456"}},
            ],
        )
        page = export_html(root)
        assert "sk-live-abcdef123456" not in page
        assert "<redacted:" in page

    def test_payloads_are_never_exported(self, chaos: Path) -> None:
        secret = chaos / "tool.drop_key" / "run-a" / "payloads"
        secret.mkdir()
        (secret / "big.json").write_text(json.dumps({"leaked": "do-not-ship"}))
        assert "do-not-ship" not in export_html(chaos)


def test_build_blob_on_an_empty_directory(tmp_path: Path) -> None:
    blob = build_blob(tmp_path)
    assert blob["runs"] == []
