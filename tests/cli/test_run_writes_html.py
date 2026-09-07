"""`alc run` leaves a viewable page behind, not just JSON.

`report.json` and `suite.json` are written for machines, and rendering them meant a
second command nobody remembers. A run now writes `index.html` beside `suite.json`, so
the way to look at a result is to open it.

The failure this must never cause: an export problem is not a run problem. The library
must never crash the run it is observing, and a page is the least important thing in the
bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loop_chaos.cli import build_parser, main


def _suite(tmp_path: Path) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "scenarios": [
                    {
                        "id": "html.demo",
                        "title": "A tool loses a required key",
                        "entrypoint": "tests.fakes.apps:build_naive",
                        "inputs": "what should I pack for Paris?",
                        "faults": [
                            {
                                "type": "ToolCorruptionFault",
                                "target": {"tool": "get_weather_data", "phase": "post"},
                                "params": {"mutation_type": "drop_key", "keys": ["temp_c"]},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class TestItWritesThePage:
    def test_a_run_leaves_an_index_html(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        out = tmp_path / ".chaos"
        main(["run", str(_suite(tmp_path)), "--judge", "rules", "--out", str(out)])

        page = out / "index.html"
        assert page.is_file(), "no page was written"
        text = page.read_text(encoding="utf-8")
        assert "window.__ALC__" in text, "the page carries no data"
        assert "html.demo" in text, "the scenario is not in the page"

    def test_the_page_is_named_in_the_summary(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        """A file nobody is told about is a file nobody opens."""
        monkeypatch.chdir(tmp_path)
        out = tmp_path / ".chaos"
        main(["run", str(_suite(tmp_path)), "--judge", "rules", "--out", str(out)])
        printed = capsys.readouterr()
        assert "index.html" in printed.out + printed.err

    def test_it_needs_no_server_to_render(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Opened from the filesystem, so the data must be inline rather than fetched."""
        monkeypatch.chdir(tmp_path)
        out = tmp_path / ".chaos"
        main(["run", str(_suite(tmp_path)), "--judge", "rules", "--out", str(out)])
        text = (out / "index.html").read_text(encoding="utf-8")
        start = text.index("window.__ALC__ = ") + len("window.__ALC__ = ")
        blob = json.loads(text[start : text.index("\n</script>", start)].rstrip().rstrip(";"))
        assert blob["suite"]["results"], "the inline blob has no results"
        assert blob["runs"], "the inline blob has no runs"


class TestItCanBeTurnedOff:
    def test_no_html_suppresses_it(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        out = tmp_path / ".chaos"
        main(["run", str(_suite(tmp_path)), "--judge", "rules", "--out", str(out), "--no-html"])
        assert not (out / "index.html").exists()

    def test_the_flag_exists(self) -> None:
        args = build_parser().parse_args(["run", "s.json", "--no-html"])
        assert args.no_html is True


class TestAnExportProblemIsNotARunProblem:
    def test_a_failing_export_does_not_fail_the_run(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        """The page is the least important thing in the bundle."""
        monkeypatch.chdir(tmp_path)
        out = tmp_path / ".chaos"

        def boom(*_a: Any, **_kw: Any) -> str:
            raise OSError("the packaged page is missing")

        monkeypatch.setattr("agent_loop_chaos.dashboard.export.export_html", boom)
        code = main(["run", str(_suite(tmp_path)), "--judge", "rules", "--out", str(out)])

        assert code == 1, "the scenario's own verdict must still decide the exit code"
        assert (out / "suite.json").is_file(), "the machine-readable bundle is still there"
        assert not (out / "index.html").exists()


class TestJsonOutputStaysClean:
    def test_json_mode_prints_exactly_one_object(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        """`--json` promises one JSON object on stdout and nothing else."""
        monkeypatch.chdir(tmp_path)
        out = tmp_path / ".chaos"
        main(["run", str(_suite(tmp_path)), "--judge", "rules", "--out", str(out), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert "results" in payload
        assert (out / "index.html").is_file(), "the page is still written in json mode"
