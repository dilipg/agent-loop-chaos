"""`alc run --record` and `--replay-cassette`, which until now did nothing at all.

Both flags were declared on the parser and read by no code. The only test covering them
asserted that they *parse*, so it passed for as long as they were dead -- a colleague
running `alc run --record tape.json` got no cassette, no warning and exit 0. The
cassette's own miss message told readers to use a flag that had never worked.

These tests assert the behaviour rather than the parsing, which is the half that was
missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos.cli import build_parser, main

pytest.importorskip("httpx")

AGENT = "tests.fakes.unwrapped_agent:answer"


def _suite(tmp_path: Path, **extra: Any) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "defaults": {
                    "entrypoint": AGENT,
                    "inputs": "what should I pack?",
                    "intercept": True,
                    **extra,
                },
                "scenarios": [
                    {
                        "id": "llm.truncated",
                        "faults": [
                            {
                                "type": "LLMTruncationFault",
                                "target": {"layer": "llm", "phase": "post"},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class TestTheFlagsExist:
    def test_record_takes_a_path(self) -> None:
        args = build_parser().parse_args(["run", "s.json", "--record", "c.json"])
        assert args.record == "c.json"

    def test_replay_cassette_takes_a_path(self) -> None:
        args = build_parser().parse_args(["run", "s.json", "--replay-cassette", "c.json"])
        assert args.replay_cassette == "c.json"


class TestRecordThenReplay:
    def test_a_recorded_suite_replays_from_the_file(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        tape = tmp_path / "tape.json"
        suite = _suite(tmp_path)

        assert main(["run", str(suite), "--judge", "rules", "--record", str(tape)]) in (0, 1)
        assert tape.is_file(), "--record wrote no cassette"
        recorded = json.loads(tape.read_text(encoding="utf-8"))
        assert recorded["entries"], "the cassette is empty"

        # Replay must not need the endpoint at all.
        assert main(["run", str(suite), "--judge", "rules", "--replay-cassette", str(tape)]) in (
            0,
            1,
        )

    def test_a_scenario_can_name_its_own_cassette(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        tape = tmp_path / "tape.json"
        main(["run", str(_suite(tmp_path)), "--judge", "rules", "--record", str(tape)])

        suite = _suite(tmp_path, cassette=str(tape))
        assert main(["run", str(suite), "--judge", "rules"]) in (0, 1)

    def test_replaying_a_cassette_that_does_not_exist_is_a_usage_error(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        """Exit 2, not 1. A missing fixture file is a configuration problem, and
        reporting it as a failing scenario blames the agent for the harness."""
        monkeypatch.chdir(tmp_path)
        code = main(
            ["run", str(_suite(tmp_path)), "--judge", "rules", "--replay-cassette", "nope.json"]
        )
        assert code == 2
        assert "--record" in capsys.readouterr().err, "the error must say how to fix itself"
