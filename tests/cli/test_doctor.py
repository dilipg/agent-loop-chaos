"""`alc doctor`: onboarding as a library feature rather than a coding-agent session.

The complaint this answers: if using the tool on a repository first requires someone to
hand-write a harness, the tool has failed. `doctor` points at a repository, attaches the
interceptors, runs the agent once with no faults, and reports the seams it found plus a
suite that targets them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_loop_chaos.cli import main

pytest.importorskip("httpx")

AGENT = "tests.fakes.unwrapped_agent:answer"


class TestTheEnvironmentReport:
    def test_with_no_target_it_reports_which_strategies_can_attach(
        self, capsys: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "httpx" in out
        assert "langchain-core" in out

    def test_json_is_exactly_one_object(self, capsys: Any, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        assert main(["doctor", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "strategies" in payload


class TestItDiscoversTheSeams:
    def test_it_finds_the_model_and_the_tool_of_an_unwrapped_agent(
        self, capsys: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """The agent wraps nothing, so everything here comes from interception."""
        monkeypatch.chdir(tmp_path)
        code = main(["doctor", AGENT, "--inputs", "what should I pack?"])
        out = capsys.readouterr().out
        assert code == 0, out
        assert "gpt-4o" in out, "the model was not discovered"
        assert "/v1/current" in out, "the HTTP tool was not discovered"

    def test_it_prints_a_suite_that_targets_what_it_found(
        self, capsys: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """A reader should be able to copy the output into a file and run it."""
        monkeypatch.chdir(tmp_path)
        main(["doctor", AGENT, "--inputs", "q"])
        out = capsys.readouterr().out
        assert "intercept" in out
        assert AGENT in out, "the suite does not name the entrypoint it was given"

    def test_json_reports_the_seams_as_data(
        self, capsys: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert main(["doctor", AGENT, "--inputs", "q", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["llm"] == ["gpt-4o"]
        assert any("/v1/current" in name for name in payload["tools"])


class TestItSaysWhenItFoundNothing:
    def test_an_agent_with_no_seams_is_reported_as_such(
        self, capsys: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """Silence is the finding. A reader must not read an empty report as success."""
        monkeypatch.chdir(tmp_path)
        code = main(["doctor", "tests.fakes.unwrapped_agent:no_seams", "--inputs", "q"])
        out = capsys.readouterr().out
        assert "no tool or llm" in out.lower()
        assert code == 1, "finding no payload seam is a finding, not a clean bill of health"

    def test_a_bad_entrypoint_is_a_usage_error(
        self, capsys: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert main(["doctor", "nope.does.not:exist"]) == 2


class TestItDistinguishesACrashFromAnAbsence:
    """Found by probing a real service with the wrong `--inputs`.

    The agent crashed before calling anything, and `doctor` reported "no tool or llm
    seam was reached" -- sending the reader after a hand-rolled client when the truth
    was one bad argument. An absence and a crash need different answers.
    """

    def test_a_crash_is_reported_as_a_crash(
        self, capsys: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.chdir(tmp_path)
        code = main(["doctor", "tests.fakes.unwrapped_agent:explodes", "--inputs", "q"])
        out = capsys.readouterr().out
        assert code == 2, "a probe that could not run is a usage problem, not a finding"
        assert "failed before reaching" in out
        assert "--inputs" in out, "the likeliest cause must be named"
        assert "No tool or llm seam was reached" not in out, "that diagnosis is misleading here"
