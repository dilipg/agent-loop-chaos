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


class TestItWorksFromAProjectRoot:
    """Found by installing into a clean venv and following the README as a newcomer.

    A service is usually run from its own root with the package importable because the
    interpreter put the working directory on the path -- which is what `python -m` and
    pytest both do. A console script does not, so `alc doctor app.service:answer` failed
    with `No module named 'app'` until the reader thought to set `PYTHONPATH`.
    """

    def test_a_module_in_the_working_directory_is_importable(
        self, tmp_path: Any, monkeypatch: Any, capsys: Any
    ) -> None:
        (tmp_path / "svc_ok.py").write_text(
            "def answer(q=None):\n    return 'fine'\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PYTHONPATH", raising=False)
        code = main(["doctor", "svc_ok:answer", "--inputs", "q"])
        assert code in (0, 1), capsys.readouterr().out


class TestAProblemInTheProjectIsNotAnInternalError:
    """Exit 3 means "the library broke". A service that cannot import is not that.

    Sending a coding agent to debug the chaos library when its own settings module
    raised is the most expensive wrong turn this tool can cause.
    """

    def test_a_failing_import_is_a_usage_error(
        self, tmp_path: Any, monkeypatch: Any, capsys: Any
    ) -> None:
        (tmp_path / "svc_boom.py").write_text(
            'raise KeyError("Missing required env var: SERVICE_DB_URI")\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        code = main(["doctor", "svc_boom:answer", "--inputs", "q"])
        err = capsys.readouterr().err
        assert code == 2, "a service that cannot import is the caller's problem, not ours"
        assert "SERVICE_DB_URI" in err, "the cause must survive into the message"
        assert "Traceback" not in err, "a raw traceback reads as a library crash"

    def test_the_same_holds_for_alc_run(self, tmp_path: Any, monkeypatch: Any) -> None:
        import json

        (tmp_path / "svc_boom2.py").write_text(
            'raise KeyError("Missing required env var: SERVICE_DB_URI")\n', encoding="utf-8"
        )
        suite = tmp_path / "suite.json"
        suite.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "scenarios": [{"id": "s.1", "entrypoint": "svc_boom2:answer", "faults": []}],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        assert main(["run", str(suite), "--judge", "rules", "--no-html"]) == 2


class TestItFindsAGraphCompiledAtImport:
    """The shape that silently loses every node and state fault.

    A service that compiles its graph once at module level hands the engine nothing, so
    the probe sees an llm seam and reports success -- while node, edge, state and
    checkpoint faults, the ones a multi-node pipeline most needs, are quietly
    unavailable. `doctor` has to say so, because nothing else will.
    """

    def test_it_names_the_graph_and_how_to_reach_it(
        self, tmp_path: Any, monkeypatch: Any, capsys: Any
    ) -> None:
        pytest.importorskip("langgraph")
        monkeypatch.chdir(tmp_path)
        main(["doctor", "tests.fakes.internal_graph:run_singleton", "--inputs", "q"])
        out = capsys.readouterr().out
        assert "COMPILED" in out, "the module-level graph was not found"
        assert "seams" in out and "graph" in out, "the fix was not named"

    def test_the_suite_it_prints_targets_a_node(
        self, tmp_path: Any, monkeypatch: Any, capsys: Any
    ) -> None:
        pytest.importorskip("langgraph")
        monkeypatch.chdir(tmp_path)
        main(["doctor", "tests.fakes.internal_graph:run_singleton", "--inputs", "q"])
        out = capsys.readouterr().out
        assert "NodeSkipFault" in out, "the suggested suite skips the graph layers"

    def test_json_reports_the_graphs(self, tmp_path: Any, monkeypatch: Any, capsys: Any) -> None:
        pytest.importorskip("langgraph")
        monkeypatch.chdir(tmp_path)
        main(["doctor", "tests.fakes.internal_graph:run_singleton", "--inputs", "q", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert any("COMPILED" in g for g in payload["graphs"])

    def test_the_suite_names_a_real_node(
        self, tmp_path: Any, monkeypatch: Any, capsys: Any
    ) -> None:
        """A placeholder makes the suite something to edit; a real name makes it
        something to run. The node names are on the graph it just found."""
        pytest.importorskip("langgraph")
        monkeypatch.chdir(tmp_path)
        main(["doctor", "tests.fakes.internal_graph:run_singleton", "--inputs", "q"])
        out = capsys.readouterr().out
        assert "gather" in out, "the graph's own node names were not shown"
        assert "your_node_name" not in out, "the suite still carries a placeholder"

    def test_the_suite_it_prints_actually_loads(
        self, tmp_path: Any, monkeypatch: Any, capsys: Any
    ) -> None:
        """The flagship flow is "copy this and run it", so the printed suite has to be
        a valid one. It was not: `seams: {graph: ...}` was accepted by the strategy and
        rejected by the schema, so `alc run` refused the file `alc doctor` had just
        written."""
        pytest.importorskip("langgraph")
        pytest.importorskip("yaml")
        from agent_loop_chaos.scenarios import load_suite

        monkeypatch.chdir(tmp_path)
        main(["doctor", "tests.fakes.internal_graph:run_singleton", "--inputs", "q"])
        printed = capsys.readouterr().out
        suite_text = printed[printed.index("version:") :]
        path = tmp_path / "printed.yaml"
        path.write_text(suite_text, encoding="utf-8")

        suite = load_suite(path)
        assert suite.scenarios, "the printed suite has no scenarios"
        assert any(s.seams for s in suite.scenarios), "the graph seam did not survive"
