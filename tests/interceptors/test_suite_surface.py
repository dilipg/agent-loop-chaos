"""`intercept: true` in a suite, and `--transport` on the command line.

Phase 2's switch is only useful if it can be reached without writing Python. A
colleague pointed at a repository writes a suite, not a harness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loop_chaos.scenarios import load_suite


def _suite(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(body, encoding="utf-8")
    return path


class TestTheSuiteKey:
    def test_a_scenario_can_ask_for_interception(self, tmp_path: Path) -> None:
        path = _suite(
            tmp_path,
            json.dumps(
                {
                    "version": "1.0",
                    "scenarios": [
                        {
                            "id": "t.on",
                            "entrypoint": "x:build",
                            "intercept": True,
                            "faults": [{"type": "LLMTruncationFault"}],
                        }
                    ],
                }
            ),
        )
        suite = load_suite(path)
        assert suite.scenarios[0].intercept is True

    def test_it_defaults_off(self, tmp_path: Path) -> None:
        path = _suite(
            tmp_path,
            json.dumps(
                {
                    "version": "1.0",
                    "scenarios": [{"id": "t.off", "entrypoint": "x:build", "faults": []}],
                }
            ),
        )
        assert load_suite(path).scenarios[0].intercept is False

    def test_defaults_can_turn_it_on_for_a_whole_suite(self, tmp_path: Path) -> None:
        """The usual shape: one repo, one interception setting, many scenarios."""
        path = _suite(
            tmp_path,
            json.dumps(
                {
                    "version": "1.0",
                    "defaults": {"entrypoint": "x:build", "intercept": True},
                    "scenarios": [{"id": "t.a", "faults": []}, {"id": "t.b", "faults": []}],
                }
            ),
        )
        assert [s.intercept for s in load_suite(path).scenarios] == [True, True]

    def test_the_schema_accepts_it(self, tmp_path: Path) -> None:
        """`scenarioBody` sets additionalProperties:false, so a new key must be added
        to the schema or every suite using it fails validation."""
        schema = json.loads(Path("schemas/scenario.schema.json").read_text(encoding="utf-8"))
        body = schema["$defs"]["scenarioBody"]
        assert "intercept" in body["properties"]
        assert body["properties"]["intercept"]["type"] == "boolean"


class TestTheCommandLineFlag:
    def test_intercept_flag_turns_it_on_for_every_scenario(self, tmp_path: Path) -> None:
        """So a colleague can try it on an existing suite without editing the file."""
        from agent_loop_chaos.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "suite.yaml", "--intercept"])
        assert args.intercept is True

    def test_it_is_off_when_not_given(self) -> None:
        from agent_loop_chaos.cli import build_parser

        args = build_parser().parse_args(["run", "suite.yaml"])
        assert args.intercept is False


class TestTheEngineGetsIt:
    def test_run_suite_passes_it_to_the_engine(self, tmp_path: Any, monkeypatch: Any) -> None:
        """The value has to survive the trip from YAML to the engine constructor."""
        seen: list[bool] = []
        import agent_loop_chaos.engine as engine_module
        import agent_loop_chaos.loop as loop_module
        from agent_loop_chaos import ChaosEngine

        class _Spy(ChaosEngine):
            def __init__(self, **kw: Any) -> None:
                seen.append(bool(kw.get("intercept")))
                super().__init__(**kw)

        # `loop` imports the engine inside the function, so the patch belongs on the
        # module the name is read from.
        monkeypatch.setattr(engine_module, "ChaosEngine", _Spy)

        from agent_loop_chaos.scenarios import Scenario

        loop_module.run_suite(
            [
                Scenario(
                    id="t.on",
                    entrypoint="tests.fakes.apps:build_good",
                    intercept=True,
                )
            ],
            out_dir=tmp_path,
            judge="rules",
        )
        # Two engines: the baseline and the faulted run. Both must be instrumented
        # the same way, or `baseline.diff` describes the instrumentation rather than
        # the fault.
        assert seen == [True, True], f"intercept did not reach every engine: {seen}"
