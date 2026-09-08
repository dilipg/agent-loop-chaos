"""An agent that could not be called is a configuration problem, not a finding.

Found in the final round of testing against a production service. Its entrypoint takes
seven arguments -- ids, two database handles, a config object -- and the suite carried
the placeholder `inputs` that `alc doctor` prints. The agent was never invoked: `steps`
was 0 and `error` said "cannot call agent 'run_user_pipeline' with the given inputs:
missing a required argument: 'user_id'".

The run reported `silent_wrong_answer` at **high** severity and wrote a work order,
because the auto-assertion `output_non_empty` failed on an output that was never
produced. Somebody handed that to a coding agent would go looking for a bug in a
function that had not run.

The engine already refuses to make this mistake about its own step limit -- "reporting
it in `error` would blame the agent for our limit, exactly as D-06 forbids". Same rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.errors import ConfigError


def needs_seven(entity_id: str, user_id: str, run_id: str, db: Any, config: Any) -> str:
    """Stands in for a real service entrypoint. Never runs in these tests."""
    return "unreachable"


class TestTheEngineRefuses:
    def test_it_raises_rather_than_scoring_a_run_that_never_happened(self, tmp_path: Path) -> None:
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False)
        with pytest.raises(ConfigError) as caught:
            engine.run(needs_seven, inputs="a string")
        message = str(caught.value)
        assert "user_id" in message, "the missing argument must be named"
        assert "Signature is" in message, "and the shape it wanted"

    @pytest.mark.asyncio
    async def test_the_async_path_refuses_too(self, tmp_path: Path) -> None:
        async def needs_two(user_id: str, db: Any) -> str:
            return "unreachable"

        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False)
        with pytest.raises(ConfigError):
            await engine.arun(needs_two, inputs="a string")

    def test_a_callable_agent_still_runs(self, tmp_path: Path) -> None:
        """The check must not reject the shapes that already worked."""
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False)
        result = engine.run(lambda question: f"answered {question}", inputs="hello")
        assert result.success
        assert "answered hello" in str(result.final_output)

    def test_an_agent_raising_its_own_error_is_still_the_agent(self, tmp_path: Path) -> None:
        """A `ConfigError` from inside the agent is agent behaviour, not our config.

        Only the failure to *call* it is ours to refuse."""

        def explodes(question: str) -> str:
            raise ConfigError("the service says its own settings are wrong")

        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False)
        result = engine.run(explodes, inputs="hello")
        assert result.success is False, "an agent that raises has failed"


class TestTheCliReportsItAsConfiguration:
    def test_it_exits_2_and_writes_no_work_order(self, tmp_path: Path, monkeypatch: Any) -> None:
        from agent_loop_chaos.cli import main

        suite = tmp_path / "suite.json"
        suite.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "scenarios": [
                        {
                            "id": "s.1",
                            "entrypoint": "tests.test_uncallable_agent:needs_seven",
                            "inputs": "your question here",
                            "faults": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        out = tmp_path / ".chaos"
        code = main(["run", str(suite), "--judge", "rules", "--out", str(out), "--no-html"])

        assert code == 2, "an uncallable agent is a usage error, not a failing scenario"
        assert not list(out.glob("**/AGENT_TASK.md")), (
            "a work order was written about a run that never happened"
        )
