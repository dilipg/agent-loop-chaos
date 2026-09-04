"""`alc replay` — re-run a stored plan, and refuse when it cannot verify.

D-31 is the whole design here. `plan_hash` covers the plan, not the experiment: the
agent's source, the model and the tool fixtures all sit outside it, and triggers are
relative to the agent's own call sequence. So a replay that only checks the hash will
happily relocate every fault while reporting a match. It has to compare what actually
happened, and say so when it differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.faults import ToolCorruptionFault
from agent_loop_chaos.targeting import Target, Trigger


def _agent(engine: ChaosEngine) -> Any:
    @engine.tool(name="weather", side_effecting=False)
    def weather(city: str = "Paris") -> dict[str, Any]:
        return {"city": city, "temp_c": 21}

    def run(question: Any = None) -> str:
        data = weather()
        return f"It is {data['temp_c']}C in {data['city']}."

    return run


def _first_run(tmp_path: Path) -> Any:
    engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
    engine.register_fault(
        ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
        target=Target(tool="weather"),
        trigger=Trigger(on_call=1),
    )
    return engine.run(_agent(engine), inputs={"question": "?"}, scenario_id="drop")


class TestTheEnvelope:
    """`reproduce.env` records what the plan hash cannot (D-31)."""

    def test_env_records_the_library_and_python_version(self, tmp_path: Path) -> None:
        env = _first_run(tmp_path).reproduce.get("env") or {}
        assert env.get("library_version")
        assert env.get("python")

    def test_env_records_the_entrypoint_source_hash(self, tmp_path: Path) -> None:
        env = _first_run(tmp_path).reproduce.get("env") or {}
        assert len(env.get("entrypoint_source_sha256", "")) == 64

    def test_the_hash_changes_when_the_agent_changes(self, tmp_path: Path) -> None:
        from agent_loop_chaos.engine import entrypoint_fingerprint

        def one(x: Any = None) -> str:
            return "a"

        def two(x: Any = None) -> str:
            return "a different body"

        assert entrypoint_fingerprint(one) != entrypoint_fingerprint(two)

    def test_an_unreadable_entrypoint_is_not_an_error(self) -> None:
        from agent_loop_chaos.engine import entrypoint_fingerprint

        # A C extension or a builtin has no source. Recording nothing beats crashing
        # the run over provenance metadata.
        assert entrypoint_fingerprint(len) == ""

    def test_the_report_is_still_schema_valid(self, tmp_path: Path) -> None:
        from agent_loop_chaos.schema import validate_obj

        assert validate_obj(_first_run(tmp_path).to_dict(), "report") == []


class TestReplay:
    def test_it_reproduces_the_original_verdict(self, tmp_path: Path) -> None:
        first = _first_run(tmp_path)
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        again = engine.replay(Path(first.artifacts["run_dir"]), target=_agent(engine))
        assert again.success == first.success
        assert again.failure_mode == first.failure_mode
        assert again.plan_hash == first.plan_hash

    def test_the_attempt_counter_advances(self, tmp_path: Path) -> None:
        first = _first_run(tmp_path)
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        again = engine.replay(Path(first.artifacts["run_dir"]), target=_agent(engine))
        assert again.attempt == first.attempt + 1
        assert again.run_id != first.run_id, "a replay is a distinct run (D-04)"

    def test_divergence_is_recorded_not_hidden(self, tmp_path: Path) -> None:
        """A different agent under the same plan must be reported, not silently run."""
        first = _first_run(tmp_path)

        def drifted(engine: ChaosEngine) -> Any:
            @engine.tool(name="weather", side_effecting=False)
            def weather(city: str = "Paris") -> dict[str, Any]:
                return {"city": city, "temp_c": 21}

            def run(question: Any = None) -> str:
                weather()  # an extra call the recorded sequence does not have
                data = weather()
                return f"It is {data['temp_c']}C in {data['city']}."

            return run

        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        again = engine.replay(Path(first.artifacts["run_dir"]), target=drifted(engine))
        events = [
            json.loads(line) for line in Path(again.artifacts["trace"]).read_text().splitlines()
        ]
        assert any(e["kind"] == "replay_divergence" for e in events), (
            "the crossing sequence changed and nothing said so"
        )
        assert any("replay_divergence" in note for note in again.schema_errors)

    def test_no_divergence_event_when_it_matches(self, tmp_path: Path) -> None:
        first = _first_run(tmp_path)
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        again = engine.replay(Path(first.artifacts["run_dir"]), target=_agent(engine))
        events = [
            json.loads(line) for line in Path(again.artifacts["trace"]).read_text().splitlines()
        ]
        assert not any(e["kind"] == "replay_divergence" for e in events)
        assert again.schema_errors == []

    def test_a_predicate_target_is_refused(self, tmp_path: Path) -> None:
        """A callable cannot be serialized, so its plan cannot be replayed (D-31)."""
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        engine.register_fault(
            ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
            target=Target(tool="weather", predicate=lambda c: c.call_index == 1),
        )
        first = engine.run(_agent(engine), inputs={"question": "?"}, scenario_id="pred")

        engine2 = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        with pytest.raises(ConfigError, match="predicate"):
            engine2.replay(Path(first.artifacts["run_dir"]), target=_agent(engine2))

    def test_a_missing_plan_is_a_config_error(self, tmp_path: Path) -> None:
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules")
        with pytest.raises(ConfigError, match="plan"):
            engine.replay(tmp_path / "nope", target=lambda x=None: "x")


class TestReplayCommand:
    def test_the_cli_replays_a_stored_run(self, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import main

        first = _first_run(tmp_path)
        code = main(
            ["replay", first.artifacts["run_dir"], "--entrypoint", "tests.cli.test_replay:_agent"]
        )
        assert code == 1, "the replayed run still fails, so the exit code says so"

    def test_a_missing_run_dir_is_a_usage_error(self, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import main

        assert main(["replay", str(tmp_path / "nothing")]) == 2


class TestReplayRestoresTheInputs:
    """A replay that changes the agent's input is not a replay.

    `plan.json` recorded the fault plan, the seed and the limits but not what the
    agent was asked. `replay` therefore re-ran with `inputs=None`, the agent fell back
    to whatever default its signature carried, and the run could pass where the
    original failed -- reporting a clean reproduction of a different experiment.
    """

    @staticmethod
    def _agent_with_default(engine: ChaosEngine) -> Any:
        @engine.tool(name="weather", side_effecting=False)
        def weather(city: str = "Paris") -> dict[str, Any]:
            return {"city": city, "temp_c": 21}

        def run(question: str = "the default question") -> str:
            data = weather()
            return f"{question}: it is {data['temp_c']}C in {data['city']}."

        return run

    def test_the_report_records_what_the_agent_was_asked(self, tmp_path: Path) -> None:
        # Not in `plan.json`: that file is exactly what `plan_hash` is computed over,
        # and the hash is documented as proving plan identity and nothing else. The
        # report schema already reserves `reproduce.scenario_yaml` for this.
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        result = engine.run(
            self._agent_with_default(engine),
            inputs={"question": "a very specific question"},
            scenario_id="inputs",
            expected_behavior="ignore_and_continue",
        )
        stored = json.loads(result.reproduce["scenario_yaml"])
        assert stored["inputs"] == {"question": "a very specific question"}

    def test_the_replay_is_asked_the_same_thing(self, tmp_path: Path) -> None:
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        first = engine.run(
            self._agent_with_default(engine),
            inputs={"question": "a very specific question"},
            scenario_id="inputs",
            expected_behavior="ignore_and_continue",
        )
        engine2 = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        again = engine2.replay(
            Path(first.artifacts["run_dir"]), target=self._agent_with_default(engine2)
        )
        assert again.final_output == first.final_output, "the replay asked the agent something else"

    def test_the_initial_state_is_restored_too(self, tmp_path: Path) -> None:
        def stateful(engine: ChaosEngine) -> Any:
            # D-19 rule 4: `initial_state` reaches the agent only through a parameter
            # named `state` or `initial_state`. Anything else and the engine holds it
            # and exposes it through `state_view` instead.
            def run(question: Any = None, state: dict[str, Any] | None = None) -> str:
                return f"city={(state or {}).get('city', 'nowhere')}"

            return run

        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        first = engine.run(
            stateful(engine),
            inputs=None,
            initial_state={"city": "Lisbon"},
            scenario_id="state",
            expected_behavior="ignore_and_continue",
        )
        engine2 = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        again = engine2.replay(Path(first.artifacts["run_dir"]), target=stateful(engine2))
        assert again.final_output == first.final_output == "city=Lisbon"
