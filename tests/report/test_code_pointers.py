"""Where to look — the section that makes a work order actionable.

`AGENT_TASK.md` section 4 promises a file:line pointer, and the whole pitch is that a
pointer turns a finding into a mechanical task. For a crash it comes from the stack.
For a **silent wrong answer** — the commonest mode, and the one this library exists to
find — there is no exception, so the section rendered as
"_no pointer met the confidence floor_" for exactly the findings that most needed one.

The fix: the engine knows which crossing was faulted and which user frame called it,
so the consumer of a corrupted value is nameable without an exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import ToolCorruptionFault
from agent_loop_chaos.targeting import Target, Trigger


def _silent_wrong_answer(engine: ChaosEngine) -> Any:
    """An agent that reports a corrupted value as fact, without raising."""

    @engine.tool(name="weather", side_effecting=False)
    def weather(city: str = "Paris") -> dict[str, Any]:
        return {"city": city, "temp_c": 21}

    def agent(question: Any = None) -> str:
        data = weather()
        return f"It is {data.get('temp_c', 19)}C in {data.get('city', 'the city')}."

    return agent


def _run(tmp_path: Path) -> Any:
    engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
    engine.register_fault(
        ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
        target=Target(tool="weather"),
        trigger=Trigger(on_call=1),
    )
    return engine.run(
        _silent_wrong_answer(engine),
        inputs={"question": "?"},
        scenario_id="silent",
        expected_behavior="graceful_degradation",
        expect={"no_unsourced_numbers": {"enabled": True, "allow_derived": False}},
    )


class TestASilentFailureStillGetsAPointer:
    def test_the_run_really_is_silent(self, tmp_path: Path) -> None:
        """Guard the premise: no exception, so the stack gives nothing."""
        result = _run(tmp_path)
        assert result.error is None
        assert result.success is False

    def test_a_pointer_is_produced(self, tmp_path: Path) -> None:
        pointers = _run(tmp_path).code_pointers
        assert pointers, "a silent wrong answer produced no pointer at all"

    def test_it_names_a_real_file_and_line(self, tmp_path: Path) -> None:
        pointer = _run(tmp_path).code_pointers[0]
        assert pointer["file"].endswith(".py"), pointer
        assert Path(pointer["file"]).is_file(), "the pointer names a file that exists"
        assert isinstance(pointer["line"], int) and pointer["line"] > 0

    def test_it_points_at_the_consumer_not_the_library(self, tmp_path: Path) -> None:
        """A pointer into `agent_loop_chaos/` is useless: the bug is the caller's."""
        for pointer in _run(tmp_path).code_pointers:
            assert "agent_loop_chaos" not in pointer["file"], pointer

    def test_it_points_at_this_test_file(self, tmp_path: Path) -> None:
        """The consumer of the faulted result is `_silent_wrong_answer.agent`."""
        pointer = _run(tmp_path).code_pointers[0]
        assert pointer["file"].endswith("test_code_pointers.py")

    def test_it_explains_why(self, tmp_path: Path) -> None:
        pointer = _run(tmp_path).code_pointers[0]
        assert pointer["why"], "a pointer with no reason is noise"
        assert "weather" in pointer["why"], "name the tool whose result was changed"

    def test_it_clears_the_confidence_floor(self, tmp_path: Path) -> None:
        """Below 0.5 the renderer drops it, which is how the section went blank."""
        assert _run(tmp_path).code_pointers[0]["confidence"] >= 0.5

    def test_the_work_order_shows_it(self, tmp_path: Path) -> None:
        from agent_loop_chaos.bundle import render_agent_task

        rendered = render_agent_task(_run(tmp_path))
        assert "no pointer met the confidence floor" not in rendered
        assert "test_code_pointers.py" in rendered


class TestACrashStillPrefersTheStack:
    """The exception frame is better evidence; it must not be displaced."""

    def test_the_traceback_pointer_comes_first(self, tmp_path: Path) -> None:
        def crashy(engine: ChaosEngine) -> Any:
            @engine.tool(name="weather", side_effecting=False)
            def weather(city: str = "Paris") -> dict[str, Any]:
                return {"city": city, "temp_c": 21}

            def agent(question: Any = None) -> str:
                return f"It is {weather()['temp_c']}C."

            return agent

        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        engine.register_fault(
            ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
            target=Target(tool="weather"),
            trigger=Trigger(on_call=1),
        )
        result = engine.run(crashy(engine), inputs={"question": "?"}, scenario_id="crash")
        assert result.error is not None
        first = result.code_pointers[0]
        assert first["confidence"] >= 0.8, "a real stack frame is the strongest evidence"

    def test_a_clean_run_gets_no_pointer(self, tmp_path: Path) -> None:
        """Nothing went wrong, so there is nowhere to look."""
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        result = engine.run(
            _silent_wrong_answer(engine),
            inputs={"question": "?"},
            scenario_id="clean",
            expected_behavior="ignore_and_continue",
        )
        assert result.success is True
        assert result.code_pointers == []


class TestPointersRenderRelatively:
    """A work order names a path you can type, not one from someone else's laptop."""

    def test_the_rendered_pointer_is_repo_relative(self, tmp_path: Path) -> None:
        from agent_loop_chaos.bundle import render_agent_task

        rendered = render_agent_task(_run(tmp_path))
        assert "tests/report/test_code_pointers.py" in rendered
        assert "/Users/" not in rendered, "an absolute path is noise in a work order"

    def test_a_path_outside_the_tree_is_left_alone(self) -> None:
        from agent_loop_chaos.bundle import display_path

        # Relativising to `../../../usr/lib/...` is worse than the absolute path.
        assert display_path("/usr/lib/python3/thing.py") == "/usr/lib/python3/thing.py"

    def test_a_path_inside_the_tree_is_shortened(self, tmp_path: Path, monkeypatch: Any) -> None:
        from agent_loop_chaos.bundle import display_path

        monkeypatch.chdir(tmp_path)
        (tmp_path / "pkg").mkdir()
        target = tmp_path / "pkg" / "agent.py"
        target.write_text("x = 1")
        assert display_path(str(target)) == "pkg/agent.py"

    def test_a_non_path_is_returned_unchanged(self) -> None:
        from agent_loop_chaos.bundle import display_path

        assert display_path("<caller of get_weather>") == "<caller of get_weather>"
