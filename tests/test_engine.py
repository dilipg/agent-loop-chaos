"""The engine: registration, the hot path, limits, dry run, and resilience.

The load-bearing test in this file is `test_a_fault_that_raises_never_reaches_the
_agent`. The library must never crash the run it is observing, and an engine bug
must never be reported as an agent failure.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, ClassVar

import pytest

from agent_loop_chaos import ChaosEngine, Limits
from agent_loop_chaos.context import Crossing, FaultContext, Layer, Phase
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.faults.base import (
    ALL_PAIRS,
    Fault,
    FaultOutcome,
    MutationLog,
    NoopFault,
)
from agent_loop_chaos.targeting import Target, Trigger


def engine(tmp_path: Path, **kw: Any) -> ChaosEngine:
    """Build an engine writing into a temp directory, with strict trace validation.

    Args:
        tmp_path: pytest's temp directory.
        **kw: Overrides for the constructor.

    Returns:
        The engine.
    """
    kw.setdefault("seed", 1337)
    kw.setdefault("out_dir", tmp_path / ".chaos")
    kw.setdefault("strict_trace", True)
    return ChaosEngine(**kw)


def kinds(result_engine: ChaosEngine) -> list[str]:  # pragma: no cover - helper
    """Unused placeholder kept for symmetry.

    Args:
        result_engine: An engine.

    Returns:
        An empty list.
    """
    return []


class ReplaceResultFault(Fault):
    """Replaces a tool result with a fixed value. A test double, not a real fault.

    **What agent weakness it proves:** nothing on its own; it exists to exercise the
    value-chaining path in `cross`.

    **What graceful behaviour looks like:** the agent validates the result rather
    than trusting its shape.
    """

    kind: ClassVar[str] = "TestReplaceResultFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "post")})

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Return the configured value.

        Args:
            crossing: The crossing being faulted.
            ctx: The fault context.

        Returns:
            A `replace_result` outcome.
        """
        new = self.params()["value"]
        return FaultOutcome(
            action="replace_result",
            value=new,
            note="replaced",
            mutation=MutationLog.of(crossing.result, new),
        )


class AppendFault(Fault):
    """Appends a marker to a string result, so chaining is observable.

    **What agent weakness it proves:** nothing; it verifies that the second fault in
    a chain sees the first one's output (D-13).

    **What graceful behaviour looks like:** not applicable.
    """

    kind: ClassVar[str] = "TestAppendFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "post")})

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Append the configured suffix to whatever arrived.

        Args:
            crossing: The crossing, carrying the previous fault's output.
            ctx: The fault context.

        Returns:
            A `replace_result` outcome.
        """
        new = f"{crossing.result}{self.params()['suffix']}"
        return FaultOutcome(
            action="replace_result",
            value=new,
            mutation=MutationLog.of(crossing.result, new),
        )


class RaisingFault(Fault):
    """Raises inside `apply`. Proves a library bug cannot reach the agent.

    **What agent weakness it proves:** none. It proves an *engine* property: an
    exception in the library becomes an `internal_error` event and the original
    value passes through untouched.

    **What graceful behaviour looks like:** the run completes exactly as if the
    fault had not been registered.
    """

    kind: ClassVar[str] = "TestRaisingFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = ALL_PAIRS

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Always raise.

        Args:
            crossing: Ignored.
            ctx: Ignored.

        Returns:
            Never returns.

        Raises:
            RuntimeError: Always, on purpose.
        """
        raise RuntimeError("deliberate fault bug")


class BoomFault(Fault):
    """Asks the engine to raise into the agent — the one thing a fault may do.

    **What agent weakness it proves:** no error handling around a tool call.

    **What graceful behaviour looks like:** the agent catches the error, retries
    with a cap, or degrades and says so.
    """

    kind: ClassVar[str] = "TestBoomFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "pre")})

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Return a `raise` outcome.

        Args:
            crossing: Ignored.
            ctx: Ignored.

        Returns:
            A terminal `raise` outcome.
        """
        return FaultOutcome(action="raise", value=ValueError("injected"), note="boom")


# --------------------------------------------------------------------- registration


def test_fault_ids_are_assigned_in_registration_order(tmp_path: Path) -> None:
    """`f1`, `f2`, … for display; the RNG keys on the hash instead (D-03)."""
    eng = engine(tmp_path)
    assert eng.register_fault(NoopFault(), target_tool="a") == "f1"
    assert eng.register_fault(NoopFault(), target_tool="b") == "f2"


def test_registration_rejects_a_layer_the_fault_does_not_accept(tmp_path: Path) -> None:
    """Eagerly, at registration — not mid-run where it could look like an agent bug."""
    with pytest.raises(ConfigError, match="does not accept layer"):
        engine(tmp_path).register_fault(ReplaceResultFault(value=1), target_llm="default")


def test_registration_rejects_a_phase_the_fault_does_not_accept(tmp_path: Path) -> None:
    """`accepts` is a set of (layer, phase) pairs, and both halves are checked."""
    with pytest.raises(ConfigError, match=r"does not accept tool\.pre"):
        engine(tmp_path).register_fault(
            ReplaceResultFault(value=1), target=Target(tool="t", phase="pre")
        )


@pytest.mark.parametrize(
    ("trigger", "message"),
    [
        (Trigger(probability=1.5), "probability"),
        (Trigger(probability=-0.1), "probability"),
        (Trigger(max_fires=0), "max_fires"),
        (Trigger(cooldown_calls=-1), "cooldown_calls"),
        (Trigger(on_call=0), "1-based"),
        (Trigger(on_step=-1), "on_step"),
    ],
)
def test_registration_rejects_impossible_triggers(
    tmp_path: Path, trigger: Trigger, message: str
) -> None:
    """The offending value appears in the message, so the fix is obvious."""
    with pytest.raises(ConfigError, match=message):
        engine(tmp_path).register_fault(NoopFault(), target_tool="t", trigger=trigger)


def test_registration_rejects_target_plus_shorthand(tmp_path: Path) -> None:
    """Two sources of truth for one target would be ambiguous."""
    with pytest.raises(ConfigError, match="not both"):
        engine(tmp_path).register_fault(NoopFault(), target=Target(tool="a"), target_tool="b")


def test_plan_hash_is_stable_and_order_sensitive(tmp_path: Path) -> None:
    """`plan_hash` proves plan identity, and nothing else (`docs/04` §3)."""
    from agent_loop_chaos.seeding import canonical_json, sha256_of

    first = engine(tmp_path)
    first.register_fault(NoopFault(), target_tool="a")
    second = engine(tmp_path)
    second.register_fault(NoopFault(), target_tool="a")
    assert sha256_of(canonical_json(first.plan())) == sha256_of(canonical_json(second.plan()))


def test_fault_key_is_independent_of_registration_order(tmp_path: Path) -> None:
    """D-03: inserting a fault must not re-key another fault's RNG stream."""
    a = engine(tmp_path)
    a.register_fault(NoopFault(), target_tool="weather")
    key_alone = a.plan()["faults"][0]["fault_key"]

    b = engine(tmp_path)
    b.register_fault(NoopFault(), target_tool="flights")
    b.register_fault(NoopFault(), target_tool="weather")
    keys = {f["target"].get("tool"): f["fault_key"] for f in b.plan()["faults"]}
    assert keys["weather"] == key_alone


def test_clear_faults_resets_the_ordinal(tmp_path: Path) -> None:
    """So a reused engine does not keep counting upward."""
    eng = engine(tmp_path)
    eng.register_fault(NoopFault(), target_tool="a")
    eng.clear_faults()
    assert eng.register_fault(NoopFault(), target_tool="a") == "f1"


# ------------------------------------------------------------------------ firing


def test_noop_fault_fires_exactly_once_with_on_call_two(tmp_path: Path) -> None:
    """The phase's headline plumbing test: a 3-call tool, one fire, on call 2."""
    eng = engine(tmp_path)
    eng.register_fault(
        NoopFault(), target=Target(tool="t", phase="post"), trigger=Trigger(on_call=2)
    )

    @eng.tool
    def t(x: int) -> dict[str, int]:
        return {"x": x}

    def agent(q: int) -> list[dict[str, int]]:
        return [t(q), t(q + 1), t(q + 2)]

    result = eng.run(agent, inputs=1)
    record = result.injected_faults[0]
    assert record["fire_count"] == 1
    assert record["fires"][0]["call_index"] == 2
    assert "skipped_reason" not in record, "D-36: emitted only when nothing fired"
    assert result.final_output == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_a_fault_that_never_matches_records_a_skip_reason(tmp_path: Path) -> None:
    """ "Why didn't my fault fire" must have an answer in the report (D-36)."""
    eng = engine(tmp_path)
    eng.register_fault(NoopFault(), target_tool="never_called", trigger=Trigger(on_call=1))

    @eng.tool
    def t() -> int:
        return 1

    result = eng.run(lambda: t())
    record = result.injected_faults[0]
    assert record["fired"] is False
    assert record["fire_count"] == 0


def test_value_replacing_faults_chain_in_registration_order(tmp_path: Path) -> None:
    """D-13: each fault sees the previous one's output, and records its own patch.

    First-wins for everything would make a multi-mutation preset exercise only its
    first fault.
    """
    eng = engine(tmp_path)
    eng.register_fault(ReplaceResultFault(value="base"), target_tool="t")
    eng.register_fault(AppendFault(suffix="-A"), target_tool="t")
    eng.register_fault(AppendFault(suffix="-B"), target_tool="t")

    @eng.tool
    def t() -> str:
        return "original"

    result = eng.run(lambda: t())
    assert result.final_output == "base-A-B"
    assert [f["fire_count"] for f in result.injected_faults] == [1, 1, 1]
    patches = [f["fires"][0]["json_patch"] for f in result.injected_faults]
    assert all(patch for patch in patches), "each fault records its own diff"


def test_a_terminal_action_supersedes_later_faults(tmp_path: Path) -> None:
    """D-13: `raise` is first-wins, and the rest are marked `superseded`."""
    eng = engine(tmp_path)
    eng.register_fault(BoomFault(), target=Target(tool="t", phase="pre"))
    eng.register_fault(NoopFault(), target=Target(tool="t", phase="pre"))

    @eng.tool
    def t() -> int:
        return 1

    def agent() -> str:
        try:
            t()
        except ValueError:
            return "handled"
        return "not raised"

    result = eng.run(agent)
    assert result.final_output == "handled"
    assert result.injected_faults[0]["fire_count"] == 1
    assert result.injected_faults[1]["skipped_reason"] == "superseded"


# -------------------------------------------------------------------- resilience


def test_a_fault_that_raises_never_reaches_the_agent(tmp_path: Path) -> None:
    """The library must never crash the run it is observing.

    An exception inside `apply` becomes an `internal_error` event, the original value
    passes through untouched, and `success` is unaffected.
    """
    eng = engine(tmp_path)
    eng.register_fault(RaisingFault(), target=Target(tool="t", phase="post"))

    @eng.tool
    def t() -> dict[str, int]:
        return {"ok": 1}

    # `ignore_and_continue` is what `completed_unaffected` satisfies (docs/11 §7).
    # M1 asserted `graceful_degradation` here while `success` was still hardcoded
    # True; now that it is computed, that expectation is the wrong one to state.
    result = eng.run(lambda: t(), expected_behavior="ignore_and_continue")
    assert result.final_output == {"ok": 1}
    assert result.error is None
    assert result.success is True

    events = [e["kind"] for e in eng_events(tmp_path, result)]
    assert "internal_error" in events


def eng_events(tmp_path: Path, result: Any) -> list[dict[str, Any]]:
    """Read the trace written for a result.

    Args:
        tmp_path: The temp directory the engine wrote into.
        result: The run's result.

    Returns:
        The trace events.
    """
    from agent_loop_chaos.trace import TraceRecorder

    return TraceRecorder.load(result.artifacts["trace"])


def test_a_bad_target_predicate_is_contained(tmp_path: Path) -> None:
    """A predicate is user code, and user code in the plan can be wrong."""

    def explode(crossing: Crossing) -> bool:
        raise RuntimeError("bad predicate")

    eng = engine(tmp_path)
    eng.register_fault(NoopFault(), target=Target(tool="t", predicate=explode))

    @eng.tool
    def t() -> int:
        return 7

    result = eng.run(lambda: t())
    assert result.final_output == 7
    assert result.error is None


def test_an_agent_exception_is_the_finding_not_a_crash(tmp_path: Path) -> None:
    """The agent's failure is what we are here to record."""
    eng = engine(tmp_path)

    def agent() -> None:
        raise KeyError("temp_c")

    result = eng.run(agent)
    assert result.error is not None
    assert result.error["type"] == "KeyError"
    assert "traceback" in result.error


# ------------------------------------------------------------------------ limits


def test_max_tool_calls_produces_limit_hit_not_an_error(tmp_path: Path) -> None:
    """D-06: a limit is a harness outcome, never reported as an agent exception."""
    eng = engine(tmp_path, limits=Limits(max_tool_calls=3))

    @eng.tool
    def t() -> int:
        return 1

    def agent() -> str:
        for _ in range(50):
            t()
        return "never"

    result = eng.run(agent)
    assert result.loop["limit_hit"] == "max_tool_calls"
    assert result.error is None


def test_max_steps_produces_limit_hit_not_an_error(tmp_path: Path) -> None:
    """One LLM call is one step under vanilla (D-05)."""
    eng = engine(tmp_path, limits=Limits(max_steps=2, max_llm_calls=99))

    @eng.llm
    def gen(prompt: str) -> str:
        return "ok"

    def agent() -> str:
        for _ in range(50):
            gen("hi")
        return "never"

    result = eng.run(agent)
    assert result.loop["limit_hit"] == "max_steps"
    assert result.error is None


def test_a_broad_except_in_the_agent_cannot_swallow_the_limit(tmp_path: Path) -> None:
    """Why `LimitExceeded` derives from `BaseException` (D-06).

    If it were an `Exception`, the run below would never stop and `LoopTrapFault`'s
    expected `limit_hit` would be unreachable.
    """
    eng = engine(tmp_path, limits=Limits(max_tool_calls=3))

    @eng.tool
    def t() -> int:
        return 1

    def agent() -> str:
        for _ in range(50):
            # A broad `except Exception` is exactly what this test is about.
            with contextlib.suppress(Exception):
                t()
        return "swallowed everything"

    result = eng.run(agent)
    assert result.loop["limit_hit"] == "max_tool_calls"


def test_tool_calls_do_not_increment_steps(tmp_path: Path) -> None:
    """D-05: a step is one agent iteration, not one tool call."""
    eng = engine(tmp_path)

    @eng.tool
    def t() -> int:
        return 1

    result = eng.run(lambda: [t(), t(), t()])
    assert result.metrics["tool_calls"] == 3
    assert result.metrics["steps"] == 0


# ----------------------------------------------------------------------- dry run


def test_dry_run_arms_faults_and_applies_none(tmp_path: Path) -> None:
    """D-12: a dry run must classify exactly as an unfaulted baseline."""
    eng = engine(tmp_path, dry_run=True)
    eng.register_fault(ReplaceResultFault(value="corrupted"), target_tool="t")

    @eng.tool
    def t() -> str:
        return "clean"

    result = eng.run(lambda: t())
    assert result.final_output == "clean"
    assert result.dry_run is True

    record = result.injected_faults[0]
    assert record["fired"] is False
    assert record["skipped_reason"] == "dry_run"

    events = [e["kind"] for e in eng_events(tmp_path, result)]
    assert "fault_armed" in events
    assert "fault_fired" not in events
    assert "mutation_applied" not in events


def test_dry_run_leaves_harness_facts_empty(tmp_path: Path) -> None:
    """A probe reading `HarnessFacts` must see a baseline (D-12)."""
    eng = engine(tmp_path, dry_run=True)
    eng.register_fault(NoopFault(), target_tool="t")

    @eng.tool
    def t() -> int:
        return 1

    captured: list[Any] = []

    def agent() -> int:
        captured.append(eng.harness_facts())
        return t()

    eng.run(agent)
    facts = captured[0]
    assert facts.dry_run is True
    assert facts.fired == ()
    assert facts.faulted_seqs == frozenset()


def test_baseline_helper_is_a_dry_run(tmp_path: Path) -> None:
    """`engine.baseline()` is the documented convenience for an unfaulted run."""
    eng = engine(tmp_path)
    eng.register_fault(ReplaceResultFault(value="corrupted"), target_tool="t")

    @eng.tool
    def t() -> str:
        return "clean"

    assert eng.baseline(lambda: t()).final_output == "clean"


# ------------------------------------------------------------------ report shape


def test_the_report_validates_against_the_schema(tmp_path: Path) -> None:
    """Doing this in M1 is the point: the schema catches shape drift immediately."""
    eng = engine(tmp_path)
    eng.register_fault(NoopFault(), target_tool="t")

    @eng.tool
    def t() -> dict[str, Any]:
        return {"a": [1, 2, {"b": None}]}

    result = eng.run(lambda: t(), scenario_id="s1")
    assert result.validate() == []
    assert result.scenario_id == "s1"


def test_the_run_writes_a_followable_bundle(tmp_path: Path) -> None:
    """The directory exists and the trace grows during the run (`docs/01` §5)."""
    eng = engine(tmp_path)

    @eng.tool
    def t() -> int:
        return 1

    result = eng.run(lambda: t())
    run_dir = Path(result.artifacts["run_dir"])
    assert (run_dir / "trace.jsonl").exists()
    assert (run_dir / "plan.json").exists()


def test_randomness_lists_only_streams_actually_drawn(tmp_path: Path) -> None:
    """D-36: `randomness.streams` lists only purposes with at least one draw."""
    eng = engine(tmp_path)
    eng.register_fault(NoopFault(), target_tool="t", trigger=Trigger(probability=1.0))

    @eng.tool
    def t() -> int:
        return 1

    assert eng.run(lambda: t()).randomness["streams"] == {}


def test_run_id_changes_with_attempt(tmp_path: Path) -> None:
    """D-04: `attempt` is an explicit parameter and part of `run_id`."""
    eng = engine(tmp_path)

    @eng.tool
    def t() -> int:
        return 1

    first = eng.run(lambda: t(), scenario_id="s", attempt=1)
    second = eng.run(lambda: t(), scenario_id="s", attempt=2)
    assert first.run_id != second.run_id
    assert first.plan_hash == second.plan_hash


def test_markers_are_inert_outside_a_run(tmp_path: Path) -> None:
    """`validated` and `note` must be safe to leave in production code."""
    eng = engine(tmp_path)
    eng.validated("x", name="check")
    eng.note("hello")
    assert eng.step() == 0


def test_an_engine_bug_in_the_routing_layer_is_not_reported_as_an_agent_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLAUDE.md: "An engine bug must never be reported as an agent failure."

    Found while wiring `invoke_target`: a missing attribute in `route_sync` raised
    inside the tool wrapper, which sits in the agent's own call stack, so the engine
    captured it as `error` and the report blamed the agent for a library bug. The
    routing layer has to contain its own failures the same way `apply()` does.
    """
    eng = engine(tmp_path)

    @eng.tool
    def fetch() -> str:
        return "real value"

    def exploding_pre(*args: Any, **kwargs: Any) -> Any:
        raise AttributeError("deliberate engine bug")

    monkeypatch.setattr(eng, "_pre_phase", exploding_pre)

    result = eng.run(lambda: fetch())
    assert result.error is None, "a library bug must not be attributed to the agent"
    assert result.final_output == "real value", "the call must pass through untouched"
    assert "internal_error" in {e["kind"] for e in eng_events(tmp_path, result)}


class TestInstrumentObjectDeclaresSideEffects:
    """`instrument_object` must be able to say which methods are side-effecting.

    Its signature took `tools: Sequence[str]`, and the wrapper registered
    `ToolInfo(side_effecting=None)` -- undeclared. `alc run --preset full` refuses to
    start when any tool leaves it undeclared (`SAFETY.md` §1 item 3, D-23), so a
    class-based agent instrumented this way could not use the preset at all, and the
    D-23 glob rule had nothing to act on.

    The workaround was to call `engine.tool(...)` first to seed the flag and rely on
    `instrument_object`'s `setdefault` preserving it. A parameter is better than a
    workaround nobody will find.
    """

    @staticmethod
    def _agent() -> Any:
        class Agent:
            def lookup(self, city: str = "Paris") -> dict[str, Any]:
                return {"city": city}

            def book(self, city: str = "Paris") -> dict[str, Any]:
                return {"held": city}

            def run(self, question: Any = None) -> str:
                self.lookup()
                return "done"

        return Agent()

    def test_a_mapping_declares_each_tool(self) -> None:
        from agent_loop_chaos import ChaosEngine

        engine = ChaosEngine(write_bundle=False)
        engine.instrument_object(self._agent(), tools={"lookup": False, "book": True})
        assert engine._tools["lookup"].side_effecting is False
        assert engine._tools["book"].side_effecting is True

    def test_a_sequence_still_works_and_leaves_it_undeclared(self) -> None:
        """The old call shape must keep working; this is an addition, not a swap."""
        from agent_loop_chaos import ChaosEngine

        engine = ChaosEngine(write_bundle=False)
        engine.instrument_object(self._agent(), tools=["lookup"])
        assert engine._tools["lookup"].side_effecting is None

    def test_the_preset_gate_accepts_a_fully_declared_object(self) -> None:
        from agent_loop_chaos import ChaosEngine

        engine = ChaosEngine(write_bundle=False)
        engine.instrument_object(self._agent(), tools={"lookup": False, "book": True})
        # Raises when anything is undeclared; returning cleanly is the assertion.
        engine.require_declared_side_effects()

    def test_the_preset_gate_still_refuses_an_undeclared_one(self) -> None:
        from agent_loop_chaos import ChaosEngine
        from agent_loop_chaos.errors import ConfigError

        engine = ChaosEngine(write_bundle=False)
        engine.instrument_object(self._agent(), tools=["lookup"])
        with pytest.raises(ConfigError, match="lookup"):
            engine.require_declared_side_effects()

    def test_an_explicit_declaration_is_not_overwritten(self) -> None:
        from agent_loop_chaos import ChaosEngine

        engine = ChaosEngine(write_bundle=False)
        agent = self._agent()
        engine.tool(agent.book, name="book", side_effecting=True)
        engine.instrument_object(agent, tools=["book"])
        assert engine._tools["book"].side_effecting is True
