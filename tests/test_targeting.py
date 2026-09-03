"""Targeting and trigger truth tables.

Targeting answers *where*, triggering answers *when*. Both are pure, and neither may
read the clock (D-47).
"""

from __future__ import annotations

import pytest

from agent_loop_chaos.context import Crossing, FaultContext, Limits, RunContext
from agent_loop_chaos.targeting import Target, Trigger, matches, should_fire, state_key_matches
from agent_loop_chaos.trace import TraceRecorder


def crossing(**kw: object) -> Crossing:
    """Build a crossing with sensible defaults.

    Args:
        **kw: Fields to override.

    Returns:
        The crossing.
    """
    defaults: dict[str, object] = {"layer": "tool", "phase": "post", "name": "get_weather"}
    defaults.update(kw)
    return Crossing(**defaults)  # type: ignore[arg-type]


def context(seed: int = 1337, *, fault_key: str = "abc123") -> FaultContext:
    """Build a minimal `FaultContext`.

    Args:
        seed: The run seed.
        fault_key: The fault's spec hash, which keys its RNG stream.

    Returns:
        The context.
    """
    ctx = RunContext(
        run_id="run-test",
        seed=seed,
        started_at="1970-01-01T00:00:00Z",
        limits=Limits(),
        trace=TraceRecorder("run-test"),
    )
    return FaultContext(
        fault_id="f1",
        fault_key=fault_key,
        run=ctx,
        limits=ctx.limits,
        counters=ctx.counters,
    )


TARGET_TABLE: list[tuple[str, Target, Crossing, bool]] = [
    ("empty target matches anything", Target(), crossing(), True),
    ("exact tool name", Target(tool="get_weather"), crossing(), True),
    ("wrong tool name", Target(tool="search"), crossing(), False),
    ("tool glob prefix", Target(tool="get_*"), crossing(), True),
    ("tool glob suffix", Target(tool="*weather"), crossing(), True),
    ("tool glob no match", Target(tool="post_*"), crossing(), False),
    ("glob is case sensitive", Target(tool="GET_*"), crossing(), False),
    ("layer only, matching", Target(layer="tool"), crossing(), True),
    ("layer only, mismatched", Target(layer="llm"), crossing(), False),
    ("phase matching", Target(phase="post"), crossing(), True),
    ("phase mismatched", Target(phase="pre"), crossing(), False),
    ("layer and phase both match", Target(layer="tool", phase="post"), crossing(), True),
    ("llm alias matches", Target(llm="default"), crossing(layer="llm", name="default"), True),
    ("llm alias on a tool crossing", Target(llm="default"), crossing(), False),
    ("node name matches", Target(node="plan"), crossing(layer="node", name="plan"), True),
    ("node glob", Target(node="pl*"), crossing(layer="node", name="plan"), True),
    (
        "state key exact",
        Target(state_key="messages"),
        crossing(layer="state", name="messages"),
        True,
    ),
    (
        "state key wildcard segment",
        Target(state_key="messages.*.content"),
        crossing(layer="state", name="messages.0.content"),
        True,
    ),
    (
        "state key wildcard wrong depth",
        Target(state_key="messages.*"),
        crossing(layer="state", name="messages.0.content"),
        False,
    ),
    (
        "state key on a tool crossing",
        Target(state_key="messages"),
        crossing(),
        False,
    ),
    ("predicate true", Target(predicate=lambda c: c.call_index == 1), crossing(), True),
    ("predicate false", Target(predicate=lambda c: c.call_index == 9), crossing(), False),
    (
        "predicate runs after the name check",
        Target(tool="nope", predicate=lambda c: True),
        crossing(),
        False,
    ),
    ("tool name implies the tool layer", Target(tool="*"), crossing(layer="llm"), False),
]


@pytest.mark.parametrize(
    ("label", "target", "cross", "expected"),
    TARGET_TABLE,
    ids=[row[0] for row in TARGET_TABLE],
)
def test_target_truth_table(label: str, target: Target, cross: Crossing, expected: bool) -> None:
    """Twenty-four (Target, Crossing) pairs, each asserted both ways."""
    assert matches(target, cross) is expected, label


@pytest.mark.parametrize(
    ("pattern", "key", "expected"),
    [
        ("a", "a", True),
        ("a.b", "a.b", True),
        ("a.*", "a.b", True),
        ("a.*", "a", False),
        ("a.**", "a.b.c.d", True),
        ("**", "anything.at.all", True),
        ("a.*.c", "a.b.c", True),
        ("a.*.c", "a.b.d", False),
        ("msg*", "msgs", True),
    ],
)
def test_state_key_matching(pattern: str, key: str, expected: bool) -> None:
    """A ``*`` matches one segment; a trailing ``**`` matches the rest."""
    assert state_key_matches(pattern, key) is expected


def test_on_call_scalar() -> None:
    """`on_call` fires on exactly that 1-based index."""
    ctx = context()
    assert should_fire(Trigger(on_call=2), crossing(call_index=2), ctx, "f1") == (True, "fired")
    assert should_fire(Trigger(on_call=2), crossing(call_index=1), ctx, "f1") == (
        False,
        "call_index_mismatch",
    )


def test_on_call_list() -> None:
    """A list of indices fires on any of them."""
    ctx = context()
    trigger = Trigger(on_call=[1, 3], max_fires=5)
    assert should_fire(trigger, crossing(call_index=1), ctx, "f1")[0]
    assert not should_fire(trigger, crossing(call_index=2), ctx, "f1")[0]
    assert should_fire(trigger, crossing(call_index=3), ctx, "f1")[0]


def test_on_step_and_after_step() -> None:
    """`on_step` is exact; `after_step` is a threshold."""
    ctx = context()
    assert should_fire(Trigger(on_step=3), crossing(step=3), ctx, "f1")[0]
    assert should_fire(Trigger(on_step=3), crossing(step=4), ctx, "f1") == (
        False,
        "step_mismatch",
    )
    assert should_fire(Trigger(after_step=2), crossing(step=3), ctx, "f1")[0]
    assert should_fire(Trigger(after_step=2), crossing(step=2), ctx, "f1") == (
        False,
        "after_step_not_reached",
    )


def test_stop_after_step() -> None:
    """A fault can be silenced past a step."""
    ctx = context()
    assert should_fire(Trigger(stop_after_step=5), crossing(step=5), ctx, "f1")[0]
    assert should_fire(Trigger(stop_after_step=5), crossing(step=6), ctx, "f1") == (
        False,
        "stopped_after_step",
    )


def test_max_fires() -> None:
    """Once the budget is spent the reason says so."""
    ctx = context()
    ctx.counters.fires["f1"] = 1
    assert should_fire(Trigger(max_fires=1), crossing(), ctx, "f1") == (
        False,
        "max_fires_reached",
    )
    assert should_fire(Trigger(max_fires=2), crossing(), ctx, "f1")[0]


def test_cooldown_calls() -> None:
    """Cooldown is measured in target calls since the last fire."""
    ctx = context()
    ctx.counters.fires["f1"] = 1
    ctx.counters.last_fire_call["f1"] = 2
    trigger = Trigger(max_fires=5, cooldown_calls=2)
    assert should_fire(trigger, crossing(call_index=3), ctx, "f1") == (False, "cooldown")
    assert should_fire(trigger, crossing(call_index=5), ctx, "f1")[0]


@pytest.mark.parametrize(("probability", "expected"), [(0.0, False), (1.0, True)])
def test_probability_extremes(probability: float, expected: bool) -> None:
    """0 never fires, 1 always does."""
    ctx = context()
    assert should_fire(Trigger(probability=probability), crossing(), ctx, "f1")[0] is expected


def test_probability_half_is_seeded_and_reproducible() -> None:
    """A coin flip is still deterministic, given the seed."""
    first = [
        should_fire(Trigger(probability=0.5, max_fires=99), crossing(), context(seed=99), "f1")[0]
        for _ in range(1)
    ]
    second = [
        should_fire(Trigger(probability=0.5, max_fires=99), crossing(), context(seed=99), "f1")[0]
        for _ in range(1)
    ]
    assert first == second


def test_the_rng_is_untouched_when_probability_is_one() -> None:
    """This is what keeps a deterministic scenario's streams stable as a suite grows."""
    ctx = context()
    should_fire(Trigger(probability=1.0, max_fires=99), crossing(), ctx, "f1")
    assert ctx.run.draws == {}
    assert ctx.run.rng_registry == {}


def test_the_rng_is_consumed_when_probability_is_below_one() -> None:
    """The complement of the assertion above."""
    ctx = context()
    should_fire(Trigger(probability=0.5, max_fires=99), crossing(), ctx, "f1")
    assert ctx.run.draws == {"abc123:trigger": 1}


def test_checks_run_before_the_rng_so_a_skip_costs_no_draw() -> None:
    """A fault skipped for a cheap reason must not perturb its own stream."""
    ctx = context()
    should_fire(Trigger(on_call=2, probability=0.5), crossing(call_index=1), ctx, "f1")
    assert ctx.run.draws == {}
