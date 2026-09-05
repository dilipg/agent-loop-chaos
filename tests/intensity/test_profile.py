"""The intensity dial: one number, 1 to 10, that decides how hard a plan pushes.

`1` is strict — the smallest blast radius that still proves something. `10` is
creative — harder parameters, relentless triggers, and extra faults pulled from the
matching preset so a one-fault scenario becomes a compound one.

Three rules hold at every level, and each has a test here:

- **3 is today.** The default level changes nothing: same params, same triggers, same
  fault set, byte-identical reports. Intensity is opt-in.
- **Nothing stops firing.** Turning the dial down never reduces a declared probability,
  because a fault that no longer fires proves nothing (D-64).
- **Plan time only.** Scaling happens once, when the plan is frozen, and is recorded.
  No probe, fault or judge reads the level at run time.
"""

from __future__ import annotations

import pytest

from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.intensity import (
    DEFAULT_LEVEL,
    MAX_LEVEL,
    MIN_LEVEL,
    profile,
    scale_params,
    scale_trigger,
)
from agent_loop_chaos.targeting import Trigger


class TestTheDial:
    def test_it_spans_one_to_ten(self) -> None:
        assert (MIN_LEVEL, MAX_LEVEL, DEFAULT_LEVEL) == (1, 10, 3)
        assert [profile(n).level for n in range(1, 11)] == list(range(1, 11))

    @pytest.mark.parametrize("level", [0, 11, -1, 100])
    def test_it_refuses_a_level_off_the_dial(self, level: int) -> None:
        with pytest.raises(ConfigError, match="intensity"):
            profile(level)

    def test_it_refuses_a_non_integer(self) -> None:
        with pytest.raises(ConfigError, match="intensity"):
            profile("high")  # type: ignore[arg-type]

    def test_it_is_monotonic(self) -> None:
        """Every dimension moves one way. A dial that goes back on itself is a lie."""
        levels = [profile(n) for n in range(1, 11)]
        for name in ("severity", "probability", "extra_faults"):
            values = [getattr(p, name) for p in levels]
            assert values == sorted(values), f"{name} is not monotonic: {values}"

    def test_each_level_has_a_label(self) -> None:
        # The dashboard shows this to someone who does not read Python.
        assert profile(1).label == "strict"
        assert profile(DEFAULT_LEVEL).label == "standard"
        assert profile(10).label == "creative"
        assert all(profile(n).label for n in range(1, 11))


class TestTheDefaultChangesNothing:
    """Level 3 is the identity. Everything below depends on this."""

    def test_triggers_are_untouched(self) -> None:
        trigger = Trigger(on_call=[1, 3], probability=0.5, max_fires=4, cooldown_calls=2)
        assert scale_trigger(trigger, profile(DEFAULT_LEVEL)) == trigger

    def test_params_are_untouched(self) -> None:
        params = {"delay_ms": 1000, "jitter_ms": 50}
        assert scale_params("ToolLatencyFault", params, profile(DEFAULT_LEVEL)) == params

    def test_no_faults_are_added(self) -> None:
        assert profile(DEFAULT_LEVEL).extra_faults == 0


class TestTriggerScaling:
    def test_high_intensity_fires_more(self) -> None:
        trigger = Trigger(on_call=1, max_fires=2)
        assert scale_trigger(trigger, profile(8)).max_fires > 2

    def test_ten_never_stops(self) -> None:
        assert scale_trigger(Trigger(max_fires=3), profile(10)).max_fires is None

    def test_low_intensity_contains_the_blast_radius(self) -> None:
        assert scale_trigger(Trigger(max_fires=99), profile(1)).max_fires < 99

    def test_it_never_scales_below_one_fire(self) -> None:
        assert scale_trigger(Trigger(max_fires=1), profile(1)).max_fires == 1

    def test_unlimited_stays_unlimited_when_turned_down(self) -> None:
        # `None` means "no cap declared". Turning the dial down bounds it rather than
        # inventing a number the scenario never asked for... but it must stay useful.
        scaled = scale_trigger(Trigger(max_fires=None), profile(1))
        assert scaled.max_fires is None or scaled.max_fires >= 1

    def test_turning_it_down_never_reduces_probability(self) -> None:
        """D-64: a fault that stops firing proves nothing."""
        for level in range(1, 11):
            scaled = scale_trigger(Trigger(probability=0.5), profile(level))
            assert scaled.probability >= 0.5, f"level {level} made the fault less likely"

    def test_high_intensity_makes_a_flaky_fault_certain(self) -> None:
        assert scale_trigger(Trigger(probability=0.3), profile(10)).probability == 1.0

    def test_probability_is_never_above_one(self) -> None:
        for level in range(1, 11):
            assert scale_trigger(Trigger(probability=0.9), profile(level)).probability <= 1.0

    def test_the_targeted_call_index_is_never_moved(self) -> None:
        # Intensity decides how hard, never what. Moving `on_call` would silently
        # retarget the experiment the scenario declared.
        for level in range(1, 11):
            assert scale_trigger(Trigger(on_call=[2, 5]), profile(level)).on_call == [2, 5]


class TestParamScaling:
    @pytest.mark.parametrize(
        ("kind", "param", "declared"),
        [
            ("ToolLatencyFault", "delay_ms", 1000),
            ("ContextNoiseFault", "tokens", 400),
            ("ToolCorruptionFault", "count", 1),
            ("StaleDataFault", "age_delta_s", 86400),
            ("DuplicateSideEffectFault", "times", 2),
        ],
    )
    def test_a_magnitude_param_grows(self, kind: str, param: str, declared: int) -> None:
        scaled = scale_params(kind, {param: declared}, profile(10))
        assert scaled[param] > declared

    @pytest.mark.parametrize(
        ("kind", "param", "declared"),
        [
            ("ContextShrinkFault", "keep_ratio", 0.4),
            ("LLMTruncationFault", "at_ratio", 0.6),
        ],
    )
    def test_an_inverted_param_shrinks(self, kind: str, param: str, declared: float) -> None:
        """Keeping less context and cutting earlier are both *harsher*."""
        scaled = scale_params(kind, {param: declared}, profile(10))
        assert 0.0 < scaled[param] < declared

    def test_a_ratio_never_leaves_its_range(self) -> None:
        for level in range(1, 11):
            value = scale_params("ContextShrinkFault", {"keep_ratio": 0.4}, profile(level))
            assert 0.0 < value["keep_ratio"] <= 1.0

    def test_a_count_never_drops_below_one(self) -> None:
        for level in range(1, 11):
            value = scale_params("ToolCorruptionFault", {"count": 1}, profile(level))
            assert value["count"] >= 1

    def test_an_unknown_param_is_left_alone(self) -> None:
        params = {"mutation_type": "drop_key", "keys": ["temp_c"], "preserve_shape": True}
        assert scale_params("ToolCorruptionFault", params, profile(10)) == params

    def test_an_unknown_fault_is_left_alone(self) -> None:
        params = {"anything": 3}
        assert scale_params("NotAFault", params, profile(10)) == params

    def test_it_does_not_mutate_the_caller_s_params(self) -> None:
        params = {"delay_ms": 1000}
        scale_params("ToolLatencyFault", params, profile(10))
        assert params == {"delay_ms": 1000}

    def test_it_is_deterministic(self) -> None:
        a = scale_params("ToolLatencyFault", {"delay_ms": 1000}, profile(7))
        b = scale_params("ToolLatencyFault", {"delay_ms": 1000}, profile(7))
        assert a == b
