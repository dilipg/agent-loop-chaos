"""One number, 1 to 10, deciding how hard a fault plan pushes.

`1` is strict: the smallest blast radius that still proves something. `10` is
creative: harsher parameters, relentless triggers, and extra faults pulled from the
matching preset so a one-fault scenario becomes a compound one.

Three rules hold at every level.

**3 is today.** The default is the identity -- same params, same triggers, same fault
set, byte-identical reports. A dial nobody turned must not change a single run.

**Nothing stops firing.** Turning the dial down never reduces a declared probability.
A scenario whose fault no longer fires proves nothing (D-64), so "strict" means a
smaller blast radius, never a fault that might not happen.

**Plan time only.** Scaling happens once, when the plan is frozen, and the level is
recorded in `plan.json` and the report. No probe, fault or judge reads it at run time,
so intensity cannot influence a verdict except through the plan it produced.

Intensity decides *how hard*, never *what*. `on_call`, `on_step` and every target stay
exactly as declared: moving them would silently retarget the experiment.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from .scenarios import FaultSpec
    from .targeting import Trigger

__all__ = [
    "DEFAULT_LEVEL",
    "MAX_LEVEL",
    "MIN_LEVEL",
    "PROFILES",
    "IntensityProfile",
    "describe",
    "expand_faults",
    "profile",
    "scale_params",
    "scale_trigger",
]

MIN_LEVEL = 1
MAX_LEVEL = 10
#: Level 3 is the identity. Changing this changes every existing report.
DEFAULT_LEVEL = 3


@dataclass(frozen=True, slots=True)
class IntensityProfile:
    """What one level of the dial means.

    Attributes:
        level: The dial position, 1 to 10.
        label: A word a non-developer can read, shown in the dashboard.
        summary: One sentence describing what this level does.
        fires: Multiplier on `max_fires`. `None` means uncapped.
        severity: Multiplier on magnitude parameters.
        probability: Multiplier on a declared trigger probability, never below 1.0.
            A large value simply means "certain".
        extra_faults: How many additional faults to pull from the matching preset.
    """

    level: int
    label: str
    summary: str
    fires: float | None
    severity: float
    probability: float
    extra_faults: int


#: A literal table rather than a formula: this is the contract, and someone tuning it
#: should be able to read what level 7 does without evaluating arithmetic.
PROFILES: dict[int, IntensityProfile] = {
    1: IntensityProfile(
        1,
        "strict",
        "One fault, one fire, the mildest parameters that still prove it.",
        fires=0.5,
        severity=0.5,
        probability=1.0,
        extra_faults=0,
    ),
    2: IntensityProfile(
        2,
        "cautious",
        "As declared, with a smaller blast radius and gentler values.",
        fires=0.75,
        severity=0.75,
        probability=1.0,
        extra_faults=0,
    ),
    3: IntensityProfile(
        3,
        "standard",
        "Exactly what the scenario declares. The default.",
        fires=1.0,
        severity=1.0,
        probability=1.0,
        extra_faults=0,
    ),
    4: IntensityProfile(
        4,
        "firm",
        "The declared faults, pushed somewhat harder.",
        fires=1.5,
        severity=1.25,
        probability=1.0,
        extra_faults=0,
    ),
    5: IntensityProfile(
        5,
        "aggressive",
        "Harder values, repeated fires, and one extra fault.",
        fires=2.0,
        severity=1.5,
        probability=1.0,
        extra_faults=1,
    ),
    6: IntensityProfile(
        6,
        "harsh",
        "Faults recur through the run; flaky triggers become likely.",
        fires=3.0,
        severity=2.0,
        probability=1.25,
        extra_faults=1,
    ),
    7: IntensityProfile(
        7,
        "relentless",
        "Sustained pressure and two extra faults compounding.",
        fires=5.0,
        severity=2.5,
        probability=1.5,
        extra_faults=2,
    ),
    8: IntensityProfile(
        8,
        "punishing",
        "Severe values, near-certain triggers, three extra faults.",
        fires=8.0,
        severity=3.0,
        probability=2.0,
        extra_faults=3,
    ),
    9: IntensityProfile(
        9,
        "brutal",
        "Everything fires, hard, most of the run.",
        fires=12.0,
        severity=4.0,
        probability=3.0,
        extra_faults=4,
    ),
    10: IntensityProfile(
        10,
        "creative",
        "Uncapped fires, extreme values, and a compound fault set.",
        fires=None,
        severity=5.0,
        probability=100.0,
        extra_faults=6,
    ),
}

#: Which parameters carry magnitude, and which way harder points.
#:
#: `up` means a bigger number is harsher (more latency, more noise, more keys
#: corrupted). `down` means a smaller one is (keeping less context, cutting a
#: response earlier, pinning a loop sooner). Anything not named here is categorical --
#: a mutation type, a mode, a list of keys -- and intensity leaves it alone, because
#: changing it would change the experiment rather than its strength.
SCALES: dict[str, dict[str, str]] = {
    "ArgumentTamperFault": {"count": "up"},
    "CheckpointRollbackFault": {"rollback_steps": "up", "times": "up"},
    "ContextNoiseFault": {"tokens": "up"},
    "ContextShrinkFault": {"keep_ratio": "down"},
    "DuplicateSideEffectFault": {"times": "up"},
    "EdgeMisrouteFault": {"times": "up"},
    "LLMTruncationFault": {"at_ratio": "down"},
    "LoopTrapFault": {"pin_after_call": "down"},
    "RateLimitFault": {"after_calls": "down", "retry_after_s": "up"},
    "StaleDataFault": {"age_delta_s": "up"},
    "StateStaleFault": {"revert_versions": "up"},
    "ToolCorruptionFault": {"count": "up", "depth": "up"},
    "ToolLatencyFault": {"delay_ms": "up", "jitter_ms": "up"},
}

#: Parameters constrained to `(0, 1]` however hard the dial is turned.
_RATIOS = frozenset({"keep_ratio", "at_ratio"})

#: Parameters that must stay a whole number of at least one.
_COUNTS = frozenset(
    {
        "count",
        "times",
        "depth",
        "rollback_steps",
        "revert_versions",
        "after_calls",
        "pin_after_call",
    }
)


def profile(level: int) -> IntensityProfile:
    """Look up one level of the dial.

    Args:
        level: The dial position.

    Returns:
        The profile.

    Raises:
        ConfigError: When the level is not an integer between 1 and 10.
    """
    if isinstance(level, bool) or not isinstance(level, int):
        raise ConfigError(
            f"intensity must be a whole number from {MIN_LEVEL} to {MAX_LEVEL}; got {level!r}"
        )
    if level not in PROFILES:
        raise ConfigError(
            f"intensity must be from {MIN_LEVEL} (strict) to {MAX_LEVEL} (creative); got {level}"
        )
    return PROFILES[level]


def describe(level: int) -> dict[str, Any]:
    """Render a level for a report or a reader.

    Args:
        level: The dial position.

    Returns:
        `{level, label, summary}` -- what the dashboard shows someone who does not
        read Python.
    """
    found = profile(level)
    return {"level": found.level, "label": found.label, "summary": found.summary}


def scale_trigger(trigger: Trigger, prof: IntensityProfile) -> Trigger:
    """Scale when and how often a fault fires.

    `on_call`, `on_step`, `after_step` and `stop_after_step` are left exactly as
    declared. Intensity decides how hard, never what.

    Args:
        trigger: The declared trigger.
        prof: The level's profile.

    Returns:
        A new trigger. The original is never mutated.
    """
    max_fires = trigger.max_fires
    if prof.fires is None:
        max_fires = None
    elif max_fires is not None:
        max_fires = max(1, math.ceil(max_fires * prof.fires))

    probability = trigger.probability
    if probability < 1.0:
        # Only ever upward: a dial turned down must not make a fault stop firing.
        probability = min(1.0, probability * max(1.0, prof.probability))

    return replace(trigger, max_fires=max_fires, probability=probability)


def scale_params(kind: str, params: Mapping[str, Any], prof: IntensityProfile) -> dict[str, Any]:
    """Scale a fault's magnitude parameters.

    Args:
        kind: The fault class name.
        params: The declared parameters.
        prof: The level's profile.

    Returns:
        A new mapping. Parameters this fault does not scale come through untouched.
    """
    scaled = dict(params)
    for name, direction in SCALES.get(kind, {}).items():
        if name not in scaled:
            continue
        value = scaled[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        factor = prof.severity if direction == "up" else 1.0 / prof.severity
        scaled[name] = _clamp(name, value * factor, isinstance(value, int))
    return scaled


def _clamp(name: str, value: float, was_int: bool) -> Any:
    """Keep a scaled parameter inside the range its fault accepts.

    Args:
        name: The parameter name.
        value: The scaled value.
        was_int: Whether the declared value was an integer.

    Returns:
        The clamped value, as the same type the scenario declared.
    """
    if name in _RATIOS:
        return min(1.0, max(0.01, round(value, 4)))
    if name in _COUNTS:
        return max(1, round(value))
    if was_int:
        return max(0, round(value))
    return max(0.0, round(value, 4))


def expand_faults(
    declared: Sequence[FaultSpec],
    prof: IntensityProfile,
    *,
    preset: str | None = None,
    allow_side_effects: Sequence[str] = (),
) -> tuple[list[FaultSpec], list[dict[str, str]]]:
    """Add faults from the matching preset, for the top of the dial.

    High intensity means compound failure -- two things going wrong at once, which is
    what actually happens in production and what a single-fault scenario never tests.
    The extras come from a preset rather than from anywhere clever, so the plan stays
    a deterministic function of `(declared faults, level)` and `plan.json` still says
    exactly what ran.

    A fault that performs a real action is never added automatically. `SAFETY.md` §1
    requires an explicit opt-in for those, and a dial is not an opt-in.

    Args:
        declared: The scenario's own faults, already scaled.
        prof: The level's profile.
        preset: The preset the scenario named, if any.
        allow_side_effects: Tools the scenario opted in for.

    Returns:
        `(extras, skipped)`. `extras` are new specs to register alongside the declared
        ones; `skipped` records what was refused and why.
    """
    from .faults.base import FAULT_REGISTRY
    from .scenarios import PRESETS

    if prof.extra_faults <= 0:
        return [], []

    name = preset if preset in PRESETS else _infer_preset(declared)
    have = {_identity(spec) for spec in declared}
    extras: list[FaultSpec] = []
    skipped: list[dict[str, str]] = []

    for spec in sorted(PRESETS[name], key=_identity):
        if len(extras) >= prof.extra_faults:
            break
        if _identity(spec) in have:
            continue
        fault_cls = FAULT_REGISTRY.get(spec.type)
        if fault_cls is None:
            skipped.append({"type": spec.type, "reason": "fault_not_registered"})
            continue
        if getattr(fault_cls, "performs_real_action", False) and not allow_side_effects:
            # SAFETY.md section 1 / D-23: a real action needs a named opt-in, and
            # turning a dial to 10 is not one.
            skipped.append({"type": spec.type, "reason": "needs_allow_side_effects"})
            continue
        have.add(_identity(spec))
        extras.append(
            replace(
                spec,
                params=scale_params(spec.type, spec.params, prof),
                origin=f"intensity:{prof.level}:{name}",
            )
        )
    return extras, skipped


def _identity(spec: FaultSpec) -> str:
    """A stable key for de-duplicating and ordering fault specs.

    Args:
        spec: The spec.

    Returns:
        Canonical JSON over type, params and target -- not the trigger, because two
        specs that differ only in when they fire are the same experiment twice.
    """
    return json.dumps(
        [spec.type, dict(spec.params or {}), dict(spec.target or {})],
        sort_keys=True,
        default=str,
    )


def _infer_preset(declared: Sequence[FaultSpec]) -> str:
    """Pick the preset a scenario's faults most look like.

    Args:
        declared: The declared faults.

    Returns:
        A preset name. `smoke` when nothing matches -- it is the smallest, so an
        unrecognised scenario gets the mildest compounding.
    """
    from .scenarios import PRESETS

    kinds = {spec.type for spec in declared}
    best, best_score = "smoke", 0
    for name in sorted(PRESETS):
        if name == "full":
            continue
        score = len(kinds & {spec.type for spec in PRESETS[name]})
        if score > best_score:
            best, best_score = name, score
    return best
