"""Scenarios, suites, presets and matrix expansion.

Presets are literal `FaultSpec` lists rather than prose (D-14), because they feed
`plan_hash` and the golden tests: a preset described in a sentence cannot be hashed.
Matrix ids follow the single canonical rule in D-15. `must_not` is validated against
the registered probe codes at load, because a typo that silently never matches is
worse than a loud failure (D-33).

YAML needs the `[yaml]` extra at runtime, but `pyyaml` is also in `[dev]` because
every gate command runs a `.yaml` suite (D-27). JSON suites need no extra at all.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

from .assertions import Expect
from .context import Limits
from .enums import ExpectedBehavior
from .errors import ConfigError, MissingExtraError
from .intensity import DEFAULT_LEVEL, profile
from .report import ChaosResult

__all__ = [
    "PRESETS",
    "ChaosSuite",
    "FaultSpec",
    "JudgeSpec",
    "Scenario",
    "SuiteResult",
    "load_suite",
    "resolve_preset",
    "slug",
]

_SLUG_ALLOWED = re.compile(r"[^a-z0-9._-]+")


def slug(value: Any) -> str:
    """Normalize a matrix value into an id segment.

    D-15: lowercase, map anything outside ``[a-z0-9._-]`` to ``-``, collapsing runs.
    No brackets -- they violate the id pattern in `scenario.schema.json`, which is
    why the old ``id[value]`` form had to go.

    Args:
        value: The matrix value.

    Returns:
        The slug.
    """
    text = str(value).lower()
    return _SLUG_ALLOWED.sub("-", text)


@dataclass(frozen=True, slots=True)
class FaultSpec:
    """One fault in a plan, as a scenario file declares it.

    Attributes:
        type: The fault's `kind`.
        params: Its constructor parameters.
        target: Where it applies.
        trigger: When it fires. Presets always set this explicitly, because an
            implicit default could change and move every plan hash with it.
        origin: Where this fault came from. Empty means the scenario declared it;
            `intensity:<level>:<preset>` means the dial added it. A reader looking at
            five faults in a plan needs to know which two the file actually asked for.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    trigger: dict[str, Any] = field(default_factory=dict)
    origin: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for a scenario's `faults` list.

        Returns:
            A plain dict with empty sections omitted.
        """
        out: dict[str, Any] = {"type": self.type, "params": dict(self.params)}
        if self.target:
            out["target"] = dict(self.target)
        if self.trigger:
            out["trigger"] = dict(self.trigger)
        if self.origin:
            out["origin"] = self.origin
        return out


@dataclass(frozen=True, slots=True)
class JudgeSpec:
    """Per-scenario judge configuration.

    Fields mirror `scenario.schema.json`'s `judge` block, which sets
    ``additionalProperties: false``.

    Attributes:
        kind: ``"rules"``, ``"slm"`` or ``"ensemble"``.
        model: The model id, for the SLM judge.
        base_url: The endpoint, normalized per transport (D-34).
        transport: ``"openai"``, ``"ollama"`` or ``"anthropic"``.
        temperature: Sampling temperature; 0 keeps a run reproducible (D-07).
        offline_fallback: Fall back to `RuleJudge` rather than raising.
    """

    kind: str = "rules"
    model: str | None = None
    base_url: str | None = None
    transport: str = "openai"
    temperature: float = 0.0
    offline_fallback: bool = True


def _t(**kw: Any) -> dict[str, Any]:
    """Shorthand for an explicit trigger in a preset.

    Args:
        **kw: Trigger fields.

    Returns:
        The trigger mapping.
    """
    return dict(kw)


# The ten recipes from `docs/03-FAULT-CATALOG.md` §D, as literal lists (D-14).
# `state_integrity` and `resume_safety` name phase-05 faults; they resolve lazily by
# `kind` string and record `fault_not_registered` until M5 lands, so this phase can
# reference all ten without owning phase 05's work.
_SMOKE = [
    FaultSpec("ToolCorruptionFault", {"mutation_type": "empty_json"}, {}, _t(on_call=1)),
    FaultSpec("ToolErrorFault", {"error_type": "exception"}, {}, _t(on_call=1)),
]
_TOOL_CONTRACT = [
    FaultSpec("ToolCorruptionFault", {"mutation_type": mutation}, {}, _t(on_call=1))
    for mutation in ("drop_key", "type_flip", "null_result", "unit_swap", "json_as_string")
]
_TRANSIENT = [
    FaultSpec("ToolErrorFault", {"error_type": "http_500"}, {}, _t(on_call=1)),
    FaultSpec("ToolErrorFault", {"error_type": "http_429"}, {}, _t(on_call=1)),
    FaultSpec("RateLimitFault", {"after_calls": 2}, {}, _t(max_fires=99)),
    FaultSpec("ToolLatencyFault", {"delay_ms": 500}, {}, _t(on_call=1)),
]
_LONG_HORIZON = [
    FaultSpec("ContextShrinkFault", {"strategy": "middle_out"}, {}, _t(after_step=2, max_fires=99)),
    FaultSpec("ContextNoiseFault", {"noise_type": "unrelated_transcript"}, {}, _t(after_step=2)),
    FaultSpec("GoalDriftFault", {"mode": "dilute"}, {}, _t(after_step=3, max_fires=99)),
]
_STRUCTURED_OUTPUT = [
    FaultSpec("LLMMalformedOutputFault", {"mode": mode}, {}, _t(on_call=1))
    for mode in (
        "prose_instead_of_json",
        "trailing_commentary",
        "markdown_fenced",
        "broken_json",
        "wrong_schema",
        "extra_fields",
        "missing_required",
        "wrong_enum_value",
        "double_encoded",
    )
] + [
    FaultSpec("LLMTruncationFault", {"cut": "mid_json"}, {}, _t(on_call=1)),
    FaultSpec("MalformedToolCallFault", {"mode": "missing_arg"}, {}, _t(on_call=1)),
]
_LOOP_SAFETY = [
    FaultSpec("LoopTrapFault", {"pin_after_call": 1}, {}, _t(max_fires=99)),
    FaultSpec("NonDeterminismFault", {"mutation_type": "reorder_list"}, {}, _t(max_fires=99)),
    FaultSpec("StateStaleFault", {}, {}, _t(max_fires=99)),
]
_ADVERSARIAL = [
    FaultSpec("PromptInjectionFault", {"objective": objective}, {}, _t(on_call=1))
    for objective in (
        "exfiltrate_secret",
        "ignore_instructions",
        "call_forbidden_tool",
        "change_output_format",
        "escalate_scope",
    )
] + [FaultSpec("ToolCorruptionFault", {"mutation_type": "unicode_noise"}, {}, _t(on_call=1))]
_STATE_INTEGRITY = [
    FaultSpec("StateDropFault", {}, {}, _t(on_step=2)),
    FaultSpec("StateTypeFault", {}, {}, _t(on_step=2)),
    FaultSpec("NodeSkipFault", {}, {}, _t(on_step=2)),
    # `__END__` is the one destination a preset can name without knowing the
    # graph: forcing an early end is a real failure mode and is always valid.
    FaultSpec("EdgeMisrouteFault", {"force_to": "__END__"}, {}, _t(on_step=2)),
]
_RESUME_SAFETY = [
    FaultSpec("CheckpointRollbackFault", {}, {}, _t(on_step=2)),
    FaultSpec("DuplicateSideEffectFault", {"times": 2}, {}, _t(on_call=1)),
]

_HALLUCINATION = [
    FaultSpec("HallucinationInducerFault", {"mode": mode}, {}, _t(on_call=1))
    for mode in (
        "false_premise",
        "unanswerable_request",
        "citation_pressure",
        "authority_bias",
        "leading_question",
        "entity_lookalike",
    )
] + [
    FaultSpec("HallucinationSeedFault", {"mode": mode}, {}, _t(on_call=1))
    for mode in ("invent_value", "contradict_tool_output", "invent_citation", "invent_tool")
]

PRESETS: dict[str, list[FaultSpec]] = {
    "smoke": _SMOKE,
    "tool_contract": _TOOL_CONTRACT,
    "transient_faults": _TRANSIENT,
    "long_horizon": _LONG_HORIZON,
    "structured_output": _STRUCTURED_OUTPUT,
    "loop_safety": _LOOP_SAFETY,
    "adversarial": _ADVERSARIAL,
    "state_integrity": _STATE_INTEGRITY,
    "resume_safety": _RESUME_SAFETY,
    "hallucination": _HALLUCINATION,
    "full": [
        *_SMOKE,
        *_TOOL_CONTRACT,
        *_TRANSIENT,
        *_LONG_HORIZON,
        *_STRUCTURED_OUTPUT,
        *_LOOP_SAFETY,
        *_ADVERSARIAL,
        *_STATE_INTEGRITY,
        *_RESUME_SAFETY,
        *_HALLUCINATION,
    ],
}


def resolve_preset(name: str) -> tuple[list[FaultSpec], list[dict[str, str]]]:
    """Resolve a preset against the registered faults.

    Args:
        name: The preset name.

    Returns:
        ``(resolved, skipped)``. A fault whose `kind` is not registered yet is
        skipped with `fault_not_registered` rather than raising, so a phase can
        reference a preset containing a later phase's faults (D-14).

    Raises:
        ConfigError: When the preset name is unknown, naming the nearest match.
    """
    if name not in PRESETS:
        near = difflib.get_close_matches(name, sorted(PRESETS), n=1)
        hint = f"; did you mean {near[0]!r}?" if near else ""
        raise ConfigError(f"unknown preset {name!r}; available: {', '.join(sorted(PRESETS))}{hint}")

    from .faults.base import FAULT_REGISTRY

    resolved: list[FaultSpec] = []
    skipped: list[dict[str, str]] = []
    for spec in PRESETS[name]:
        if spec.type in FAULT_REGISTRY:
            resolved.append(spec)
        else:
            skipped.append({"type": spec.type, "reason": "fault_not_registered"})
    return resolved, skipped


def _validate_must_not(codes: Sequence[str]) -> list[str]:
    """Check `must_not` entries against the registered probe codes (D-33).

    A typo loads clean, never matches, and the scenario passes -- while the
    refinement loop hashes the list as if it were protecting something. Failing loudly
    at load is the only way that mistake ever surfaces.

    Args:
        codes: The declared codes.

    Returns:
        The codes unchanged.

    Raises:
        ConfigError: Naming the unknown code and its nearest match.
    """
    from .probes import PROBE_PRECEDENCE

    known = set(PROBE_PRECEDENCE)
    for code in codes:
        if code not in known:
            near = difflib.get_close_matches(code, sorted(known), n=1)
            hint = f"; did you mean {near[0]!r}?" if near else ""
            raise ConfigError(
                f"must_not names an unknown probe code {code!r}{hint}. "
                f"Registered codes: {', '.join(sorted(known))}"
            )
    return list(codes)


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    """Write a dotted path into a nested scenario body.

    Args:
        target: The scenario body, mutated in place.
        path: A dotted path such as ``faults.0.params.mutation_type``.
        value: The value to write.

    Raises:
        ConfigError: When the path addresses nothing in this scenario.
    """
    parts = path.split(".")
    node: Any = target
    try:
        for part in parts[:-1]:
            node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
        last = parts[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    except (IndexError, KeyError, ValueError, TypeError, AttributeError) as exc:
        # A bare IndexError mid-expansion tells an author nothing. The commonest
        # cause is a matrix key naming a fault index the scenario never declared.
        raise ConfigError(
            f"matrix key {path!r} does not address anything in this scenario "
            f"({type(exc).__name__}). Check the index and that the fault is declared."
        ) from exc


@dataclass(slots=True)
class Scenario:
    """One experiment: an entrypoint, a fault list, and what was expected of it.

    v0.1 scenarios are single-invocation only (D-41).
    """

    id: str
    entrypoint: str | Callable[..., Any]
    inputs: Any = None
    initial_state: dict[str, Any] | None = None
    faults: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 1337
    expected_behavior: ExpectedBehavior = "graceful_degradation"
    must_not: list[str] = field(default_factory=list)
    expect: Expect | None = None
    expected_errors: list[str] = field(default_factory=list)
    allow_side_effects: list[str] = field(default_factory=list)
    objective_state_key: str = "query"
    #: How hard the plan pushes, 1 (strict) to 10 (creative). 3 is the identity: a
    #: dial nobody turned changes nothing. See `agent_loop_chaos.intensity`.
    intensity: int = DEFAULT_LEVEL
    #: The preset this scenario named, kept so a high intensity knows where to draw
    #: extra faults from. `resolve_preset` already folded its faults into `faults`.
    preset: str | None = None
    dry_run: bool = False
    limits: Limits = field(default_factory=Limits)
    judge: JudgeSpec | None = None
    tags: dict[str, str] = field(default_factory=dict)
    description: str | None = None
    matrix: dict[str, list[Any]] | None = None
    preset_skipped: list[dict[str, str]] = field(default_factory=list)
    # Set by `load_suite`, unset for a suite built in Python. `RefinementLoop` hashes
    # this file between rounds: editing the scenario is the easiest way to make a
    # chaos run pass, and a flip that follows an edit is a regression, not a fix
    # (`docs/05` §9).
    source_path: str | None = None

    def __post_init__(self) -> None:
        """Validate the fields that can only be checked against the registry.

        Raises:
            ConfigError: When `must_not` names an unknown probe code (D-33), or when
                `intensity` is off the dial.
        """
        self.must_not = _validate_must_not(self.must_not)
        profile(self.intensity)

    def to_body(self) -> dict[str, Any]:
        """Render the scenario as a plain body, for matrix substitution.

        Returns:
            A deep-copied mapping of the mutable fields.
        """
        rendered = json.dumps(
            {
                "id": self.id,
                "inputs": self.inputs,
                "initial_state": self.initial_state,
                "faults": self.faults,
                "seed": self.seed,
            },
            default=str,
        )
        body: dict[str, Any] = json.loads(rendered)
        return body

    def expand(self) -> list[Scenario]:
        """Apply `matrix`, producing one scenario per combination.

        Products are ordered by sorted matrix key, then by declared value order, and
        ids follow ``<base_id>-<last_key_segment>-<slug(value)>`` (D-15). The base
        scenario is never mutated: it is reused across every product.

        Returns:
            The expanded scenarios, or ``[self]`` when there is no matrix.
        """
        if not self.matrix:
            return [self]

        keys = sorted(self.matrix)
        out: list[Scenario] = []
        for combination in product(*(self.matrix[key] for key in keys)):
            body = self.to_body()
            suffix = ""
            for key, value in zip(keys, combination, strict=True):
                _set_path(body, key, value)
                suffix += f"-{key.split('.')[-1]}-{slug(value)}"
            out.append(
                replace(
                    self,
                    id=f"{self.id}{suffix}",
                    inputs=body["inputs"],
                    initial_state=body["initial_state"],
                    faults=body["faults"],
                    seed=int(body["seed"]),
                    matrix=None,
                )
            )
        return out


@dataclass(slots=True)
class SuiteResult:
    """The aggregate of one suite run, written to `<out_dir>/suite.json`.

    That file is what a CI job, an optimizer, or the dashboard polls.
    """

    results: list[ChaosResult] = field(default_factory=list)
    out_dir: Path = field(default_factory=Path)

    @property
    def passed(self) -> int:
        """How many scenarios passed.

        Returns:
            The count.
        """
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        """How many scenarios failed.

        Returns:
            The count.
        """
        return sum(1 for r in self.results if not r.success)

    @property
    def failure_modes(self) -> dict[str, int]:
        """A histogram of failure modes across the suite.

        Returns:
            Failure mode to count, failures only.
        """
        out: dict[str, int] = {}
        for result in self.results:
            if not result.success:
                out[result.failure_mode] = out.get(result.failure_mode, 0) + 1
        return out

    @property
    def coverage(self) -> dict[str, int]:
        """How many times each fault kind actually fired.

        A preset that never fires is a preset that proves nothing, so this is the
        number that says the suite did any work.

        Returns:
            Fault kind to fire count.
        """
        out: dict[str, int] = {}
        for result in self.results:
            for fault in result.injected_faults:
                if fault.get("fired"):
                    kind = str(fault.get("type"))
                    out[kind] = out.get(kind, 0) + int(fault.get("fire_count", 1))
        return out

    def worst(self, n: int = 5) -> list[ChaosResult]:
        """The most severe results, worst first.

        Args:
            n: How many to return.

        Returns:
            Up to `n` failing results, ordered by severity.
        """
        from .probes import SEVERITY_ORDER

        failures = [r for r in self.results if not r.success]
        return sorted(failures, key=lambda r: -SEVERITY_ORDER.get(r.severity, 0))[:n]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the `suite.json` shape.

        Returns:
            The suite result as a plain dict.
        """
        return {
            "passed": self.passed,
            "failed": self.failed,
            "failure_modes": self.failure_modes,
            "coverage": self.coverage,
            "out_dir": str(self.out_dir),
        }


class ChaosSuite:
    """A collection of scenarios run together."""

    def __init__(self, scenarios: Sequence[Scenario], *, engine: Any | None = None) -> None:
        """Initialise a suite.

        Args:
            scenarios: The scenarios to run, already expanded.
            engine: A configured `ChaosEngine`, or `None` for a default.
        """
        self.scenarios = list(scenarios)
        self.engine = engine

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChaosSuite:
        """Load a suite from YAML or JSON.

        Args:
            path: Path to the suite file.

        Returns:
            The loaded suite.

        Raises:
            ConfigError: When the file is missing, unparseable, or fails schema
                validation.
            MissingExtraError: For a YAML file when `pyyaml` is absent.
        """
        return load_suite(path)


def _load_document(path: Path) -> dict[str, Any]:
    """Read a suite file as JSON or YAML.

    Args:
        path: The file to read.

    Returns:
        The parsed document.

    Raises:
        ConfigError: When the file is missing or unparseable.
        MissingExtraError: For a YAML file with `pyyaml` absent.
    """
    if not path.is_file():
        raise ConfigError(f"suite file not found: {path}")
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise MissingExtraError(
                "reading a YAML suite needs pyyaml: install agent-loop-chaos[yaml]. "
                "JSON suites work with no extras."
            ) from exc
        try:
            loaded = yaml.safe_load(text)
        except Exception as exc:
            raise ConfigError(f"{path}: could not parse as YAML: {exc}") from exc
    else:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}: could not parse as JSON: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: a suite file must be a mapping, got {type(loaded).__name__}")
    return loaded


def _merge_defaults(defaults: Mapping[str, Any], body: Mapping[str, Any]) -> dict[str, Any]:
    """Merge suite defaults into one scenario body.

    Fault lists **concatenate**; everything else is replaced by the scenario's own
    value. That asymmetry is deliberate: defaults exist so a suite can arm one fault
    everywhere, and replacing the list would silently drop it.

    Args:
        defaults: The suite-level defaults.
        body: The scenario's own body.

    Returns:
        The merged body.
    """
    merged = {**dict(defaults), **dict(body)}
    default_faults = list(defaults.get("faults") or [])
    own_faults = list(body.get("faults") or [])
    if default_faults or own_faults:
        merged["faults"] = [*default_faults, *own_faults]
    return merged


def _build(name: str, cls: Any, body: Any) -> Any:
    """Construct a scenario sub-block, turning a bad key into a `ConfigError`.

    A `TypeError` from a dataclass constructor names the argument but not the file,
    the scenario, or the block -- which is most of what an author needs.

    Args:
        name: The block name, for the message.
        cls: The dataclass to build.
        body: The mapping from the scenario file, or `None`.

    Returns:
        The constructed object, or `None` when `body` is not a mapping.

    Raises:
        ConfigError: When a key is unknown to the block.
    """
    if not isinstance(body, Mapping):
        return None
    try:
        return cls(**dict(body))
    except TypeError as exc:
        raise ConfigError(f"invalid `{name}` block: {exc}") from exc


def _scenario_from_body(body: Mapping[str, Any]) -> Scenario:
    """Build a `Scenario` from a validated body.

    Args:
        body: The merged scenario body.

    Returns:
        The scenario.

    Raises:
        ConfigError: When a preset name or a `must_not` code is unknown.
    """
    data = dict(body)
    faults = [dict(f) for f in data.pop("faults", []) or []]
    skipped: list[dict[str, str]] = []

    preset = data.pop("preset", None)
    if preset:
        resolved, skipped = resolve_preset(str(preset))
        faults = [*(spec.to_dict() for spec in resolved), *faults]

    expect = _build("expect", Expect, data.pop("expect", None))
    judge = _build("judge", JudgeSpec, data.pop("judge", None))
    limits = _build("limits", Limits, data.pop("limits", None)) or Limits()

    known = {
        "id",
        "entrypoint",
        "inputs",
        "initial_state",
        "seed",
        "expected_behavior",
        "must_not",
        "expected_errors",
        "allow_side_effects",
        "objective_state_key",
        "intensity",
        "dry_run",
        "tags",
        "description",
        "matrix",
    }
    return Scenario(
        faults=faults,
        expect=expect,
        judge=judge,
        limits=limits,
        preset_skipped=skipped,
        preset=str(preset) if preset else None,
        **{k: v for k, v in data.items() if k in known},
    )


def load_suite(path: str | Path) -> ChaosSuite:
    """Load and expand a suite file.

    The document is validated against `scenario.schema.json` before use, so a
    malformed file fails at load with a JSON pointer rather than mid-run with a
    `KeyError`.

    Args:
        path: Path to a YAML or JSON suite.

    Returns:
        The loaded suite, with every matrix already expanded -- so the scenario count
        is what the runner will actually execute.

    Raises:
        ConfigError: On a missing file, a parse failure, or a schema violation.
    """
    from .schema import validate_obj

    path = Path(path)
    document = _load_document(path)

    errors = validate_obj(document, "scenario")
    if errors:
        raise ConfigError(f"{path} does not satisfy scenario.schema.json: {errors[:3]}")

    defaults = document.get("defaults") or {}
    scenarios: list[Scenario] = []
    for body in document.get("scenarios") or []:
        scenarios.extend(_scenario_from_body(_merge_defaults(defaults, body)).expand())
    for scenario in scenarios:
        scenario.source_path = str(path)
    return ChaosSuite(scenarios)
