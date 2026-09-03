"""The `Fault` base, its outcome types, the registry, and `NoopFault`.

`NoopFault` exists so the whole plumbing — targeting, triggering, chaining,
recording, reporting — can be tested end to end before a single real fault exists.
The 26 real faults arrive in M2, M3 and M5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeVar

from ..context import Crossing, FaultContext, Layer, Phase
from ..enums import Severity
from ..errors import ConfigError
from ..jsonpatch import diff, is_jsonable
from ..seeding import canonical_json, sha256_of

__all__ = [
    "FAULT_REGISTRY",
    "TERMINAL_ACTIONS",
    "VALUE_ACTIONS",
    "Action",
    "Fault",
    "FaultInfo",
    "FaultOutcome",
    "FaultRecord",
    "MutationLog",
    "NoopFault",
    "fault_from_dict",
    "fault_key_for",
    "list_faults",
    "register_fault",
]

Action = Literal[
    "replace_result",
    "replace_args",
    "raise",
    "delay",
    "replace_state",
    "replace_messages",
    "invoke_target",
    "resume_from_checkpoint",
    "noop",
]

# Value-replacing actions chain: each fault receives the previous one's output and
# records its own MutationLog. Terminal actions are first-wins and supersede the
# rest, which is what makes the multi-mutation presets exercise every mutation
# instead of only the first (D-13).
VALUE_ACTIONS: frozenset[str] = frozenset(
    {"replace_result", "replace_args", "replace_state", "replace_messages"}
)
TERMINAL_ACTIONS: frozenset[str] = frozenset(
    {"raise", "delay", "invoke_target", "resume_from_checkpoint"}
)

ALL_PAIRS: frozenset[tuple[Layer, Phase]] = frozenset(
    (layer, phase)
    for layer in ("tool", "llm", "state", "node", "edge", "checkpoint")
    for phase in ("pre", "post", "error")
)


@dataclass(slots=True)
class MutationLog:
    """What a fault changed, exactly.

    Attributes:
        payload_before: The value as it arrived.
        payload_after: The value the fault produced.
        json_patch: An RFC 6902 subset patch. Empty when the values are equal, or
            when either side is not JSON-representable.
        unrepresentable: True when the payload could not be represented as JSON, so
            an empty `json_patch` is not mistaken for "nothing changed".
        mutation_skipped_uncopyable: True when the fault could not deep-copy the
            value and passed the original through untouched.
    """

    payload_before: Any = None
    payload_after: Any = None
    json_patch: list[dict[str, Any]] = field(default_factory=list)
    unrepresentable: bool = False
    mutation_skipped_uncopyable: bool = False

    @classmethod
    def of(cls, before: Any, after: Any) -> MutationLog:
        """Build a log by diffing two values.

        Args:
            before: The original value.
            after: The mutated value.

        Returns:
            The populated log, with `unrepresentable` set when a patch could not be
            computed.
        """
        representable = is_jsonable(before) and is_jsonable(after)
        return cls(
            payload_before=before,
            payload_after=after,
            json_patch=diff(before, after) if representable else [],
            unrepresentable=not representable,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the trace and the report.

        Returns:
            A plain dict of the log.
        """
        return {
            "payload_before": self.payload_before,
            "payload_after": self.payload_after,
            "json_patch": self.json_patch,
            "unrepresentable": self.unrepresentable,
            "mutation_skipped_uncopyable": self.mutation_skipped_uncopyable,
        }


@dataclass(slots=True)
class FaultOutcome:
    """What a fault asks the engine to do.

    Attributes:
        action: Which of the nine actions to take.
        value: The new result, args, exception, state or messages.
        delay_ms: How long to sleep, for ``delay``.
        note: A short human note, e.g. ``"dropped key temp_c"``.
        mutation: What changed, for the report's payload diff.
        params: Extra action parameters — ``times``/``return_from`` for
            ``invoke_target``, ``rollback_steps`` for ``resume_from_checkpoint``
            (D-10).
    """

    action: Action = "noop"
    value: Any = None
    delay_ms: int = 0
    note: str | None = None
    mutation: MutationLog | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FaultRecord:
    """One fault's identity, plan and fire history, as it lands in the report.

    Carries both ids on purpose (D-03): `fault_id` for display and
    cross-referencing, `fault_key` for RNG streams and `randomness.streams`.
    """

    fault_id: str
    fault_key: str
    type: str
    params: dict[str, Any]
    target: dict[str, Any]
    trigger: dict[str, Any]
    fired: bool = False
    fire_count: int = 0
    skipped_reason: str | None = None
    fires: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the `injected_faults[]` shape.

        `skipped_reason` is emitted only when nothing fired (D-36).

        Returns:
            A plain dict matching `chaos_report.schema.json`.
        """
        out: dict[str, Any] = {
            "fault_id": self.fault_id,
            "fault_key": self.fault_key,
            "type": self.type,
            "params": self.params,
            "target": self.target,
            "trigger": self.trigger,
            "fired": self.fired,
            "fire_count": self.fire_count,
            "fires": self.fires,
        }
        if self.fire_count == 0:
            out["skipped_reason"] = self.skipped_reason
        return out


@dataclass(frozen=True, slots=True)
class FaultInfo:
    """Catalog metadata, powering `alc list-faults`."""

    kind: str
    layer_phases: tuple[str, ...]
    severity_hint: Severity
    summary: str


class Fault(ABC):
    """Base class for every fault.

    Subclasses state in their docstring **what agent weakness the fault proves** and
    **what graceful behaviour looks like** — that is what turns a finding into an
    actionable work order.

    Attributes:
        kind: Stable name used in reports and YAML. Renaming one breaks stored
            reports, so it is part of the public contract.
        accepts: The ``(layer, phase)`` pairs this fault handles. The engine refuses
            a mismatched registration with `ConfigError` at registration time.
        severity_hint: Default severity if this fault causes a failure.
        performs_real_action: True when the fault performs an operation the agent
            never requested -- a real tool call, a repeated call, a replayed node.
            The D-23 gate refuses these against a `side_effecting=True` tool without
            an explicit opt-in (`SAFETY.md` §1).
    """

    kind: ClassVar[str] = "Fault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset()
    severity_hint: ClassVar[Severity] = "medium"
    performs_real_action: ClassVar[bool] = False

    def __init__(self, **params: Any) -> None:
        """Initialise and validate parameters eagerly.

        Args:
            **params: Fault-specific parameters. See `docs/03-FAULT-CATALOG.md`.

        Raises:
            ConfigError: On an invalid parameter.
        """
        self._params: dict[str, Any] = dict(params)
        self.validate()

    def validate(self) -> None:  # noqa: B027 - optional hook; most faults need no checks
        """Check parameters. Subclasses override; the base accepts anything.

        Raises:
            ConfigError: On an invalid parameter.
        """

    def params(self) -> dict[str, Any]:
        """Return the parameters as serialized into the report and the plan hash.

        Returns:
            A copy of the parameters, so a caller cannot mutate the plan.
        """
        return dict(self._params)

    @abstractmethod
    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Apply the fault to a deep copy of the crossing's payload.

        Never mutates the caller's object. A fault that cannot copy a value records
        `mutation_skipped_uncopyable` and passes the original through.

        Args:
            crossing: The crossing being faulted.
            ctx: The fault context (D-01), carrying the seeded RNG, counters,
                history, the tool registry and the state view.

        Returns:
            The outcome the engine should act on.
        """

    def __repr__(self) -> str:
        """Readable representation including parameters.

        Returns:
            e.g. ``NoopFault()``.
        """
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self._params.items()))
        return f"{type(self).__name__}({args})"


FAULT_REGISTRY: dict[str, type[Fault]] = {}

F = TypeVar("F", bound=type[Fault])


def register_fault(cls: F) -> F:
    """Register a fault class under its `kind`.

    Args:
        cls: The `Fault` subclass to register.

    Returns:
        The class unchanged, so this works as a decorator.

    Raises:
        ConfigError: If `kind` is missing, or already registered by another class.
    """
    kind = getattr(cls, "kind", None)
    if not kind or kind == "Fault":
        raise ConfigError(f"{cls.__name__} must define a non-empty class-level `kind`")
    existing = FAULT_REGISTRY.get(kind)
    if existing is not None and existing is not cls:
        raise ConfigError(f"fault kind {kind!r} is already registered by {existing.__name__}")
    FAULT_REGISTRY[kind] = cls
    return cls


def fault_key_for(
    kind: str,
    params: Mapping[str, Any],
    target: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> str:
    """Compute the stable hash that keys a fault's RNG stream.

    Keyed on the spec rather than the registration ordinal, so inserting a fault,
    reordering YAML, or expanding a preset never re-keys another fault's stream
    (D-03).

    Args:
        kind: The fault's `kind`.
        params: Its parameters.
        target: Its serialized target.
        trigger: Its serialized trigger.

    Returns:
        The first 6 hex characters of the spec hash.
    """
    spec = {
        "type": kind,
        "params": dict(params),
        "target": dict(target),
        "trigger": dict(trigger),
    }
    return sha256_of(canonical_json(spec)).removeprefix("sha256:")[:6]


def fault_from_dict(spec: Mapping[str, Any]) -> Fault:
    """Build a fault from a YAML or JSON spec.

    Args:
        spec: ``{"type": "ToolCorruptionFault", "params": {...}}``.

    Returns:
        The constructed fault.

    Raises:
        ConfigError: If `type` is missing or names an unregistered fault.
    """
    kind = spec.get("type")
    if not kind:
        raise ConfigError(f"fault spec is missing `type`: {dict(spec)!r}")
    cls = FAULT_REGISTRY.get(str(kind))
    if cls is None:
        known = ", ".join(sorted(FAULT_REGISTRY)) or "(none registered yet)"
        raise ConfigError(f"unknown fault type {kind!r}; registered kinds: {known}")
    return cls(**dict(spec.get("params") or {}))


def list_faults() -> list[FaultInfo]:
    """Describe every registered fault, for `alc list-faults`.

    Returns:
        One `FaultInfo` per registered kind, sorted by kind.
    """
    out: list[FaultInfo] = []
    for kind in sorted(FAULT_REGISTRY):
        cls = FAULT_REGISTRY[kind]
        summary = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
        out.append(
            FaultInfo(
                kind=kind,
                layer_phases=tuple(sorted(f"{layer}.{phase}" for layer, phase in cls.accepts)),
                severity_hint=cls.severity_hint,
                summary=summary,
            )
        )
    return out


@register_fault
class NoopFault(Fault):
    """Fires and changes nothing. The plumbing test double.

    **What agent weakness it proves:** none. It exists so targeting, triggering,
    chaining, recording and reporting can be tested end to end before any real fault
    exists, and so a scenario can prove the harness itself is inert.

    **What graceful behaviour looks like:** identical to an unfaulted run. A probe
    that fires on a `NoopFault` run has a false positive.
    """

    kind: ClassVar[str] = "NoopFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = ALL_PAIRS
    severity_hint: ClassVar[Severity] = "info"

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Record a no-change mutation and let the value through.

        Args:
            crossing: The crossing being faulted.
            ctx: The fault context. Deliberately unused: drawing from the RNG here
                would perturb streams a test relies on.

        Returns:
            A `noop` outcome carrying a `MutationLog` with an empty patch.
        """
        observed = crossing.result if crossing.phase == "post" else crossing.args
        return FaultOutcome(
            action="noop",
            value=None,
            note="noop",
            mutation=MutationLog.of(observed, observed),
        )
