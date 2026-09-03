"""Tool-execution faults (catalog section A).

Every fault here answers two questions in its docstring: what agent weakness it
proves, and what graceful handling looks like. That pairing is what turns a finding
into a work order somebody can act on.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..context import Crossing, FaultContext, Layer, Phase
from ..enums import Severity
from ..errors import ConfigError, SimulatedToolError
from ..mutations import MUTATIONS
from .base import Fault, FaultOutcome, MutationLog, register_fault

__all__ = [
    "ArgumentTamperFault",
    "DuplicateSideEffectFault",
    "LoopTrapFault",
    "NonDeterminismFault",
    "RateLimitFault",
    "StaleDataFault",
    "ToolCorruptionFault",
    "ToolErrorFault",
    "ToolLatencyFault",
    "ToolTimeoutFault",
]

# Reads as English in a work order: "dropped key "temp_c" from 3 of 3 records".
_VERBS: dict[str, str] = {
    "drop_key": "dropped key",
    "drop_required_key": "dropped required key",
    "rename_key": "renamed key",
    "null_fields": "nulled field",
    "type_flip": "retyped field",
    "stringify_numbers": "stringified number in",
    "negative_numbers": "flipped the sign of",
    "nan_numbers": "made non-finite",
    "unit_swap": "unit-swapped",
    "truncate_string": "truncated string",
    "unicode_noise": "inserted invisible characters into",
}

_WHOLE_PAYLOAD: dict[str, str] = {
    "empty_json": "replaced the result with an empty object",
    "empty_list": "replaced the result with an empty list",
    "null_result": "replaced the result with null",
    "truncate_list": "kept only the first element of the list",
    "duplicate_items": "duplicated every element of the list",
    "reorder_list": "reordered the list",
    "deep_nest": "wrapped the result in extra nesting levels",
    "wrong_schema": "replaced the result with a different, plausible object",
    "json_as_string": "returned the result as a JSON string",
    "malformed_json_string": "returned a string of broken JSON",
    "whitespace_only": "returned whitespace only",
}


def _leaf_names(patch: list[dict[str, Any]]) -> list[str]:
    """Extract the distinct field names a patch touched.

    Args:
        patch: An RFC 6902 subset patch.

    Returns:
        Sorted leaf names, list indices excluded.
    """
    names: set[str] = set()
    for op in patch:
        parts = [p for p in str(op.get("path", "")).split("/") if p]
        leaves = [p for p in parts if not p.isdigit()]
        if leaves:
            names.add(leaves[-1])
    return sorted(names)


def _record_span(before: Any, patch: list[dict[str, Any]]) -> tuple[int, int]:
    """Count how many records were affected, out of how many there were.

    Args:
        before: The payload as it arrived.
        patch: The patch describing the change.

    Returns:
        ``(affected, total)``.
    """
    total = len(before) if isinstance(before, list) else 1
    if not isinstance(before, list):
        return (1 if patch else 0), total
    touched = {
        parts[0]
        for op in patch
        if (parts := [p for p in str(op.get("path", "")).split("/") if p]) and parts[0].isdigit()
    }
    return len(touched), total


def describe_mutation(
    mutation_type: str, before: Any, after: Any, patch: list[dict[str, Any]]
) -> str:
    """Build the one-clause `note` a work order quotes.

    Args:
        mutation_type: Which mutation ran.
        before: The payload as it arrived.
        after: The payload the mutation produced.
        patch: The recorded diff.

    Returns:
        Plain English, e.g. ``dropped key "temp_c" from 3 of 3 records``. Falls back
        to naming the mutation when the change cannot be summarized.
    """
    if mutation_type in _WHOLE_PAYLOAD:
        return _WHOLE_PAYLOAD[mutation_type]
    if not patch:
        return f"applied {mutation_type} but nothing changed"

    verb = _VERBS.get(mutation_type, f"applied {mutation_type} to")
    names = _leaf_names(patch)
    subject = ", ".join(f'"{name}"' for name in names) if names else "the payload"
    affected, total = _record_span(before, patch)
    if total > 1:
        return f"{verb} {subject} from {affected} of {total} records"
    return f"{verb} {subject}"


@register_fault
class ToolCorruptionFault(Fault):
    """Corrupt a tool's result: drop a key, retype it, swap its units.

    **What agent weakness it proves:** the agent uses tool output without validating
    its shape, its types, or the plausibility of its magnitudes. `unit_swap` is the
    sharpest of these because nothing about the payload's shape is wrong.

    **What graceful behaviour looks like:** detect the schema violation, retry or
    degrade, and *tell the user the data was unusable*. Never fabricate the missing
    value.

    Expected failure modes when handled badly: `crash_unhandled_exception` from a
    `KeyError` or `TypeError`, `hallucination_on_corrupt_data`, and
    `silent_wrong_answer` — especially for `unit_swap`.
    """

    kind: ClassVar[str] = "ToolCorruptionFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "post")})
    severity_hint: ClassVar[Severity] = "high"

    def validate(self) -> None:
        """Check the mutation name and the numeric bounds.

        Raises:
            ConfigError: With the offending value, so a scenario file can be fixed
                without reading the source.
        """
        params = self.params()
        mutation_type = params.get("mutation_type", "drop_key")
        if mutation_type != "random" and mutation_type not in MUTATIONS:
            known = ", ".join(sorted(MUTATIONS))
            raise ConfigError(
                f"unknown mutation_type {mutation_type!r}; expected 'random' or one of: {known}"
            )
        for name in ("count", "depth"):
            if name in params and int(params[name]) < 1:
                raise ConfigError(f"{self.kind}.{name} must be >= 1; got {params[name]!r}")
        keys = params.get("keys")
        if keys is not None and not isinstance(keys, (list, tuple)):
            raise ConfigError(f"{self.kind}.keys must be a list of dotted paths; got {keys!r}")

    def _resolve_mutation(self, ctx: FaultContext) -> str:
        """Pick the mutation to run, drawing from the RNG only for ``"random"``.

        Args:
            ctx: The fault context.

        Returns:
            A mutation name present in `MUTATIONS`.
        """
        mutation_type = str(self.params().get("mutation_type", "drop_key"))
        if mutation_type != "random":
            return mutation_type
        chosen = sorted(MUTATIONS)[ctx.rng("mutation_type").randrange(len(MUTATIONS))]
        # An RNG-derived choice is a decision; an explicitly configured one is not (D-36).
        ctx.record("mutation_type", chosen)
        return chosen

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Mutate a deep copy of the tool's result.

        Args:
            crossing: The `(tool, post)` crossing carrying the result.
            ctx: The fault context, supplying the seeded RNG.

        Returns:
            A `replace_result` outcome with a populated `MutationLog` and a note
            naming the field and the record count.
        """
        params = self.params()
        mutation_type = self._resolve_mutation(ctx)
        before = crossing.result

        kwargs: dict[str, Any] = {
            "count": int(params.get("count", 1)),
            "depth": int(params.get("depth", 3)),
        }
        if params.get("keys") is not None:
            kwargs["keys"] = list(params["keys"])
        if mutation_type == "drop_required_key":
            kwargs["baseline_prompts"] = params.get("baseline_prompts")

        after = MUTATIONS[mutation_type](before, ctx.rng(mutation_type), **kwargs)
        log = MutationLog.of(before, after)
        note = describe_mutation(mutation_type, before, after, log.json_patch)
        if mutation_type == "drop_required_key" and not params.get("baseline_prompts"):
            note = f"{note} (no baseline; fell back to drop_key)"
        return FaultOutcome(action="replace_result", value=after, note=note, mutation=log)


# The allow-list. Never `eval` a class name from a scenario file: these are
# user-authored strings, and evaluating one would be a code-execution hole.
_EXC_CLASSES: dict[str, type[BaseException]] = {
    "RuntimeError": RuntimeError,
    "TimeoutError": TimeoutError,
    "ConnectionError": ConnectionError,
    "ValueError": ValueError,
    "KeyError": KeyError,
    "SimulatedToolError": SimulatedToolError,
}

_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "exception",
        "error_payload",
        "http_500",
        "http_429",
        "http_401",
        "timeout",
        "connection_reset",
        "malformed_json",
    }
)

_HTTP_STATUS: dict[str, int] = {"http_500": 500, "http_429": 429, "http_401": 401}

_DEFAULT_MESSAGES: dict[str, str] = {
    "http_500": "500 Internal Server Error",
    "http_429": "429 Too Many Requests",
    "http_401": "401 Unauthorized",
    "timeout": "the tool did not respond in time",
    "connection_reset": "connection reset by peer",
    "malformed_json": "the tool returned a body that is not valid JSON",
    "exception": "the tool failed",
    "error_payload": "the tool reported an error",
}


def _build_exception(exc_class: str, message: str) -> BaseException:
    """Instantiate an allow-listed exception.

    Args:
        exc_class: A key of `_EXC_CLASSES`, validated at construction.
        message: The exception message.

    Returns:
        The exception instance.
    """
    return _EXC_CLASSES[exc_class](message)


def _validate_exc_class(kind: str, exc_class: str) -> None:
    """Reject a class name outside the allow-list.

    Args:
        kind: The fault's kind, for the message.
        exc_class: The requested class name.

    Raises:
        ConfigError: Listing the permitted names.
    """
    if exc_class not in _EXC_CLASSES:
        allowed = ", ".join(sorted(_EXC_CLASSES))
        raise ConfigError(
            f"{kind}.exc_class must be one of: {allowed}; got {exc_class!r}. "
            "Class names are never evaluated, so only these are available."
        )


@register_fault
class ToolErrorFault(Fault):
    """Make a tool fail, either by raising or by returning an error envelope.

    **What agent weakness it proves:** missing error handling; ignoring an in-band
    error envelope and treating it as data; failing to retry a transient class; and
    retrying a terminal one such as `http_401` until the budget is gone.

    **What graceful behaviour looks like:** distinguish transient from terminal,
    retry the transient kind with backoff a bounded number of times, and surface
    terminal errors explicitly rather than guessing.

    Expected failure modes: `crash_unhandled_exception`,
    `no_retry_on_transient_error`, `retry_storm`, and `silent_wrong_answer` when an
    in-band error is read as data.
    """

    kind: ClassVar[str] = "ToolErrorFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "pre")})
    severity_hint: ClassVar[Severity] = "high"

    def validate(self) -> None:
        """Check `error_type` and `exc_class`.

        Raises:
            ConfigError: With the offending value.
        """
        params = self.params()
        error_type = str(params.get("error_type", "exception"))
        if error_type not in _ERROR_TYPES:
            allowed = ", ".join(sorted(_ERROR_TYPES))
            raise ConfigError(
                f"{self.kind}.error_type must be one of: {allowed}; got {error_type!r}"
            )
        _validate_exc_class(self.kind, str(params.get("exc_class", "RuntimeError")))

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Fail the call, in the shape `error_type` asks for.

        Args:
            crossing: The `(tool, pre)` crossing; the real tool is not called.
            ctx: The fault context.

        Returns:
            A `raise` outcome, or a `replace_result` outcome carrying an error
            envelope when ``error_type="error_payload"``.
        """
        params = self.params()
        error_type = str(params.get("error_type", "exception"))
        message = str(params.get("message") or _DEFAULT_MESSAGES[error_type])
        status = _HTTP_STATUS.get(error_type)

        if error_type == "error_payload":
            payload: dict[str, Any] = params.get("status_payload") or {
                "error": message,
                "code": status or "tool_error",
            }
            log = MutationLog.of(None, payload)
            return FaultOutcome(
                action="replace_result",
                value=payload,
                note=f"returned an in-band error envelope instead of data: {message}",
                mutation=log,
                # The report says so explicitly: an in-band error the agent believed
                # is a different finding from a crash.
                params={"in_band": True, "error_type": error_type},
            )

        if error_type == "malformed_json":
            broken = '{"data": [{"temp_c": 21,'
            return FaultOutcome(
                action="replace_result",
                value=broken,
                note="returned a body that is not valid JSON",
                mutation=MutationLog.of(None, broken),
                params={"in_band": True, "error_type": error_type},
            )

        exc_class = str(params.get("exc_class", "RuntimeError"))
        if error_type == "timeout":
            exc_class = "TimeoutError"
        elif error_type == "connection_reset":
            exc_class = "ConnectionError"

        detail = f"{message} (status {status})" if status is not None else message
        retry_after = params.get("retry_after_s")
        if retry_after is not None:
            detail = f"{detail}; retry after {retry_after}s"
        return FaultOutcome(
            action="raise",
            value=_build_exception(exc_class, detail),
            note=f"raised {exc_class} instead of calling the tool: {detail}",
            params={"error_type": error_type, "status": status},
        )


@register_fault
class ToolTimeoutFault(Fault):
    """Make a tool time out immediately.

    Distinct from `ToolLatencyFault`: that one spends the budget, this one tests the
    *handling* path without waiting.

    **What agent weakness it proves:** no per-tool timeout handling, so a timeout
    propagates as an unhandled exception or is retried without limit.

    **What graceful behaviour looks like:** catch the timeout, retry a bounded
    number of times, then continue with partial results and say what is missing.
    """

    kind: ClassVar[str] = "ToolTimeoutFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "pre")})
    severity_hint: ClassVar[Severity] = "medium"

    def validate(self) -> None:
        """Check `exc_class` and `after_ms`.

        Raises:
            ConfigError: With the offending value.
        """
        params = self.params()
        _validate_exc_class(self.kind, str(params.get("exc_class", "TimeoutError")))
        if int(params.get("after_ms", 0)) < 0:
            raise ConfigError(f"{self.kind}.after_ms must be >= 0; got {params['after_ms']!r}")

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Raise a timeout in place of the call.

        Args:
            crossing: The `(tool, pre)` crossing.
            ctx: The fault context.

        Returns:
            A `raise` outcome carrying the timeout.
        """
        params = self.params()
        exc_class = str(params.get("exc_class", "TimeoutError"))
        after_ms = int(params.get("after_ms", 0))
        message = f"{crossing.name} timed out after {after_ms}ms"
        return FaultOutcome(
            action="raise",
            value=_build_exception(exc_class, message),
            note=f"timed out {crossing.name!r} after {after_ms}ms",
            params={"after_ms": after_ms},
        )


@register_fault
class RateLimitFault(Fault):
    """Throttle a tool or model after a budget of calls.

    Stateful across calls: the first `after_calls` invocations pass through and
    everything after is throttled, so a retry loop meets the limit rather than a
    one-off blip.

    **What agent weakness it proves:** whether the agent backs off, whether it
    respects `Retry-After`, and whether it burns its whole budget hammering a
    limited endpoint.

    **What graceful behaviour looks like:** honour `retry_after_s`, back off
    exponentially, cap the attempts, and report the limit rather than looping.

    Expected failure modes: `retry_storm`, `no_retry_on_transient_error`.
    """

    kind: ClassVar[str] = "RateLimitFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "pre"), ("llm", "pre")})
    severity_hint: ClassVar[Severity] = "medium"

    def validate(self) -> None:
        """Check the numeric bounds and the error shape.

        Raises:
            ConfigError: With the offending value.
        """
        params = self.params()
        for name, minimum in (("after_calls", 0), ("retry_after_s", 0), ("status", 100)):
            if name in params and int(params[name]) < minimum:
                raise ConfigError(f"{self.kind}.{name} must be >= {minimum}; got {params[name]!r}")
        error_type = str(params.get("error_type", "error_payload"))
        if error_type not in {"error_payload", "exception"}:
            raise ConfigError(
                f"{self.kind}.error_type must be 'error_payload' or 'exception'; got {error_type!r}"
            )

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Pass the call through until the budget is spent, then throttle.

        Args:
            crossing: The `(tool, pre)` or `(llm, pre)` crossing.
            ctx: The fault context, whose counters carry the call index.

        Returns:
            A `noop` outcome while under budget, otherwise a 429-shaped
            `replace_result` or a `raise`.
        """
        params = self.params()
        after_calls = int(params.get("after_calls", 2))
        if crossing.call_index <= after_calls:
            return FaultOutcome(
                action="noop",
                note=f"call {crossing.call_index} is within the rate-limit budget",
            )

        status = int(params.get("status", 429))
        retry_after = int(params.get("retry_after_s", 30))
        message = f"{status} Too Many Requests; retry after {retry_after}s"

        if str(params.get("error_type", "error_payload")) == "exception":
            return FaultOutcome(
                action="raise",
                value=_build_exception("RuntimeError", message),
                note=f"rate-limited {crossing.name!r} after {after_calls} calls",
                params={"status": status, "retry_after_s": retry_after},
            )

        payload = {
            "error": "rate_limited",
            "status": status,
            "retry_after_s": retry_after,
            "message": message,
        }
        return FaultOutcome(
            action="replace_result",
            value=payload,
            note=f"rate-limited {crossing.name!r} after {after_calls} calls",
            mutation=MutationLog.of(None, payload),
            params={"in_band": True, "status": status, "retry_after_s": retry_after},
        )


@register_fault
class ToolLatencyFault(Fault):
    """Make a tool slow.

    Clamped by `Limits.max_injected_delay_ms`, and the delay is accumulated into
    `metrics.injected_delay_ms` so every budget comparison can subtract it (R3).
    There is deliberately no latency *probe*: a timing-derived finding would make
    `success` wall-clock dependent, which is why that probe was removed (D-46).

    **What agent weakness it proves:** no per-tool timeout budget, unbounded
    sequential fan-out, and no notion of partial results.

    **What graceful behaviour looks like:** enforce a per-tool timeout and continue
    with what has already arrived, saying what is missing.

    Expected failure mode: `timeout`, surfaced through `loop.limit_hit`.
    """

    kind: ClassVar[str] = "ToolLatencyFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "pre")})
    severity_hint: ClassVar[Severity] = "low"

    def validate(self) -> None:
        """Check the delay bounds and the mode.

        Raises:
            ConfigError: With the offending value.
        """
        params = self.params()
        for name in ("delay_ms", "jitter_ms"):
            if name in params and int(params[name]) < 0:
                raise ConfigError(f"{self.kind}.{name} must be >= 0; got {params[name]!r}")
        mode = str(params.get("mode", "fixed"))
        if mode not in {"fixed", "ramp", "spike"}:
            raise ConfigError(f"{self.kind}.mode must be 'fixed', 'ramp' or 'spike'; got {mode!r}")

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Compute the delay, clamped, and ask the engine to sleep.

        Args:
            crossing: The `(tool, pre)` crossing.
            ctx: The fault context, for the limit and the seeded jitter.

        Returns:
            A `delay` outcome recording both the requested and the applied delay,
            so a clamped run does not understate what the scenario asked for.
        """
        params = self.params()
        base = int(params.get("delay_ms", 1000))
        mode = str(params.get("mode", "fixed"))
        if mode == "ramp":
            base *= max(1, crossing.call_index)
        elif mode == "spike":
            base *= 5 if crossing.call_index == 1 else 1

        jitter = int(params.get("jitter_ms", 0))
        requested = base + (ctx.rng("jitter").randrange(jitter + 1) if jitter else 0)
        applied = min(requested, ctx.limits.max_injected_delay_ms)
        clamped = " (clamped)" if applied < requested else ""
        return FaultOutcome(
            action="delay",
            delay_ms=applied,
            note=f"delayed {crossing.name!r} by {applied}ms{clamped}",
            params={"requested_delay_ms": requested, "applied_delay_ms": applied, "mode": mode},
        )


@register_fault
class StaleDataFault(Fault):
    """Serve an earlier call's payload, backdated so the staleness is detectable.

    The backdating matters: serving old data with a current timestamp would be
    undetectable, and a fault nobody can detect proves nothing.

    **What agent weakness it proves:** the agent ignores the freshness metadata it
    was handed and treats a cached answer as current.

    **What graceful behaviour looks like:** notice the timestamp, refresh, or caveat
    the answer as possibly out of date.

    Expected failure mode: `silent_wrong_answer`.
    """

    kind: ClassVar[str] = "StaleDataFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "post")})
    severity_hint: ClassVar[Severity] = "medium"

    def validate(self) -> None:
        """Check the call index and the age delta.

        Raises:
            ConfigError: With the offending value.
        """
        params = self.params()
        if int(params.get("serve_call_index", 1)) < 1:
            raise ConfigError(
                f"{self.kind}.serve_call_index is 1-based; got {params['serve_call_index']!r}"
            )
        if int(params.get("age_delta_s", 86400)) < 0:
            raise ConfigError(f"{self.kind}.age_delta_s must be >= 0")

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Replay an earlier result for this tool.

        Args:
            crossing: The `(tool, post)` crossing.
            ctx: The fault context, whose `pre_fault_history` holds what the tool
                really returned on each earlier call.

        Returns:
            A `replace_result` outcome, or `noop` when there is no earlier call to
            replay.
        """
        params = self.params()
        index = int(params.get("serve_call_index", 1))
        history = ctx.pre_fault_history.get(crossing.name, [])
        if len(history) < index:
            return FaultOutcome(
                action="noop",
                note=f"no call {index} of {crossing.name!r} to replay yet",
            )

        import copy as _copy

        try:
            stale = _copy.deepcopy(history[index - 1])
            copied = True
        except Exception:
            stale, copied = history[index - 1], False

        age_field = params.get("age_field")
        delta = int(params.get("age_delta_s", 86400))
        backdated = ""
        if age_field and isinstance(stale, dict) and isinstance(stale.get(age_field), (int, float)):
            stale[age_field] = stale[age_field] - delta
            backdated = f" and backdated {age_field!r} by {delta}s"

        log = MutationLog.of(crossing.result, stale)
        log.mutation_skipped_uncopyable = not copied
        return FaultOutcome(
            action="replace_result",
            value=stale,
            note=f"served the result of call {index} of {crossing.name!r}{backdated}",
            mutation=log,
            params={"served_call_index": index, "age_delta_s": delta},
        )


@register_fault
class NonDeterminismFault(Fault):
    """Return a different answer to the same question on every call.

    Different every call, identical every run: the variation is drawn from the
    seeded stream, so a suite stays reproducible while the agent sees instability.

    **What agent weakness it proves:** the agent assumes reads are idempotent;
    caching bugs; comparison or dedup logic that breaks when order changes.

    **What graceful behaviour looks like:** treat a read as potentially unstable —
    compare on content rather than order, and stop re-checking after a bounded
    number of attempts.

    Expected failure modes: `infinite_loop` when the agent keeps re-checking, and
    `silent_wrong_answer`.
    """

    kind: ClassVar[str] = "NonDeterminismFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "post")})
    severity_hint: ClassVar[Severity] = "medium"

    def validate(self) -> None:
        """Check the variant list and the fallback mutation.

        Raises:
            ConfigError: With the offending value.
        """
        params = self.params()
        variants = params.get("variants")
        if variants is not None and not isinstance(variants, (list, tuple)):
            raise ConfigError(f"{self.kind}.variants must be a list; got {variants!r}")
        mutation_type = str(params.get("mutation_type", "reorder_list"))
        if mutation_type not in MUTATIONS:
            raise ConfigError(f"{self.kind}.mutation_type {mutation_type!r} is not a mutation")

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Pick a variant, or re-mutate the real result.

        Args:
            crossing: The `(tool, post)` crossing.
            ctx: The fault context, supplying the per-call seeded draw.

        Returns:
            A `replace_result` outcome.
        """
        params = self.params()
        variants = params.get("variants")
        stream = ctx.rng(f"variant:{crossing.call_index}")

        if variants:
            chosen = list(variants)[stream.randrange(len(variants))]
            ctx.record("variant", chosen)
            log = MutationLog.of(crossing.result, chosen)
            return FaultOutcome(
                action="replace_result",
                value=chosen,
                note=f"returned a different value on call {crossing.call_index}",
                mutation=log,
            )

        mutation_type = str(params.get("mutation_type", "reorder_list"))
        after = MUTATIONS[mutation_type](crossing.result, stream)
        log = MutationLog.of(crossing.result, after)
        return FaultOutcome(
            action="replace_result",
            value=after,
            note=(
                f"applied {mutation_type} on call {crossing.call_index}, so the same "
                "input gave a different answer"
            ),
            mutation=log,
        )


@register_fault
class ArgumentTamperFault(Fault):
    """Mutate the arguments the agent sent, then let the real tool run.

    **This performs a real operation the agent never requested.** A sign-flipped
    amount or a dropped `where` clause reaches the actual tool, which is why the
    D-23 gate refuses it against a `side_effecting=True` tool without an explicit
    opt-in (`SAFETY.md` §1).

    It doubles as an agent-bug detector: if the agent already sent something
    invalid, that is recorded as `pre_existing_invalid_args` *before* the fault
    muddies the evidence.

    **What agent weakness it proves:** the agent trusts the round trip and never
    validates what it sends.

    **What graceful behaviour looks like:** validate tool inputs at the boundary and
    fail loudly on a mismatch rather than passing them along.

    Expected failure modes: `silent_wrong_answer`, `schema_violation_downstream`.
    """

    kind: ClassVar[str] = "ArgumentTamperFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "pre")})
    severity_hint: ClassVar[Severity] = "high"
    performs_real_action: ClassVar[bool] = True

    def validate(self) -> None:
        """Check the mutation name and the argument list.

        Raises:
            ConfigError: With the offending value.
        """
        params = self.params()
        mutation_type = str(params.get("mutation_type", "type_flip"))
        if mutation_type not in MUTATIONS:
            known = ", ".join(sorted(MUTATIONS))
            raise ConfigError(
                f"{self.kind}.mutation_type {mutation_type!r} is not a mutation; one of: {known}"
            )
        arg_names = params.get("arg_names")
        if arg_names is not None and not isinstance(arg_names, (list, tuple)):
            raise ConfigError(f"{self.kind}.arg_names must be a list; got {arg_names!r}")

    @staticmethod
    def _invalid_args(kwargs: dict[str, Any], arg_names: list[str] | None) -> list[str]:
        """Find arguments that were already unusable before the fault touched them.

        Args:
            kwargs: The keyword arguments the agent sent.
            arg_names: The arguments under consideration, or `None` for all.

        Returns:
            Sorted names whose value is `None` or an empty string.
        """
        candidates = arg_names if arg_names is not None else list(kwargs)
        return sorted(
            name
            for name in candidates
            if name in kwargs and (kwargs[name] is None or kwargs[name] == "")
        )

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Mutate the outbound keyword arguments.

        Args:
            crossing: The `(tool, pre)` crossing carrying `args` and `kwargs`.
            ctx: The fault context.

        Returns:
            A `replace_args` outcome whose value is the mutated `kwargs`, with the
            original recorded when `also_record_original` is set.
        """
        params = self.params()
        arg_names = list(params["arg_names"]) if params.get("arg_names") is not None else None
        mutation_type = str(params.get("mutation_type", "type_flip"))

        before = dict(crossing.kwargs)
        pre_existing = self._invalid_args(before, arg_names)

        kwargs: dict[str, Any] = {"count": int(params.get("count", 1)), "depth": 1}
        if arg_names is not None:
            kwargs["keys"] = arg_names
        after = MUTATIONS[mutation_type](before, ctx.rng(mutation_type), **kwargs)

        log = MutationLog.of(before, after)
        note = describe_mutation(mutation_type, before, after, log.json_patch)
        outcome_params: dict[str, Any] = {"target": "kwargs", "mutation_type": mutation_type}
        if pre_existing:
            outcome_params["pre_existing_invalid_args"] = pre_existing
            note = f"{note}; the agent had already sent invalid {pre_existing}"
        if params.get("also_record_original", True):
            outcome_params["original_kwargs"] = before
        return FaultOutcome(
            action="replace_args", value=after, note=note, mutation=log, params=outcome_params
        )


@register_fault
class DuplicateSideEffectFault(Fault):
    """Call the real tool more than once for a single request.

    **This performs a real operation the agent never requested.** Pointed at
    `charge_card` or `place_order` this is a double charge and a duplicate order,
    which is why the D-23 gate covers it.

    The repeat is performed by the **adapter**, not by `apply()`, via the
    `invoke_target` action (D-10): only the adapter knows whether to `await`. Every
    invocation the harness makes is recorded in
    `HarnessFacts.harness_invocation_seqs`, so the `duplicate_side_effect` probe can
    exclude the harness's own calls — without that, the probe would fire even
    against a perfectly idempotent agent.

    **What agent weakness it proves:** no idempotency key on a write tool.

    **What graceful behaviour looks like:** the tool layer is idempotent, or the
    agent detects the duplicate and reconciles.

    Expected failure mode: `duplicate_side_effect`.
    """

    kind: ClassVar[str] = "DuplicateSideEffectFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "pre")})
    severity_hint: ClassVar[Severity] = "critical"
    performs_real_action: ClassVar[bool] = True

    def validate(self) -> None:
        """Check `times` and `return_from`.

        Raises:
            ConfigError: With the offending value.
        """
        params = self.params()
        if int(params.get("times", 2)) < 2:
            raise ConfigError(
                f"{self.kind}.times must be >= 2 to be a duplicate; got {params['times']!r}"
            )
        return_from = str(params.get("return_from", "first"))
        if return_from not in {"first", "last"}:
            raise ConfigError(
                f"{self.kind}.return_from must be 'first' or 'last'; got {return_from!r}"
            )

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Ask the adapter to invoke the tool repeatedly.

        Args:
            crossing: The `(tool, pre)` crossing.
            ctx: The fault context.

        Returns:
            An `invoke_target` outcome carrying `times` and `return_from`.
        """
        params = self.params()
        times = int(params.get("times", 2))
        return_from = str(params.get("return_from", "first"))
        return FaultOutcome(
            action="invoke_target",
            note=f"called {crossing.name!r} {times} times for one request",
            params={"times": times, "return_from": return_from},
        )


@register_fault
class LoopTrapFault(Fault):
    """Return the identical payload forever, so nothing the agent does progresses.

    The pinned value is a deep copy, so an agent that mutates the payload in place
    cannot accidentally escape the trap.

    **What agent weakness it proves:** no cycle detection, no max-iteration
    fallback, no check that the state actually changed.

    **What graceful behaviour looks like:** notice the repeat within three
    iterations, break, and explain. The `loop_repeat_cycle` probe therefore fires at
    **four or more** identical signatures, and only when the agent did not break the
    cycle itself — three observations then an abort is correct behaviour and must
    not be reported as a failure.

    Expected failure modes: `infinite_loop`, `max_iterations_exhausted`. This fault
    is *expected* to exhaust `Limits.max_steps`; that abort is the signal, not an
    error, and it must never appear in `error`.
    """

    kind: ClassVar[str] = "LoopTrapFault"
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]] = frozenset({("tool", "post")})
    severity_hint: ClassVar[Severity] = "high"

    def __init__(self, **params: Any) -> None:
        """Initialise, and set up the per-instance pin.

        Args:
            **params: `pin_after_call`, `pin_value`, `max_repeats`.
        """
        self._pinned: Any = None
        self._has_pin = False
        self._repeats = 0
        super().__init__(**params)

    def validate(self) -> None:
        """Check the call index and the repeat cap.

        Raises:
            ConfigError: With the offending value.
        """
        params = self.params()
        if int(params.get("pin_after_call", 1)) < 1:
            raise ConfigError(f"{self.kind}.pin_after_call is 1-based; got {params!r}")
        max_repeats = params.get("max_repeats")
        if max_repeats is not None and int(max_repeats) < 1:
            raise ConfigError(f"{self.kind}.max_repeats must be >= 1; got {max_repeats!r}")

    def apply(self, crossing: Crossing, ctx: FaultContext) -> FaultOutcome:
        """Pin on the configured call, then serve the pinned value.

        Args:
            crossing: The `(tool, post)` crossing.
            ctx: The fault context.

        Returns:
            A `replace_result` outcome carrying a fresh copy of the pinned value, or
            `noop` once `max_repeats` is spent.
        """
        import copy as _copy

        params = self.params()
        pin_after = int(params.get("pin_after_call", 1))
        max_repeats = params.get("max_repeats")

        if not self._has_pin:
            if crossing.call_index < pin_after:
                return FaultOutcome(action="noop", note="not yet pinned")
            source = params.get("pin_value")
            source = crossing.result if source is None else source
            try:
                self._pinned = _copy.deepcopy(source)
            except Exception:
                self._pinned = source
            self._has_pin = True

        if max_repeats is not None and self._repeats >= int(max_repeats):
            return FaultOutcome(action="noop", note="max_repeats reached; released the trap")
        self._repeats += 1

        try:
            served = _copy.deepcopy(self._pinned)
            copied = True
        except Exception:
            served, copied = self._pinned, False

        log = MutationLog.of(crossing.result, served)
        log.mutation_skipped_uncopyable = not copied
        return FaultOutcome(
            action="replace_result",
            value=served,
            note=(
                f"pinned {crossing.name!r} to its call-{pin_after} result, so nothing "
                f"the agent does makes progress (repeat {self._repeats})"
            ),
            mutation=log,
            params={"repeat": self._repeats},
        )
