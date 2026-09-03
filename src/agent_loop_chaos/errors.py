"""Exception hierarchy.

One base, `ChaosError`, for everything the library raises deliberately. Two
exceptions sit outside it on purpose: `LimitExceeded`, which must not be catchable
by the agent under test, and `ExplicitError`, which the agent raises to prove it
detected a problem.
"""

from __future__ import annotations

__all__ = [
    "AdapterError",
    "ChaosError",
    "ConfigError",
    "ExplicitError",
    "JudgeError",
    "LimitExceeded",
    "MissingExtraError",
    "SchemaError",
]


class ChaosError(Exception):
    """Base class for every error the library raises deliberately."""


class ConfigError(ChaosError):
    """A scenario, fault, target or trigger is invalid.

    Raised eagerly — at `register_fault` or suite-load time — never mid-run, so a
    misconfiguration cannot masquerade as an agent failure.
    """


class MissingExtraError(ChaosError):
    """An optional dependency is needed but not installed.

    The message always names the extra to install, e.g.
    ``install agent-loop-chaos[langgraph]``.
    """


class SchemaError(ChaosError):
    """A payload failed JSON Schema validation, or a schema could not be resolved."""


class JudgeError(ChaosError):
    """A judge could not produce a verdict.

    Never fatal to a run: the engine records it and falls back to `RuleJudge`.
    """


class AdapterError(ChaosError):
    """A framework could not be instrumented.

    The message names the installed framework version, because the usual cause is
    an attribute path that moved between releases.
    """


class LimitExceeded(BaseException):
    """A run limit was hit at a crossing; the run must stop now.

    Derives from `BaseException`, not `Exception`, so a broad ``except Exception``
    in the agent under test cannot swallow it — which would otherwise leave the run
    unbounded and make `LoopTrapFault`'s expected ``limit_hit="max_steps"``
    unreachable (D-06).

    Caught only by the engine's outermost frame, converted to `loop.limit_hit`, and
    never surfaced in the report's `error` field.
    """

    def __init__(self, limit: str, message: str | None = None) -> None:
        """Initialise.

        Args:
            limit: Which limit was hit, e.g. ``"max_steps"`` or ``"timeout_s"``.
                Lands in `loop.limit_hit` verbatim.
            message: Optional human detail. Defaults to a message naming the limit.
        """
        self.limit = limit
        super().__init__(message or f"limit exceeded: {limit}")


class ExplicitError(Exception):
    """Raised by the agent under test to signal it *detected* a problem.

    This is positive evidence, not a failure: an agent that raises `ExplicitError`
    when a tool returns corrupt data has behaved correctly. The engine classifies it
    as `explicit_error_surfaced` rather than `crash_unhandled_exception`
    (`docs/11-OUTCOMES-AND-ASSERTIONS.md` §3.1).
    """
