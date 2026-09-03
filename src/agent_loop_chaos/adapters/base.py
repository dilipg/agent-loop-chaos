"""The adapter protocol.

An adapter's only job is to turn its framework's hooks into `Crossing` objects, so
the core never learns which framework is in play. That is why one engine serves six
layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - types only
    from ..context import RunContext

__all__ = ["Adapter"]


@runtime_checkable
class Adapter(Protocol):
    """What every adapter implements.

    Attributes:
        name: Short framework name, as it appears in `target.framework`.
        version: The instrumented framework's version, recorded in
            `target.adapter_version` so a compat break is diagnosable from a report.
    """

    name: str
    version: str

    def instrument(self, target: Any, engine: Any, ctx: RunContext) -> Any:
        """Wrap the target's interception points.

        Args:
            target: The agent callable, object, or compiled graph.
            engine: The `ChaosEngine` routing crossings.
            ctx: The run context.

        Returns:
            The instrumented target.
        """
        ...

    def run(self, instrumented: Any, inputs: Any, ctx: RunContext) -> Any:
        """Invoke the instrumented target.

        Args:
            instrumented: What `instrument` returned.
            inputs: The payload for the agent.
            ctx: The run context.

        Returns:
            The agent's final output.
        """
        ...

    def state_of(self, instrumented: Any, ctx: RunContext) -> Any:
        """Read the target's visible state.

        Args:
            instrumented: What `instrument` returned.
            ctx: The run context.

        Returns:
            The state, or `None` when the framework exposes none.
        """
        ...
