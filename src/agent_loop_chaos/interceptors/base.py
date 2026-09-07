"""The attach registry: patch what is installed, restore it, and say what happened.

Onboarding a real service is where this module comes from. A ten-node graph reached
its model through a constructor argument and its data through a repository call, so
`engine.llm` and `engine.tool` had nothing to wrap: every payload fault armed and never
fired, and the run was green. Two hundred lines of hand-written harness fixed it, which
is two hundred lines too many.

A strategy patches somewhere a call must pass through -- `httpx.Client.send`,
`BaseChatModel.agenerate` -- and turns what it sees into a crossing. The registry picks
the strategies that can work here, patches on the way in, restores on the way out, and
reports what attached.

**Silence is loud.** An `AttachReport` distinguishes three states per strategy: could not
help and why, patched but saw no traffic, and patched and used. `require()` turns the
first two into a `ConfigError` naming all of them, because a scenario whose fault had
nothing to attach to proves nothing (D-64) and must not read as a pass.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..context import Layer
from ..errors import ConfigError

__all__ = ["AttachReport", "Attachment", "Registry", "Strategy"]

logger = logging.getLogger("agent_loop_chaos")


@dataclass(frozen=True, slots=True)
class Attachment:
    """One patched interception point.

    Attributes:
        layer: The crossing layer this point produces.
        strategy: The strategy that patched it.
        target: What was patched, for the diagnosis, e.g. `httpx.Client.send`.
    """

    layer: Layer
    strategy: str
    target: str


@runtime_checkable
class Strategy(Protocol):
    """An attach strategy. Stateful: it holds what it needs to restore."""

    name: str

    def available(self) -> str | None:
        """Report usability.

        Returns:
            `None` when the strategy can attach here, else the reason it cannot,
            phrased for a diagnosis (``"not installed"``).
        """
        ...

    def attach(self, engine: Any, seen: dict[str, int]) -> list[Attachment]:
        """Patch the interception points.

        Args:
            engine: The engine to route crossings through.
            seen: Shared call tally, keyed by `Attachment.target`. A strategy bumps
                its own key on every call it intercepts.

        Returns:
            One `Attachment` per patched point.
        """
        ...

    def detach(self) -> None:
        """Restore everything this strategy patched. Must tolerate being called twice."""
        ...


@dataclass(frozen=True, slots=True)
class AttachReport:
    """What attached, what could not, and what each hook saw.

    Attributes:
        attached: Every patched point.
        unavailable: Per strategy name, why it could not attach.
        seen: Call tally per `Attachment.target`, live until `detach()`.
    """

    attached: Sequence[Attachment] = ()
    unavailable: dict[str, str] = field(default_factory=dict)
    seen: dict[str, int] = field(default_factory=dict)

    def calls(self, target: str) -> int:
        """Report how many calls a patched point intercepted.

        Args:
            target: The `Attachment.target` to look up.

        Returns:
            The call count, zero when the point was never reached.
        """
        return self.seen.get(target, 0)

    def layers(self) -> set[Layer]:
        """Report which crossing layers are attached.

        Returns:
            The set of layers, empty when nothing attached.
        """
        return {a.layer for a in self.attached}

    def used_layers(self) -> set[Layer]:
        """Report which layers actually saw traffic.

        A layer that attached and saw nothing cannot carry a fault, so the two sets
        are not interchangeable.

        Returns:
            The set of layers with at least one intercepted call.
        """
        return {a.layer for a in self.attached if self.calls(a.target)}

    def diagnose(self, layer: Layer) -> str:
        """Explain why a layer cannot carry a fault.

        Args:
            layer: The layer the scenario needs.

        Returns:
            A multi-line message naming every strategy tried and its outcome.
        """
        lines = [f"no {layer} layer attached; this scenario cannot prove anything."]
        width = max(
            [len(name) for name in self.unavailable]
            + [len(a.strategy) for a in self.attached]
            + [1]
        )
        for name, reason in sorted(self.unavailable.items()):
            lines.append(f"  {name:<{width}}  {reason}")
        for attachment in self.attached:
            calls = self.calls(attachment.target)
            detail = (
                f"patched {attachment.target}, saw {calls} calls"
                if attachment.layer == layer
                else f"attached the {attachment.layer} layer, not {layer}"
            )
            lines.append(f"  {attachment.strategy:<{width}}  {detail}")
        lines.append(
            f'hint: name the seam explicitly -- seams: {{{layer}: ["your.module:YourClient.call"]}}'
        )
        return "\n".join(lines)

    def require(self, layer: Layer) -> None:
        """Refuse to run when a needed layer cannot carry a fault.

        Args:
            layer: The layer the scenario needs.

        Raises:
            ConfigError: When no strategy attached that layer, or none saw traffic.
        """
        if layer in self.used_layers():
            return
        raise ConfigError(self.diagnose(layer))


class Registry:
    """Holds the strategies for one run and guarantees they are restored."""

    def __init__(self, strategies: Iterable[Strategy] | None = None) -> None:
        """Build a registry.

        Args:
            strategies: The strategies to try, in order. Defaults to every built-in
                one, which is what a run wants.
        """
        self._strategies = list(strategies) if strategies is not None else default_strategies()
        self._active: list[Strategy] = []

    def attach(self, engine: Any) -> AttachReport:
        """Patch every strategy that can work here.

        A strategy that raises while attaching is reported as unavailable rather than
        allowed to break the run: the library must never crash the run it observes.

        Args:
            engine: The engine to route crossings through.

        Returns:
            The report. Its `seen` tallies stay live until `detach()`.
        """
        attached: list[Attachment] = []
        unavailable: dict[str, str] = {}
        seen: dict[str, int] = {}
        for strategy in self._strategies:
            reason = strategy.available()
            if reason is not None:
                unavailable[strategy.name] = reason
                continue
            try:
                points = strategy.attach(engine, seen)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("interceptor %s failed to attach: %s", strategy.name, exc)
                unavailable[strategy.name] = f"failed to attach: {exc}"
                continue
            self._active.append(strategy)
            attached.extend(points)
        return AttachReport(attached=attached, unavailable=unavailable, seen=seen)

    def detach(self) -> None:
        """Restore every patch, whatever else goes wrong.

        One strategy raising must not leave the others patched, so each is restored in
        its own `try`. Safe to call twice.
        """
        while self._active:
            strategy = self._active.pop()
            try:
                strategy.detach()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("interceptor %s failed to detach: %s", strategy.name, exc)

    def __enter__(self) -> AttachReport:
        """Not used by the engine, which needs the report before the body runs."""
        raise NotImplementedError("use attach()/detach(); the engine owns the ordering")

    def __exit__(self, *_: object) -> None:  # pragma: no cover - see __enter__
        self.detach()


def default_strategies() -> list[Strategy]:
    """Build the built-in strategies, in attach order.

    Returns:
        The strategies. Each reports its own availability, so an absent optional
        dependency costs nothing here.
    """
    from .http import HttpxStrategy
    from .langchain import LangChainStrategy

    return [HttpxStrategy(), LangChainStrategy()]
