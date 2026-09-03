"""The refinement loop.

baseline -> chaos matrix -> probes -> judge -> `AGENT_TASK.md`, repeated. The loop
never edits source itself: it writes work orders and reports what flipped.

Arrives in M7 (`prompts/07-refinement-loop.md`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from .report import ChaosResult
from .scenarios import ChaosSuite

__all__ = ["LoopReport", "RefinementLoop"]

_M7 = "arrives in M7 (prompts/07-refinement-loop.md)"


class RefinementLoop:
    """Runs a suite repeatedly, handing findings to a caller between rounds."""

    def __init__(
        self,
        suite: ChaosSuite,
        *,
        engine: Any | None = None,
        max_rounds: int = 3,
        stop_when: Literal["all_pass", "no_new_failures", "rounds"] = "no_new_failures",
        on_findings: Callable[[Sequence[ChaosResult]], None] | None = None,
    ) -> None:
        """Initialise a loop.

        Args:
            suite: The suite to run each round.
            engine: A configured `ChaosEngine`, or `None` for a default.
            max_rounds: Hard cap on rounds.
            stop_when: Termination condition.
            on_findings: Hand-off hook. The caller may shell out to a coding agent,
                apply patches and return; the loop then re-runs and reports what
                flipped. A scenario whose `plan_hash` changed between rounds is
                reported as a regression, not a fix.

        Raises:
            NotImplementedError: Until M7.
        """
        raise NotImplementedError(f"RefinementLoop {_M7}")

    def run(self) -> LoopReport:
        """Run rounds until `stop_when` is satisfied or `max_rounds` is reached.

        Returns:
            The `LoopReport`.

        Raises:
            NotImplementedError: Until M7.
        """
        raise NotImplementedError(f"RefinementLoop.run {_M7}")


class LoopReport:
    """What changed across rounds: what was fixed, what regressed, what was written."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the loop report.

        Returns:
            The report as a plain dict.

        Raises:
            NotImplementedError: Until M7.
        """
        raise NotImplementedError(f"LoopReport.to_dict {_M7}")

    def markdown(self) -> str:
        """Render the loop report as markdown.

        Returns:
            The markdown body.

        Raises:
            NotImplementedError: Until M7.
        """
        raise NotImplementedError(f"LoopReport.markdown {_M7}")

    @property
    def tasks_written(self) -> list[Path]:
        """The `AGENT_TASK.md` paths written across every round.

        Returns:
            The paths, in write order.

        Raises:
            NotImplementedError: Until M7.
        """
        raise NotImplementedError(f"LoopReport.tasks_written {_M7}")
