"""Scenarios, suites and the matrix expansion.

YAML loading needs the `[yaml]` extra at runtime, but `pyyaml` is also in `[dev]`
because every gate command runs a `.yaml` suite (D-27). JSON suites work with no
extras at all.

Everything here arrives in M4/M7.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .assertions import Expect
from .context import Limits
from .enums import ExpectedBehavior
from .report import ChaosResult

__all__ = ["ChaosSuite", "Scenario", "SuiteResult", "load_suite"]

_LATER = "arrives in M4 (prompts/04-report-and-probes.md)"


class Scenario:
    """One experiment: an entrypoint, a fault list, and what was expected of it.

    v0.1 scenarios are single-invocation only (D-41).
    """

    def __init__(
        self,
        id: str,
        entrypoint: str | Callable[..., Any],
        *,
        inputs: Any = None,
        initial_state: Mapping[str, Any] | None = None,
        faults: Sequence[Mapping[str, Any]] | None = None,
        seed: int = 1337,
        expected_behavior: ExpectedBehavior = "graceful_degradation",
        must_not: Sequence[str] | None = None,
        expect: Expect | None = None,
        expected_errors: Sequence[str] | None = None,
        allow_side_effects: Sequence[str] | None = None,
        objective_state_key: str = "query",
        dry_run: bool = False,
        limits: Limits | None = None,
        judge: Mapping[str, Any] | None = None,
        tags: Mapping[str, str] | None = None,
        description: str | None = None,
        matrix: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        """Initialise a scenario. Fields mirror `scenario.schema.json`.

        Args:
            id: Scenario identifier, used in the run path and the report.
            entrypoint: ``"module:attr"`` or a callable.
            inputs: Payload for the agent.
            initial_state: Starting state.
            faults: Fault specs, each ``{"type": ..., "params": {...}}``.
            seed: Root seed for this scenario.
            expected_behavior: What good behaviour looks like here.
            must_not: Probe codes that always fail the run. Validated against the
                registered probe codes (D-33).
            expect: Declarative assertions.
            expected_errors: Exception names that count as an explicit error.
            allow_side_effects: Tools the D-23 gate may target.
            objective_state_key: State key holding the objective, for goal faults.
            dry_run: Register faults but fire none (D-12).
            limits: Run guard rails. Defaults to `Limits()`.
            judge: Judge configuration for this scenario.
            tags: Free-form labels.
            description: Human description.
            matrix: Cartesian expansion over fault params and seeds.

        Raises:
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"Scenario {_LATER}")

    def expand(self) -> list[Scenario]:
        """Apply `matrix`, producing one scenario per combination.

        Ids gain a canonical ``[k=v]`` suffix, ordered by the one matrix-id rule in
        D-15 so the same matrix always yields the same ids.

        Returns:
            The expanded scenarios, or ``[self]`` when there is no matrix.

        Raises:
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"Scenario.expand {_LATER}")


class ChaosSuite:
    """A collection of scenarios run together."""

    def __init__(self, scenarios: Sequence[Scenario], *, engine: Any | None = None) -> None:
        """Initialise a suite.

        Args:
            scenarios: The scenarios to run.
            engine: A configured `ChaosEngine`, or `None` for a default.

        Raises:
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"ChaosSuite {_LATER}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChaosSuite:
        """Load a suite from YAML or JSON.

        Args:
            path: Path to the suite file.

        Returns:
            The loaded suite.

        Raises:
            MissingExtraError: For a YAML file when `pyyaml` is absent.
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"ChaosSuite.from_yaml {_LATER}")

    def run(
        self,
        *,
        jobs: int = 1,
        fail_fast: bool = False,
        baseline: bool = True,
        filter: str | None = None,
    ) -> SuiteResult:
        """Run every scenario.

        Args:
            jobs: Parallel workers. Each run gets its own RNG registry, so streams
                are never shared (D-02).
            fail_fast: Stop at the first failing scenario.
            baseline: Run an unfaulted baseline per scenario first.
            filter: Glob over scenario ids.

        Returns:
            The aggregated `SuiteResult`.

        Raises:
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"ChaosSuite.run {_LATER}")


class SuiteResult:
    """The aggregate of one suite run, written to `<out_dir>/suite.json`.

    That file is what a CI job, an optimizer, or the dashboard polls.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the `suite.json` shape.

        Returns:
            The suite result as a plain dict.

        Raises:
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"SuiteResult.to_dict {_LATER}")

    def worst(self, n: int = 5) -> list[ChaosResult]:
        """Return the most severe results, worst first.

        Args:
            n: How many to return.

        Returns:
            Up to `n` results, ordered by severity then failure mode precedence.

        Raises:
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"SuiteResult.worst {_LATER}")


def load_suite(path: str | Path) -> ChaosSuite:
    """Load a suite from a file. Module-level convenience for `ChaosSuite.from_yaml`.

    Args:
        path: Path to a YAML or JSON suite.

    Returns:
        The loaded suite.

    Raises:
        NotImplementedError: Until M4.
    """
    raise NotImplementedError(f"load_suite {_LATER}")
