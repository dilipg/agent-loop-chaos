"""Fault classes.

The 26 concrete faults in `docs/03-FAULT-CATALOG.md` arrive in M2 (tool and loop),
M3 (LLM, prompt and injection) and M5 (graph state). Per D-51, only the `Fault` base
lives here in M0; the family modules (`tool.py`, `llm.py`, `state.py`, `loop.py`,
`injection.py`, `registry.py`) are created by the phase that fills them, and
re-export through this package so no user-facing import path changes.

Every fault: operates on a deep copy and never mutates the caller's object; draws
all randomness from `ctx.rng(purpose)`; and states in its docstring what agent
weakness it proves and what graceful behaviour looks like.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..context import Crossing, Layer, Phase
from ..enums import Severity

__all__ = ["Fault"]


class Fault(ABC):
    """Base class for every fault.

    Attributes:
        kind: Stable name used in reports and in YAML scenario files. Renaming one
            breaks stored reports, so it is part of the public contract.
        accepts: The ``(layer, phase)`` pairs this fault handles. The engine refuses
            a mismatched `register_fault` with `ConfigError` at registration time.
        severity_hint: Default severity if this fault causes a failure.
    """

    kind: ClassVar[str]
    accepts: ClassVar[frozenset[tuple[Layer, Phase]]]
    severity_hint: ClassVar[Severity] = "medium"

    def __init__(self, **params: Any) -> None:
        """Initialise and validate parameters eagerly.

        Args:
            **params: Fault-specific parameters. See
                `docs/03-FAULT-CATALOG.md` for each fault's set.

        Raises:
            ConfigError: On an invalid or unknown parameter.
            NotImplementedError: Until M2.
        """
        raise NotImplementedError("Fault arrives in M2 (prompts/02-tool-faults.md)")

    def params(self) -> dict[str, Any]:
        """Return the parameters as serialized into the report and the plan hash.

        Returns:
            The parameter mapping, canonically ordered.

        Raises:
            NotImplementedError: Until M2.
        """
        raise NotImplementedError("Fault.params arrives in M2 (prompts/02-tool-faults.md)")

    @abstractmethod
    def apply(self, crossing: Crossing, ctx: Any) -> Any:
        """Apply the fault to a deep copy of the crossing's payload.

        Args:
            crossing: The crossing being faulted.
            ctx: The `FaultContext` (D-01), carrying the seeded RNG, counters,
                history, the tool registry and the state view.

        Returns:
            A `FaultOutcome` describing what the engine should do — replace a
            result, replace args, raise, delay, replace state or messages, or noop —
            together with the `MutationLog` recording
            `payload_before`/`payload_after`/`json_patch`.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError
