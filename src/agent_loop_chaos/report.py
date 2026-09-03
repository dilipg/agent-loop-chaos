"""`ChaosResult` — the report object.

The schema is the product; this class exists to fill
`schemas/chaos_report.schema.json`. The dataclass and the schema must not drift, and
M4 adds a test asserting every dataclass field appears in the schema and vice versa.

`ChaosResult` and its nested value objects arrive in M4
(`prompts/04-report-and-probes.md`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["ChaosResult"]

_M4 = "arrives in M4 (prompts/04-report-and-probes.md)"


class ChaosResult:
    """One run's full finding, schema-valid on serialization.

    `success` is computed — from the probes, the assertions layer and the scenario's
    `expected_behavior`, exactly as `docs/11-OUTCOMES-AND-ASSERTIONS.md` specifies.
    A language model never decides it. The judge may contribute `failure_mode`,
    `root_cause_hypothesis`, `refinement_hint` and narration only, and when the
    judge and the probes disagree the probes win and the disagreement is recorded in
    `verdict.judge_disagreement`.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the `chaos_report.schema.json` shape.

        Floats are rounded to 6 decimals and non-finite floats are encoded as
        ``{"__float__": "NaN" | "Infinity" | "-Infinity"}``, so a report is always
        valid JSON for a non-Python consumer (D-09).

        Returns:
            The report as a plain dict with sorted keys.

        Raises:
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"ChaosResult.to_dict {_M4}")

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON.

        Args:
            indent: Indentation for the on-disk form. `None` gives the compact
                canonical form used for hashing.

        Returns:
            The report as a JSON string, always with `allow_nan=False`.

        Raises:
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"ChaosResult.to_json {_M4}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ChaosResult:
        """Rebuild a result from its serialized form.

        Args:
            d: A dict matching `chaos_report.schema.json`.

        Returns:
            The reconstructed `ChaosResult`.

        Raises:
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"ChaosResult.from_dict {_M4}")

    def validate(self) -> list[str]:
        """Validate against the packaged report schema.

        Returns:
            Human-readable error strings, each carrying a JSON pointer and the
            failing value truncated to 200 chars. Empty when valid.

        Raises:
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"ChaosResult.validate {_M4}")

    def agent_task_markdown(self) -> str:
        """Render the `AGENT_TASK.md` work order.

        This artifact is the product: if a coding agent would have to ask a question
        before starting work, it is not good enough.

        Returns:
            The markdown body.

        Raises:
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"ChaosResult.agent_task_markdown {_M4}")

    def summary_line(self) -> str:
        """Render the one-line CLI summary.

        Returns:
            A single line naming the scenario, outcome and failure mode.

        Raises:
            NotImplementedError: Until M4.
        """
        raise NotImplementedError(f"ChaosResult.summary_line {_M4}")
