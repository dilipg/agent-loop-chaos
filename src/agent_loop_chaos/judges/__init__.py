"""Judges.

Rules decide; models explain. A judge is **advisory only**: it may author
`failure_mode`, `root_cause_hypothesis`, `refinement_hint`, narration and ranked
fixes, and it never touches `success`. When a judge contradicts the probes, the
probes win and the disagreement is recorded in `verdict.judge_disagreement`
(`docs/11-OUTCOMES-AND-ASSERTIONS.md` §1).

Arrives in M6 (`prompts/06-judge-slm.md`). Per D-51 the protocol and the three judge
classes are declared here in M0; `base.py`, `rules.py`, `slm.py` and `ensemble.py`
are created by M6 and re-export through this package.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = ["EnsembleJudge", "Judge", "RuleJudge", "SLMJudge", "Verdict"]

_M6 = "arrives in M6 (prompts/06-judge-slm.md)"


@runtime_checkable
class Judge(Protocol):
    """What every judge implements."""

    name: str

    def judge(self, ev: Any) -> Verdict:
        """Turn probe output and trace evidence into a verdict.

        Args:
            ev: The `JudgeEvidence` assembled by the engine. Untrusted spans inside
                it are fenced before they reach a model (D-21).

        Returns:
            The `Verdict`. `passed` is supplied by the engine, never by a model.
        """
        ...


class Verdict:
    """A classified outcome plus its narration.

    `passed` is authoritative but is computed by the probes and the scenario's
    `expected_behavior` — a judge only ever carries it, never decides it.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the `judge_verdict.schema.json` shape.

        Returns:
            The verdict as a plain dict.

        Raises:
            NotImplementedError: Until M6.
        """
        raise NotImplementedError(f"Verdict.to_dict {_M6}")


class RuleJudge:
    """Deterministic judge. No model, no network, no clock.

    This is the default and it is fully supported on its own: `--judge rules` works
    offline, and a socket-blocking test fixture proves it makes no network call.
    """

    def __init__(self, *, templates: Mapping[str, str] | None = None) -> None:
        """Initialise.

        Args:
            templates: Override the narration templates.

        Raises:
            NotImplementedError: Until M6.
        """
        raise NotImplementedError(f"RuleJudge {_M6}")


class SLMJudge:
    """Small-language-model judge, over an OpenAI-compatible or Ollama transport."""

    def __init__(
        self,
        *,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        transport: Literal["openai", "ollama", "anthropic"] = "openai",
        temperature: float = 0.0,
        max_tokens: int = 900,
        timeout_s: float = 60.0,
        retries: int = 2,
        prompt_dir: str | Path | None = None,
        offline_fallback: bool = True,
    ) -> None:
        """Initialise.

        Args:
            model: Model identifier.
            base_url: Endpoint. Normalized per transport (D-34). A non-loopback
                endpoint additionally requires `allow_remote_judge=True` before it
                may receive code context (D-22).
            api_key: Credential, if the endpoint needs one.
            transport: Wire protocol.
            temperature: Sampling temperature. Defaults to 0 for reproducibility.
            max_tokens: Response cap.
            timeout_s: Per-request timeout.
            retries: Retry count on a transport error.
            prompt_dir: Override the packaged prompt assets.
            offline_fallback: Fall back to `RuleJudge` instead of raising when the
                endpoint is unreachable.

        Raises:
            MissingExtraError: When `httpx` is absent.
            NotImplementedError: Until M6.
        """
        raise NotImplementedError(f"SLMJudge {_M6}")


class EnsembleJudge:
    """Rules decide, the model explains.

    The default when ``judge="ensemble"``, or when `judge` is `None` and a model
    endpoint is reachable; otherwise `RuleJudge`. Reachability is probed once per
    process with a short timeout and cached.
    """

    def __init__(self, *, rules: RuleJudge | None = None, model_judge: Any | None = None) -> None:
        """Initialise.

        Args:
            rules: The authoritative rule judge. Defaults to `RuleJudge()`.
            model_judge: The narrating judge. Defaults to `SLMJudge()`.

        Raises:
            NotImplementedError: Until M6.
        """
        raise NotImplementedError(f"EnsembleJudge {_M6}")
