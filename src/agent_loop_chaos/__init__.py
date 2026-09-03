"""agent-loop-chaos — chaos engineering for agent loops.

Breaks an agent on purpose and hands the bug report to a coding agent. The output is
designed for machines: a schema-valid `report.json`, a `trace.jsonl`, and on failure
an `AGENT_TASK.md` work order.

Importing this package pulls in nothing third-party except `jsonschema`. `langgraph`,
`httpx`, `pyyaml` and the model SDKs are optional extras, imported locally by the
adapter or judge that needs them.

Everything listed in `__all__` is public and matches `docs/02-API.md` §1 name for
name. Anything not listed is private and may change without a MAJOR bump.
"""

from __future__ import annotations

import logging

from .assertions import AssertionResult, Expect, HarnessFacts
from .context import Crossing, Limits
from .enums import ExpectedBehavior, FailureMode, Severity
from .errors import (
    AdapterError,
    ChaosError,
    ConfigError,
    ExplicitError,
    JudgeError,
    LimitExceeded,
    MissingExtraError,
    SchemaError,
)
from .faults import Fault
from .judges import EnsembleJudge, Judge, RuleJudge, SLMJudge, Verdict
from .loop import LoopReport, RefinementLoop
from .report import ChaosResult
from .scenarios import ChaosSuite, Scenario, SuiteResult, load_suite
from .targeting import Target, Trigger
from .version import __version__

# A library must not configure logging for its host application; it only ensures a
# handler exists so "No handlers could be found" never appears.
logging.getLogger("agent_loop_chaos").addHandler(logging.NullHandler())

# Order mirrors docs/02-API.md §1, which is normative and grouped by concern.
# Not sorted, deliberately: the doc is the source of truth for this list.
__all__ = [  # noqa: RUF022
    "__version__",
    # engine
    "ChaosEngine",
    "ChaosResult",
    "Limits",
    "Crossing",
    "HarnessFacts",
    # targeting
    "Target",
    "Trigger",
    # scenarios
    "Scenario",
    "ChaosSuite",
    "SuiteResult",
    "load_suite",
    # loop
    "RefinementLoop",
    "LoopReport",
    # judges
    "Judge",
    "Verdict",
    "RuleJudge",
    "SLMJudge",
    "EnsembleJudge",
    # faults (re-exported for convenience)
    "Fault",
    # assertions
    "Expect",
    "AssertionResult",
    # errors
    "ChaosError",
    "ConfigError",
    "MissingExtraError",
    "SchemaError",
    "JudgeError",
    "AdapterError",
    "LimitExceeded",
    "ExplicitError",
    # enums
    "Severity",
    "ExpectedBehavior",
    "FailureMode",
]


def __getattr__(name: str) -> object:
    """Import `ChaosEngine` lazily.

    `engine.py` imports `report`, `scenarios` and `assertions`; resolving it at
    package-import time would make the import graph cyclic. Deferring it keeps
    ``import agent_loop_chaos`` cheap and acyclic while leaving
    ``from agent_loop_chaos import ChaosEngine`` working normally.

    Args:
        name: The attribute being looked up.

    Returns:
        The resolved attribute.

    Raises:
        AttributeError: If `name` is not a public attribute of this package.
    """
    if name == "ChaosEngine":
        from .engine import ChaosEngine

        return ChaosEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
