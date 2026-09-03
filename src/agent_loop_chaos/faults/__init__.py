"""Fault classes.

The 26 concrete faults in `docs/03-FAULT-CATALOG.md` arrive in M2 (tool and loop),
M3 (LLM, prompt and injection) and M5 (graph state). M1 ships only the base
machinery plus `NoopFault`, the plumbing test double.

Per D-51 the family modules are created by the phase that fills them and re-export
through this package, so no user-facing import path changes as they land.

Every fault: operates on a deep copy and never mutates the caller's object; draws
all randomness from `ctx.rng(purpose)`; reads no clock; and states in its docstring
what agent weakness it proves and what graceful behaviour looks like.
"""

from __future__ import annotations

from .base import (
    FAULT_REGISTRY,
    TERMINAL_ACTIONS,
    VALUE_ACTIONS,
    Action,
    Fault,
    FaultInfo,
    FaultOutcome,
    FaultRecord,
    MutationLog,
    NoopFault,
    fault_from_dict,
    fault_key_for,
    list_faults,
    register_fault,
)

__all__ = [
    "FAULT_REGISTRY",
    "TERMINAL_ACTIONS",
    "VALUE_ACTIONS",
    "Action",
    "Fault",
    "FaultInfo",
    "FaultOutcome",
    "FaultRecord",
    "MutationLog",
    "NoopFault",
    "fault_from_dict",
    "fault_key_for",
    "list_faults",
    "register_fault",
]
