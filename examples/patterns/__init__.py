"""A pool of agent implementations, one per shape people actually ship.

The library's claim is that it attaches to an agent loop without caring how that loop
is written. This package is the evidence: each module is a different *shape* --
a ReAct text loop, an OpenAI tool-calling loop, an async agent, a class with state on
`self`, a supervisor delegating to specialists, a fixed pipeline with no loop, a
streaming accumulator, a retrieve-rerank-generate chain -- and `tests/test_patterns.py`
runs every one of them through the same fault battery.

A pattern that cannot be instrumented, or whose faults never fire, is a gap in the
library, not a gap in the example. That is what makes this pool a conformance suite
rather than a gallery.

Every module ships the same three things:

- `build(engine)` -- the naive implementation, with one honest weakness a real
  codebase would have. Registered tools and the model go through `engine`.
- `build_fixed(engine)` -- the same agent, hardened. It is the negative control: a
  probe that fires on it has a false positive.
- a `PatternSpec` in `PATTERNS`, naming the weakness and the fault that catches it.

Both builders follow the D-61 convention: a callable whose first parameter is named
`engine` is a *builder*, handed the engine and returning the real agent. That is what
lets `entrypoint: examples.patterns.react_loop:build` work from a suite file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["PATTERNS", "PatternSpec", "register"]


@dataclass(frozen=True, slots=True)
class PatternSpec:
    """One agent shape, and what should break it.

    Attributes:
        name: Registry key, matching the module name.
        description: One line: what shape this is and where it is seen in the wild.
        build: Builder for the naive agent.
        build_fixed: Builder for the hardened twin.
        inputs: What to pass as the agent's payload.
        initial_state: Starting state, for shapes that carry one.
        tools: Tool names the engine will see, so a scenario can target them.
        llm_names: Model names the engine will see.
        is_async: Whether the entrypoint is a coroutine function. Async and sync
            parity is a hard requirement, so at least one pattern must set this.
        weakness: One line naming what is wrong with `build`.
        faults: Fault specs that bite `build` and must not bite `build_fixed`.
        expected_behavior: What good behaviour looks like under those faults.
    """

    name: str
    description: str
    build: Callable[[Any], Any]
    build_fixed: Callable[[Any], Any]
    weakness: str
    faults: tuple[Mapping[str, Any], ...]
    inputs: Any = None
    initial_state: Mapping[str, Any] | None = None
    tools: tuple[str, ...] = ()
    llm_names: tuple[str, ...] = ()
    is_async: bool = False
    expected_behavior: str = "graceful_degradation"


PATTERNS: dict[str, PatternSpec] = {}


def register(spec: PatternSpec) -> PatternSpec:
    """Add a pattern to the registry.

    Args:
        spec: The pattern.

    Returns:
        The same spec, so a module can assign it in one statement.

    Raises:
        ValueError: On a duplicate name, which would silently drop a pattern from
            the conformance run.
    """
    if spec.name in PATTERNS:
        raise ValueError(f"duplicate pattern {spec.name!r}")
    PATTERNS[spec.name] = spec
    return spec


def load_all() -> dict[str, PatternSpec]:
    """Import every pattern module so the registry is populated.

    Returns:
        The registry.
    """
    import importlib
    import pkgutil

    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"{__name__}.{info.name}")
    return PATTERNS
