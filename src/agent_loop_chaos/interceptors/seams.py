"""Name a call by dotted path when nothing else can find it.

The `httpx` strategy reaches anything that speaks HTTP; the `langchain-core` strategy
reaches anything that inherits its base classes. Neither sees a hand-rolled client: an
in-house `LLMClient.chat`, a repository function reading Mongo through a driver that is
not HTTP, an SDK that ships its own transport. Plenty of real services are built exactly
that way, and "does your code follow a convention we recognise" is not a fair question
to make a colleague answer before they can test anything.

    seams:
      llm:   ["app.llm:LLMClient.chat"]
      tools: ["app.repositories:fetch_*"]

A spec is `module:attr`, where `attr` may name a function, a `Class.method`, or a glob
over a module's attributes. Each match is wrapped with `engine.llm` or `engine.tool` for
the duration of the run and restored afterwards, and the seam is named by its attribute
path -- so `tool: fetch_weather` and `llm: "LLMClient.chat"` target it as written.

Everything is validated at attach time. A typo in a dotted path must surface as a
`ConfigError` before the run, never as an agent failure mid-run.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from typing import Any

from ..context import Layer
from ..errors import ConfigError
from .base import Attachment
from .tape import taped

__all__ = ["SeamsStrategy"]

#: Suite key to crossing layer. `tools` is plural because a suite lists many.
_LAYERS: Mapping[str, Layer] = {"llm": "llm", "tool": "tool", "tools": "tool"}


def _split(spec: str) -> tuple[str, str]:
    """Split a seam spec into its module and attribute path.

    Args:
        spec: A ``module:attr`` string.

    Returns:
        ``(module, attr_path)``.

    Raises:
        ConfigError: When the spec has no colon, or either half is empty.
    """
    module, _, attr = str(spec).partition(":")
    if not module or not attr:
        raise ConfigError(
            f"seam {spec!r} must be 'module:attr' -- for example "
            "'app.llm:LLMClient.chat' or 'app.repositories:fetch_*'"
        )
    return module, attr


def _resolve(spec: str) -> list[tuple[Any, str, str]]:
    """Find every attribute a seam spec names.

    Args:
        spec: A ``module:attr`` string, where `attr` may be `Class.method` or a glob.

    Returns:
        One ``(owner, name, label)`` per match. `owner` is the module or class holding
        the attribute, and `label` is the attribute path, used to name the seam.

    Raises:
        ConfigError: When the module cannot be imported, or the attribute path matches
            nothing. Both are typos, and both must surface before the run.
    """
    module_name, attr_path = _split(spec)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"seam {spec!r}: cannot import {module_name!r} ({exc})") from exc

    *parents, leaf = attr_path.split(".")
    owner: Any = module
    walked: list[str] = []
    for part in parents:
        owner = getattr(owner, part, None)
        walked.append(part)
        if owner is None:
            raise ConfigError(f"seam {spec!r}: {module_name}:{'.'.join(walked)} does not exist")

    prefix = ".".join([*parents, ""]) if parents else ""
    if any(ch in leaf for ch in "*?["):
        matches = [
            name
            for name in dir(owner)
            if not name.startswith("_")
            and fnmatchcase(name, leaf)
            and callable(getattr(owner, name, None))
        ]
        if not matches:
            raise ConfigError(f"seam {spec!r}: nothing in {module_name} matches {leaf!r}")
        return [(owner, name, f"{prefix}{name}") for name in sorted(matches)]

    if not callable(getattr(owner, leaf, None)):
        raise ConfigError(f"seam {spec!r}: {attr_path} is not a callable attribute")
    return [(owner, leaf, attr_path)]


class SeamsStrategy:
    """Wrap the callables a suite names, for the duration of a run."""

    name = "seams"

    def __init__(self, seams: Mapping[str, Sequence[str]] | None = None) -> None:
        """Build the strategy.

        Args:
            seams: Layer name to dotted-path specs, as a suite declares them.
        """
        self._seams = {str(k): list(v) for k, v in (seams or {}).items()}
        self._originals: list[tuple[Any, str, Any]] = []

    def available(self) -> str | None:
        """Report whether any seam was declared.

        Returns:
            `None` when there is something to patch, else the reason there is not --
            which reads correctly in a diagnosis: nothing was named.
        """
        if not any(self._seams.values()):
            return "no seams declared"
        return None

    def attach(self, engine: Any, seen: dict[str, int]) -> list[Attachment]:
        """Wrap every named callable.

        Args:
            engine: The engine to route crossings through.
            seen: Shared call tally.

        Returns:
            One `Attachment` per patched callable.

        Raises:
            ConfigError: On an unknown layer or an unresolvable path. Raised before
                anything runs, so a typo never masquerades as an agent failure.
        """
        points: list[Attachment] = []
        for key, specs in self._seams.items():
            layer = _LAYERS.get(key)
            if layer is None:
                allowed = ", ".join(sorted(_LAYERS))
                raise ConfigError(f"seams: unknown layer {key!r}; expected one of: {allowed}")
            for spec in specs:
                for owner, attribute, label in _resolve(spec):
                    target = f"{_split(spec)[0]}:{label}"
                    self._originals.append((owner, attribute, getattr(owner, attribute)))
                    setattr(
                        owner,
                        attribute,
                        _wrap(engine, seen, getattr(owner, attribute), layer, label, target),
                    )
                    points.append(Attachment(layer=layer, strategy=self.name, target=target))
        return points

    def detach(self) -> None:
        """Restore every wrapped callable. Safe to call twice."""
        while self._originals:
            owner, attribute, original = self._originals.pop()
            setattr(owner, attribute, original)


def _wrap(
    engine: Any, seen: dict[str, int], original: Any, layer: Layer, label: str, target: str
) -> Any:
    """Wrap one named callable so its calls become crossings.

    The engine's own wrapper handles async parity, payload normalization and the
    crossing itself. This adds only the tally, and the fact that an unarmed engine
    passes the call straight through.

    Args:
        engine: The engine.
        seen: Shared call tally.
        original: The callable being replaced.
        layer: Which crossing layer it produces.
        label: The seam's name, as a target would write it.
        target: The tally key.

    Returns:
        The wrapper, a coroutine function when `original` is one.
    """
    import functools
    import inspect

    wrap = engine.llm if layer == "llm" else engine.tool
    # The cassette goes *under* the engine's wrapper, replacing the real call. Faults
    # then apply on top of whatever the tape supplied, which is what makes a recorded
    # run testable rather than merely reproducible.
    routed = wrap(taped(engine, original, layer, label), name=label)

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            if not engine.is_active():
                return await original(*args, **kwargs)
            seen[target] = seen.get(target, 0) + 1
            return await routed(*args, **kwargs)

        return awrapper

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not engine.is_active():
            return original(*args, **kwargs)
        seen[target] = seen.get(target, 0) + 1
        return routed(*args, **kwargs)

    return wrapper
