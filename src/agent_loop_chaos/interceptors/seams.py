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


def _split(spec: str) -> tuple[str | None, str, str]:
    """Split a seam spec into its optional alias, module and attribute path.

    An alias exists because a suite naturally targets `llm: summarizer`, not
    `llm: "StubModel.ainvoke"`. Without one, a scenario written the obvious way matches
    nothing and passes having tested nothing.

    Args:
        spec: ``module:attr``, or ``alias=module:attr``.

    Returns:
        ``(alias, module, attr_path)``, with `alias` `None` when none was given.

    Raises:
        ConfigError: When the spec has no colon, or any part is empty.
    """
    text = str(spec)
    alias: str | None = None
    if "=" in text and text.index("=") < (text.index(":") if ":" in text else len(text)):
        alias, _, text = text.partition("=")
        alias = alias.strip()
    module, _, attr = text.partition(":")
    if not module or not attr or (alias is not None and not alias):
        raise ConfigError(
            f"seam {spec!r} must be 'module:attr', optionally 'name=module:attr' -- for "
            "example 'summarizer=app.llm:LLMClient.chat' or 'app.repositories:fetch_*'"
        )
    return alias, module, attr


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
    alias, module_name, attr_path = _split(spec)
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
    is_glob = any(ch in leaf for ch in "*?[")
    if alias and is_glob:
        raise ConfigError(
            f"seam {spec!r}: a name cannot stand for a glob -- {leaf!r} may match several "
            "callables, and naming only the first would be worse than refusing"
        )
    if is_glob:
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
    return [(owner, leaf, alias or attr_path)]


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
                    target = spec if "=" in spec else f"{_split(spec)[1]}:{label}"
                    self._originals.append((owner, attribute, getattr(owner, attribute)))
                    setattr(
                        owner,
                        attribute,
                        _wrap(
                            engine,
                            seen,
                            getattr(owner, attribute),
                            layer,
                            label,
                            target,
                            is_method=isinstance(owner, type),
                        ),
                    )
                    points.append(Attachment(layer=layer, strategy=self.name, target=target))
        return points

    def detach(self) -> None:
        """Restore every wrapped callable. Safe to call twice."""
        while self._originals:
            owner, attribute, original = self._originals.pop()
            setattr(owner, attribute, original)


def _wrap(
    engine: Any,
    seen: dict[str, int],
    original: Any,
    layer: Layer,
    label: str,
    target: str,
    *,
    is_method: bool,
) -> Any:
    """Wrap one named callable so its calls become crossings.

    The engine's own wrapper handles async parity, payload normalization and the
    crossing itself. This adds the tally, the pass-through when the engine is unarmed,
    and one thing that is easy to get wrong.

    **A method is bound before the engine sees it.** Patching a class attribute means
    the wrapper is called as ``(self, messages)``, so the engine would take `self` for
    the payload and disable every prompt-side fault with an unnormalizable payload.
    Binding first makes the payload the argument the caller actually passed. The engine
    wrapper is then built per call, which is safe: `call_index` lives on the run context
    keyed by ``layer:name``, not on the wrapper.

    Args:
        engine: The engine.
        seen: Shared call tally.
        original: The callable being replaced.
        layer: Which crossing layer it produces.
        label: The seam's name, as a target would write it.
        target: The tally key.
        is_method: Whether `original` was found on a class rather than a module.

    Returns:
        The wrapper, a coroutine function when `original` is one.
    """
    import functools
    import inspect

    wrap = engine.llm if layer == "llm" else engine.tool

    def route(bound: Any) -> Any:
        return wrap(taped(engine, bound, layer, label), name=label)

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            if not engine.is_active():
                return await original(*args, **kwargs)
            seen[target] = seen.get(target, 0) + 1
            if is_method and args:
                instance, rest = args[0], args[1:]

                async def bound(*inner: Any, **inner_kw: Any) -> Any:
                    return await original(instance, *inner, **inner_kw)

                return await route(bound)(*rest, **kwargs)
            return await route(original)(*args, **kwargs)

        return awrapper

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not engine.is_active():
            return original(*args, **kwargs)
        seen[target] = seen.get(target, 0) + 1
        if is_method and args:
            instance, rest = args[0], args[1:]
            return route(functools.partial(original, instance))(*rest, **kwargs)
        return route(original)(*args, **kwargs)

    return wrapper
