"""Put a cassette under an intercepted call.

The chosen answer to "how does a colleague supply fake data" is that they do not write
any: the library records what their app really did, once, and every later run replays
that. This is the piece that makes the recording happen at the interception point, so
it covers whatever the strategies cover rather than needing a hook per integration.

Ordering is the whole design. The cassette replaces the **real call**; the engine's
wrapper sits above it. So a replayed payload is still faulted, and a recorded run is
testable rather than merely reproducible. The other way round -- faults inside, cassette
outside -- would record the faulted value and replay a fixed finding forever.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from ..cassettes import crossing_key

__all__ = ["taped"]


def taped(engine: Any, original: Any, layer: str, name: str) -> Any:
    """Wrap a callable so its result comes from the engine's cassette.

    Args:
        engine: The engine, which may carry a cassette.
        original: The real callable.
        layer: The crossing layer, part of the key.
        name: The seam's name, part of the key.

    Returns:
        `original` unchanged when there is no cassette -- no cassette must cost
        nothing -- else a wrapper that records or replays, matching coroutine-ness.
    """
    if getattr(engine, "cassette", None) is None:
        return original

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            cassette = engine.cassette
            key = crossing_key(layer, name, {"args": list(args), "kwargs": dict(kwargs)})
            return await cassette.aplay_key(key, live=lambda: original(*args, **kwargs))

        return awrapper

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cassette = engine.cassette
        key = crossing_key(layer, name, {"args": list(args), "kwargs": dict(kwargs)})
        return cassette.play_key(key, live=lambda: original(*args, **kwargs))

    return wrapper
