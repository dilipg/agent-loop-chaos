"""A tool that may raise, called with no error handling.

**Planted weakness:** no `try`/`except` anywhere, so a transient failure propagates
as an unhandled exception instead of being retried or surfaced.
"""

from __future__ import annotations

from typing import Any


def run(*, raises: BaseException | None = None, result: Any = "ok") -> Any:
    """Call the tool once and use whatever comes back.

    Args:
        raises: An exception the tool raises, if any.
        result: What it returns otherwise.

    Returns:
        The tool's result.

    Raises:
        BaseException: Whatever the tool raised, unhandled.
    """
    if raises is not None:
        raise raises
    return result
