"""A shim entrypoint: build what a service needs, then call its real pipeline.

The pattern `alc doctor` recommends for a service whose own entrypoint takes live
handles. The graph is one import away, which is what makes this worth a test.
"""

from __future__ import annotations

from typing import Any

from tests.fakes.internal_graph import run_singleton


def arun(query: Any = None) -> Any:
    """Call the real pipeline with everything it needs.

    Args:
        query: Ignored; the fixture supplies its own input.

    Returns:
        Whatever the pipeline returned.
    """
    return run_singleton(query or "a realistic input")
