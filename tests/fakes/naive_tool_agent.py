"""One tool, one LLM call, and no shape check at all.

**Planted weakness:** indexes tool output directly as `data[0]["temp_c"]`, so an
empty list or a dropped key becomes a `KeyError`/`IndexError` rather than a
degradation.
"""

from __future__ import annotations

from typing import Any


def run(result: dict[str, Any]) -> str:
    """Summarise a weather payload without validating it.

    Args:
        result: The tool result.

    Returns:
        A one-line summary.

    Raises:
        KeyError: When the payload lost the field.
        IndexError: When the payload lost the record.
        TypeError: When the payload changed shape.
    """
    return f"It is {result['data'][0]['temp_c']}C."
