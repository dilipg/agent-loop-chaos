"""Parses the model's response as JSON, unguarded.

**Planted weakness:** `json.loads(response)` with no try/except and no repair, so
prose or a markdown fence crashes it.
"""

from __future__ import annotations

import json
from typing import Any


def run(response: str) -> Any:
    """Parse a model response.

    Args:
        response: The raw response body.

    Returns:
        The parsed object.

    Raises:
        json.JSONDecodeError: Whenever the body is not valid JSON.
    """
    return json.loads(response)
