"""Retries a failing tool forever.

**Planted weakness:** no attempt cap and no backoff, so it hammers a limited
endpoint until an external budget stops it.
"""

from __future__ import annotations


def run(*, fail_times: int, budget: int) -> int:
    """Retry until the tool succeeds, or until the budget runs out.

    Args:
        fail_times: How many calls fail before one succeeds.
        budget: An external cap, standing in for the engine's limits.

    Returns:
        How many attempts were made.
    """
    attempts = 0
    while attempts < budget:
        attempts += 1
        if attempts > fail_times:
            return attempts
    return attempts
