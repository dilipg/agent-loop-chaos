"""Shared test fixtures.

The default run makes no network calls, ever. That is enforced rather than trusted:
`no_network` replaces `socket.socket` for every test, so an accidental outbound call
fails loudly instead of silently passing on a machine that happens to be online
(`docs/07-TESTING.md` §1).
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make any socket creation raise, except in tests marked `live`.

    Args:
        request: The pytest request, used to honour the `live` marker.
        monkeypatch: Patching helper.

    Yields:
        None, for the duration of the test.
    """
    if request.node.get_closest_marker("live"):
        yield
        return

    real_socket = socket.socket
    blocked_families = {socket.AF_INET, socket.AF_INET6}

    def guarded(family: Any = socket.AF_INET, *args: Any, **kwargs: Any) -> Any:
        # Only IP families are blocked. asyncio's event loop builds an AF_UNIX
        # socketpair for its self-pipe, so refusing every family would break every
        # async test while proving nothing about network access.
        if family in blocked_families:
            raise RuntimeError(
                "network access is blocked in the default test run; mark the test "
                "`@pytest.mark.live` if it genuinely needs an endpoint"
            )
        return real_socket(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", guarded)
    try:
        yield
    finally:
        monkeypatch.setattr(socket, "socket", real_socket)


@pytest.fixture
def repo_root() -> Path:
    """The repository root.

    Returns:
        Path to the checkout root.
    """
    return REPO_ROOT
