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
from typing import Any, NoReturn

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

    def blocked(*args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError(
            "network access is blocked in the default test run; mark the test "
            "`@pytest.mark.live` if it genuinely needs an endpoint"
        )

    monkeypatch.setattr(socket, "socket", blocked)
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
