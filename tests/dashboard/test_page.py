"""The single page, checked as text.

There is no browser in the default test run, so this asserts the properties that can
be established by reading the file: it is self-contained, it never hands
payload-derived text to `innerHTML`, and the escaping helpers a hostile payload has
to get past are actually present (D-38).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGE = Path("src/agent_loop_chaos/dashboard/static/index.html")


@pytest.fixture
def page(repo_root: Path) -> str:
    return (repo_root / PAGE).read_text(encoding="utf-8")


def test_it_ships_with_the_package(repo_root: Path) -> None:
    from agent_loop_chaos.dashboard import server

    assert server._STATIC.is_file()


def test_it_references_no_external_asset(page: str) -> None:
    assert "http" + "://" not in page
    assert "http" + "s://" not in page
    assert "//cdn" not in page


def test_it_never_uses_innerhtml(page: str) -> None:
    # The trace is guaranteed to contain attacker-controlled markup, so there is no
    # "safe" call site for innerHTML on this page and the rule is absolute.
    assert "innerHTML" not in page
    assert "outerHTML" not in page
    assert "insertAdjacentHTML" not in page
    assert "document.write" not in page


def test_it_renders_invisible_characters_visibly(page: str) -> None:
    # A bidi override reorders what a human reads without changing the string, which
    # is exactly how a prompt-injection payload hides in plain sight.
    assert "u202e" in page.lower()
    assert "u200b" in page.lower()
    assert "RLO" in page and "ZWSP" in page


def test_it_paints_both_themes(page: str) -> None:
    assert "prefers-color-scheme: dark" in page
    assert re.search(r"body\s*\{[^}]*background:\s*var\(--bg\)", page)


def test_it_binds_the_documented_keys(page: str) -> None:
    for key in ('"j"', '"k"', '"f"', '"/"', '"g"', '"G"', '"?"'):
        assert key in page


def test_it_names_the_named_sse_frames(page: str) -> None:
    # The server sends `event: trace`, and a named frame never reaches onmessage.
    assert "addEventListener(kind, onFrame)" in page
    for kind in ("trace", "run_started", "run_finished", "suite", "reset", "desync"):
        assert f'"{kind}"' in page


def test_it_has_an_empty_state_that_says_what_to_do(page: str) -> None:
    assert "no runs yet" in page and "alc run" in page


def test_the_package_exports_its_public_names() -> None:
    from agent_loop_chaos import dashboard

    assert dashboard.RunDirWatcher.__name__ == "RunDirWatcher"
    assert dashboard.DashboardServer.__name__ == "DashboardServer"
    assert callable(dashboard.export_html)
    with pytest.raises(AttributeError):
        dashboard.not_a_thing  # noqa: B018  attribute access is the assertion
