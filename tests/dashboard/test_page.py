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


class TestTheReportView:
    """The view for someone who did not write the agent.

    `docs/10` §6 gives the developer three panes of trace. That is the wrong first
    thing to show a person deciding whether to ship: `failure_mode:
    unverified_claim_emitted` is a stable identifier, not an explanation. The report
    view renders every code through the glossary the library ships.
    """

    def test_it_is_the_landing_view(self, page: str) -> None:
        assert 'setView("report")' in page
        assert 'id="report" class="on"' in page

    def test_both_views_are_reachable(self, page: str) -> None:
        assert 'id="viewreport"' in page and 'id="viewtrace"' in page

    def test_it_renders_codes_through_the_glossary(self, page: str) -> None:
        # A raw enum name must never reach this reader, so every code goes through
        # `say()`, which looks it up in the shipped glossary.
        assert "function say(" in page
        for table in ("failure_modes", "observed", "expected", "severities", "probes"):
            assert f'say("{table}"' in page

    def test_it_says_how_hard_the_run_pushed(self, page: str) -> None:
        assert "Pressure was set to" in page

    def test_it_offers_the_trace_as_the_next_step(self, page: str) -> None:
        assert "See what happened, step by step" in page

    def test_it_offers_the_work_order(self, page: str) -> None:
        assert "Copy the work order" in page

    def test_it_uses_plain_headings(self, page: str) -> None:
        for heading in ("What we broke", "What it did", "What to fix", "How we know"):
            assert heading in page

    def test_it_still_escapes_everything(self, page: str) -> None:
        # The report view renders fault notes and symptom details, which are
        # payload-derived. Same rule as the timeline: nothing is assigned as markup.
        assert "innerHTML" not in page


class TestTheListingHelp:
    def test_the_trace_view_explains_itself(self, page: str) -> None:
        assert "How to read this" in page

    def test_the_help_explains_the_colours(self, page: str) -> None:
        assert "we broke something here" in page
        assert "the agent called a tool" in page

    def test_the_help_names_the_keys(self, page: str) -> None:
        assert "jumps to the next break" in page
