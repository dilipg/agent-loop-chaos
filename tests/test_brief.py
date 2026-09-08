"""`BRIEF.md` is the page a colleague with repo access reads first.

A short document goes stale faster than a long one, because nobody re-reads it. These
tests hold it to the same standard as the README: every command it shows exists, every
key it names is real, and every file it links to is there.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BRIEF = REPO / "BRIEF.md"


@pytest.fixture(scope="module")
def brief() -> str:
    return BRIEF.read_text(encoding="utf-8")


class TestItIsAccurate:
    def test_every_command_it_shows_exists(self, brief: str) -> None:
        from agent_loop_chaos.cli import build_parser

        parser = build_parser()
        subcommands = set(parser._subparsers._group_actions[0].choices)  # type: ignore[union-attr,attr-defined]
        for shown in set(re.findall(r"\balc ([a-z-]+)", brief)):
            assert shown in subcommands, f"BRIEF.md shows `alc {shown}`, which does not exist"

    def test_every_flag_it_shows_exists(self, brief: str) -> None:
        from agent_loop_chaos.cli import build_parser

        choices = build_parser()._subparsers._group_actions[0].choices  # type: ignore[union-attr,attr-defined]
        known = {
            opt
            for sub in choices.values()
            for action in sub._actions
            for opt in action.option_strings
        }
        for flag in set(re.findall(r"(--[a-z][a-z-]+)", brief)):
            assert flag in known, f"BRIEF.md shows `{flag}`, which no subcommand takes"

    def test_every_seam_layer_it_shows_is_real(self, brief: str) -> None:
        from agent_loop_chaos.interceptors import seams

        block = brief.split("  seams:")[1].split("```")[0]
        shown = set(re.findall(r"^\s{4}([a-z]+):", block, re.M))
        assert shown, "the seams example lost its layers"
        assert shown <= set(seams._LAYERS) | {"graph"}, f"unknown seam layers: {shown}"

    def test_the_scenario_keys_it_shows_are_real(self, brief: str) -> None:
        from agent_loop_chaos import Scenario

        fields = {f.name for f in dataclasses.fields(Scenario)}
        for key in ("intercept", "seams"):
            assert f"  {key}:" in brief, f"the brief no longer shows `{key}:`"
            assert key in fields

    def test_the_install_line_pins_the_current_version(self, brief: str) -> None:
        from agent_loop_chaos import __version__

        pinned = re.search(r"@v(\d+\.\d+\.\d+)", brief)
        assert pinned and pinned.group(1) == __version__, (
            f"BRIEF.md installs v{pinned.group(1) if pinned else '?'} but the code is {__version__}"
        )

    def test_the_exit_codes_it_quotes_are_the_real_ones(self, brief: str) -> None:
        from agent_loop_chaos import cli

        assert cli.EXIT_OK == 0
        assert cli.EXIT_USAGE == 2
        assert cli.EXIT_INTERNAL == 3
        assert "`2` your configuration" in brief

    def test_the_warning_it_quotes_is_the_one_the_library_prints(self, brief: str) -> None:
        printed = "this scenario proves nothing"
        source = (REPO / "src" / "agent_loop_chaos" / "cli.py").read_text(encoding="utf-8")
        assert printed in source, "the warning moved; update the brief's quote with it"
        assert printed in brief


class TestItsLinksResolve:
    def test_every_file_it_links_to_exists(self, brief: str) -> None:
        for target in re.findall(r"\]\(([^)#]+\.md)[^)]*\)", brief):
            assert (REPO / target).is_file(), f"BRIEF.md links to {target}, which is missing"

    def test_every_readme_anchor_it_uses_exists(self, brief: str) -> None:
        """A link to a heading that was renamed is a dead end for the reader."""
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        headings = {
            re.sub(r"[^a-z0-9 -]", "", line.lstrip("# ").lower()).replace(" ", "-")
            for line in readme.splitlines()
            if line.startswith("#")
        }
        for anchor in re.findall(r"\]\(README\.md#([a-z0-9-]+)\)", brief):
            assert anchor in headings, f"README.md has no heading anchored at #{anchor}"


class TestItStaysBrief:
    def test_it_is_short_enough_to_read(self, brief: str) -> None:
        """The point of a brief is that someone finishes it. The README is the long one."""
        assert len(brief.splitlines()) < 200, "BRIEF.md is turning into a second README"


def test_the_readme_points_at_it() -> None:
    """A brief nobody is told about is a brief nobody reads."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "BRIEF.md" in readme


def test_it_warns_about_patching_the_definition_instead_of_the_call_site(brief: str) -> None:
    """The gotcha that cost a real run three silent scenarios.

    Seams named at the definition attached and never fired, because the calling module
    had imported the function by name. Without this note a reader repeats it.
    """
    assert "Name the call site, not the definition" in brief
