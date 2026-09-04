"""The mustache-lite renderer (`docs/05` §5).

Ten lines of substitution, deliberately not Jinja: the prompt assets embed JSON
schema examples in double braces, and a real template engine would try to evaluate
them.
"""

from __future__ import annotations

from agent_loop_chaos.judges.base import render


def test_replaces_named_placeholders() -> None:
    assert render("seed: {{seed}}", {"seed": 7}) == "seed: 7"


def test_missing_key_becomes_empty_string() -> None:
    assert render("a{{nope}}b", {}) == "ab"


def test_leaves_single_braces_alone() -> None:
    assert render('{"a": 1} {{x}}', {"x": "v"}) == '{"a": 1} v'


def test_json_example_blocks_survive_untouched() -> None:
    # A `{{"kind": …}}` inside a prompt asset is a JSON example, not a placeholder.
    template = 'return {"kind": "other", "confidence": 0.0}'
    assert render(template, {"kind": "SHOULD NOT APPEAR"}) == template


def test_substitution_is_not_recursive() -> None:
    # A value that itself looks like a placeholder must not be expanded again;
    # untrusted evidence reaches this renderer.
    assert render("{{a}}", {"a": "{{b}}", "b": "boom"}) == "{{b}}"


def test_repeated_placeholder_filled_every_time() -> None:
    assert render("{{x}}-{{x}}", {"x": "1"}) == "1-1"


def test_non_string_values_are_stringified() -> None:
    assert render("{{n}} {{b}} {{none}}", {"n": 3, "b": True, "none": None}) == "3 True None"
