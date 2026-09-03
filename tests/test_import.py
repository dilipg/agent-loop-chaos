"""The public import surface.

`docs/02-API.md` §1 is normative: every name in it is frozen, and `__all__` must
match it name for name. This test parses the doc rather than restating the list, so
the doc cannot drift from the code silently.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import agent_loop_chaos

REPO_ROOT = Path(__file__).resolve().parent.parent
API_DOC = REPO_ROOT / "docs" / "02-API.md"


def _documented_all(doc: Path) -> list[str]:
    """Extract `__all__` from the first python block in `docs/02-API.md` §1.

    Args:
        doc: Path to the API document.

    Returns:
        The documented export names, in documented order.
    """
    text = doc.read_text(encoding="utf-8")
    section = text.split("## 1. Top-level exports", 1)[1].split("## 2.", 1)[0]
    match = re.search(r"```python\n(.*?)```", section, re.S)
    assert match, "docs/02-API.md §1 has no python block"
    tree = ast.parse(match.group(1))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            return [name for name in value if isinstance(name, str)]
    raise AssertionError("docs/02-API.md §1 python block has no __all__ assignment")


@pytest.mark.skipif(not API_DOC.exists(), reason="docs/ is not shipped in the wheel")
def test_all_matches_the_api_doc_name_for_name() -> None:
    """`__all__` equals `docs/02-API.md` §1, as a set and as a sorted list."""
    documented = _documented_all(API_DOC)
    assert sorted(agent_loop_chaos.__all__) == sorted(documented)


def test_every_exported_name_actually_resolves() -> None:
    """Every name in `__all__` is importable from the package."""
    missing = [name for name in agent_loop_chaos.__all__ if not hasattr(agent_loop_chaos, name)]
    assert missing == []


def test_all_has_no_duplicates() -> None:
    """`__all__` lists each name once."""
    names = list(agent_loop_chaos.__all__)
    assert len(names) == len(set(names))


def test_version_is_a_dotted_string() -> None:
    """`__version__` is present and looks like a release version."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", agent_loop_chaos.__version__)


def test_private_names_are_not_exported() -> None:
    """`ObservedBehavior` exists but is deliberately not public (D-51)."""
    from agent_loop_chaos import enums

    assert hasattr(enums, "ObservedBehavior")
    assert "ObservedBehavior" not in agent_loop_chaos.__all__


def test_unknown_attribute_raises_attribute_error() -> None:
    """The lazy `__getattr__` does not swallow typos."""
    with pytest.raises(AttributeError):
        _ = agent_loop_chaos.NoSuchName  # type: ignore[attr-defined]


def test_chaos_engine_is_implemented_as_of_m1() -> None:
    """M1 built the engine, so constructing one must no longer raise.

    This is the half of the old M0 stub test that changed: `ChaosEngine` was a stub
    then and is real now.
    """
    engine = agent_loop_chaos.ChaosEngine(seed=1, write_bundle=False)
    assert engine.seed == 1
    assert engine.is_active() is False


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: agent_loop_chaos.RuleJudge(), id="RuleJudge (M6)"),
        pytest.param(lambda: agent_loop_chaos.SLMJudge(), id="SLMJudge (M6)"),
        pytest.param(lambda: agent_loop_chaos.EnsembleJudge(), id="EnsembleJudge (M6)"),
        pytest.param(
            lambda: agent_loop_chaos.ChaosEngine(write_bundle=False).replay("d"),
            id="replay (M7)",
        ),
    ],
)
def test_remaining_stubs_still_fail_loudly(call: object) -> None:
    """A stub must raise. Silently returning `None` is forbidden by CLAUDE.md.

    Each of these lands in a later milestone; until then, calling one must say so
    rather than hand back a `None` that surfaces as a confusing error later.

    `Scenario`, `ChaosSuite` and `load_suite` left this list in M4, which implemented
    them.
    """
    with pytest.raises(NotImplementedError, match=r"M\d"):
        call()  # type: ignore[operator]


def test_limit_exceeded_is_not_catchable_as_exception() -> None:
    """`LimitExceeded` derives from `BaseException` so the agent cannot swallow it (D-06)."""
    assert issubclass(agent_loop_chaos.LimitExceeded, BaseException)
    assert not issubclass(agent_loop_chaos.LimitExceeded, Exception)


def test_explicit_error_is_an_ordinary_exception() -> None:
    """`ExplicitError` is raised *by* the agent, so it must be a normal `Exception`."""
    assert issubclass(agent_loop_chaos.ExplicitError, Exception)


def test_error_hierarchy_has_one_base() -> None:
    """Every deliberate error except the two special cases derives from `ChaosError`."""
    for name in (
        "ConfigError",
        "MissingExtraError",
        "SchemaError",
        "JudgeError",
        "AdapterError",
    ):
        assert issubclass(getattr(agent_loop_chaos, name), agent_loop_chaos.ChaosError)
