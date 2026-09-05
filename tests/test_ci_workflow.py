"""CI's own configuration, checked the way any other config is.

Two jobs were failing on every commit and had been since they were written, for
reasons that could not show up locally:

- `lint` installed only `[dev]`, so `mypy --strict` could not find `langgraph`,
  `langchain_core` or `httpx` and reported `import-not-found` for the three modules
  that import them. A maintainer's venv has those extras, so `make check` was green
  and the job was red.
- A FAQ snippet imports the LangGraph adapter, which raises `MissingExtraError`
  without the extra — so the `langgraph: none` column, the one that proves the core
  has no framework dependency, failed on documentation *about* the framework.

Neither is reachable from a normal working copy. The checks below are, which is the
point.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

CI = Path(".github/workflows/ci.yml")


@pytest.fixture
def ci(repo_root: Path) -> dict[str, Any]:
    yaml = pytest.importorskip("yaml", reason="the yaml extra is not installed")
    return dict(yaml.safe_load((repo_root / CI).read_text(encoding="utf-8")))


def _runs(job: dict[str, Any]) -> str:
    return " ; ".join(str(step.get("run", "")) for step in job["steps"])


def _typecheck_extras(repo_root: Path) -> set[str]:
    """Extras `src/` imports, from the refusals it raises when they are missing.

    Derived rather than listed: `MissingExtraError("install agent-loop-chaos[x]")` is
    already the single place each optional import declares what it needs, so a new
    optional import brings its own requirement here with it.
    """
    names: set[str] = set()
    for path in (repo_root / "src").rglob("*.py"):
        names |= set(re.findall(r"agent-loop-chaos\[([a-z,]+)\]", path.read_text(encoding="utf-8")))
    return {part for group in names for part in group.split(",")}


def _extras_in(spec: str, repo_root: Path) -> set[str]:
    """Expand the extras an install spec pulls in, following `all`.

    Args:
        spec: The install line, e.g. `pip install -e ".[dev,all]"`.
        repo_root: The checkout root.

    Returns:
        Every extra name the spec resolves to.
    """
    named = {
        part
        for group in re.findall(r'pip install -e "\.\[([a-z,]+)\]"', spec)
        for part in group.split(",")
    }
    text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    aggregate = re.search(r"^all = \[(.*?)\]", text, re.S | re.M)
    if "all" in named and aggregate:
        named |= set(re.findall(r"agent-loop-chaos\[([a-z,]+)", aggregate.group(1))[0].split(","))
    return named


class TestTheLintJobCanActuallyTypeCheck:
    def test_it_installs_every_optional_extra(self, ci: dict[str, Any], repo_root: Path) -> None:
        """`mypy --strict` reports `import-not-found` for an extra it cannot see, and
        typing those imports as `Any` instead would silence the checks that matter
        most, in the code with the least coverage."""
        install = _runs(ci["jobs"]["lint"])
        have = _extras_in(install, repo_root)
        missing = _typecheck_extras(repo_root) - have
        assert not missing, f"the lint job cannot type-check without {sorted(missing)}"

    def test_it_still_runs_all_three_checks(self, ci: dict[str, Any]) -> None:
        install = _runs(ci["jobs"]["lint"])
        for command in ("ruff check", "ruff format --check", "mypy --strict"):
            assert command in install


class TestTheCoreOnlyColumnStaysHonest:
    def test_one_column_installs_no_framework(self, ci: dict[str, Any]) -> None:
        """It is the only thing proving the core has no hard framework dependency."""
        matrix = ci["jobs"]["test"]["strategy"]["matrix"]
        assert "none" in matrix["langgraph"]

    def test_that_column_installs_dev_only(self, ci: dict[str, Any]) -> None:
        steps = ci["jobs"]["test"]["steps"]
        core = [s for s in steps if "core only" in str(s.get("name", ""))]
        assert core, "no core-only install step"
        assert core[0]["run"].strip() == 'pip install -e ".[dev]"'

    def test_the_matrix_matches_the_supported_pythons(
        self, ci: dict[str, Any], repo_root: Path
    ) -> None:
        text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        floor = re.search(r'requires-python = ">=(\d+)\.(\d+)"', text)
        assert floor, "no requires-python"
        lowest = f"{floor.group(1)}.{floor.group(2)}"
        assert lowest in ci["jobs"]["test"]["strategy"]["matrix"]["python"], (
            "the oldest supported Python is not in the matrix, so nothing tests it"
        )


class TestTheGateIsNotWeakened:
    def test_coverage_is_enforced_not_reported(self, ci: dict[str, Any]) -> None:
        assert "--cov" in _runs(ci["jobs"]["test"])

    def test_the_schema_job_runs_the_pack_check(self, ci: dict[str, Any]) -> None:
        assert "verify_pack.py --strict" in _runs(ci["jobs"]["schema"])

    def test_no_job_swallows_a_failure(self, repo_root: Path) -> None:
        """`|| true` and `continue-on-error` turn a gate into a decoration."""
        text = (repo_root / CI).read_text()
        assert "continue-on-error" not in text
        assert "|| true" not in text
