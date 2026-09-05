"""The release workflow, checked as configuration.

Nothing here runs GitHub Actions. What it checks are the properties that, if wrong,
are only discovered at the moment of an irreversible publish: a version number on PyPI
can be yanked but never reused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

WORKFLOW = Path(".github/workflows/release.yml")


@pytest.fixture
def workflow(repo_root: Path) -> dict[str, Any]:
    yaml = pytest.importorskip("yaml", reason="the yaml extra is not installed")
    return dict(yaml.safe_load((repo_root / WORKFLOW).read_text(encoding="utf-8")))


class TestItCannotPublishUnverified:
    def test_publish_waits_for_the_build(self, workflow: dict[str, Any]) -> None:
        assert workflow["jobs"]["publish"]["needs"] == "build"

    def test_the_build_waits_for_the_gate(self, workflow: dict[str, Any]) -> None:
        assert workflow["jobs"]["build"]["needs"] == "verify"

    def test_the_gate_is_the_real_gate(self, workflow: dict[str, Any]) -> None:
        """A tag can point anywhere. "green on main" is a different statement."""
        steps = " ".join(str(s.get("run", "")) for s in workflow["jobs"]["verify"]["steps"])
        for command in (
            "ruff check",
            "ruff format --check",
            "mypy --strict",
            "pytest",
            "verify_pack",
        ):
            assert command in steps, f"the tag gate does not run {command}"

    def test_it_refuses_a_tag_that_disagrees_with_the_code(self, workflow: dict[str, Any]) -> None:
        steps = " ".join(str(s.get("run", "")) for s in workflow["jobs"]["verify"]["steps"])
        assert "__version__" in steps and "GITHUB_REF_NAME" in steps

    def test_it_refuses_a_tag_with_no_changelog_section(self, workflow: dict[str, Any]) -> None:
        steps = " ".join(str(s.get("run", "")) for s in workflow["jobs"]["verify"]["steps"])
        assert "CHANGELOG.md" in steps

    def test_the_wheel_is_used_before_it_is_published(self, workflow: dict[str, Any]) -> None:
        """The gate that caught D-123: install it where there is no source tree."""
        steps = " ".join(str(s.get("run", "")) for s in workflow["jobs"]["build"]["steps"])
        assert "venv /tmp/clean" in steps
        assert "alc init" in steps and "alc validate" in steps


class TestNoStoredCredential:
    def test_it_uses_trusted_publishing(self, workflow: dict[str, Any]) -> None:
        """OIDC, so there is no token to leak, rotate, or accidentally print."""
        assert workflow["jobs"]["publish"]["permissions"] == {"id-token": "write"}

    def test_no_job_reads_a_pypi_secret(self, repo_root: Path) -> None:
        text = (repo_root / WORKFLOW).read_text()
        assert "PYPI_API_TOKEN" not in text
        assert "password:" not in text

    def test_the_default_permission_is_read(self, workflow: dict[str, Any]) -> None:
        assert workflow["permissions"] == {"contents": "read"}

    def test_publishing_is_environment_gated(self, workflow: dict[str, Any]) -> None:
        """An environment can require a human approval before an irreversible upload."""
        assert workflow["jobs"]["publish"]["environment"]


class TestItCanBeRehearsed:
    def test_testpypi_is_reachable_without_a_tag(self, workflow: dict[str, Any]) -> None:
        triggers = workflow[True]  # PyYAML reads the `on:` key as the boolean True
        assert "workflow_dispatch" in triggers
        options = triggers["workflow_dispatch"]["inputs"]["target"]["options"]
        assert "testpypi" in options

    def test_a_rehearsal_does_not_default_to_the_real_index(self, workflow: dict[str, Any]) -> None:
        triggers = workflow[True]
        assert triggers["workflow_dispatch"]["inputs"]["target"]["default"] == "testpypi"

    def test_a_tag_publishes_for_real(self, workflow: dict[str, Any]) -> None:
        triggers = workflow[True]
        assert triggers["push"]["tags"] == ["v*"]


def test_the_setup_it_needs_is_written_down(repo_root: Path) -> None:
    """A workflow whose one-time setup lives only in someone's head is not done."""
    text = (repo_root / WORKFLOW).read_text()
    for needed in ("Trusted Publishing", "Pending publishers", "Environment", "release.yml"):
        assert needed in text
