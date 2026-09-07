"""The reporting path, checked as configuration.

Colleagues will find bugs — the last two hours found seven without leaving this
machine. What decides whether those become fixes is whether the report arrives with
enough in it to act on, and whether the reporter believes it is worth sending.

The false-positive template matters most. A probe firing on a correct agent is the one
failure that makes every other verdict untrustworthy, and a reporter's default
assumption is that they misconfigured something.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

TEMPLATES = Path(".github/ISSUE_TEMPLATE")


@pytest.fixture
def templates(repo_root: Path) -> dict[str, Any]:
    yaml = pytest.importorskip("yaml", reason="the yaml extra is not installed")
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((repo_root / TEMPLATES).glob("*.yml"))
    }


def _text(template: dict[str, Any]) -> str:
    return " ".join(
        str(field.get("attributes", {}).get("value", ""))
        + str(field.get("attributes", {}).get("label", ""))
        + str(field.get("attributes", {}).get("description", ""))
        for field in template.get("body", [])
    )


class TestTheFalsePositivePath:
    def test_it_exists_and_is_labelled(self, templates: dict[str, Any]) -> None:
        template = templates["false-positive.yml"]
        assert "false-positive" in template["labels"]

    def test_it_tells_the_reporter_it_is_our_bug(self, templates: dict[str, Any]) -> None:
        """A reporter's default assumption is that they misconfigured something."""
        assert "our bug, not yours" in _text(templates["false-positive.yml"])

    def test_it_cites_the_time_this_really_happened(self, templates: dict[str, Any]) -> None:
        """Evidence that the report is wanted beats a promise that it is."""
        assert "redacted_value_in_output" in _text(templates["false-positive.yml"])

    def test_it_says_the_report_is_safe_to_attach(self, templates: dict[str, Any]) -> None:
        text = _text(templates["false-positive.yml"])
        assert "redacted before it is written" in text
        assert "redact_keys" in text

    def test_it_asks_for_what_only_the_reporter_knows(self, templates: dict[str, Any]) -> None:
        labels = [
            f["attributes"]["label"]
            for f in templates["false-positive.yml"]["body"]
            if "label" in f.get("attributes", {})
        ]
        assert any("did, and why it was right" in label for label in labels)

    def test_it_asks_for_the_scenario_and_the_report(self, templates: dict[str, Any]) -> None:
        ids = {f.get("id") for f in templates["false-positive.yml"]["body"]}
        assert {"version", "probe", "scenario", "report"} <= ids


class TestTheGeneralBugPath:
    def test_it_warns_about_the_two_traps_that_look_like_user_error(
        self, templates: dict[str, Any]
    ) -> None:
        text = _text(templates["bug.yml"])
        assert "json_patch" in text, "the empty-patch trap is not mentioned"
        assert "make doctor" in text, "the iCloud trap is not mentioned"

    def test_it_asks_for_the_version_and_the_platform(self, templates: dict[str, Any]) -> None:
        ids = {f.get("id") for f in templates["bug.yml"]["body"]}
        assert {"version", "python", "repro"} <= ids


class TestTheSignposts:
    def test_they_point_at_files_that_exist(
        self, templates: dict[str, Any], repo_root: Path
    ) -> None:
        for link in templates["config.yml"]["contact_links"]:
            target = link["url"].rsplit("/main/", 1)[-1]
            assert (repo_root / target).is_file(), f"{link['name']} points at {target}"


class TestTheFaqCoversTheFirstHour:
    def test_it_troubleshoots_a_fault_that_did_nothing(self, repo_root: Path) -> None:
        faq = (repo_root / "docs" / "FAQ.md").read_text()
        assert "nothing happened" in faq
        assert "skipped_reason" in faq

    def test_it_troubleshoots_the_import_error(self, repo_root: Path) -> None:
        faq = (repo_root / "docs" / "FAQ.md").read_text()
        assert "ModuleNotFoundError" in faq and "make doctor" in faq

    def test_it_troubleshoots_a_false_secret_leak(self, repo_root: Path) -> None:
        faq = (repo_root / "docs" / "FAQ.md").read_text()
        assert "leaked a secret" in faq and "redact_keys" in faq

    def test_the_readme_points_at_the_reporting_path(self, repo_root: Path) -> None:
        readme = (repo_root / "README.md").read_text()
        assert "false-positive" in readme
