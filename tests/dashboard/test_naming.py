"""A run has to say what it was, not only what it is called.

`pattern.async_agent.fixed` is a stable identifier: it names a directory, feeds
`--filter`, and must never change. It is also meaningless to everyone who has not
read the suite file — which is most people opening the dashboard, developers
included.

So a scenario carries a `title` (a short human name) and its existing `description`
(why the scenario exists), both reach the report, and the reader-facing views lead
with the title and keep the id as secondary text. Where a suite gives neither, the
page derives something readable rather than showing a bare dotted string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos.loop import run_suite
from agent_loop_chaos.scenarios import Scenario, load_suite

FAULT: dict[str, Any] = {
    "type": "ToolCorruptionFault",
    "params": {"mutation_type": "drop_key", "keys": ["temp_c"]},
    "trigger": {"on_call": 1},
}


def _scenario(**kw: Any) -> Scenario:
    return Scenario(
        id="tool.drop_required_key",
        entrypoint="tests.fakes.apps:build_naive",
        inputs=None,
        faults=[dict(FAULT)],
        expected_behavior="graceful_degradation",
        **kw,
    )


class TestTheScenarioCarriesIt:
    def test_title_defaults_to_nothing(self) -> None:
        assert _scenario().title is None

    def test_a_suite_file_can_set_both(self, tmp_path: Path) -> None:
        path = tmp_path / "suite.json"
        path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "scenarios": [
                        {
                            "id": "s",
                            "entrypoint": "m:a",
                            "title": "A tool drops a field the agent needs",
                            "description": "Proves the agent checks a tool result.",
                            "faults": [FAULT],
                        }
                    ],
                }
            )
        )
        scenario = load_suite(path).scenarios[0]
        assert scenario.title == "A tool drops a field the agent needs"
        assert scenario.description == "Proves the agent checks a tool result."


class TestItReachesTheReport:
    def test_both_land_in_the_report(self, tmp_path: Path) -> None:
        [result] = run_suite(
            [_scenario(title="A tool drops a field", description="Why this matters.")],
            out_dir=tmp_path,
            judge="rules",
        )
        document = result.to_dict()
        assert document["scenario_title"] == "A tool drops a field"
        assert document["scenario_description"] == "Why this matters."

    def test_they_are_absent_rather_than_empty_when_unset(self, tmp_path: Path) -> None:
        [result] = run_suite([_scenario()], out_dir=tmp_path, judge="rules")
        document = result.to_dict()
        assert document["scenario_title"] is None
        assert document["scenario_description"] is None

    def test_the_id_is_untouched(self, tmp_path: Path) -> None:
        """The id names a directory and feeds `--filter`. A title never replaces it."""
        [result] = run_suite([_scenario(title="Anything at all")], out_dir=tmp_path, judge="rules")
        assert result.scenario_id == "tool.drop_required_key"
        assert (tmp_path / "tool.drop_required_key").is_dir()


class TestTheRunListShowsIt:
    def test_the_row_carries_the_title(self, tmp_path: Path) -> None:
        from agent_loop_chaos.dashboard import api
        from agent_loop_chaos.dashboard.watcher import RunDirWatcher

        run_suite([_scenario(title="A tool drops a field")], out_dir=tmp_path, judge="rules")
        watcher = RunDirWatcher(tmp_path)
        watcher.poll()
        [row] = api.runs(watcher)
        assert row["title"] == "A tool drops a field"
        assert row["scenario_id"] == "tool.drop_required_key"


class TestTheShippedSuitesReadWell:
    """A suite that ships as an example is also documentation."""

    @pytest.mark.parametrize("name", ["demo_suite.yaml", "patterns_suite.yaml"])
    def test_every_scenario_has_a_title(self, repo_root: Path, name: str) -> None:
        suite = load_suite(repo_root / "examples" / "scenarios" / name)
        untitled = [s.id for s in suite.scenarios if not s.title]
        assert not untitled, f"{name}: no title for {untitled}"

    @pytest.mark.parametrize("name", ["demo_suite.yaml", "patterns_suite.yaml"])
    def test_no_title_is_just_the_id_again(self, repo_root: Path, name: str) -> None:
        suite = load_suite(repo_root / "examples" / "scenarios" / name)
        for scenario in suite.scenarios:
            title = str(scenario.title)
            assert title != scenario.id
            assert "_" not in title, f"{scenario.id}: {title!r} still reads as code"
            assert title[0].isupper(), f"{scenario.id}: {title!r} is not a sentence"

    @pytest.mark.parametrize("name", ["demo_suite.yaml", "patterns_suite.yaml"])
    def test_titles_are_short_enough_to_be_a_heading(self, repo_root: Path, name: str) -> None:
        suite = load_suite(repo_root / "examples" / "scenarios" / name)
        for scenario in suite.scenarios:
            assert len(str(scenario.title)) <= 78, f"{scenario.id}: title is a paragraph"


class TestThePageLeadsWithTheName:
    """`docs/10` §6: the id is not a heading."""

    def test_it_has_a_namer(self, repo_root: Path) -> None:
        page = (repo_root / "src/agent_loop_chaos/dashboard/static/index.html").read_text()
        assert "function nameOf(" in page

    def test_both_views_use_it(self, repo_root: Path) -> None:
        page = (repo_root / "src/agent_loop_chaos/dashboard/static/index.html").read_text()
        assert page.count("nameOf(") >= 3  # defined, used in the card, used in the list

    def test_the_id_is_still_shown(self, repo_root: Path) -> None:
        """Developers need it: it names a directory and feeds `--filter`."""
        page = (repo_root / "src/agent_loop_chaos/dashboard/static/index.html").read_text()
        assert '"sid"' in page and ".sid" in page

    def test_the_filter_matches_the_title_too(self, repo_root: Path) -> None:
        page = (repo_root / "src/agent_loop_chaos/dashboard/static/index.html").read_text()
        assert "r.title" in page


class TestMatrixExpansionKeepsTitlesDistinct:
    """Four rows reading "Hidden instructions arrive inside retrieved content" is not
    a listing, it is a wall. The id already disambiguates; the title has to as well."""

    def _matrix(self) -> Scenario:
        return Scenario(
            id="adversarial.injection_corpus",
            entrypoint="m:a",
            title="Hidden instructions arrive inside retrieved content",
            faults=[{"type": "PromptInjectionFault", "params": {"objective": "x"}}],
            matrix={"faults.0.params.objective": ["exfiltrate_secret", "ignore_instructions"]},
        )

    def test_each_product_gets_its_own_title(self) -> None:
        titles = [s.title for s in self._matrix().expand()]
        assert len(set(titles)) == 2, titles

    def test_the_axis_value_is_readable_in_the_title(self) -> None:
        titles = [str(s.title) for s in self._matrix().expand()]
        assert any("exfiltrate secret" in t for t in titles), titles
        assert all(t.startswith("Hidden instructions") for t in titles)

    def test_the_id_is_unchanged(self) -> None:
        """D-15 fixes the id shape; a title must not move it."""
        ids = [s.id for s in self._matrix().expand()]
        assert ids == [
            "adversarial.injection_corpus-objective-exfiltrate_secret",
            "adversarial.injection_corpus-objective-ignore_instructions",
        ]

    def test_an_untitled_matrix_stays_untitled(self) -> None:
        base = self._matrix()
        base.title = None
        assert all(s.title is None for s in base.expand())
