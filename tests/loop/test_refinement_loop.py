"""`RefinementLoop` — run, hand off, re-run, report what actually changed.

The loop's whole value is the last part. A round-over-round diff is easy; telling a
real fix from a weakened test is not, and `docs/05` §9 says the countermeasures are
implemented rather than documented. That is what most of this file checks.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from agent_loop_chaos.loop import LoopReport, RefinementLoop
from agent_loop_chaos.report import ChaosResult
from agent_loop_chaos.scenarios import ChaosSuite, Scenario
from tests.fakes import swappable

DROP_KEY = {
    "type": "ToolCorruptionFault",
    "params": {"mutation_type": "drop_key", "keys": ["temp_c"]},
    "trigger": {"on_call": 1},
}


@pytest.fixture(autouse=True)
def _reset_fakes() -> Iterator[None]:
    swappable.reset()
    yield
    swappable.reset()


def _suite(*, with_sibling: bool = False, with_control: bool = False) -> ChaosSuite:
    scenarios = [
        Scenario(
            id="tool.drop_key",
            entrypoint="tests.fakes.swappable:build",
            inputs=None,
            faults=[DROP_KEY],
            expected_behavior="graceful_degradation",
        )
    ]
    if with_sibling:
        scenarios.append(
            Scenario(
                id="tool.sibling",
                entrypoint="tests.fakes.swappable:build_sibling",
                inputs=None,
                faults=[],
                expected_behavior="ignore_and_continue",
            )
        )
    if with_control:
        scenarios.insert(
            0,
            Scenario(
                id="control.dry_run",
                entrypoint="tests.fakes.swappable:build_sibling",
                inputs=None,
                faults=[],
                dry_run=True,
                expected_behavior="ignore_and_continue",
            ),
        )
    return ChaosSuite(scenarios)


def _loop(suite: ChaosSuite, tmp_path: Path, **over: object) -> RefinementLoop:
    kwargs: dict[str, object] = {"out_dir": tmp_path, "max_rounds": 3, "judge": "rules"}
    kwargs.update(over)
    return RefinementLoop(suite, **kwargs)  # type: ignore[arg-type]


class TestTwoRoundFix:
    """The happy path: a finding is handed out, fixed, and the flip is reported."""

    @staticmethod
    def _fix(_results: Sequence[ChaosResult]) -> None:
        swappable.FIXED = True

    def test_round_one_fails_and_round_two_passes(self, tmp_path: Path) -> None:
        report = _loop(_suite(), tmp_path, on_findings=self._fix).run()
        assert len(report.rounds) == 2
        assert report.rounds[0].failed == 1
        assert report.rounds[1].failed == 0

    def test_the_flip_is_recorded(self, tmp_path: Path) -> None:
        report = _loop(_suite(), tmp_path, on_findings=self._fix).run()
        assert report.fixed_between_rounds == [[], ["tool.drop_key"]]
        assert report.regressions == []

    def test_outcome_classification(self, tmp_path: Path) -> None:
        report = _loop(_suite(), tmp_path, on_findings=self._fix).run()
        assert report.rounds[0].outcomes["tool.drop_key"] == "fail"
        assert report.rounds[1].outcomes["tool.drop_key"] == "flipped_to_pass"

    def test_the_final_round_re_runs_everything(self, tmp_path: Path) -> None:
        # Not just the failures: a fix that broke something else must be catchable.
        report = _loop(_suite(with_sibling=True), tmp_path, on_findings=self._fix).run()
        assert {r.scenario_id for r in report.rounds[-1].results} == {
            "tool.drop_key",
            "tool.sibling",
        }

    def test_on_findings_receives_only_the_failures(self, tmp_path: Path) -> None:
        seen: list[list[str | None]] = []

        def record(results: Sequence[ChaosResult]) -> None:
            seen.append([r.scenario_id for r in results])
            swappable.FIXED = True

        _loop(_suite(with_sibling=True), tmp_path, on_findings=record).run()
        assert seen == [["tool.drop_key"]]

    def test_tasks_written_lists_the_work_orders(self, tmp_path: Path) -> None:
        # Deliberately *not* using `_fix`: a scenario that ends the loop passing has
        # no outstanding work order, which `TestTasksWrittenIsActionable` covers.
        report = _loop(_suite(), tmp_path, max_rounds=2, stop_when="rounds").run()
        assert report.tasks_written
        assert all(Path(p).name == "AGENT_TASK.md" for p in report.tasks_written)
        assert all(Path(p).is_file() for p in report.tasks_written)


class TestSeedsAreStable:
    def test_plan_hash_is_identical_across_rounds(self, tmp_path: Path) -> None:
        report = _loop(_suite(), tmp_path, max_rounds=2, stop_when="rounds").run()
        hashes = {r.results[0].plan_hash for r in report.rounds}
        assert len(hashes) == 1, "an unmodified scenario must plan identically every round"

    def test_run_ids_differ_only_by_attempt(self, tmp_path: Path) -> None:
        report = _loop(_suite(), tmp_path, max_rounds=2, stop_when="rounds").run()
        assert report.rounds[0].results[0].attempt == 1
        assert report.rounds[1].results[0].attempt == 2
        assert report.rounds[0].results[0].run_id != report.rounds[1].results[0].run_id


class TestRegressions:
    def test_a_fix_that_breaks_a_sibling_is_a_regression(self, tmp_path: Path) -> None:
        def bad_fix(_results: Sequence[ChaosResult]) -> None:
            swappable.FIXED = True
            swappable.BROKEN_SIBLING = True

        report = _loop(_suite(with_sibling=True), tmp_path, on_findings=bad_fix).run()
        assert any("tool.sibling" in line for line in report.regressions)
        assert report.rounds[1].outcomes["tool.sibling"] == "flipped_to_fail"

    def test_still_failing_is_distinguished_from_a_first_failure(self, tmp_path: Path) -> None:
        report = _loop(_suite(), tmp_path, max_rounds=2, stop_when="rounds").run()
        assert report.rounds[0].outcomes["tool.drop_key"] == "fail"
        assert report.rounds[1].outcomes["tool.drop_key"] == "still_failing"


class TestTamperDetection:
    """`docs/05` §9. A coding agent that weakens the test is the threat model."""

    def test_a_flip_after_the_plan_changed_is_a_regression_not_a_fix(self, tmp_path: Path) -> None:
        suite = _suite()

        def weaken(_results: Sequence[ChaosResult]) -> None:
            # The realistic cheat: delete the fault so nothing is injected.
            swappable.FIXED = True
            suite.scenarios[0].faults = []

        report = _loop(suite, tmp_path, on_findings=weaken).run()
        assert report.fixed_between_rounds[-1] == [], "a modified scenario is never a fix"
        assert any("plan_hash changed" in line for line in report.regressions)
        assert any("tool.drop_key" in line for line in report.regressions)

    def test_a_changed_must_not_list_is_detected(self, tmp_path: Path) -> None:
        suite = _suite()

        def weaken(_results: Sequence[ChaosResult]) -> None:
            swappable.FIXED = True
            suite.scenarios[0].must_not = []
            suite.scenarios[0].expected_behavior = "ignore_and_continue"

        report = _loop(suite, tmp_path, on_findings=weaken).run()
        assert report.regressions

    def test_a_changed_scenario_file_is_detected(self, tmp_path: Path) -> None:
        source = tmp_path / "suite.json"
        source.write_text(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "id": "tool.drop_key",
                            "entrypoint": "tests.fakes.swappable:build",
                            "expected_behavior": "graceful_degradation",
                            "faults": [DROP_KEY],
                        }
                    ]
                }
            )
        )
        from agent_loop_chaos.scenarios import load_suite

        suite = load_suite(source)

        def edit_the_file(_results: Sequence[ChaosResult]) -> None:
            swappable.FIXED = True
            source.write_text(source.read_text().replace("temp_c", "humidity"))

        report = _loop(suite, tmp_path / "out", on_findings=edit_the_file).run()
        assert any("scenario file changed" in line for line in report.regressions)

    def test_an_unmodified_scenario_records_stable_hashes(self, tmp_path: Path) -> None:
        report = _loop(_suite(), tmp_path, max_rounds=2, stop_when="rounds").run()
        assert report.tamper["modified"] == []
        assert report.tamper["probes_changed"] is False
        assert len(report.tamper["probes_sha256"]) == 64

    def test_probe_module_hash_change_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_loop_chaos import loop as loop_module

        calls = iter(["a" * 64, "b" * 64, "b" * 64])
        monkeypatch.setattr(loop_module, "probes_fingerprint", lambda: next(calls))
        report = _loop(_suite(), tmp_path, max_rounds=2, stop_when="rounds").run()
        assert report.tamper["probes_changed"] is True
        assert any("probe" in line.lower() for line in report.regressions)


class TestControlAbort:
    def test_a_failing_control_aborts_the_loop(self, tmp_path: Path) -> None:
        swappable.BROKEN_SIBLING = True  # the control scenario now crashes
        report = _loop(_suite(with_control=True), tmp_path, max_rounds=3).run()
        assert report.aborted is True
        assert report.tamper["control_failed"] == "control.dry_run"
        assert len(report.rounds) == 1, "the loop must not continue past a failed control"

    def test_the_abort_reason_names_the_harness(self, tmp_path: Path) -> None:
        swappable.BROKEN_SIBLING = True
        report = _loop(_suite(with_control=True), tmp_path).run()
        assert report.abort_reason is not None
        assert "harness" in report.abort_reason.lower()

    def test_a_passing_control_does_not_abort(self, tmp_path: Path) -> None:
        report = _loop(_suite(with_control=True), tmp_path, max_rounds=1).run()
        assert report.aborted is False
        assert report.tamper["control_failed"] is None


class TestStopConditions:
    def test_all_pass_stops_as_soon_as_everything_passes(self, tmp_path: Path) -> None:
        def fix(_results: Sequence[ChaosResult]) -> None:
            swappable.FIXED = True

        report = _loop(
            _suite(), tmp_path, max_rounds=5, stop_when="all_pass", on_findings=fix
        ).run()
        assert len(report.rounds) == 2

    def test_no_new_failures_stops_when_a_round_adds_nothing(self, tmp_path: Path) -> None:
        # Nothing is fixed, so round 2 surfaces exactly what round 1 did.
        report = _loop(_suite(), tmp_path, max_rounds=5, stop_when="no_new_failures").run()
        assert len(report.rounds) == 2
        assert report.new_failures_per_round == [1, 0]

    def test_rounds_runs_the_full_count(self, tmp_path: Path) -> None:
        report = _loop(_suite(), tmp_path, max_rounds=3, stop_when="rounds").run()
        assert len(report.rounds) == 3

    def test_max_rounds_caps_no_new_failures_too(self, tmp_path: Path) -> None:
        broken = iter([False, False, False, True, True])

        def flaky(_results: Sequence[ChaosResult]) -> None:
            swappable.BROKEN_SIBLING = next(broken)

        report = _loop(_suite(with_sibling=True), tmp_path, max_rounds=2, on_findings=flaky).run()
        assert len(report.rounds) == 2

    def test_a_single_round_calls_no_hook(self, tmp_path: Path) -> None:
        called = []
        report = _loop(
            _suite(), tmp_path, max_rounds=1, on_findings=lambda r: called.append(r)
        ).run()
        assert len(report.rounds) == 1
        assert called == [], "there is no round after the last one to benefit from a fix"


class TestResilience:
    def test_a_raising_hook_does_not_kill_the_loop(self, tmp_path: Path) -> None:
        def explode(_results: Sequence[ChaosResult]) -> None:
            raise RuntimeError("the coding agent crashed")

        report = _loop(_suite(), tmp_path, max_rounds=2, on_findings=explode).run()
        assert len(report.rounds) == 2
        assert any("on_findings" in line for line in report.regressions)

    def test_the_loop_never_edits_source(self, tmp_path: Path) -> None:
        # There is no code path that writes to a .py file. This asserts the promise
        # by checking the module's own source for the obvious ways to break it.
        from agent_loop_chaos import loop as loop_module

        source = Path(loop_module.__file__).read_text()
        assert "subprocess" not in source
        assert "shutil" not in source


class TestLoopReportSerialization:
    def test_to_dict_is_json_serializable(self, tmp_path: Path) -> None:
        report = _loop(_suite(), tmp_path, max_rounds=1).run()
        json.dumps(report.to_dict())

    def test_suite_json_carries_the_loop_fields(self, tmp_path: Path) -> None:
        from agent_loop_chaos.schema import validate_obj

        def fix(_results: Sequence[ChaosResult]) -> None:
            swappable.FIXED = True

        _loop(_suite(), tmp_path, on_findings=fix).run()
        document = json.loads((tmp_path / "suite.json").read_text())
        assert document["schema_version"] == "1.1"
        assert document["rounds_planned"] == 3
        assert document["round"] == 2
        assert document["flipped"] == [{"scenario_id": "tool.drop_key", "round": 2}]
        assert "tamper" in document
        assert validate_obj(document, "suite") == []

    def test_out_dir_is_recorded(self, tmp_path: Path) -> None:
        report = _loop(_suite(), tmp_path, max_rounds=1).run()
        assert Path(report.out_dir) == tmp_path


class TestMarkdown:
    def test_the_table_has_a_column_per_round(self, tmp_path: Path) -> None:
        def fix(_results: Sequence[ChaosResult]) -> None:
            swappable.FIXED = True

        markdown = _loop(_suite(), tmp_path, on_findings=fix).run().markdown()
        assert "| scenario | r1 | r2 | outcome |" in markdown
        assert "| tool.drop_key | FAIL | PASS | fixed in r2 |" in markdown

    def test_still_failing_names_the_failure_mode(self, tmp_path: Path) -> None:
        markdown = _loop(_suite(), tmp_path, max_rounds=2, stop_when="rounds").run().markdown()
        assert "still failing (crash_unhandled_exception)" in markdown

    def test_a_regression_is_shouted(self, tmp_path: Path) -> None:
        def bad_fix(_results: Sequence[ChaosResult]) -> None:
            swappable.FIXED = True
            swappable.BROKEN_SIBLING = True

        markdown = _loop(_suite(with_sibling=True), tmp_path, on_findings=bad_fix).run().markdown()
        assert "REGRESSION in r2" in markdown

    def test_the_header_carries_the_totals_a_human_scans_for(self, tmp_path: Path) -> None:
        markdown = _loop(_suite(), tmp_path, max_rounds=1).run().markdown()
        for expected in ("rounds", "regressions", "judge disagreement", "AGENT_TASK.md"):
            assert expected in markdown, expected

    def test_it_is_valid_markdown_with_no_stray_pipes(self, tmp_path: Path) -> None:
        markdown = _loop(_suite(), tmp_path, max_rounds=1).run().markdown()
        rows = [line for line in markdown.splitlines() if line.startswith("| ")]
        widths = {line.count("|") for line in rows}
        assert len(widths) == 1, f"ragged table: {widths}"


def test_loop_report_is_the_documented_shape() -> None:
    """`docs/02-API.md` §8 freezes these names."""
    for field in (
        "rounds",
        "new_failures_per_round",
        "fixed_between_rounds",
        "regressions",
        "tasks_written",
        "out_dir",
    ):
        assert field in LoopReport.__dataclass_fields__, field


class TestTasksWrittenIsActionable:
    """`tasks_written` is what a human pastes to a coding agent. It has to be current.

    Every round rewrites a still-failing scenario's work order into a new run
    directory. Listing all of them hands out two orders for one bug, the older
    describing a run that has since been superseded -- and `docs/05` §8's pattern A
    is literally `print("\\n".join(report.tasks_written))`.
    """

    def test_one_work_order_per_scenario_not_one_per_round(self, tmp_path: Path) -> None:
        report = _loop(_suite(with_sibling=True), tmp_path, max_rounds=3, stop_when="rounds").run()
        assert len(report.rounds) == 3
        assert len(report.tasks_written) == 1, "one still-failing scenario, one work order"

    def test_it_is_the_newest_run(self, tmp_path: Path) -> None:
        report = _loop(_suite(), tmp_path, max_rounds=2, stop_when="rounds").run()
        newest = report.rounds[-1].results[0]
        assert report.tasks_written == [Path(newest.artifacts["agent_task"])]

    def test_a_scenario_fixed_mid_loop_leaves_no_work_order(self, tmp_path: Path) -> None:
        def fix(_results: Sequence[ChaosResult]) -> None:
            swappable.FIXED = True

        report = _loop(_suite(), tmp_path, on_findings=fix).run()
        assert report.tasks_written == [], "it passes now; there is nothing to hand out"

    def test_every_listed_path_exists(self, tmp_path: Path) -> None:
        report = _loop(_suite(), tmp_path, max_rounds=2, stop_when="rounds").run()
        assert all(p.is_file() for p in report.tasks_written)


class TestInitialStateIsIsolatedPerRun:
    """A scenario's `initial_state` must not carry mutations between runs.

    `dict(scenario.initial_state)` is a *shallow* copy, so a nested list or dict is
    shared by every run of that scenario. An agent that appends to
    `state["claims"]` -- an ordinary thing for a scratchpad to do -- then sees the
    previous round's entries, and the loop's round-over-round comparison is
    comparing two different starting conditions while reporting them as the same
    scenario at the same seed.
    """

    @staticmethod
    def _suite() -> ChaosSuite:
        return ChaosSuite(
            [
                Scenario(
                    id="stateful",
                    entrypoint="tests.fakes.swappable:build_appending",
                    inputs=None,
                    initial_state={"claims": []},
                    faults=[],
                    expected_behavior="ignore_and_continue",
                )
            ]
        )

    def test_each_round_starts_from_the_declared_state(self, tmp_path: Path) -> None:
        suite = self._suite()
        report = _loop(suite, tmp_path, max_rounds=3, stop_when="rounds").run()
        answers = [r.results[0].final_output for r in report.rounds]
        assert answers == ["claims=1", "claims=1", "claims=1"], (
            f"the scratchpad leaked between rounds: {answers}"
        )

    def test_the_nested_value_the_agent_mutated_is_a_copy(self, tmp_path: Path) -> None:
        # The direct reproduction: hand the engine the *same* nested list twice and
        # assert the second run does not see the first run's writes.
        from agent_loop_chaos import ChaosEngine
        from tests.fakes.swappable import build_appending

        shared: dict[str, list[str]] = {"claims": []}
        answers = []
        for attempt in range(3):
            engine = ChaosEngine(seed=1337, out_dir=tmp_path, write_bundle=False, judge="rules")
            result = engine.run(
                build_appending(engine),
                inputs=None,
                initial_state=shared,
                scenario_id="stateful",
                expected_behavior="ignore_and_continue",
                attempt=attempt + 1,
            )
            answers.append(result.final_output)
        assert answers == ["claims=1"] * 3, f"the caller's list was shared: {answers}"
        assert shared == {"claims": []}, "the engine mutated the caller's object"

    def test_the_scenarios_own_state_is_not_mutated(self, tmp_path: Path) -> None:
        suite = self._suite()
        _loop(suite, tmp_path, max_rounds=2, stop_when="rounds").run()
        assert suite.scenarios[0].initial_state == {"claims": []}
