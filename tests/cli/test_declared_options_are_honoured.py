"""Every declared option does something, and two sweeps keep it that way.

`--record` and `--replay-cassette` were declared on the parser and read by no code for
months (D-144). The only test covering them asserted that they *parse*, so it passed
for exactly as long as they were dead. Looking for others found five more flags and four
schema keys in the same state -- including `--dry-run`, whose help promises "arm every
fault but fire none" while every fault fired, and `skip: true`, which ran the scenario
anyway.

The two sweeps below are the guard. A flag nobody reads and a schema key nothing stores
are both silent lies to a user who took the trouble to set them.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import Scenario
from agent_loop_chaos.cli import build_parser, main
from agent_loop_chaos.scenarios import load_suite

SRC = Path(__file__).resolve().parents[2] / "src" / "agent_loop_chaos"
SCHEMA = Path(__file__).resolve().parents[2] / "schemas" / "scenario.schema.json"

# Read through a loop over names rather than as `args.<dest>`, in `_judge_options`.
_READ_INDIRECTLY = {"model", "base_url", "transport"}


def _library_source() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py"))


class TestEveryFlagIsRead:
    """The sweep that would have caught `--record`."""

    def test_no_flag_is_declared_and_never_read(self) -> None:
        body = _library_source()
        parser = build_parser()
        sub = parser._subparsers._group_actions[0]  # type: ignore[union-attr]

        dead: list[str] = []
        for command, subparser in sub.choices.items():  # type: ignore[attr-defined]
            for action in subparser._actions:
                dest = action.dest
                if dest in {"help", "==SUPPRESS=="} or dest in _READ_INDIRECTLY:
                    continue
                read = re.search(rf"args\.{dest}\b", body) or re.search(
                    rf"""getattr\(\s*args\s*,\s*["']{dest}["']""", body
                )
                if not read:
                    dead.append(f"alc {command} --{dest.replace('_', '-')}")
        assert not dead, (
            "these flags are declared and read by nothing, so setting them does "
            f"nothing: {sorted(set(dead))}"
        )


class TestEverySchemaKeyIsStored:
    """The same sweep for the suite format.

    A key the schema accepts and the loader drops is worse than one it rejects: the
    author gets no error and no effect.
    """

    def test_every_scenario_body_key_has_a_field(self) -> None:
        props = set(json.loads(SCHEMA.read_text())["$defs"]["scenarioBody"]["properties"])
        fields = {f.name for f in dataclasses.fields(Scenario)}
        # `preset` expands into `faults` at load time and is kept for provenance.
        missing = props - fields - {"faults", "preset"}
        assert not missing, f"the schema accepts keys nothing stores: {sorted(missing)}"


def _suite(tmp_path: Path, **body: Any) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "scenarios": [
                    {
                        "id": "s.1",
                        "entrypoint": "tests.fakes.apps:build_naive",
                        "inputs": "pack for Paris",
                        "faults": [
                            {
                                "type": "ToolCorruptionFault",
                                "target": {"tool": "fetch_weather", "phase": "post"},
                                "params": {"mutation_type": "drop_key", "keys": ["temp_c"]},
                            }
                        ],
                        **body,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _fired(out: Path) -> list[tuple[bool, Any]]:
    report = next(out.glob("s.1/run-*/report.json"))
    data = json.loads(report.read_text(encoding="utf-8"))
    return [(f["fired"], f.get("skipped_reason")) for f in data["injected_faults"]]


class TestDryRun:
    """The one with teeth: its help promises "arm every fault but fire none"."""

    def test_the_flag_stops_every_fault_from_firing(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        out = tmp_path / ".chaos"
        main(["run", str(_suite(tmp_path)), "--judge", "rules", "--dry-run", "--out", str(out)])
        assert _fired(out) == [(False, "dry_run")]

    def test_without_it_the_fault_fires(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Otherwise the test above would pass for the wrong reason."""
        monkeypatch.chdir(tmp_path)
        out = tmp_path / ".chaos"
        main(["run", str(_suite(tmp_path)), "--judge", "rules", "--out", str(out)])
        assert _fired(out) == [(True, None)]


class TestSkip:
    def test_a_scenario_marked_skip_does_not_run(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        out = tmp_path / ".chaos"
        code = main(
            ["run", str(_suite(tmp_path, skip=True)), "--judge", "rules", "--out", str(out)]
        )
        assert not list(out.glob("s.1/run-*")), "a skipped scenario ran"
        assert code == 0, "skipping everything is not a failure"

    def test_the_loader_keeps_it(self, tmp_path: Path) -> None:
        assert load_suite(_suite(tmp_path, skip=True)).scenarios[0].skip is True


class TestTraceLevel:
    def test_the_flag_reaches_the_engine(self, tmp_path: Path, monkeypatch: Any) -> None:
        """`trace_level` is not a report field, so the observable is the engine call."""
        import agent_loop_chaos.engine as engine_module
        from agent_loop_chaos import ChaosEngine
        from agent_loop_chaos.loop import run_suite

        seen: list[Any] = []

        class _Spy(ChaosEngine):
            def __init__(self, **kw: Any) -> None:
                seen.append(kw.get("trace_level"))
                super().__init__(**kw)

        monkeypatch.setattr(engine_module, "ChaosEngine", _Spy)
        scenarios = load_suite(_suite(tmp_path, trace_level="minimal")).scenarios
        run_suite(scenarios, out_dir=tmp_path / ".chaos", judge="rules")
        assert "minimal" in seen, f"trace_level never reached the engine: {seen}"

    def test_a_scenario_can_set_it(self, tmp_path: Path) -> None:
        assert load_suite(_suite(tmp_path, trace_level="verbose")).scenarios[0].trace_level == (
            "verbose"
        )


class TestAllowSideEffects:
    def test_the_flag_reaches_the_scenario(self, tmp_path: Path) -> None:
        """Dropping an explicit safety opt-in leaves the user arguing with a gate they
        thought they had opened."""
        from agent_loop_chaos.cli import _apply_run_overrides

        scenarios = load_suite(_suite(tmp_path)).scenarios
        args = build_parser().parse_args(
            ["run", "s.json", "--allow-side-effects", "hold_booking,charge_card"]
        )
        _apply_run_overrides(args, scenarios)
        assert scenarios[0].allow_side_effects == ["hold_booking", "charge_card"]


class TestAdapterAndBaseline:
    def test_a_scenario_can_name_its_adapter(self, tmp_path: Path) -> None:
        assert load_suite(_suite(tmp_path, adapter="vanilla")).scenarios[0].adapter == "vanilla"

    def test_a_scenario_can_decline_a_baseline(self, tmp_path: Path, monkeypatch: Any) -> None:
        assert load_suite(_suite(tmp_path, baseline=False)).scenarios[0].baseline is False


class TestPreset:
    def test_the_flag_replaces_the_declared_faults(self, tmp_path: Path) -> None:
        from agent_loop_chaos.cli import _apply_run_overrides

        scenarios = load_suite(_suite(tmp_path)).scenarios
        before = [f["type"] for f in scenarios[0].faults]
        args = build_parser().parse_args(["run", "s.json", "--preset", "smoke"])
        _apply_run_overrides(args, scenarios)
        after = [f["type"] for f in scenarios[0].faults]
        assert after and after != before, "the preset did not replace the fault list"

    def test_an_unknown_preset_is_a_usage_error(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        assert main(["run", str(_suite(tmp_path)), "--preset", "nope", "--judge", "rules"]) == 2


class TestTheDeadFlagIsGone:
    def test_suggest_fixes_is_no_longer_offered(self) -> None:
        """It promised "a ranked fix list (one extra model call per failure)" and no
        judge accepted such an option. `suggested_fixes` already comes from the SLM and
        ensemble judges, so the flag offered a feature that did not exist."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "s.json", "--suggest-fixes"])
