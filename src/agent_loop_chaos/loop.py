"""The refinement loop.

baseline -> chaos matrix -> probes -> judge -> `AGENT_TASK.md`, repeated. The loop
never edits source itself: it writes work orders, hands them to a caller, and reports
what flipped.

Most of this module is the reporting half, and specifically telling a real fix from a
weakened test. That is not paranoia: the natural way for a coding agent to make a
chaos scenario pass is to delete the fault, widen the `must_not` list, or edit the
scenario file, and every one of those looks exactly like a fix from the outside. So
before each round the loop records what the scenario *is* -- its plan hash, its file's
hash, its `must_not` list -- and a flip whose fingerprint changed is reported as a
regression (`docs/05` §9).

`run_suite` lives here rather than in `cli.py` because the loop is its second caller
and two copies of the scenario-running logic would drift (D-74).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from .errors import ConfigError
from .report import ChaosResult
from .scenarios import ChaosSuite, Scenario

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .targeting import Trigger

__all__ = [
    "LoopReport",
    "RefinementLoop",
    "RoundResult",
    "probes_fingerprint",
    "run_suite",
    "scenario_fingerprint",
]

log = logging.getLogger("agent_loop_chaos")

StopWhen = Literal["all_pass", "no_new_failures", "rounds"]
Outcome = Literal["pass", "fail", "flipped_to_pass", "flipped_to_fail", "still_failing", "skipped"]

# A scenario whose id starts with this is a control: it must pass in every round, and
# its failure means the harness itself was altered rather than the agent. The spec
# names `control.dry_run`; treating the whole prefix as controls means the convention
# scales to a suite with several of them (D-75).
_CONTROL_PREFIX = "control."


def probes_fingerprint() -> str:
    """Hash the installed probe module's source.

    A coding agent editing the chaos library to make its own agent pass is the
    failure mode `docs/05` §9 exists to catch, and the probes are where it would
    edit. Comparing this between rounds catches it.

    Returns:
        A sha256 hex digest, or 64 zeros when the source cannot be read (a zipped or
        frozen install). An unreadable module is reported as unchanged rather than as
        tampering: a false accusation is worse than a missed one here, and the
        scenario fingerprints still apply.
    """
    from . import probes

    try:
        source = Path(probes.__file__).read_bytes()
    except (OSError, TypeError):  # pragma: no cover - frozen or zipped install
        return "0" * 64
    return hashlib.sha256(source).hexdigest()


def scenario_fingerprint(scenario: Scenario, plan_hash: str) -> dict[str, Any]:
    """Record what a scenario *is*, so a change to it can be spotted next round.

    Args:
        scenario: The scenario as it stood this round.
        plan_hash: The hash the engine computed for its fault plan.

    Returns:
        ``{plan_hash, scenario_file, must_not}``. `scenario_file` is `None` for a
        suite built in Python rather than loaded from a file.
    """
    source = getattr(scenario, "source_path", None)
    file_hash: str | None = None
    if source:
        try:
            file_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        except OSError:
            file_hash = None
    must_not = json.dumps(sorted(scenario.must_not), sort_keys=True)
    return {
        "plan_hash": plan_hash,
        "scenario_file": file_hash,
        "must_not": hashlib.sha256(must_not.encode()).hexdigest(),
    }


# --------------------------------------------------------------------- the runner


def build_fault(spec: Mapping[str, Any]) -> Any:
    """Construct a fault from a scenario's fault spec.

    Args:
        spec: The spec.

    Returns:
        The fault instance.
    """
    from .faults.base import fault_from_dict

    return fault_from_dict(dict(spec))


def plan_specs(scenario: Scenario) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Apply the intensity dial to a scenario's declared faults.

    This is the one place the dial does anything. It scales each declared fault's
    trigger and magnitude parameters, then -- above level 4 -- adds faults from the
    matching preset so a one-fault scenario becomes a compound one. Level 3 is the
    identity and returns the declared list unchanged.

    Args:
        scenario: The scenario, carrying `intensity` and any `preset` it named.

    Returns:
        `(specs, skipped)`. `specs` are ready for `build_fault`; `skipped` records
        anything the dial refused to add, notably a real-action fault with no opt-in.
    """
    from .intensity import DEFAULT_LEVEL, expand_faults, profile, scale_params, scale_trigger
    from .scenarios import FaultSpec
    from .targeting import Trigger

    declared = [dict(spec) for spec in scenario.faults]
    if scenario.intensity == DEFAULT_LEVEL:
        return declared, []

    prof = profile(scenario.intensity)
    scaled: list[dict[str, Any]] = []
    for spec in declared:
        trigger = Trigger(**(spec.get("trigger") or {}))
        out = dict(spec)
        out["params"] = scale_params(str(spec.get("type")), spec.get("params") or {}, prof)
        out["trigger"] = _trigger_dict(scale_trigger(trigger, prof))
        scaled.append(out)

    extras, skipped = expand_faults(
        [
            FaultSpec(
                type=str(s.get("type")),
                params=dict(s.get("params") or {}),
                target=dict(s.get("target") or {}),
                trigger=dict(s.get("trigger") or {}),
            )
            for s in scaled
        ],
        prof,
        preset=scenario.preset,
        allow_side_effects=scenario.allow_side_effects,
    )
    return [*scaled, *(spec.to_dict() for spec in extras)], skipped


def _trigger_dict(trigger: Trigger) -> dict[str, Any]:
    """Render a trigger back to the mapping a fault spec carries.

    Args:
        trigger: The trigger.

    Returns:
        Only the fields that differ from the default, so a scaled spec stays as
        readable as the one the scenario wrote.
    """
    from .targeting import Trigger

    default = Trigger()
    return {
        name: getattr(trigger, name)
        for name in (
            "on_call",
            "on_step",
            "after_step",
            "probability",
            "max_fires",
            "cooldown_calls",
            "stop_after_step",
        )
        if getattr(trigger, name) != getattr(default, name)
    }


def target_kwargs(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a spec's target and trigger into `register_fault` arguments.

    Args:
        spec: The fault spec.

    Returns:
        Keyword arguments for `register_fault`.
    """
    from .targeting import Target, Trigger

    target = spec.get("target") or {}
    trigger = spec.get("trigger") or {}
    return {
        "target": Target(**target) if target else None,
        "trigger": Trigger(**trigger) if trigger else None,
    }


def resolve_entrypoint(entrypoint: Any, engine: Any) -> Any:
    """Import a ``module:attr`` entrypoint, building it when it wants the engine.

    A vanilla agent's tools have to be wrapped by *this run's* engine before any tool
    fault can fire, and a bare function has no way to do that. The convention: a
    resolved callable whose first parameter is named `engine` is a **builder** -- it
    is handed the engine and returns the real agent. Anything else is the agent
    itself (D-61).

    Args:
        entrypoint: A callable, or a ``module:attr`` string.
        engine: The run's engine, passed to a builder.

    Returns:
        The agent callable.

    Raises:
        ConfigError: When the module or attribute cannot be imported.
    """
    import inspect

    resolved = entrypoint
    if not callable(entrypoint):
        import importlib

        text = str(entrypoint)
        if ":" not in text:
            raise ConfigError(f"entrypoint must be 'module:attr'; got {text!r}")
        module_name, _, attr = text.partition(":")
        try:
            resolved = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            raise ConfigError(f"cannot import entrypoint {text!r}: {exc}") from exc

    try:
        parameters = list(inspect.signature(resolved).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return resolved
    if parameters and parameters[0] == "engine":
        return resolved(engine)
    return resolved


def _baseline_key(scenario: Scenario) -> str:
    """The cache key a scenario's baseline lives under.

    Args:
        scenario: The scenario.

    Returns:
        A key over the entrypoint and inputs -- the only two things that decide what
        an unfaulted run produces.
    """
    return json.dumps([str(scenario.entrypoint), scenario.inputs], sort_keys=True, default=str)


def drive(engine: Any, agent: Any, **kwargs: Any) -> ChaosResult:
    """Run an agent through whichever of `run`/`arun` it needs.

    A suite names `module:attr` and does not get to pick, so the loop picks. Calling a
    coroutine function from the sync path builds a coroutine, never awaits it, and
    reports a verdict about a run that did not happen -- exactly the failure this
    library exists to catch (D-114).

    Args:
        engine: The engine to run on.
        agent: The resolved entrypoint.
        **kwargs: Forwarded to `run` or `arun`.

    Returns:
        The `ChaosResult`.
    """
    if inspect.iscoroutinefunction(agent):
        return cast("ChaosResult", asyncio.run(engine.arun(agent, **kwargs)))
    return cast("ChaosResult", engine.run(agent, **kwargs))


def _shared_baseline(cache: dict[str, Any], scenario: Scenario, out_dir: Path) -> Any:
    """Run the scenario's entrypoint unfaulted once, and reuse it.

    Args:
        cache: The per-suite cache, keyed by entrypoint and inputs.
        scenario: The scenario needing a baseline.
        out_dir: Where run directories are written.

    Returns:
        The baseline `ChaosResult`, or `None` when it could not be produced. A
        baseline is a convenience, not a requirement: failing to get one must not
        fail the scenario it was meant to inform.
    """
    from .engine import ChaosEngine

    key = _baseline_key(scenario)
    if key in cache:
        return cache[key]
    try:
        engine = ChaosEngine(
            seed=scenario.seed,
            out_dir=out_dir,
            limits=scenario.limits,
            dry_run=True,
            write_bundle=False,
            strict_schema=False,
            judge="rules",
        )
        cache[key] = drive(
            engine,
            resolve_entrypoint(scenario.entrypoint, engine),
            inputs=scenario.inputs,
            initial_state=dict(scenario.initial_state or {}) or None,
            scenario_id=f"{scenario.id}.baseline",
            expected_behavior="ignore_and_continue",
        )
    except Exception as exc:
        log.info("baseline for %s could not be produced: %s", scenario.id, exc)
        cache[key] = None
    return cache[key]


def run_suite(
    scenarios: Sequence[Scenario],
    *,
    out_dir: Path,
    seed: int | None = None,
    attempt: int = 1,
    no_baseline: bool = False,
    fail_fast: bool = False,
    jobs: int = 1,
    judge: Any | str | None = None,
    judge_options: Mapping[str, Any] | None = None,
    narrate_all: bool = False,
    allow_remote_judge: bool = False,
    on_result: Callable[[ChaosResult], None] | None = None,
) -> list[ChaosResult]:
    """Run every scenario once, in order.

    The single-round path for `alc run` and one round of `RefinementLoop`. Kept in
    one place on purpose: the loop's guarantee that "the same seeds run every round"
    is only true if both callers plan a scenario identically (D-74).

    Args:
        scenarios: The scenarios to run, already expanded and filtered.
        out_dir: Where run bundles are written.
        seed: Override every scenario's seed. `None` keeps each scenario's own.
        attempt: Threaded into `run_id`; `RefinementLoop` passes the round number so
            re-running a scenario does not collide with its earlier bundle (D-04).
        no_baseline: Skip the unfaulted reference run.
        fail_fast: Stop after the first failing scenario. Forces serial execution,
            since "stop at the first failure" needs an order to be first in.
        jobs: How many scenarios to run at once. Scenarios are independent runs with
            their own engine, and `run_id` derives from
            `(seed, scenario_id, plan_hash, attempt)` rather than from order -- so a
            verdict cannot depend on scheduling, and a test asserts that. Baselines
            are computed serially first, because they are shared and would otherwise
            race.
        judge: Judge selection, per `ChaosEngine(judge=…)`.
        judge_options: Extra keyword arguments for a constructed `SLMJudge`.
        narrate_all: Narrate passing runs too.
        allow_remote_judge: Consent to a non-loopback judge endpoint (D-22).
        on_result: Called with each result as it completes, for progress output.

    Returns:
        One result per scenario that ran.
    """
    from .engine import ChaosEngine

    out_dir = Path(out_dir)
    results: list[ChaosResult] = []
    # One baseline per (entrypoint, inputs). Recomputing it per scenario would double
    # the cost of every suite for an answer that cannot have changed.
    baselines: dict[str, Any] = {}

    planned = [s.id for s in scenarios]

    def _publish(status: str, current: str | None) -> None:
        """Write `suite.json` so a viewer can see the run in progress.

        The engine does not know the dashboard exists and never will; this is a file
        the suite writes for anyone who cares to read it (`docs/10` §1).

        Args:
            status: `running` or `completed`.
            current: The scenario in flight, or `None`.
        """
        from .bundle import write_suite_json

        try:
            write_suite_json(
                out_dir,
                list(results),
                seed=seed or 0,
                status=status,
                current=current,
                planned=planned,
            )
        except Exception as exc:  # pragma: no cover - disk trouble only
            log.debug("could not publish suite.json: %s", exc)

    def _one(scenario: Scenario) -> ChaosResult:
        """Run a single scenario in its own engine."""
        engine = ChaosEngine(
            seed=seed if seed is not None else scenario.seed,
            out_dir=out_dir,
            limits=scenario.limits,
            dry_run=scenario.dry_run,
            allow_side_effects=scenario.allow_side_effects,
            strict_schema=False,
            judge=judge,
            judge_options=dict(judge_options or {}),
            narrate_all=narrate_all,
            allow_remote_judge=allow_remote_judge,
            intensity=scenario.intensity,
        )
        specs, dial_skipped = plan_specs(scenario)
        for spec in specs:
            engine.register_fault(
                build_fault(spec), origin=str(spec.get("origin") or ""), **target_kwargs(spec)
            )
        if dial_skipped:
            log.debug("intensity %s skipped %s", scenario.intensity, dial_skipped)
        return drive(
            engine,
            resolve_entrypoint(scenario.entrypoint, engine),
            inputs=scenario.inputs,
            initial_state=dict(scenario.initial_state or {}) or None,
            scenario_id=scenario.id,
            scenario_title=scenario.title,
            scenario_description=scenario.description,
            expected_behavior=scenario.expected_behavior,
            must_not=scenario.must_not,
            expect=scenario.expect,
            expected_errors=scenario.expected_errors,
            allow_side_effects=scenario.allow_side_effects,
            baseline=baselines.get(_baseline_key(scenario)),
            attempt=attempt,
        )

    _publish("running", planned[0] if planned else None)

    if not no_baseline:
        # Serially, before anything runs in parallel: the cache is shared, and two
        # workers racing to fill the same key would run the baseline twice and give
        # two scenarios different references.
        for scenario in scenarios:
            if not scenario.dry_run:
                _shared_baseline(baselines, scenario, out_dir)

    if jobs > 1 and not fail_fast and len(scenarios) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            # Mapped in submission order, so the output does not depend on which
            # worker finished first. `_ACTIVE` is a ContextVar, which is per-thread,
            # so two runs cannot see each other's state.
            results = list(pool.map(_one, scenarios))
        for result in results:
            if on_result is not None:
                on_result(result)
        _publish("completed", None)
        return results

    for index, scenario in enumerate(scenarios):
        _publish("running", scenario.id)
        result = _one(scenario)
        results.append(result)
        remaining = planned[index + 1] if index + 1 < len(planned) else None
        _publish("running" if remaining else "completed", remaining)
        if on_result is not None:
            on_result(result)
        if fail_fast and not result.success:
            break
    _publish("completed", None)
    return results


# ---------------------------------------------------------------------- the report


@dataclass(slots=True)
class RoundResult:
    """One round of the suite.

    Attributes:
        round: 1-based round number.
        results: Every scenario's result, in run order.
        outcomes: Scenario id to its classification against the previous round.
        wall_ms: Wall time for the round. A timing field.
    """

    round: int
    results: list[ChaosResult] = field(default_factory=list)
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    wall_ms: int = 0

    @property
    def passed(self) -> int:
        """How many scenarios passed.

        Returns:
            The count.
        """
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        """How many scenarios failed.

        Returns:
            The count.
        """
        return sum(1 for r in self.results if not r.success)

    @property
    def failure_modes(self) -> dict[str, int]:
        """Histogram over the failing scenarios.

        Returns:
            Failure mode to count.
        """
        modes: dict[str, int] = {}
        for result in self.results:
            if not result.success:
                modes[result.failure_mode] = modes.get(result.failure_mode, 0) + 1
        return modes

    def to_dict(self) -> dict[str, Any]:
        """Serialize to `suite.schema.json`'s `rounds[]` shape.

        Returns:
            A plain dict.
        """
        return {
            "round": self.round,
            "passed": self.passed,
            "failed": self.failed,
            "wall_ms": self.wall_ms,
            "failure_modes": self.failure_modes,
            "outcomes": dict(self.outcomes),
        }


@dataclass(slots=True)
class LoopReport:
    """What changed across rounds: what was fixed, what regressed, what was written.

    Field names are frozen by `docs/02-API.md` §8.
    """

    rounds: list[RoundResult] = field(default_factory=list)
    new_failures_per_round: list[int] = field(default_factory=list)
    fixed_between_rounds: list[list[str]] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    tasks_written: list[Path] = field(default_factory=list)
    out_dir: Path = Path(".chaos")
    tamper: dict[str, Any] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str | None = None
    wall_ms: int = 0

    @property
    def flipped(self) -> list[dict[str, Any]]:
        """Scenarios that went from failing to passing, with the round they did it in.

        Returns:
            One entry per genuine flip. A flip whose fingerprint changed is excluded
            -- it is in `regressions` instead.
        """
        return [
            {"scenario_id": scenario_id, "round": index + 1}
            for index, ids in enumerate(self.fixed_between_rounds)
            for scenario_id in ids
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the loop report.

        Returns:
            The report as a plain dict, JSON-serializable throughout.
        """
        return {
            "rounds": [r.to_dict() for r in self.rounds],
            "new_failures_per_round": list(self.new_failures_per_round),
            "fixed_between_rounds": [list(ids) for ids in self.fixed_between_rounds],
            "flipped": self.flipped,
            "regressions": list(self.regressions),
            "tasks_written": [str(p) for p in self.tasks_written],
            "out_dir": str(self.out_dir),
            "tamper": dict(self.tamper),
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "wall_ms": self.wall_ms,
        }

    def markdown(self) -> str:
        """Render the round-by-round table a human reads after an unattended loop.

        Returns:
            A markdown document: header totals, the per-scenario table, regressions,
            and the work orders written.
        """
        return _render_markdown(self)


def _outcome_label(history: Sequence[tuple[int, Outcome]], result: ChaosResult | None) -> str:
    """Describe a scenario's final state in the words the table uses.

    A regression outranks whatever came after it. Labelling from the last round alone
    hid "regressed in r2" behind r3's "still failing", which is exactly the fact a
    human is scanning the table for.

    Args:
        history: ``(round, outcome)`` for every round the scenario ran in, in order.
        result: Its last result, for the failure mode.

    Returns:
        The `outcome` cell.
    """
    if not history:
        return "skipped"
    # A flip is the fact a human is scanning this column for, and it happened in some
    # earlier round -- reading only the last round turns "fixed in r2" into a bland
    # "passing" by r3, and buries "REGRESSION in r2" under "still failing".
    regression = next((n for n, outcome in history if outcome == "flipped_to_fail"), None)
    if regression is not None:
        return f"**REGRESSION in r{regression}**"
    fixed = next((n for n, outcome in history if outcome == "flipped_to_pass"), None)
    if fixed is not None and history[-1][1] in {"pass", "flipped_to_pass"}:
        return f"fixed in r{fixed}"
    outcome = history[-1][1]
    if outcome in {"fail", "still_failing"}:
        mode = result.failure_mode if result else "unknown"
        return f"still failing ({mode})"
    if outcome == "skipped":
        return "skipped"
    return "passing"


def _render_markdown(report: LoopReport) -> str:
    """Build the loop report's markdown.

    Args:
        report: The finished report.

    Returns:
        The document.
    """
    rounds = report.rounds
    order: list[str] = []
    for round_result in rounds:
        for result in round_result.results:
            name = result.scenario_id or result.run_id
            if name not in order:
                order.append(name)

    lines = ["# Refinement loop", ""]
    if report.aborted:
        lines += [f"> **ABORTED** — {report.abort_reason}", ""]

    last = rounds[-1] if rounds else None
    lines += [
        f"- rounds: {len(rounds)}",
        f"- final round: {last.passed if last else 0} passed, {last.failed if last else 0} failed",
        f"- fixed: {len(report.flipped)}",
        f"- regressions: {len(report.regressions)}",
        f"- judge disagreement: {_disagreement_rate(rounds):.0%}",
        f"- wall time: {report.wall_ms} ms",
        "",
    ]

    headers = " | ".join(f"r{r.round}" for r in rounds)
    lines += [
        f"| scenario | {headers} | outcome |",
        "|---" * (len(rounds) + 2) + "|",
    ]
    for name in order:
        cells: list[str] = []
        history: list[tuple[int, Outcome]] = []
        last_result: ChaosResult | None = None
        for round_result in rounds:
            match = next(
                (r for r in round_result.results if (r.scenario_id or r.run_id) == name), None
            )
            if match is None:
                cells.append("—")
                continue
            cells.append("PASS" if match.success else "FAIL")
            history.append((round_result.round, round_result.outcomes.get(name, "skipped")))
            last_result = match
        label = _outcome_label(history, last_result)
        lines.append(f"| {name} | {' | '.join(cells)} | {label} |")

    if report.regressions:
        lines += ["", "## Regressions", ""]
        lines += [f"- {line}" for line in report.regressions]

    lines += ["", "## Work orders written", ""]
    if report.tasks_written:
        lines += [f"- `{path}`" for path in report.tasks_written]
    else:
        lines.append("_No AGENT_TASK.md was written; every scenario passed._")
    lines += [
        "",
        "Hand a work order to a coding agent as-is. Do not edit the scenario, the "
        "`must_not` list, or the probes to make a run pass: the loop hashes all three "
        "between rounds and reports a flip that follows a change as a regression.",
        "",
    ]
    return "\n".join(lines)


def _disagreement_rate(rounds: Sequence[RoundResult]) -> float:
    """Fraction of judged runs where the model contradicted the probes.

    Args:
        rounds: Every round.

    Returns:
        The rate, or 0.0 when nothing was judged.
    """
    judged = [r for round_ in rounds for r in round_.results if (r.verdict or {}).get("judge_meta")]
    if not judged:
        return 0.0
    return sum(1 for r in judged if (r.verdict or {}).get("judge_disagreement")) / len(judged)


# ------------------------------------------------------------------------ the loop


class RefinementLoop:
    """Runs a suite repeatedly, handing findings to a caller between rounds.

    The loop never edits source. `on_findings` is deliberately dumb: it receives the
    failing results and may do whatever it likes, including shelling out to a coding
    agent. What the loop guarantees is that the next round runs the *same* scenarios
    with the *same* seeds, and that a scenario which changed shape in the meantime is
    reported as a regression rather than a fix.
    """

    def __init__(
        self,
        suite: ChaosSuite,
        *,
        engine: Any | None = None,
        max_rounds: int = 3,
        stop_when: StopWhen = "no_new_failures",
        on_findings: Callable[[Sequence[ChaosResult]], None] | None = None,
        out_dir: str | Path = ".chaos",
        judge: Any | str | None = None,
        judge_options: Mapping[str, Any] | None = None,
        no_baseline: bool = False,
    ) -> None:
        """Initialise a loop.

        Args:
            suite: The suite to run each round.
            engine: A configured `ChaosEngine`. Its `out_dir` and judge settings are
                used for every round; the loop still constructs a fresh engine per
                scenario, because a fault plan is per-scenario state.
            max_rounds: Hard cap on rounds.
            stop_when: Termination condition.
            on_findings: Hand-off hook. The caller may shell out to a coding agent,
                apply patches and return; the loop then re-runs and reports what
                flipped. A scenario whose `plan_hash` changed between rounds is
                reported as a regression, not a fix.
            out_dir: Where run bundles and `suite.json` are written. Ignored when
                `engine` is given, which carries its own.
            judge: Judge selection, per `ChaosEngine(judge=…)`.
            judge_options: Extra keyword arguments for a constructed `SLMJudge`.
            no_baseline: Skip the unfaulted reference run each round.
        """
        self.suite = suite
        self.engine = engine
        self.max_rounds = max(1, int(max_rounds))
        self.stop_when: StopWhen = stop_when
        self.on_findings = on_findings
        self.out_dir = Path(getattr(engine, "out_dir", None) or out_dir)
        self.judge = judge if engine is None else engine.judge
        self.judge_options = dict(judge_options or getattr(engine, "judge_options", {}) or {})
        self.no_baseline = no_baseline

    def run(self) -> LoopReport:
        """Run rounds until `stop_when` is satisfied or `max_rounds` is reached.

        Returns:
            The `LoopReport`. A failing control scenario or a detected probe-module
            change stops the loop early with `aborted` set; every other kind of
            trouble is recorded and the loop continues, because a loop that dies on
            the first surprise is not useful unattended.
        """
        report = LoopReport(out_dir=self.out_dir)
        fingerprints: dict[str, dict[str, Any]] = {}
        modified: set[str] = set()
        probes_before = probes_fingerprint()
        # Scenario id to its latest work order, so the list stays one-per-bug.
        tasks: dict[str, Path] = {}
        started = time.monotonic()
        previous: dict[str, bool] = {}

        for round_number in range(1, self.max_rounds + 1):
            round_started = time.monotonic()
            results = run_suite(
                list(self.suite.scenarios),
                out_dir=self.out_dir,
                attempt=round_number,
                no_baseline=self.no_baseline,
                judge=self.judge,
                judge_options=self.judge_options,
            )
            round_result = RoundResult(round=round_number, results=results)
            round_result.wall_ms = int((time.monotonic() - round_started) * 1000)

            changed = self._fingerprint_round(results, fingerprints, report)
            modified |= changed
            round_result.outcomes = self._classify(results, previous, changed, report, round_number)
            report.rounds.append(round_result)
            report.new_failures_per_round.append(
                sum(
                    1
                    for r in results
                    if not r.success and previous.get(str(r.scenario_id), True) is not False
                )
                if round_number > 1
                else round_result.failed
            )
            report.fixed_between_rounds.append(
                sorted(
                    str(name)
                    for name, outcome in round_result.outcomes.items()
                    if outcome == "flipped_to_pass"
                )
            )
            # Keyed by scenario, not appended: every round rewrites a still-failing
            # scenario's work order into a fresh run directory, and listing all of
            # them hands a coding agent two orders for one bug -- the older
            # describing a run that has since been superseded. A scenario that has
            # started passing drops out entirely.
            for result in results:
                name = str(result.scenario_id or result.run_id)
                task = result.artifacts.get("agent_task")
                if task:
                    tasks[name] = Path(task)
                else:
                    tasks.pop(name, None)
            report.tasks_written = list(tasks.values())
            previous = {str(r.scenario_id): r.success for r in results}

            probes_now = probes_fingerprint()
            if probes_now != probes_before:
                report.regressions.append(
                    f"the probe module changed during the loop ({probes_before[:8]} -> "
                    f"{probes_now[:8]}); results from this round are not comparable"
                )
                probes_before = probes_now
            report.tamper = {
                "probes_sha256": probes_now,
                "probes_changed": any("probe module changed" in r for r in report.regressions),
                "control_failed": None,
                "modified": sorted(modified),
                "scenarios": fingerprints,
            }

            control = self._failing_control(results)
            if control is not None:
                report.aborted = True
                report.abort_reason = (
                    f"control scenario {control!r} failed. A control must pass in every round; "
                    "its failure means the harness itself was altered, not the agent. "
                    "Nothing after this point can be trusted, so the loop stopped."
                )
                report.tamper["control_failed"] = control
                report.regressions.append(report.abort_reason)
                break

            if round_number == self.max_rounds or self._should_stop(report, round_result):
                break
            self._hand_off(results, report)

        report.wall_ms = int((time.monotonic() - started) * 1000)
        self._write_suite_json(report)
        return report

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _failing_control(results: Sequence[ChaosResult]) -> str | None:
        """Find a failing control scenario.

        Args:
            results: This round's results.

        Returns:
            The first failing control's id, or `None`.
        """
        for result in results:
            name = str(result.scenario_id or "")
            if name.startswith(_CONTROL_PREFIX) and not result.success:
                return name
        return None

    def _fingerprint_round(
        self,
        results: Sequence[ChaosResult],
        fingerprints: dict[str, dict[str, Any]],
        report: LoopReport,
    ) -> set[str]:
        """Record each scenario's fingerprint and report what changed since last round.

        Args:
            results: This round's results.
            fingerprints: The store, updated in place.
            report: The report, for regression lines.

        Returns:
            Ids whose fingerprint changed.
        """
        by_id = {s.id: s for s in self.suite.scenarios}
        changed: set[str] = set()
        for result in results:
            name = str(result.scenario_id or "")
            scenario = by_id.get(name)
            if scenario is None:
                continue
            current = scenario_fingerprint(scenario, result.plan_hash)
            previous = fingerprints.get(name)
            fingerprints[name] = current
            if previous is None or previous == current:
                continue
            changed.add(name)
            for key, phrase in (
                ("plan_hash", "plan_hash changed"),
                ("scenario_file", "scenario file changed"),
                ("must_not", "must_not list changed"),
            ):
                if previous.get(key) != current.get(key):
                    report.regressions.append(f"{name}: {phrase} between rounds")
        return changed

    @staticmethod
    def _classify(
        results: Sequence[ChaosResult],
        previous: Mapping[str, bool],
        changed: set[str],
        report: LoopReport,
        round_number: int,
    ) -> dict[str, Outcome]:
        """Classify each scenario against the previous round.

        A flip to pass by a scenario whose fingerprint changed is **not** a fix. It
        is the single case this whole module exists to catch, so it is recorded as a
        regression and classified `still_failing` -- claiming otherwise would let a
        deleted fault read as a green run (`docs/05` §9).

        Args:
            results: This round's results.
            previous: Scenario id to its previous success, empty in round 1.
            changed: Ids whose fingerprint changed this round.
            report: The report, for regression lines.
            round_number: Which round this is.

        Returns:
            Scenario id to outcome.
        """
        outcomes: dict[str, Outcome] = {}
        for result in results:
            name = str(result.scenario_id or result.run_id)
            was = previous.get(name)
            if was is None:
                outcomes[name] = "pass" if result.success else "fail"
                continue
            if result.success and not was:
                if name in changed:
                    report.regressions.append(
                        f"{name}: passed after the scenario was modified "
                        f"(round {round_number}); counted as still failing, not as a fix"
                    )
                    outcomes[name] = "still_failing"
                else:
                    outcomes[name] = "flipped_to_pass"
            elif not result.success and was:
                report.regressions.append(
                    f"{name}: regressed in round {round_number} "
                    f"({result.failure_mode}); it passed in round {round_number - 1}"
                )
                outcomes[name] = "flipped_to_fail"
            elif result.success:
                outcomes[name] = "pass"
            else:
                outcomes[name] = "still_failing"
        return outcomes

    def _should_stop(self, report: LoopReport, round_result: RoundResult) -> bool:
        """Decide whether the loop has done its job.

        Args:
            report: The report so far.
            round_result: The round just completed.

        Returns:
            True to stop.
        """
        if self.stop_when == "all_pass":
            return round_result.failed == 0
        if self.stop_when == "no_new_failures":
            return len(report.rounds) > 1 and report.new_failures_per_round[-1] == 0
        return False

    def _hand_off(self, results: Sequence[ChaosResult], report: LoopReport) -> None:
        """Give the failing results to the caller.

        Args:
            results: This round's results.
            report: The report, for recording a hook that blew up.
        """
        if self.on_findings is None:
            return
        failures = [r for r in results if not r.success]
        if not failures:
            return
        try:
            self.on_findings(failures)
        except Exception as exc:
            # The hook is caller code shelling out to a coding agent. Its failure is
            # information, not a reason to abandon a loop that may still be useful.
            log.warning("on_findings raised: %s", exc)
            report.regressions.append(f"on_findings raised {type(exc).__name__}: {exc}")

    def _write_suite_json(self, report: LoopReport) -> None:
        """Write the enriched `suite.json` for the loop.

        Args:
            report: The finished report.
        """
        from .bundle import write_suite_json

        last = report.rounds[-1] if report.rounds else None
        try:
            write_suite_json(
                self.out_dir,
                last.results if last else [],
                seed=self.suite.scenarios[0].seed if self.suite.scenarios else 0,
                loop=report,
                rounds_planned=self.max_rounds,
            )
        except Exception as exc:  # pragma: no cover - disk trouble only
            log.warning("could not write suite.json: %s", exc)
