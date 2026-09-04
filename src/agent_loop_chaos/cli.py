"""The `alc` command-line interface.

Argument parsing is complete in M0; every subcommand except `--version` reports that
it is not implemented and exits 2. The behaviour arrives with the phases that own
it, and `alc` is finished in M9.

Exit codes are a contract, because CI pipes them: 0 all passed, 1 at least one
scenario failed, 2 configuration or usage error, 3 internal error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .errors import ConfigError, MissingExtraError
from .version import __version__

__all__ = ["build_parser", "main"]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERNAL = 3

log = logging.getLogger("agent_loop_chaos")


def build_parser() -> argparse.ArgumentParser:
    """Build the full `alc` argument parser.

    Every subcommand in `docs/02-API.md` §10 is present, so `--help` is accurate
    from M0 onward even where the implementation is not.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="alc",
        description="Break your agent on purpose and hand the bug report to your coding agent.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="raise log verbosity; repeatable"
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    run = sub.add_parser("run", help="run a suite, a scenario, or a module:attr entrypoint")
    run.add_argument("target", help="suite.yaml | scenario.yaml | module:attr")
    run.add_argument("--seed", type=int, help="override the seed")
    run.add_argument("--jobs", type=int, default=1, help="parallel scenarios")
    run.add_argument("--judge", choices=("rules", "slm", "ensemble"), help="judge to use")
    run.add_argument("--model", help="model id for the SLM judge")
    run.add_argument("--base-url", dest="base_url", help="judge endpoint")
    run.add_argument("--out", help="output directory (default .chaos)")
    run.add_argument(
        "--trace-level", dest="trace_level", choices=("minimal", "standard", "verbose")
    )
    run.add_argument("--filter", help="glob over scenario ids")
    run.add_argument("--fail-fast", dest="fail_fast", action="store_true")
    run.add_argument("--no-baseline", dest="no_baseline", action="store_true")
    run.add_argument("--dry-run", dest="dry_run", action="store_true")
    run.add_argument("--json", action="store_true", help="emit one JSON object and nothing else")
    run.add_argument("--rounds", type=int, help="run the refinement loop for N rounds")
    run.add_argument(
        "--stop-when", dest="stop_when", choices=("all_pass", "no_new_failures", "rounds")
    )
    run.add_argument("--dashboard", action="store_true", help="serve the live dashboard (M10)")
    run.add_argument("--port", type=int, default=7717)
    run.add_argument("--linger", type=int, help="seconds to keep serving after the run")

    replay = sub.add_parser("replay", help="re-run a stored plan")
    replay.add_argument("run_dir")
    replay.add_argument("--seed", type=int)

    judge = sub.add_parser("judge", help="re-judge a run without re-running the agent")
    judge.add_argument("run_dir")
    judge.add_argument("--judge", dest="judge_kind", choices=("rules", "slm", "ensemble"))
    judge.add_argument("--model")

    explain = sub.add_parser("explain", help="human-readable narrative to stdout")
    explain.add_argument("run_dir")

    report = sub.add_parser("report", help="render a run or suite directory")
    report.add_argument("path")
    report.add_argument("--format", choices=("md", "json", "html"), default="md")
    report.add_argument("-o", "--output", help="write to a file instead of stdout")
    report.add_argument("--max-trace-events", dest="max_trace_events", type=int)

    validate = sub.add_parser("validate", help="validate a report or a suite file")
    validate.add_argument("path")

    list_faults = sub.add_parser("list-faults", help="list the registered fault kinds")
    list_faults.add_argument("--json", action="store_true")

    sub.add_parser("init", help="scaffold a chaos/ directory and an example scenario")

    dashboard = sub.add_parser("dashboard", help="serve the read-only trace dashboard (M10)")
    dashboard.add_argument("--out", help="run directory to serve")
    dashboard.add_argument("--port", type=int, default=7717)
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--open", action="store_true")
    dashboard.add_argument("--poll-ms", dest="poll_ms", type=int, default=250)
    dashboard.add_argument("--max-events", dest="max_events", type=int)
    dashboard.add_argument("--no-sse", dest="no_sse", action="store_true")
    dashboard.add_argument("--once", action="store_true")

    return parser


def _list_faults(*, as_json: bool) -> int:
    """Print the registered fault catalog.

    Args:
        as_json: Emit one JSON document and nothing else, so CI can pipe it.

    Returns:
        `EXIT_OK`.
    """
    # Imported here so `alc --version` does not pay for building the registry.
    from .faults import list_faults

    infos = list_faults()
    if as_json:
        payload = [
            {
                "kind": info.kind,
                "accepts": list(info.layer_phases),
                "severity_hint": info.severity_hint,
                "summary": info.summary,
            }
            for info in infos
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_OK

    width = max((len(info.kind) for info in infos), default=0)
    for info in infos:
        accepts = ", ".join(info.layer_phases)
        print(f"{info.kind:<{width}}  {accepts}")
        print(f"{'':<{width}}  {info.severity_hint}: {info.summary}")
    print(f"\n{len(infos)} fault kinds registered.")
    return EXIT_OK


def _load_report(run_dir: Path) -> tuple[dict[str, Any], Path]:
    """Read a stored report, accepting either a run directory or the file itself.

    Args:
        run_dir: A run directory or a path to `report.json`.

    Returns:
        ``(report, path)``.

    Raises:
        ConfigError: When no report can be found.
    """
    path = run_dir if run_dir.is_file() else run_dir / "report.json"
    if not path.is_file():
        raise ConfigError(f"no report.json found at {run_dir}")
    return json.loads(path.read_text(encoding="utf-8")), path


def _run(args: argparse.Namespace) -> int:
    """Run a suite and write its bundles.

    Args:
        args: Parsed arguments.

    Returns:
        `EXIT_OK` when every scenario passed, `EXIT_FAILED` otherwise.
    """
    from .bundle import write_suite_json
    from .engine import ChaosEngine
    from .scenarios import load_suite

    suite = load_suite(args.target)
    scenarios = [s for s in suite.scenarios if _selected(s.id, args.filter)]
    out_dir = Path(args.out or ".chaos")
    results = []

    for scenario in scenarios:
        engine = ChaosEngine(
            seed=args.seed if args.seed is not None else scenario.seed,
            out_dir=out_dir,
            limits=scenario.limits,
            dry_run=scenario.dry_run,
            allow_side_effects=scenario.allow_side_effects,
            strict_schema=False,
        )
        for spec in scenario.faults:
            engine.register_fault(_build_fault(spec), **_target_kwargs(spec))
        result = engine.run(
            _resolve_entrypoint(scenario.entrypoint, engine),
            inputs=scenario.inputs,
            initial_state=dict(scenario.initial_state or {}) or None,
            scenario_id=scenario.id,
            expected_behavior=scenario.expected_behavior,
            must_not=scenario.must_not,
            expect=scenario.expect,
            expected_errors=scenario.expected_errors,
            allow_side_effects=scenario.allow_side_effects,
        )
        results.append(result)
        if not args.json:
            print(result.summary_line())
        if args.fail_fast and not result.success:
            break

    write_suite_json(out_dir, results, seed=args.seed or 0)
    if args.json:
        print(
            json.dumps(
                {
                    "passed": sum(1 for r in results if r.success),
                    "failed": sum(1 for r in results if not r.success),
                    "results": [
                        {
                            "scenario_id": r.scenario_id,
                            "success": r.success,
                            "failure_mode": r.failure_mode,
                            "run_dir": r.artifacts.get("run_dir"),
                        }
                        for r in results
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return EXIT_OK if all(r.success for r in results) else EXIT_FAILED


def _selected(scenario_id: str, pattern: str | None) -> bool:
    """Report whether a scenario matches the `--filter` glob.

    Args:
        scenario_id: The scenario's id.
        pattern: The glob, or `None` for everything.

    Returns:
        True when the scenario should run.
    """
    from fnmatch import fnmatch

    if not pattern:
        return True
    return fnmatch(scenario_id, pattern) or fnmatch(scenario_id, f"*{pattern}*")


def _build_fault(spec: dict[str, Any]) -> Any:
    """Construct a fault from a scenario's fault spec.

    Args:
        spec: The spec.

    Returns:
        The fault instance.
    """
    from .faults.base import fault_from_dict

    return fault_from_dict(spec)


def _target_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
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


def _resolve_entrypoint(entrypoint: Any, engine: Any) -> Any:
    """Import a ``module:attr`` entrypoint, building it when it wants the engine.

    A vanilla agent's tools have to be wrapped by *this run's* engine before any tool
    fault can fire, and a bare function has no way to do that. The convention: a
    resolved callable whose first parameter is named `engine` is a **builder** -- it
    is handed the engine and returns the real agent. Anything else is the agent
    itself.

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


def _report(args: argparse.Namespace) -> int:
    """Render a stored report, with no re-run.

    Args:
        args: Parsed arguments.

    Returns:
        `EXIT_OK`.
    """
    from .bundle import render_agent_task
    from .report import ChaosResult

    report, _path = _load_report(Path(args.path))
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return EXIT_OK
    rendered = render_agent_task(ChaosResult.from_dict(report))
    print(rendered or f"# {report.get('scenario_id')}\n\nThis run passed; no work order.")
    return EXIT_OK


def _validate(args: argparse.Namespace) -> int:
    """Validate a report or a suite file against its schema.

    Args:
        args: Parsed arguments.

    Returns:
        `EXIT_OK` when valid, `EXIT_FAILED` with the errors printed otherwise.
    """
    from .schema import validate_obj

    path = Path(args.path)
    if not path.is_file():
        raise ConfigError(f"no such file: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        from .scenarios import load_suite

        load_suite(path)
        print(f"{path}: valid suite")
        return EXIT_OK

    document = json.loads(path.read_text(encoding="utf-8"))
    name = "scenario" if "scenarios" in document else "report"
    errors = validate_obj(document, name)  # type: ignore[arg-type]
    if errors:
        for error in errors:
            print(f"{path}: {error}", file=sys.stderr)
        return EXIT_FAILED
    print(f"{path}: valid {name}")
    return EXIT_OK


def _explain(args: argparse.Namespace) -> int:
    """Print a human-readable narrative of a stored run.

    Auto-assertions are listed explicitly, so an author can see what was checked and
    override it (`docs/11` §4.4).

    Args:
        args: Parsed arguments.

    Returns:
        `EXIT_OK`.
    """
    report, _path = _load_report(Path(args.run_dir))
    verdict = report.get("verdict") or {}
    status = "PASS" if report.get("success") else "FAIL"
    print(f"{report.get('scenario_id') or report.get('run_id')}: {status}")
    print(f"  expected  {verdict.get('expected_behavior')}")
    print(f"  observed  {verdict.get('observed_behavior')}")
    print(f"  mode      {report.get('failure_mode')} (severity {report.get('severity')})")
    print(f"\n  {report.get('chaos_narrative') or 'No narrative was recorded.'}")

    print("\n  assertions:")
    for assertion in report.get("assertions") or []:
        mark = "ok  " if assertion.get("ok") else "FAIL"
        source = assertion.get("source", "scenario")
        print(f"    {mark} {assertion.get('check')} ({source}) — {assertion.get('detail', '')}")
    if not report.get("assertions"):
        print("    (none)")

    print("\n  probes:")
    for symptom in report.get("symptoms") or []:
        seqs = ", ".join(str(e.get("seq")) for e in symptom.get("evidence") or [])
        print(f"    {symptom.get('severity'):<8} {symptom.get('code')} @ seq {seqs}")
    if not report.get("symptoms"):
        print("    (none fired)")
    return EXIT_OK


def _configure_logging(verbosity: int) -> None:
    """Set the library logger's level from a repeat-count flag.

    Args:
        verbosity: How many times ``-v`` was given.
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `alc` console script.

    Args:
        argv: Argument vector, defaulting to `sys.argv[1:]`.

    Returns:
        A process exit code: 0 all passed, 1 a scenario failed, 2 configuration or
        usage error, 3 internal error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        return EXIT_USAGE

    if args.command == "list-faults":
        return _list_faults(as_json=bool(args.json))

    handlers = {
        "run": _run,
        "report": _report,
        "validate": _validate,
        "explain": _explain,
    }
    handler = handlers.get(args.command)
    if handler is not None:
        try:
            return handler(args)
        except (ConfigError, MissingExtraError) as exc:
            print(f"alc {args.command}: {exc}", file=sys.stderr)
            return EXIT_USAGE
        except Exception:
            log.exception("alc %s failed", args.command)
            return EXIT_INTERNAL

    # Each remaining command lands with the phase that owns it, and `alc` is
    # complete in M9 (prompts/09-cli-and-release.md).
    print(f"alc {args.command}: not implemented", file=sys.stderr)
    return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())
