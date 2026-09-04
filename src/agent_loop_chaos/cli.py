"""The `alc` command-line interface.

Argument parsing is complete in M0; every subcommand except `--version` reports that
it is not implemented and exits 2. The behaviour arrives with the phases that own
it, and `alc` is finished in M9.

Exit codes are a contract, because CI pipes them: 0 all passed, 1 at least one
scenario failed, 2 configuration or usage error, 3 internal error.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
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
# D-26. A control scenario failed or a scenario changed shape mid-loop: the harness
# itself was altered, so nothing in the report can be trusted.
EXIT_TAMPERED = 4

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
    run.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="run this many scenarios at once (ignored with --fail-fast, which needs an order)",
    )
    run.add_argument("--judge", choices=("rules", "slm", "ensemble"), help="judge to use")
    run.add_argument("--model", help="model id for the SLM judge")
    run.add_argument("--base-url", dest="base_url", help="judge endpoint")
    run.add_argument(
        "--transport", choices=("openai", "ollama", "anthropic"), help="judge wire protocol"
    )
    run.add_argument(
        "--allow-remote-judge",
        dest="allow_remote_judge",
        action="store_true",
        help="consent to a non-loopback judge endpoint receiving payloads and source (D-22)",
    )
    run.add_argument(
        "--narrate-all",
        dest="narrate_all",
        action="store_true",
        help="narrate passing runs too; off by default, one model call per failure",
    )
    run.add_argument("--out", help="output directory (default .chaos)")
    run.add_argument(
        "--trace-level",
        dest="trace_level",
        choices=("minimal", "standard", "verbose"),
        help="how much payload detail to record (default: standard)",
    )
    run.add_argument("--filter", help="glob over scenario ids")
    cassette = run.add_mutually_exclusive_group()
    cassette.add_argument(
        "--record",
        metavar="CASSETTE",
        help="record every model response to a cassette file, so the suite replays later",
    )
    cassette.add_argument(
        "--replay-cassette",
        dest="replay_cassette",
        metavar="CASSETTE",
        help="replay model responses from a cassette; never calls the model",
    )
    run.add_argument(
        "--quiet", "-q", action="store_true", help="summary only, no per-scenario lines"
    )
    run.add_argument("--preset", help="run a named preset instead of a scenario file")
    run.add_argument(
        "--allow-side-effects",
        dest="allow_side_effects",
        help="comma-separated tools a real-action fault may target (SAFETY.md section 1)",
    )
    run.add_argument(
        "--suggest-fixes",
        dest="suggest_fixes",
        action="store_true",
        help="ask the judge for a ranked fix list (one extra model call per failure)",
    )
    run.add_argument(
        "--entrypoint",
        help="override every scenario's entrypoint (MODULE:ATTR); how the same suite "
        "is run against a corrected tree",
    )
    run.add_argument(
        "--fail-fast",
        dest="fail_fast",
        action="store_true",
        help="stop after the first failing scenario",
    )
    run.add_argument(
        "--no-baseline",
        dest="no_baseline",
        action="store_true",
        help="skip the unfaulted reference run",
    )
    run.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="arm every fault but fire none; the run must classify as a clean baseline",
    )
    run.add_argument("--json", action="store_true", help="emit one JSON object and nothing else")
    run.add_argument("--rounds", type=int, help="run the refinement loop for N rounds")
    run.add_argument(
        "--stop-when",
        dest="stop_when",
        choices=("all_pass", "no_new_failures", "rounds"),
        help="when to stop a --rounds loop (default: no_new_failures)",
    )
    run.add_argument("--dashboard", action="store_true", help="serve the live dashboard (M10)")
    run.add_argument("--port", type=int, default=7717, help="dashboard port (M10)")
    run.add_argument("--linger", type=int, help="seconds to keep serving after the run")

    replay = sub.add_parser(
        "replay",
        help="re-run a stored plan",
        description="Rebuild a run's fault plan from its plan.json and run it again.",
        epilog=(
            "example:\n"
            "  alc replay .chaos/tool.drop_key/run-7d303952 \\\n"
            "      --entrypoint examples.trip_planner.app:build_app\n\n"
            "The plan hash covers the plan, not the experiment: your agent's source, "
            "the model and the fixtures all sit outside it. Replay compares the "
            "observed crossing sequence against the original and reports divergence "
            "rather than pretending the reproduction was faithful."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    replay.add_argument("run_dir", help="a run directory containing plan.json")
    replay.add_argument("--seed", type=int, help="override the recorded seed")
    replay.add_argument("--entrypoint", help="the agent to re-run, as MODULE:ATTR (required)")
    replay.add_argument("--out", help="output directory (default: alongside the original)")
    replay.add_argument("--json", action="store_true", help="emit one JSON object")

    judge = sub.add_parser("judge", help="re-judge a run without re-running the agent")
    judge.add_argument("run_dir")
    judge.add_argument(
        "--judge",
        dest="judge_kind",
        choices=("rules", "slm", "ensemble"),
        help="which judge to re-judge with (default: rules)",
    )
    judge.add_argument("--model", help="model id for the SLM judge")
    judge.add_argument("--base-url", dest="base_url", help="judge endpoint")
    judge.add_argument(
        "--transport",
        choices=("openai", "ollama", "anthropic"),
        help="judge wire protocol",
    )
    judge.add_argument(
        "--allow-remote-judge",
        dest="allow_remote_judge",
        action="store_true",
        help="consent to a non-loopback endpoint receiving payloads and source (D-22)",
    )

    explain = sub.add_parser("explain", help="human-readable narrative to stdout")
    explain.add_argument("run_dir")

    report = sub.add_parser("report", help="render a run or suite directory")
    report.add_argument("path")
    report.add_argument(
        "--format",
        choices=("md", "json", "html"),
        default="md",
        help="output format (default: md)",
    )
    report.add_argument("-o", "--output", help="write to a file instead of stdout")
    report.add_argument(
        "--max-trace-events",
        dest="max_trace_events",
        type=int,
        help="cap the trace excerpt at N events",
    )

    validate = sub.add_parser(
        "validate",
        help="validate a report or a suite file",
        description="Check a report.json or a suite file against its JSON Schema.",
        epilog="example:\n  alc validate .chaos/tool.drop_key/run-7d30/report.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate.add_argument("--json", action="store_true", help="emit one JSON object")
    validate.add_argument("path")

    list_faults = sub.add_parser("list-faults", help="list the registered fault kinds")
    list_faults.add_argument("--json", action="store_true", help="emit one JSON object")

    init = sub.add_parser(
        "init",
        help="scaffold a chaos/ directory and an example scenario",
        description="Write chaos/quickstart.yaml and chaos/README.md into this directory.",
        epilog="example:\n  alc init && alc run chaos/quickstart.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    init.add_argument("--dir", default="chaos", help="where to scaffold (default: chaos)")
    init.add_argument("--force", action="store_true", help="overwrite existing files")

    dashboard = sub.add_parser("dashboard", help="serve the read-only trace dashboard (M10)")
    dashboard.add_argument("--out", help="run directory to serve")
    dashboard.add_argument("--port", type=int, default=7717, help="port to serve on")
    dashboard.add_argument("--host", default="127.0.0.1", help="bind address; loopback by default")
    dashboard.add_argument(
        "--open", action="store_true", help="open a browser when the server starts"
    )
    dashboard.add_argument(
        "--poll-ms",
        dest="poll_ms",
        type=int,
        default=250,
        help="how often to re-read the run directory",
    )
    dashboard.add_argument(
        "--max-events", dest="max_events", type=int, help="cap the events held in the page"
    )
    dashboard.add_argument(
        "--no-sse", dest="no_sse", action="store_true", help="poll instead of streaming"
    )
    dashboard.add_argument("--once", action="store_true", help="render a static snapshot and exit")

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
        # An object, not a bare array: `--json` is documented as one JSON *object*
        # (docs/02-API.md section 10), and an object leaves room to add a field
        # without breaking every consumer's parser.
        print(json.dumps({"faults": payload, "count": len(payload)}, indent=2, sort_keys=True))
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
    """Run a suite and write its bundles, once or as a refinement loop.

    Args:
        args: Parsed arguments.

    Returns:
        `EXIT_OK` when every scenario passed, `EXIT_FAILED` otherwise, `EXIT_TAMPERED`
        when the loop detected that the harness itself was altered.
    """
    from .loop import run_suite
    from .scenarios import ChaosSuite, load_suite

    suite = load_suite(args.target)
    scenarios = [s for s in suite.scenarios if _selected(s.id, args.filter)]
    if getattr(args, "entrypoint", None):
        for scenario in scenarios:
            scenario.entrypoint = args.entrypoint
    out_dir = Path(args.out or ".chaos")

    if args.rounds:
        return _run_loop(args, ChaosSuite(scenarios), out_dir)

    server = _serve(args, out_dir) if getattr(args, "dashboard", False) else None

    colour = use_colour() and not args.json
    quiet = getattr(args, "quiet", False)
    total = len(scenarios)
    done = 0

    def report(result: Any) -> None:
        nonlocal done
        done += 1
        if not args.json and not quiet:
            mark = "pass" if result.success else "fail"
            print(_paint(result.summary_line(), mark, colour=colour))
        elif not args.json and total > 5:
            # Progress on stderr, so `alc run > out.txt` still shows movement and
            # `--json` piping stays clean.
            print(f"  [{done}/{total}] {result.scenario_id}", file=sys.stderr)
        if not quiet:
            _warn_if_nothing_fired(result)

    results = run_suite(
        scenarios,
        out_dir=out_dir,
        seed=args.seed,
        no_baseline=args.no_baseline,
        fail_fast=args.fail_fast,
        jobs=max(1, int(getattr(args, "jobs", 1) or 1)),
        judge=args.judge,
        judge_options=_judge_options(args),
        narrate_all=getattr(args, "narrate_all", False),
        allow_remote_judge=getattr(args, "allow_remote_judge", False),
        on_result=report,
    )

    # `run_suite` already published `suite.json` -- live during the run and once more
    # as `completed` at the end (`docs/10` §2). Writing it again here would clobber
    # the 1.1 live document with a 1.0 one.
    if server is not None:
        linger = float(getattr(args, "linger", 0.0) or 0.0)
        if linger:
            print(f"dashboard: serving for {linger:g}s more — ctrl-c to stop", file=sys.stderr)
            with contextlib.suppress(KeyboardInterrupt):
                time.sleep(linger)
        server.stop()
    if not args.json and not quiet:
        _print_summary(results, out_dir, colour=colour)
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


def _run_loop(args: argparse.Namespace, suite: Any, out_dir: Path) -> int:
    """Run the refinement loop with no hand-off hook.

    `alc run --rounds N` writes work orders and re-runs; it never calls a coding
    agent itself. Automating that is what `examples/refine_with_claude_code.py` is
    for, precisely so the decision to let something edit source stays the user's.

    Args:
        args: Parsed arguments.
        suite: The filtered suite.
        out_dir: Where bundles are written.

    Returns:
        `EXIT_TAMPERED` when the loop aborted, `EXIT_FAILED` when the final round
        still has a failure, `EXIT_OK` otherwise.
    """
    from .loop import RefinementLoop

    report = RefinementLoop(
        suite,
        max_rounds=args.rounds,
        stop_when=args.stop_when or "no_new_failures",
        out_dir=out_dir,
        judge=args.judge,
        judge_options=_judge_options(args),
        no_baseline=args.no_baseline,
    ).run()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
    else:
        print(report.markdown())

    if report.aborted:
        print(f"alc run: {report.abort_reason}", file=sys.stderr)
        return EXIT_TAMPERED
    last = report.rounds[-1] if report.rounds else None
    return EXIT_OK if last and last.failed == 0 else EXIT_FAILED


def _shared_baseline(cache: dict[str, Any], scenario: Any, out_dir: Path) -> Any:
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

    key = json.dumps([str(scenario.entrypoint), scenario.inputs], sort_keys=True, default=str)
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
        )
        cache[key] = engine.run(
            resolve_entrypoint(scenario.entrypoint, engine),
            inputs=scenario.inputs,
            initial_state=dict(scenario.initial_state or {}) or None,
            scenario_id=f"{scenario.id}.baseline",
            expected_behavior="ignore_and_continue",
        )
    except Exception:
        log.exception("baseline run failed for %s", scenario.id)
        cache[key] = None
    return cache[key]


def _warn_if_nothing_fired(result: Any) -> None:
    """Warn when a scenario armed faults and none of them fired.

    Such a run still fails -- `completed_unaffected` does not satisfy
    `graceful_degradation` -- but it demonstrates nothing, and without a warning it
    reads as a real finding. The commonest cause is a tool or llm name in the target
    that does not match the agent.

    Args:
        result: The scenario's result.
    """
    faults = result.injected_faults
    if not faults or any(f.get("fired") for f in faults):
        return
    reasons = sorted({str(f.get("skipped_reason")) for f in faults if f.get("skipped_reason")})
    print(
        f"  warning: {result.scenario_id}: no fault fired "
        f"({', '.join(reasons) or 'no reason recorded'}); this scenario proves nothing. "
        "Check the target's tool/llm name.",
        file=sys.stderr,
    )


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


def build_fault(spec: dict[str, Any]) -> Any:
    """Construct a fault from a scenario's fault spec.

    Args:
        spec: The spec.

    Returns:
        The fault instance.
    """
    from .faults.base import fault_from_dict

    return fault_from_dict(spec)


def target_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
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

    if args.format == "html":
        from .dashboard.export import export_html

        path = Path(args.path)
        # A run directory exports as itself; anything else exports the whole suite,
        # so `alc report .chaos --format html` is the common case.
        root = path.parent.parent if (path / "report.json").is_file() else path
        page = export_html(root, max_trace_events=args.max_trace_events or 5_000)
        if args.output:
            Path(args.output).write_text(page, encoding="utf-8")
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            print(page)
        return EXIT_OK

    report, _path = _load_report(Path(args.path))
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return EXIT_OK
    rendered = render_agent_task(ChaosResult.from_dict(report))
    print(rendered or f"# {report.get('scenario_id')}\n\nThis run passed; no work order.")
    return EXIT_OK


def _serve(args: argparse.Namespace, out_dir: Path) -> Any:
    """Start the dashboard, or explain why it could not start.

    Args:
        args: Parsed arguments.
        out_dir: The directory to serve.

    Returns:
        The started server, or `None` when the port was unavailable. A dashboard is
        never a reason for a suite to fail (`docs/10` §1), so a bind failure is a
        message on stderr and nothing more.
    """
    from .dashboard.server import DashboardServer

    server = DashboardServer(
        out_dir,
        host=args.host if getattr(args, "host", None) else "127.0.0.1",
        port=args.port,
        poll_ms=getattr(args, "poll_ms", 250) or 250,
        max_events=getattr(args, "max_events", None) or 20_000,
        sse=not getattr(args, "no_sse", False),
        strict_port="--port" in sys.argv,
    )
    if server.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"alc dashboard: binding {server.host} exposes every run under {out_dir} "
            "to anything that can reach this machine",
            file=sys.stderr,
        )
    try:
        server.start()
    except OSError as exc:
        print(f"alc dashboard: could not bind port {args.port}: {exc}", file=sys.stderr)
        return None
    print(f"dashboard: {server.url}", file=sys.stderr)
    if getattr(args, "open", False) and not (
        os.environ.get("SSH_CONNECTION") or os.environ.get("CI")
    ):
        import webbrowser

        webbrowser.open(server.url)
    return server


def _dashboard(args: argparse.Namespace) -> int:
    """Serve a `.chaos` directory, read-only.

    Args:
        args: Parsed arguments.

    Returns:
        `EXIT_OK`.
    """
    out_dir = Path(args.out or ".chaos")
    if not out_dir.is_dir():
        raise ConfigError(f"no such directory: {out_dir}")

    if args.once:
        from .dashboard import api
        from .dashboard.watcher import RunDirWatcher

        watcher = RunDirWatcher(out_dir, max_events=getattr(args, "max_events", None) or 20_000)
        watcher.poll()
        print(json.dumps({"suite": api.suite(watcher), "runs": api.runs(watcher)}, indent=2))
        return EXIT_OK

    server = _serve(args, out_dir)
    if server is None:
        return EXIT_USAGE
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
    finally:
        server.stop()
    return EXIT_OK


def _validate(args: argparse.Namespace) -> int:
    """Validate a report or a suite file against its schema.

    Args:
        args: Parsed arguments.

    Returns:
        `EXIT_OK` when valid, `EXIT_FAILED` with the errors printed otherwise.
    """
    from .schema import validate_obj

    as_json = getattr(args, "json", False)
    path = Path(args.path)
    if not path.is_file():
        raise ConfigError(f"no such file: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        from .scenarios import load_suite

        suite = load_suite(path)
        if as_json:
            print(
                json.dumps(
                    {
                        "path": str(path),
                        "kind": "suite",
                        "valid": True,
                        "errors": [],
                        "scenarios": len(suite.scenarios),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"{path}: valid suite ({len(suite.scenarios)} scenarios)")
        return EXIT_OK

    document = json.loads(path.read_text(encoding="utf-8"))
    name = "scenario" if "scenarios" in document else "report"
    errors = validate_obj(document, name)  # type: ignore[arg-type]
    if as_json:
        print(
            json.dumps(
                {"path": str(path), "kind": name, "valid": not errors, "errors": errors},
                indent=2,
                sort_keys=True,
            )
        )
        return EXIT_FAILED if errors else EXIT_OK
    if errors:
        for error in errors:
            print(f"{path}: {error}", file=sys.stderr)
        return EXIT_FAILED
    print(f"{path}: valid {name}")
    return EXIT_OK


QUICKSTART = """# Your first chaos scenario.
#
#   alc run chaos/quickstart.yaml --judge rules
#
# Point `entrypoint` at your agent. A callable whose first parameter is named
# `engine` is treated as a *builder*: it is handed the ChaosEngine and returns the
# real agent, which is how your tools get wrapped so a tool fault can reach them.
scenarios:
  - id: quickstart.drop_required_key
    description: A field the agent reads simply stops being returned.
    entrypoint: your_package.agent:build
    inputs: { question: "what is the weather in Paris?" }
    expected_behavior: graceful_degradation
    expect:
      # The answer must not state a number no tool produced...
      no_unsourced_numbers: { enabled: true }
      # ...and must say what it could not determine.
      output_matches: ["(?i)unavailab|missing|could not"]
    faults:
      - type: ToolCorruptionFault
        target: { tool: get_weather }
        trigger: { on_call: 1 }
        params: { mutation_type: drop_key, keys: [temp_c] }
"""

SCAFFOLD_README = """# chaos/

    alc run chaos/quickstart.yaml --judge rules

Edit `entrypoint` and the tool name in `quickstart.yaml` to match your agent. Every
failure writes an `AGENT_TASK.md` under `.chaos/` — hand it to a coding agent as-is.
`alc list-faults` shows what else you can inject.
"""


def _init(args: argparse.Namespace) -> int:
    """Scaffold a chaos directory, refusing to clobber anything.

    Args:
        args: Parsed arguments.

    Returns:
        `EXIT_OK`, or `EXIT_USAGE` when a file already exists and `--force` was not
        given. Overwriting someone's edited scenario silently is the one thing a
        scaffolding command must never do.
    """
    root = Path(args.dir)
    files = {root / "quickstart.yaml": QUICKSTART, root / "README.md": SCAFFOLD_README}
    existing = [p for p in files if p.exists()]
    if existing and not args.force:
        for path in existing:
            print(f"alc init: {path} already exists", file=sys.stderr)
        print("alc init: nothing written; pass --force to overwrite", file=sys.stderr)
        return EXIT_USAGE

    root.mkdir(parents=True, exist_ok=True)
    for path, body in files.items():
        path.write_text(body, encoding="utf-8")
        print(f"  wrote {path}")
    print(f"\nNext: edit {root / 'quickstart.yaml'} to point at your agent, then\n")
    print(f"  alc run {root / 'quickstart.yaml'} --judge rules\n")
    return EXIT_OK


def _replay(args: argparse.Namespace) -> int:
    """Re-run a stored plan.

    Args:
        args: Parsed arguments.

    Returns:
        `EXIT_OK` when the replayed run passed, `EXIT_FAILED` otherwise.

    Raises:
        ConfigError: When the run directory or the entrypoint cannot be used.
    """
    from .engine import ChaosEngine
    from .loop import resolve_entrypoint

    run_dir = Path(args.run_dir)
    if not (run_dir / "plan.json").is_file():
        raise ConfigError(f"no plan.json in {run_dir}; nothing to replay")
    if not args.entrypoint:
        raise ConfigError("replay needs the agent to re-run: pass --entrypoint MODULE:ATTR")

    engine = ChaosEngine(
        seed=args.seed if args.seed is not None else 1337,
        out_dir=Path(args.out) if args.out else run_dir.parent.parent,
        judge=getattr(args, "judge", None) or "rules",
        strict_schema=False,
    )
    result = engine.replay(run_dir, target=resolve_entrypoint(args.entrypoint, engine))

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))
    else:
        print(result.summary_line())
        for note in result.schema_errors:
            if note.startswith("replay_divergence"):
                print(f"  warning: {note}", file=sys.stderr)
    return EXIT_OK if result.success else EXIT_FAILED


#: ANSI codes, used only when `use_colour()` says a human is watching.
_COLOURS = {"pass": "\x1b[32m", "fail": "\x1b[31m", "dim": "\x1b[2m", "off": "\x1b[0m"}


def use_colour() -> bool:
    """Decide whether to emit ANSI escapes.

    Colour helps a person scan thirty scenarios and corrupts a pipe. `NO_COLOR` is
    honoured unconditionally -- https://no-color.org -- because a user who set it has
    already told us the answer.

    Returns:
        True only on a TTY with `NO_COLOR` unset.
    """
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _paint(text: str, kind: str, *, colour: bool) -> str:
    """Wrap text in an ANSI colour, or leave it alone.

    Args:
        text: What to print.
        kind: A key of `_COLOURS`.
        colour: Whether colour is wanted.

    Returns:
        The text, coloured or not.
    """
    if not colour or kind not in _COLOURS:
        return text
    return f"{_COLOURS[kind]}{text}{_COLOURS['off']}"


def _print_summary(results: Sequence[Any], out_dir: Path, *, colour: bool) -> None:
    """Print the block a person reads after a suite finishes.

    Totals, the failure-mode histogram, and the work orders -- in that order, because
    the work orders are the product and a reader who stops early should still have
    seen them.

    Args:
        results: Every scenario's result.
        out_dir: Where the bundles were written.
        colour: Whether to colour the totals.
    """
    failed = [r for r in results if not r.success]
    passed = len(results) - len(failed)
    print()
    print(
        f"  {_paint(f'{passed} passed', 'pass', colour=colour)}, "
        f"{_paint(f'{len(failed)} failed', 'fail' if failed else 'dim', colour=colour)}"
        f" of {len(results)} scenario(s)"
    )
    if failed:
        modes: dict[str, int] = {}
        for result in failed:
            modes[result.failure_mode] = modes.get(result.failure_mode, 0) + 1
        print()
        for mode, count in sorted(modes.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {count:>3}  {mode}")

    tasks = [r.artifacts["agent_task"] for r in results if r.artifacts.get("agent_task")]
    if tasks:
        print(f"\n  {len(tasks)} work order(s) written. Hand one to a coding agent as-is:")
        for path in tasks[:5]:
            print(f"    {path}")
        if len(tasks) > 5:
            print(f"    ... and {len(tasks) - 5} more under {out_dir}")
    print()


def _judge_options(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the judge flags that were actually given.

    Only present flags are returned, so an unset one falls through to `SLMJudge`'s
    own default rather than overriding it with `None`.

    Args:
        args: Parsed arguments.

    Returns:
        Keyword arguments for `SLMJudge`.
    """
    options: dict[str, Any] = {}
    for flag, key in (("model", "model"), ("base_url", "base_url"), ("transport", "transport")):
        value = getattr(args, flag, None)
        if value:
            options[key] = value
    if getattr(args, "allow_remote_judge", False):
        options["allow_remote_judge"] = True
    return options


def _judge(args: argparse.Namespace) -> int:
    """Re-judge a stored run without re-running the agent.

    The whole pipeline after the trace is pure over `(trace, plan)`, so a better
    model can be pointed at last week's run. `success`, `failure_mode` and `severity`
    are *not* recomputed: they were decided by the probes and the assertions layer at
    run time, and a judge has no authority over them (`docs/11` §1).

    Args:
        args: Parsed arguments.

    Returns:
        `EXIT_OK`, or `EXIT_FAILED` when the re-judged run had failed.

    Raises:
        ConfigError: When no report can be found.
    """
    from .bundle import write_agent_task
    from .judges import select_judge
    from .judges.base import build_evidence
    from .report import ChaosResult, extract_code_context

    document, path = _load_report(Path(args.run_dir))
    result = ChaosResult.from_dict(document)
    run_dir = path.parent

    judge = select_judge(args.judge_kind, **_judge_options(args))
    evidence = build_evidence(result, code_context=extract_code_context(result.code_pointers))
    verdict = judge.judge(evidence)

    # The judge carries these; it never decides them.
    verdict.passed = result.success
    verdict.observed_behavior = result.verdict["observed_behavior"]
    verdict.expected_behavior = result.verdict["expected_behavior"]
    verdict.failure_mode = result.failure_mode
    verdict.severity = result.severity

    result.verdict = verdict.to_dict()
    result.root_cause_hypothesis = verdict.root_cause_hypothesis
    result.refinement_hint = verdict.refinement_hint
    result.suggested_fixes = list(verdict.suggested_fixes)
    if verdict.narrative:
        result.chaos_narrative = verdict.narrative

    (run_dir / "report.json").write_text(result.to_json(), encoding="utf-8")
    last_call = dict(getattr(judge, "last_call", {}) or {})
    if last_call:
        (run_dir / "judge.json").write_text(
            json.dumps({**last_call, "verdict": verdict.to_dict()}, indent=2, default=str),
            encoding="utf-8",
        )
    write_agent_task(result, run_dir)

    print(f"{result.scenario_id or result.run_id}: re-judged by {verdict.judge_meta.kind}")
    return EXIT_OK if result.success else EXIT_FAILED


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

    print("\n  faults:")
    for fault in report.get("injected_faults") or []:
        if fault.get("fired"):
            fires = fault.get("fires") or []
            note = (fires[0].get("note") if fires else None) or "fired"
            print(f"    fired  {fault.get('fault_id')} {fault.get('type')} — {note}")
        else:
            reason = fault.get("skipped_reason") or "no reason recorded"
            print(f"    never fired  {fault.get('fault_id')} {fault.get('type')} — {reason}")
    if not report.get("injected_faults"):
        print("    (none armed)")

    print("\n  probes:")
    for symptom in report.get("symptoms") or []:
        seqs = ", ".join(str(e.get("seq")) for e in symptom.get("evidence") or [])
        print(f"    {symptom.get('severity'):<8} {symptom.get('code')} @ seq {seqs}")
    if not report.get("symptoms"):
        print("    (none fired)")

    # The diff is what makes this readable without opening JSON: it shows exactly
    # what the agent was handed that it was not expecting.
    diffs = [
        (fault, fire)
        for fault in report.get("injected_faults") or []
        for fire in (fault.get("fires") or [])
        if fire.get("json_patch")
    ]
    if diffs:
        print("\n  what changed:")
        for fault, fire in diffs[:3]:
            print(f"    {fault.get('fault_id')} {fault.get('type')} on {fire.get('name')!r}")
            for op in (fire.get("json_patch") or [])[:6]:
                value = op.get("value")
                shown = (
                    "" if op.get("op") == "remove" else f" = {json.dumps(value, default=str)[:60]}"
                )
                print(f"      {op.get('op'):<7} {op.get('path')}{shown}")

    hint = report.get("refinement_hint")
    print("\n  refinement hint:")
    print(f"    {hint}" if hint else "    (none produced)")
    if report.get("root_cause_hypothesis"):
        print(f"\n  hypothesis: {report['root_cause_hypothesis']}")

    task = (report.get("artifacts") or {}).get("agent_task")
    if task:
        print(f"\n  work order: {task}")
    print()
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
        "judge": _judge,
        "replay": _replay,
        "init": _init,
        "dashboard": _dashboard,
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
