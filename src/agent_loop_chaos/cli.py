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

    # Each remaining command lands with the phase that owns it, and `alc` is
    # complete in M9 (prompts/09-cli-and-release.md).
    print(f"alc {args.command}: not implemented", file=sys.stderr)
    return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())
