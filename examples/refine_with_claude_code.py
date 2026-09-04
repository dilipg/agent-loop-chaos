#!/usr/bin/env python3
"""Find real bugs in an agent, write work orders, and optionally let a coding agent fix them.

This is the library's value proposition in one file:

    1. Run the suite. Some scenarios fail -- those are real bugs, not flaky tests.
    2. Each failure becomes an `AGENT_TASK.md`: a work order a coding agent can
       execute without asking a question.
    3. With `--apply`, hand each work order to a coding agent, then re-run the whole
       suite and print what actually got fixed -- including what regressed.

**Default mode is read-only.** It runs the suite, prints the findings, and stops. It
does not shell out, touch git, or write outside `--out`.

`--apply` is different and says so loudly: it lets an external coding agent edit your
source files. It refuses to run on a dirty tree or on `main`/`master`, so the changes
are always reviewable as a diff on a branch.

The library itself never edits source. That decision stays here, in an example you
can read, rather than behind a flag inside the engine.

Usage:

    python examples/refine_with_claude_code.py
    python examples/refine_with_claude_code.py --suite my_suite.yaml
    git switch -c chaos-fixes
    python examples/refine_with_claude_code.py --apply --rounds 3
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:  # running from a checkout without an install
    sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The suite this ships against, with a fallback so the script is runnable in a fresh
# checkout before the demo agent exists.
DEFAULT_SUITES = (
    REPO / "examples" / "scenarios" / "demo_suite.yaml",
    REPO / "tests" / "data" / "fake_suite.yaml",
)
DEFAULT_AGENT = "claude"
REFERENCE = "examples/revenue_review_fixed/"

WARNING = """
================================================================================
  --apply WILL EDIT YOUR SOURCE FILES.

  Every work order below is handed to `{agent}`, which may change any file in
  this repository. Nothing here is reviewed first. You are on branch {branch!r}
  with a clean tree, so `git diff` afterwards shows exactly what changed and
  `git restore .` undoes all of it.

  A coding agent can also "fix" a chaos failure by weakening the scenario. The
  loop hashes every scenario file, fault plan and must_not list between rounds,
  and reports a pass that follows an edit as a REGRESSION, never as a fix.
================================================================================
"""


def git_status() -> str:
    """Read the working tree's porcelain status.

    Returns:
        The output of `git status --porcelain`, or the empty string when git is
        unavailable or this is not a repository.
    """
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip()


def git_branch() -> str:
    """Read the current branch name.

    Returns:
        The branch, or the empty string when git is unavailable.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip()


def default_suite() -> Path:
    """Pick the suite to run when `--suite` was not given.

    Returns:
        The first suite that exists, preferring the demo suite over the fake one.
    """
    for candidate in DEFAULT_SUITES:
        if candidate.is_file():
            return candidate
    return DEFAULT_SUITES[-1]


def check_apply_preconditions(agent: str) -> tuple[bool, str | None]:
    """Decide whether `--apply` may proceed.

    Args:
        agent: The coding-agent executable name.

    Returns:
        ``(ok, refusal)``. `refusal` is a message for stderr when `ok` is False.
    """
    branch = git_branch()
    if branch in {"main", "master"}:
        return False, (
            f"refusing to run --apply on branch {branch!r}. A coding agent is about to "
            "edit your source; do it on a branch so the diff is reviewable:\n"
            "    git switch -c chaos-fixes"
        )
    status = git_status()
    if status:
        return False, (
            "refusing to run --apply with uncommitted changes. The whole safety story "
            "is that `git diff` afterwards shows only what the coding agent did:\n"
            f"{status[:400]}"
        )
    if shutil.which(agent) is None:
        return False, None  # not an error: fall back to printing the tasks
    return True, None


def print_findings(results: Sequence[Any]) -> list[Path]:
    """Report what the suite found, and print one work order in full.

    Args:
        results: Every scenario's result.

    Returns:
        The `AGENT_TASK.md` paths written.
    """
    failures = [r for r in results if not r.success]
    tasks = [Path(r.artifacts["agent_task"]) for r in failures if r.artifacts.get("agent_task")]

    print(f"\n{len(failures)} failure(s) in {len(results)} scenario(s).\n")
    for result in failures:
        print(f"  FAIL  {result.scenario_id}  {result.failure_mode} ({result.severity})")
        if result.refinement_hint:
            print(f"        {result.refinement_hint}")
    if not failures:
        print("  Every scenario passed. Nothing to hand to a coding agent.")
        return tasks

    print("\nWork orders written:\n")
    for path in tasks:
        print(f"  {path}")

    if tasks:
        print(f"\n{'-' * 78}\nOne finding in full ({tasks[0]}):\n{'-' * 78}\n")
        print(tasks[0].read_text(encoding="utf-8"))
    return tasks


def hand_off(agent: str, results: Sequence[Any]) -> None:
    """Give each work order to the coding agent.

    Args:
        agent: The executable to invoke.
        results: The failing results for this round.
    """
    for result in results:
        task = result.artifacts.get("agent_task")
        if not task:
            continue
        print(f"\n  -> {agent} <- {task}")
        try:
            subprocess.run(
                [agent, "-p", Path(task).read_text(encoding="utf-8")],
                cwd=REPO,
                timeout=1800,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"     {agent} failed: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--suite", help="suite file to run (default: the demo suite)")
    parser.add_argument("--out", default=".chaos-refine", help="output directory")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="let a coding agent edit your source files (needs a clean tree, not on main)",
    )
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="coding-agent executable")
    parser.add_argument("--rounds", type=int, default=2, help="rounds to run with --apply")
    parser.add_argument("--judge", default="rules", help="judge to use")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the suite, print the findings, and optionally apply fixes.

    Args:
        argv: Argument vector, defaulting to `sys.argv[1:]`.

    Returns:
        0 on success -- including when the suite found failures, which is the job.
        2 when `--apply` was refused.
    """
    from agent_loop_chaos.loop import RefinementLoop, run_suite
    from agent_loop_chaos.scenarios import load_suite

    args = build_parser().parse_args(argv)
    suite_path = Path(args.suite) if args.suite else default_suite()
    out_dir = Path(args.out)

    print(f"Running {suite_path} -> {out_dir}")
    suite = load_suite(suite_path)

    if not args.apply:
        results = run_suite(suite.scenarios, out_dir=out_dir, judge=args.judge)
        print_findings(results)
        print(
            f"\nNothing was modified. To let a coding agent act on these:\n"
            f"    git switch -c chaos-fixes\n"
            f"    python {Path(__file__).relative_to(REPO)} --apply\n"
            f"\nThe reference destination -- what a fixed agent looks like -- is "
            f"{REFERENCE}\n"
        )
        return 0

    ok, refusal = check_apply_preconditions(args.agent)
    if refusal is not None:
        print(f"refine: {refusal}", file=sys.stderr)
        return 2

    print(WARNING.format(agent=args.agent, branch=git_branch() or "?"))

    if not ok:
        # No coding-agent CLI installed. Do the run, print the work orders, and say
        # plainly what --apply would have done. Exiting non-zero here would fail a CI
        # job over a missing optional tool.
        results = run_suite(suite.scenarios, out_dir=out_dir, judge=args.judge)
        print_findings(results)
        print(
            f"\n{args.agent!r} is not on PATH, so nothing was applied.\n"
            f"--apply would have run `{args.agent} -p <work order>` once per file above, "
            f"then re-run the whole suite and printed what flipped.\n"
            f"\nThe reference destination is {REFERENCE}\n"
        )
        return 0

    report = RefinementLoop(
        suite,
        max_rounds=max(2, args.rounds),
        stop_when="all_pass",
        out_dir=out_dir,
        judge=args.judge,
        on_findings=lambda results: hand_off(args.agent, results),
    ).run()

    print()
    print(report.markdown())
    print(f"\nThe reference destination -- what a fixed agent looks like -- is {REFERENCE}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
