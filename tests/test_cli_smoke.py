"""The `alc` entry point.

M0 ships the parser only. What must hold already: `--version` works, `--help` is
accurate for every documented subcommand, and no invocation ends in a traceback.
Exit codes are a contract because CI pipes them (`docs/02-API.md` §10).
"""

from __future__ import annotations

import pytest

from agent_loop_chaos import __version__
from agent_loop_chaos.cli import EXIT_USAGE, build_parser, main

DOCUMENTED_COMMANDS = [
    "run",
    "replay",
    "judge",
    "explain",
    "report",
    "validate",
    "list-faults",
    "init",
    "dashboard",
]


def test_version_flag_prints_the_version_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`alc --version` prints `0.1.0` and exits 0."""
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_bare_invocation_prints_help_and_exits_usage(capsys: pytest.CaptureFixture[str]) -> None:
    """`alc` with no command is a usage error, not a crash."""
    assert main([]) == EXIT_USAGE
    assert "usage: alc" in capsys.readouterr().out


@pytest.mark.parametrize("command", DOCUMENTED_COMMANDS)
def test_every_documented_subcommand_is_registered(command: str) -> None:
    """Every subcommand in `docs/02-API.md` §10 parses, so `--help` is honest."""
    parser = build_parser()
    actions = [a for a in parser._subparsers._actions if hasattr(a, "choices") and a.choices]
    registered: set[str] = set()
    for action in actions:
        registered |= set(action.choices)
    assert command in registered


def test_list_faults_is_implemented_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """M2 implemented it, so it now prints the catalog and succeeds.

    This replaces the M0 assertion that it exited 2 as unimplemented -- the command
    changed, not the standard.
    """
    assert main(["list-faults"]) == 0
    assert "fault kinds registered" in capsys.readouterr().out


IMPLEMENTED_COMMANDS = {"list-faults"}


def test_an_unimplemented_command_never_raises() -> None:
    """No documented command may escape as an exception; the CLI owns its exit codes."""
    for command in DOCUMENTED_COMMANDS:
        if command in IMPLEMENTED_COMMANDS:
            continue
        argv = [command]
        if command in {"run", "replay", "judge", "explain", "report", "validate"}:
            argv.append("placeholder")
        assert main(argv) == EXIT_USAGE


def test_unknown_command_is_a_usage_error() -> None:
    """argparse rejects an unknown subcommand with exit code 2."""
    with pytest.raises(SystemExit) as exc:
        main(["no-such-command"])
    assert exc.value.code == EXIT_USAGE
