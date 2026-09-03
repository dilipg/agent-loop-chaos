"""The core has no framework dependency.

`jsonschema` is the only required third-party package. This is the test the
`langgraph = none` CI column exists to run, and it is why the library can be
installed with `pip install agent-loop-chaos` and used offline.

Each probe runs in a fresh interpreter, so the assertion is about *the package's*
import graph rather than whatever pytest and its plugins loaded into this process.
`PYTHONPATH` points at the source tree so a probe cannot be derailed by the state of
an editable install; that the built distribution installs correctly is verified
separately by building a wheel and installing it into a clean venv.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"

FORBIDDEN = ("langgraph", "langchain", "langchain_core", "httpx", "yaml", "anthropic", "pydantic")

PROBE = """
import sys
import agent_loop_chaos
forbidden = {forbidden!r}
leaked = sorted({{m for m in sys.modules if m.split('.')[0] in forbidden}})
print(",".join(leaked))
"""


def _run_probe(snippet: str) -> str:
    """Run a probe program in a fresh interpreter.

    Args:
        snippet: The program to run.

    Returns:
        Its stripped stdout.
    """
    # Fixed argv, no shell: the only variable is the snippet this module builds.
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    return proc.stdout.strip()


def test_importing_the_package_pulls_in_no_framework() -> None:
    """`import agent_loop_chaos` must not import any optional extra."""
    out = _run_probe(PROBE.format(forbidden=FORBIDDEN))
    assert out == "", f"these optional dependencies were imported: {out}"


def test_importing_the_package_does_not_even_need_jsonschema_eagerly() -> None:
    """The one required dependency is still imported lazily, by `schema.py` only."""
    snippet = "import sys, agent_loop_chaos; print('yes' if 'jsonschema' in sys.modules else 'no')"
    assert _run_probe(snippet) == "no"


@pytest.mark.parametrize(
    "module",
    ["agent_loop_chaos.faults", "agent_loop_chaos.judges", "agent_loop_chaos.adapters"],
)
def test_family_packages_import_without_extras(module: str) -> None:
    """`faults`, `judges` and `adapters` import with no optional dependency installed.

    `adapters` matters most: importing it must not pull in `langgraph`, which is what
    makes the CI `langgraph = none` column meaningful.
    """
    importlib.import_module(module)
    snippet = (
        f"import sys, {module}\n"
        f"leaked = [m for m in sys.modules if m.split('.')[0] in {FORBIDDEN!r}]\n"
        "print(','.join(sorted(leaked)))\n"
    )
    assert _run_probe(snippet) == ""
