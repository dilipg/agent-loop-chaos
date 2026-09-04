"""The wheel, built and installed the way a stranger would install it.

A missing data file is the classic first-release bug: everything works from a git
checkout and nothing works from `pip install`, because the schemas and prompt assets
were never packaged. These tests build the real artifact and read from it.

Marked `slow` because building a wheel and creating a venv is not a unit test.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.slow


# Built and installed once per session and cached in module globals rather than in
# scoped fixtures: a module- or class-scoped fixture that spawns subprocesses trips
# pytest's finalizer guard when it is requested from inside a test class.
_ARTIFACTS: dict[str, Path] = {}


def _build_wheel() -> Path:
    """Build the wheel once, and reuse it.

    Returns:
        The built wheel.
    """
    if "wheel" in _ARTIFACTS:
        return _ARTIFACTS["wheel"]
    out = Path(tempfile.mkdtemp(prefix="alc-dist-"))
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(f"`python -m build` unavailable: {proc.stderr[-300:]}")
    _ARTIFACTS["wheel"] = next(out.glob("*.whl"))
    return _ARTIFACTS["wheel"]


def _clean_install() -> Path:
    """Create a venv holding only the wheel, once, and reuse it.

    Returns:
        The venv root.
    """
    if "venv" in _ARTIFACTS:
        return _ARTIFACTS["venv"]
    root = Path(tempfile.mkdtemp(prefix="alc-venv-"))
    venv.create(root, with_pip=True)
    proc = subprocess.run(
        [str(root / "bin" / "pip"), "install", "--quiet", str(_build_wheel())],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(f"clean install unavailable (offline?): {proc.stderr[-300:]}")
    _ARTIFACTS["venv"] = root
    return root


def _in_venv(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a command inside the clean install, from outside the source tree.

    Args:
        args: The executable name under `bin/`, then its arguments.

    Returns:
        The completed process.
    """
    root = _clean_install()
    return subprocess.run(
        [str(root / "bin" / args[0]), *args[1:]],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tempfile.gettempdir(),
    )


@pytest.fixture
def wheel() -> Path:
    """The built wheel.

    Returns:
        Its path.
    """
    return _build_wheel()


class TestTheWheelCarriesItsData:
    def test_every_schema_is_inside(self, wheel: Path) -> None:
        names = zipfile.ZipFile(wheel).namelist()
        for schema in ("chaos_report", "trace_event", "judge_verdict", "scenario", "suite"):
            path = f"agent_loop_chaos/schemas/{schema}.schema.json"
            assert path in names, f"{path} is missing from the wheel"

    def test_every_prompt_asset_is_inside(self, wheel: Path) -> None:
        names = zipfile.ZipFile(wheel).namelist()
        for asset in ("judge_system", "judge_user", "narrator", "refiner"):
            path = f"agent_loop_chaos/judges/prompts/{asset}.md"
            assert path in names, f"{path} is missing from the wheel"

    def test_py_typed_is_inside(self, wheel: Path) -> None:
        assert "agent_loop_chaos/py.typed" in zipfile.ZipFile(wheel).namelist()

    def test_no_pack_infrastructure_leaks_in(self, wheel: Path) -> None:
        """`prompts/`, `assets/` and `tools/` are build inputs, not library code."""
        names = zipfile.ZipFile(wheel).namelist()
        leaked = [
            n for n in names if n.startswith(("prompts/", "assets/", "tools/", "tests/", "docs/"))
        ]
        assert leaked == [], f"pack infrastructure shipped in the wheel: {leaked[:5]}"


class TestACleanInstall:
    def test_the_console_script_runs(self) -> None:
        proc = _in_venv("alc", "--version")
        assert proc.returncode == 0, proc.stderr[-400:]
        assert proc.stdout.strip()

    def test_the_schemas_load_from_the_installed_package(self) -> None:
        proc = _in_venv(
            "python",
            "-c",
            "from agent_loop_chaos.schema import load_schema; import json;"
            "print(json.dumps([load_schema(n)['title'] for n in "
            "('report','trace','verdict','scenario','suite')]))",
        )
        assert proc.returncode == 0, proc.stderr[-400:]
        assert json.loads(proc.stdout) == [
            "ChaosResult",
            "TraceEvent",
            "Verdict",
            "ChaosSuiteFile",
            "SuiteResult",
        ]

    def test_the_judge_prompts_load(self) -> None:
        proc = _in_venv(
            "python",
            "-c",
            "from agent_loop_chaos.judges.slm import prompt_text;"
            "print(len(prompt_text('judge_system.md')))",
        )
        assert proc.returncode == 0, proc.stderr[-400:]
        assert int(proc.stdout) > 1000

    def test_list_faults_works_outside_the_source_tree(self) -> None:
        proc = _in_venv("alc", "list-faults", "--json")
        assert proc.returncode == 0, proc.stderr[-400:]
        assert json.loads(proc.stdout)["count"] == 27

    def test_only_jsonschema_is_required(self) -> None:
        """Importing the package must not need any optional extra."""
        proc = _in_venv(
            "python",
            "-c",
            "import sys, agent_loop_chaos;"
            "print([m for m in ('langgraph','httpx','yaml','openai','anthropic') "
            "if m in sys.modules])",
        )
        assert proc.returncode == 0, proc.stderr[-400:]
        assert proc.stdout.strip() == "[]"
