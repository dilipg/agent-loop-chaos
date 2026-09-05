"""`alc init` must not scaffold a file the very next command cannot read.

A clean `pip install agent-loop-chaos` has no `pyyaml`: it is an optional runtime
extra (D-27). `alc init` writes `chaos/quickstart.yaml`, and the next line of every
quickstart is `alc run chaos/quickstart.yaml`, which then fails with "reading a YAML
suite needs pyyaml". Scaffolding an unreadable file and letting the user find out one
command later is the worst possible ordering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent_loop_chaos.cli import main


class TestWithoutPyyaml:
    @pytest.fixture
    def no_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make `import yaml` fail, as a base install does."""
        monkeypatch.setitem(sys.modules, "yaml", None)

    def test_it_says_so_at_scaffold_time(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        no_yaml: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert main(["init"]) == 0
        captured = capsys.readouterr()
        printed = captured.out + captured.err
        assert "yaml" in printed.lower()

    def test_it_scaffolds_something_runnable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_yaml: None
    ) -> None:
        """JSON needs no extra, so the scaffold falls back to it rather than to a
        file the reader cannot use."""
        monkeypatch.chdir(tmp_path)
        main(["init"])
        scaffolded = sorted(p.name for p in (tmp_path / "chaos").iterdir())
        assert "quickstart.json" in scaffolded
        document = json.loads((tmp_path / "chaos" / "quickstart.json").read_text())
        assert document["scenarios"][0]["id"]

    def test_the_json_scaffold_validates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_yaml: None
    ) -> None:
        monkeypatch.chdir(tmp_path)
        main(["init"])
        assert main(["validate", "chaos/quickstart.json"]) == 0


class TestWithPyyaml:
    def test_it_still_writes_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("yaml")
        monkeypatch.chdir(tmp_path)
        main(["init"])
        scaffolded = sorted(p.name for p in (tmp_path / "chaos").iterdir())
        assert "quickstart.yaml" in scaffolded
        assert "quickstart.json" not in scaffolded

    def test_the_yaml_scaffold_validates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("yaml")
        monkeypatch.chdir(tmp_path)
        main(["init"])
        assert main(["validate", "chaos/quickstart.yaml"]) == 0


def test_the_readme_install_line_mentions_the_yaml_extra(repo_root: Path) -> None:
    """The scaffold is YAML by default, so the install that produces it must be
    the one the README leads with."""
    readme = (repo_root / "README.md").read_text()
    install = readme.split("## Install")[1].split("## ")[0]
    assert "[yaml]" in install
