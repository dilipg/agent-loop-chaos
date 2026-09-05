"""`tools/doctor.py` — the environment check.

It exists for one failure mode that costs an hour every time: on macOS with iCloud's
Desktop & Documents sync on, the file provider sets `UF_HIDDEN` on files it manages,
CPython's `site.py` skips hidden `.pth` files, and an editable install inside a synced
directory silently stops being on `sys.path`. The symptom is
`ModuleNotFoundError: No module named 'agent_loop_chaos'`, which looks exactly like a
broken install and is not one.

The detection is what is tested. The advice is prose and changes; a false "no problems
found" is what would waste the hour.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

# `tools/` is pack infrastructure, not library code, so it is not importable by name.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import doctor


def _site_packages(root: Path) -> Path:
    path = root / "lib" / "python3.12" / "site-packages"
    path.mkdir(parents=True)
    return path


class TestHiddenPthDetection:
    def test_a_visible_pth_is_not_a_problem(self, tmp_path: Path) -> None:
        (_site_packages(tmp_path) / "thing.pth").write_text("/somewhere\n")
        assert doctor.hidden_pth_files(tmp_path) == []

    @pytest.mark.skipif(not hasattr(os, "chflags"), reason="UF_HIDDEN is a BSD/macOS flag")
    def test_a_hidden_pth_is_found(self, tmp_path: Path) -> None:
        path = _site_packages(tmp_path) / "thing.pth"
        path.write_text("/somewhere\n")
        os.chflags(path, stat.UF_HIDDEN)  # type: ignore[attr-defined]
        assert doctor.hidden_pth_files(tmp_path) == [path]

    def test_a_venv_with_no_pth_files_is_fine(self, tmp_path: Path) -> None:
        _site_packages(tmp_path)
        assert doctor.hidden_pth_files(tmp_path) == []

    def test_a_missing_venv_does_not_raise(self, tmp_path: Path) -> None:
        assert doctor.hidden_pth_files(tmp_path / "nope") == []


class TestItRunsAtAll:
    def test_it_reports_on_this_environment(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = doctor.main()
        printed = capsys.readouterr().out
        assert "python" in printed and "venv" in printed
        assert code in (0, 1)

    def test_a_clean_environment_exits_zero(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(doctor, "hidden_pth_files", lambda prefix: [])
        monkeypatch.setattr(doctor, "icloud_documents_sync_is_on", lambda: False)
        assert doctor.main() == 0
        assert "no problems found" in capsys.readouterr().out

    def test_a_hidden_pth_exits_nonzero_and_names_the_fix(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(doctor, "hidden_pth_files", lambda prefix: [tmp_path / "x.pth"])
        assert doctor.main() == 1
        printed = capsys.readouterr().out
        assert "iCloud" in printed
        assert "make venv VENV=" in printed
        assert "chflags nohidden` does not stick" in printed

    def test_the_sync_check_never_raises(self) -> None:
        assert doctor.icloud_documents_sync_is_on() in (True, False)


def test_the_makefile_offers_both_targets(repo_root: Path) -> None:
    makefile = (repo_root / "Makefile").read_text()
    assert "\nvenv:" in makefile and "\ndoctor:" in makefile
    assert "VENV ?= .venv" in makefile
