#!/usr/bin/env python3
"""Explain a development environment that looks broken but is not.

One failure mode is worth a whole tool: on macOS with iCloud's "Desktop & Documents"
sync on, the file provider sets `UF_HIDDEN` on files it manages. CPython's `site.py`
skips a hidden `.pth` file — so an editable install inside a synced directory stops
being on `sys.path` and every command fails with

    ModuleNotFoundError: No module named 'agent_loop_chaos'

which looks exactly like a broken install. `chflags nohidden` does not stick; the
provider re-applies it. The fix is to put the *virtualenv* outside the synced tree —
the checkout itself can stay where it is, because only `.pth` files are skipped, never
source files.

    python3 tools/doctor.py
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

OK = "ok  "
WARN = "WARN"
FAIL = "FAIL"


def hidden_pth_files(prefix: Path) -> list[Path]:
    """Find `.pth` files that `site.py` will silently skip.

    Args:
        prefix: A virtualenv root.

    Returns:
        The hidden `.pth` files, if any.
    """
    found: list[Path] = []
    for path in prefix.glob("lib/python*/site-packages/*.pth"):
        try:
            flags = os.lstat(path).st_flags  # type: ignore[attr-defined]  # macOS only
        except (AttributeError, OSError):
            continue
        if flags & stat.UF_HIDDEN:
            found.append(path)
    return found


def icloud_documents_sync_is_on() -> bool:
    """Whether macOS is syncing Desktop & Documents to iCloud.

    Returns:
        True when the Finder default says so. `~/Documents` is not a symlink when this
        is on -- the provider syncs in place -- so checking for one proves nothing.
    """
    if sys.platform != "darwin":
        return False
    try:
        out = subprocess.run(
            ["defaults", "read", "com.apple.finder", "FXICloudDriveDocuments"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.stdout.strip() == "1"


def main() -> int:
    """Report on the environment.

    Returns:
        `0` when nothing is wrong, `1` when something needs fixing.
    """
    prefix = Path(sys.prefix)
    problems = 0

    print(f"{OK}  python      {sys.version.split()[0]} at {sys.executable}")
    print(f"{OK}  venv        {prefix}")

    try:
        import agent_loop_chaos

        print(f"{OK}  import      agent_loop_chaos {agent_loop_chaos.__version__}")
    except ImportError:
        print(f"{FAIL}  import      agent_loop_chaos is not importable")
        problems += 1

    hidden = hidden_pth_files(prefix)
    if hidden:
        problems += 1
        print(f"{FAIL}  path        {len(hidden)} .pth file(s) are hidden, so site.py skips them:")
        for path in hidden:
            print(f"              {path}")
        print()
        print("      This is macOS's iCloud file provider, not a broken install.")
        print("      `chflags nohidden` does not stick -- the provider re-applies it.")
        print()
        print("      Fix: put the virtualenv outside the synced tree. The checkout can")
        print("      stay exactly where it is; only .pth files are skipped, never source.")
        print()
        print("          make venv VENV=~/.venvs/agent-loop-chaos")
        print("          . ~/.venvs/agent-loop-chaos/bin/activate")
        print()
        print("      Or, without the Makefile:")
        print()
        print("          python3 -m venv ~/.venvs/agent-loop-chaos")
        print('          ~/.venvs/agent-loop-chaos/bin/pip install -e ".[dev]"')
    elif icloud_documents_sync_is_on() and str(prefix).startswith(
        (str(Path.home() / "Documents"), str(Path.home() / "Desktop"))
    ):
        print(f"{WARN}  path        the venv is under iCloud-synced Desktop & Documents.")
        print("              Nothing is hidden yet, but the provider may hide the .pth")
        print("              at any time. `make venv VENV=~/.venvs/agent-loop-chaos`.")
    else:
        print(f"{OK}  path        no hidden .pth files")

    print()
    print("no problems found" if not problems else f"{problems} problem(s) found")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
