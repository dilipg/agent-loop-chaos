"""The live trace dashboard.

`report.json` is for machines; this is the one surface built for a human watching a
suite run. Stdlib only, read-only, localhost by default.

The dependency arrow is one-way and stays that way: this package reads the report and
trace contracts and never changes them, and nothing here imports `faults/` or
`judges/`. The engine does not know the dashboard exists -- if the dashboard crashes,
the suite keeps going (`docs/10` §1).
"""

from __future__ import annotations

__all__ = ["DashboardServer", "RunDirWatcher", "export_html"]


def __getattr__(name: str) -> object:
    """Import the watcher lazily.

    Keeps `import agent_loop_chaos.dashboard` free of work for a CLI that only wants
    to know the module exists.

    Args:
        name: The attribute being looked up.

    Returns:
        The attribute.

    Raises:
        AttributeError: For anything not exported.
    """
    if name == "RunDirWatcher":
        from .watcher import RunDirWatcher

        return RunDirWatcher
    if name == "DashboardServer":
        from .server import DashboardServer

        return DashboardServer
    if name == "export_html":
        from .export import export_html

        return export_html
    raise AttributeError(name)
