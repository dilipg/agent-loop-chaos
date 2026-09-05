"""Single source of truth for the library version.

`pyproject.toml` reads `__version__` from here via hatchling, so the version is
bumped in exactly one place (M9 release checklist, step 2).
"""

from __future__ import annotations

__version__ = "0.2.1"
