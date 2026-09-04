"""Executable documentation.

A code block in `docs/` that no longer runs is worse than no example: a reader copies
it, it fails, and they conclude the library is broken rather than the doc is stale.
Any fenced `python` block preceded by `<!-- test -->` is extracted and executed here.

Marking is opt-in on purpose. Plenty of blocks in the docs are deliberately partial —
a signature, a `...` stub, a shape being described rather than run — and forcing
those to execute would push the docs toward whatever happens to be runnable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
MARKER = "<!-- test -->"

_BLOCK = re.compile(
    re.escape(MARKER) + r"\s*\n```python\n(?P<code>.*?)```",
    re.DOTALL,
)


def _snippets() -> list[tuple[str, str]]:
    """Every marked block in `docs/` and the README, as `(label, code)`."""
    found: list[tuple[str, str]] = []
    for path in sorted([*DOCS.glob("*.md"), REPO / "README.md"]):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for index, match in enumerate(_BLOCK.finditer(text), start=1):
            line = text[: match.start()].count("\n") + 1
            found.append((f"{path.name}:{line}#{index}", match.group("code")))
    return found


SNIPPETS = _snippets()


def test_at_least_one_snippet_is_marked() -> None:
    """A silently empty extractor would make this whole file pass vacuously."""
    assert SNIPPETS, f"no `{MARKER}` blocks found; the extractor is not doing anything"


@pytest.mark.parametrize(("label", "code"), SNIPPETS, ids=[s[0] for s in SNIPPETS])
def test_the_snippet_runs(label: str, code: str, tmp_path: Path, monkeypatch: Any) -> None:
    """Execute one documented block in a temp directory."""
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {"__name__": "__doc_snippet__"}
    try:
        exec(compile(code, label, "exec"), namespace)
    except Exception as exc:  # pragma: no cover - the failure message is the product
        pytest.fail(f"{label} no longer runs: {type(exc).__name__}: {exc}\n\n{code}")
