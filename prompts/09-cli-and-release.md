# Phase 09 — CLI completion, docs, release (M9)

The library works. This phase makes it something a stranger can install, understand
in two minutes, and adopt.

**Read first:** `docs/02-API.md` §10 (CLI), `docs/08-ROADMAP.md` (M9 row and the
release checklist), `docs/00-VISION.md` (for the README's voice).

## Scope

Full `alc` implementation, README rewrite, docs polish, packaging, release
artifacts. No new library behaviour except what the CLI needs.

## Build

### CLI
Every subcommand from `docs/02-API.md` §10, fully implemented:

`run`, `replay`, `judge`, `explain`, `report`, `validate`, `list-faults`, `init`.

Quality bar:

- `--help` on every subcommand names its flags with one-line descriptions and one
  example invocation. Someone should be able to use the tool from `--help` alone.
- Default human output is compact and scannable: one line per scenario
  (`PASS`/`FAIL`, id, failure mode, severity, wall time), then a summary block with
  the failure-mode histogram and the paths to the task files. Colour only when the
  stream is a TTY; honour `NO_COLOR`.
- `--json` prints exactly one JSON object to stdout and nothing else — no log lines,
  no progress. Tested by piping to `json.loads`.
- Progress on stderr for suites over 5 scenarios; suppressed with `--quiet`.
- Exit codes: 0 pass, 1 failures, 2 usage/config, 3 internal, 4 tampering.
- `alc init` scaffolds `chaos/quickstart.yaml` plus a three-line README snippet into
  the current directory, refusing to overwrite existing files.
- `alc explain <run_dir>` prints the narrative, the fault diff, and the refinement
  hint — the "what happened here" command, readable without opening JSON.
- Every command works on a run directory produced by an older run in the same minor
  version (no re-run required).

### README (rewrite, not patch)
Structure, in order:

1. Name, one-sentence pitch from `docs/00-VISION.md`, badges.
2. The problem, in four sentences: an agent that invents a value when a tool returns
   `{}` is not caught by any test you have.
3. Quickstart: install, `alc init`, `alc run`, and the terminal output — real, pasted.
4. **One full `AGENT_TASK.md`, collapsed in a `<details>` block.** This is the pitch;
   give it room.
5. The two integration snippets (LangGraph, vanilla) from `docs/02-API.md` §11.
6. The fault catalog as a compact table with a link to `docs/03-FAULT-CATALOG.md`.
7. The demo results from phase 08, with the real numbers.
8. Judge configuration: local SLM setup in three lines, and the rules-only offline
   mode.
9. How the output is meant to be consumed: schema link, `suite.json`, the loop.
10. Status, roadmap link, contributing, license.

Voice: plain, specific, no hype. Every claim must be reproducible with a command in
the README. Cut any sentence that would embarrass you if a reader tested it.

### Docs polish
- Fold `docs/DECISIONS.md` entries into the relevant docs where they change a stated
  contract; keep the dated log.
- Add `docs/FAQ.md` covering: why not just eval, why the model does not decide
  pass/fail, how to add a fault, how to run offline, why my probe has a false
  positive, how to pin a scenario's seed, what to do when LangGraph breaks the
  adapter.
- Verify every code block in `docs/` still matches the implementation. Add
  `tests/test_docs_snippets.py` that extracts fenced `python` blocks tagged
  `<!-- test -->` and executes them.

### Packaging
Per the release checklist in `docs/08-ROADMAP.md`. Confirm `schemas/*.json` and
`judges/prompts/*.md` are inside the wheel (`unzip -l dist/*.whl | grep -E 'schemas|prompts'`),
because a missing data file is the classic first-release bug.

## Tests

- One test per subcommand exercising the happy path and one error path.
- `--json` output parses for `run`, `report`, `list-faults`, `validate`.
- Exit codes asserted for all five values.
- `alc init` in a temp dir creates the files and refuses on the second call.
- `NO_COLOR` and non-TTY produce no ANSI escapes.
- Wheel-install test: build, install into a fresh venv, run
  `alc run examples/scenarios/quickstart.yaml` from outside the source tree.
- Docs-snippet test passes.

## Acceptance checklist

- [ ] `make check` green; `pytest -m slow` green
- [ ] all eight subcommands implemented, tested, and documented in `--help`
- [ ] `--json` is machine-clean for every command that offers it
- [ ] all five exit codes verified
- [ ] README rewritten; every command in it run and its real output pasted
- [ ] `docs/FAQ.md` added; doc snippets execute in CI
- [ ] wheel contains the schemas and prompt assets; clean-venv install runs the demo
- [ ] `CHANGELOG.md` has a complete `## 0.1.0` section
- [ ] version `0.1.0` in one place, read dynamically by the build
- [ ] `twine check dist/*` passes

## Verify

```bash
make check && pytest -m slow -q
python -m build && twine check dist/*
unzip -l dist/*.whl | grep -E 'schemas/|prompts/'
deactivate; python -m venv /tmp/v && /tmp/v/bin/pip install dist/*.whl
cd /tmp && /tmp/v/bin/alc list-faults && /tmp/v/bin/alc --help
```

## Final report

In your last message, give me:

1. The acceptance checklist, ticked.
2. The demo suite's real failure-mode histogram.
3. Anything in `docs/` that the implementation contradicts, with the file and line.
4. The three weakest parts of the library, in your judgement, and what you would do
   about each. Be blunt; this is what the next round of work is built from.
