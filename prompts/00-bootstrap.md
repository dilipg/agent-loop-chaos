# Phase 00 — Bootstrap the repository (M0)

You are starting a new open-source Python library called **agent-loop-chaos**. This
folder contains its complete specification. Build only what this phase asks for.

**Read first, and treat as binding:** `docs/DECISIONS.md` — pre-seeded errata from an adversarial review of this spec; it overrides any doc it contradicts.

**Read first, in full:** `CLAUDE.md`, `docs/00-VISION.md`, `docs/01-ARCHITECTURE.md`
(module map section), `docs/04-SCHEMAS.md`, `docs/08-ROADMAP.md` (M0 row).

## Scope

Repository skeleton, packaging, tooling, CI, and the vendored schemas. No fault
logic, no engine behaviour, no adapters. Stubs are fine and expected — but every
stub must have a real signature from `docs/02-API.md` and raise
`NotImplementedError`, never silently return `None`.

## Create

```
pyproject.toml            hatchling; name "agent-loop-chaos"; requires-python ">=3.10"
                          dynamic version read from src/agent_loop_chaos/version.py
                          dependencies = ["jsonschema>=4.18"]
                          optional-dependencies:
                            langgraph = ["langgraph>=0.2,<0.7", "langchain-core>=0.3"]
                            slm       = ["httpx>=0.27"]
                            yaml      = ["pyyaml>=6"]
                            anthropic = ["anthropic>=0.40"]
                            dev       = [pytest, pytest-asyncio, pytest-cov, ruff, mypy,
                                         hypothesis, build, twine, pyyaml]
                                         # pyyaml IS in dev: every gate command runs a
                                         # .yaml suite (D-27). It stays an optional
                                         # RUNTIME extra as [yaml].
                            all       = [langgraph, slm, yaml]
                          [project.scripts] alc = "agent_loop_chaos.cli:main"
                          [tool.ruff] line-length = 100, select E,F,I,N,UP,B,SIM,RUF
                          [tool.mypy] strict = true, files = "src/agent_loop_chaos"
                          [tool.pytest.ini_options] markers = ["live", "slow"], asyncio_mode = "auto"
                          [tool.coverage] fail_under = 85 (source src/agent_loop_chaos)
                          package data: schemas/*.json, judges/prompts/*.md
Makefile                  check, test, lint, fmt, typecheck, cov, golden-update, demo, build, clean
README.md                 name, one-line pitch from docs/00-VISION.md, install, the two
                          quickstart snippets from docs/02-API.md §11, output example,
                          "status: pre-alpha", link to docs/
LICENSE                   Apache-2.0
NOTICE
CHANGELOG.md              "## Unreleased"
CONTRIBUTING.md           how to add a fault and a probe (from docs/01 §7)
CODE_OF_CONDUCT.md        Contributor Covenant 2.1
.gitignore                python, .venv, dist, coverage, AND `.chaos*/` — the run
                          directory holds full prompts, payloads and source excerpts
                          and must never be committed (D-24, SAFETY.md §2)
.github/workflows/ci.yml  jobs: lint, test (matrix 3.10-3.13 x langgraph{pinned,none}), schema
src/agent_loop_chaos/
  __init__.py             __all__ exactly as in docs/02-API.md §1; stubs re-exported
  version.py              __version__ = "0.1.0"
  errors.py              ChaosError + ConfigError, MissingExtraError, SchemaError,
                          JudgeError, AdapterError; plus LimitExceeded(BaseException)
                          and ExplicitError(Exception) — see D-06 and docs/11 §3.1
  py.typed
  schemas/                the four .json files copied VERBATIM from ../../schemas/
  schema.py               registry + validator_for() + validate_obj() per docs/04 §1
  cli.py                  argparse skeleton: all subcommands from docs/02-API.md §10
                          parse args, print "not implemented", exit 2. `--version` works.
tests/
  test_import.py          package imports; __all__ matches docs/02-API.md §1 exactly
  test_no_hard_deps.py    after `import agent_loop_chaos`, assert no module named
                          langgraph/langchain/httpx/yaml/anthropic/pydantic in sys.modules
  test_schemas.py         all four schemas load, compile, and cross-$ref resolve;
                          schemas/examples/report_failing.json validates against the report
                          schema; trace_excerpt.jsonl validates line by line
  test_cli_smoke.py       `alc --version`, `alc list-faults` exit cleanly
docs/DECISIONS.md         empty with a header
```

Substitute the real GitHub org/user for `OWNER` in every schema `$id` in this
pass, once — changing an `$id` later breaks the provenance of every stored report
(D-39). Confirm the PyPI name `agent-loop-chaos` is actually free before you do
(`pip index versions agent-loop-chaos`); the adjacent name `agent-loop-detector`
already exists, so do not assume.

Copy `../schemas/examples/*` into `tests/data/` so the schema tests have fixtures
inside the package tree. Keep the originals in `schemas/examples/` as the reference.

## Notes

- `schema.py` must build one `referencing.Registry` from the packaged schemas keyed
  by `$id`, cached at module level, with **no network access** — pass a retrieve
  function that raises. A test asserts that resolving an unknown `$id` raises rather
  than fetching.
- The `langgraph = none` CI column installs only `[dev]`; `test_no_hard_deps.py` and
  every core test must pass there.
- Do not create `faults/`, `judges/`, `adapters/` module bodies yet beyond empty
  `__init__.py` files with a docstring.

## Acceptance checklist (tick each in your final message)

- [ ] `pip install -e ".[dev]"` succeeds in a clean venv
- [ ] `make check` is green (ruff check, ruff format --check, mypy --strict, pytest)
- [ ] `alc --version` prints `0.1.0`
- [ ] `python -c "import agent_loop_chaos"` imports nothing third-party but jsonschema
- [ ] the four schemas validate their example instances via `schema.py`, offline
- [ ] `__all__` matches `docs/02-API.md` §1 name for name
- [ ] CI workflow runs all three jobs on push and PR
- [ ] `CHANGELOG.md` has an Unreleased entry for this phase

## Verify

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]" && make check
alc --version && alc list-faults
python -c "import sys, agent_loop_chaos; print([m for m in sys.modules if m.split('.')[0] in {'langgraph','httpx','yaml','pydantic'}])"
```

## Out of scope

Fault classes, engine behaviour, probes, judges, adapters, the demo agent. If you
find yourself writing a mutation function, stop.
