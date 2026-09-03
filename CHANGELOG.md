# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[Semantic Versioning](https://semver.org/).

`schema_version` in the report and trace schemas versions independently of the
library — see `docs/04-SCHEMAS.md` §2.

## Unreleased

### Added

- **M0 — repository skeleton (phase 00).** Packaging via hatchling with a dynamic
  version read from `src/agent_loop_chaos/version.py`; `jsonschema>=4.18` as the
  only required dependency; `langgraph`, `slm`, `yaml`, `anthropic`, `dev` and `all`
  optional extras.
- `agent_loop_chaos.errors` — `ChaosError` and its subclasses, plus
  `LimitExceeded(BaseException)` (D-06) and `ExplicitError(Exception)`.
- `agent_loop_chaos.schema` — offline `referencing.Registry` over the four packaged
  schemas keyed by `$id`, with `validator_for()` and `validate_obj()`. Resolving an
  unknown `$id` raises rather than fetching.
- The four JSON Schemas vendored into `src/agent_loop_chaos/schemas/` as package
  data, with `OWNER` in every `$id` substituted for `dilipgdt` (D-50).
- `alc` CLI skeleton covering every subcommand in `docs/02-API.md` §10.
  `--version` works; the rest exit 2 as not implemented.
- Public API stubs re-exported from `agent_loop_chaos.__init__` with `__all__`
  matching `docs/02-API.md` §1 name for name. Every stub carries its real signature
  and raises `NotImplementedError`.
- Test suite: import surface, no-hard-dependency assertion, schema compilation and
  example validation, CLI smoke.
- CI workflow with `lint`, `test` (3.10–3.13 × langgraph {pinned, none}) and
  `schema` jobs.

### Changed

- The build handover pack moved from `README.md` to `PACK.md`; `README.md` is now
  the library's own readme (D-49). `tools/verify_pack.py`'s "every phase prompt is
  listed" check retargets accordingly, and its doc-reference scan now covers
  `PACK.md` too.
- `tools/verify_pack.py` brought up to the repo's lint bar (it predates `ruff`
  here), and its pack scan now prunes `.venv/`, `.git/` and cache directories so
  the reported file count means something.

### Fixed

- `schema.validate_obj` capped only the appended `value=` repr, but `jsonschema`
  embeds the offending instance in `message` as well, so a 5 kB bad value produced a
  5 kB error string. Both halves are now clipped at 200 characters, honouring
  `docs/04-SCHEMAS.md` §1.

### Decisions recorded

- **D-48** — phase 00 does not reset `docs/DECISIONS.md`; the Create-list line
  predates the file being pre-seeded and would have deleted D-01…D-47.
- **D-49** — `README.md` belongs to the library; the build pack moves to `PACK.md`.
- **D-50** — `OWNER` resolves to `dilipgdt`. PyPI `agent-loop-chaos` is free;
  `agent-loop-detector` is taken at 0.1.0; GitHub org `delightree` does not exist.
- **D-51** — where phase 00's stubs live, plus the two modules the architecture map
  was missing (`enums.py`, `assertions.py`).

### Notes

- `ruff` ≥0.16 formats Python code blocks inside Markdown, which silently rewrites
  the hand-aligned code blocks in the normative docs. Markdown is therefore excluded
  from `ruff` in `pyproject.toml`; do not remove that exclusion.
- **Editable installs silently break on macOS.** Files written into `.venv` acquire
  the `UF_HIDDEN` flag, and current CPython — 3.12 and 3.14 both observed — skips
  `.pth` files flagged hidden, so `import agent_loop_chaos` fails with nothing but a
  `ModuleNotFoundError` despite `pip install -e` succeeding. Because the flag is
  applied after the file is written, it can surface mid-session. `pytest` is now
  immune via `pythonpath = ["src"]`; for the `alc` console script, run
  `chflags nohidden .venv/lib/python3.*/site-packages/*.pth`. See `RUNBOOK.md` §0.
  Wheel installs are unaffected. Develop on 3.10–3.13, the supported matrix.
- Nothing in this release does anything yet. Faults, engine behaviour, probes,
  judges and adapters arrive in M1–M6.
