# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[Semantic Versioning](https://semver.org/).

`schema_version` in the report and trace schemas versions independently of the
library — see `docs/04-SCHEMAS.md` §2.

## Unreleased

### Added

- **M3 — LLM/prompt faults and the injection corpus (phase 03).** Built
  red-green-refactor, with every `pre` fault's effect asserted on the messages the
  model actually received.
- `faults/_messages.py` — the normalized message form and its helpers, with a
  lossless `denormalize(normalize(x)) == x` round trip via `_extra`, so a fault
  never silently discards a field the provider needed.
- All ten section-B faults: `ContextShrinkFault`, `ContextNoiseFault`,
  `GoalDriftFault`, `LLMMalformedOutputFault`, `LLMRefusalFault`, `LLMEmptyFault`,
  `LLMTruncationFault`, `MalformedToolCallFault`, `HallucinationSeedFault`,
  `PromptInjectionFault`.
- A 17-payload injection corpus covering all five objectives and all five
  placements, with every `detect` regex asserted against a real canary (D-16) and
  every `check` mechanically decidable — a semantic constraint is a judge
  hypothesis, never a probe.
- The canary is planted in `initial_state["_alc_canary"]`, exempt from redaction,
  and the redactor now recognises its shape so the exemption is not a no-op (D-59).
- `alc list-faults` shows 21 kinds: 10 tool/loop, 10 LLM/prompt, and `NoopFault`.

### Fixed in M3 development

- The `json_key` injection placement truncated the payload to 120 characters, which
  dropped the canary out of every longer exfiltration payload and made them
  permanently undetectable (D-60).

### Added

- **M2 — tool-execution faults and the mutation library (phase 02).** Built
  red-green-refactor throughout: every test written first and watched failing for
  the expected reason before any implementation.
- The D-23 side-effect gate, built first as the phase requires: a real-action fault
  refuses a `side_effecting=True` tool without an explicit per-tool opt-in, a
  wildcard target never matches one, and `engine.require_declared_side_effects()`
  refuses a broad preset while any tool is undeclared.
- `mutations.py` — all 22 names from catalog A1, pure and deep-copying, with
  deterministic `rng.sample(sorted(paths), count)` selection. `unit_swap` converts
  the value and leaves the label intact; `nan_numbers` emits the D-09 encoding;
  `drop_required_key` implements D-17's baseline heuristic with its documented
  fallback.
- All ten section-A faults: `ToolCorruptionFault`, `ToolErrorFault`,
  `ToolLatencyFault`, `ToolTimeoutFault`, `ArgumentTamperFault`, `StaleDataFault`,
  `NonDeterminismFault`, `DuplicateSideEffectFault`, `LoopTrapFault`,
  `RateLimitFault`.
- `errors.SimulatedToolError`, so a scenario can raise something unambiguously ours.
  Exception classes come from an allow-list; a class name from a scenario file is
  never evaluated.
- `alc list-faults` is implemented, with `--json` for CI.
- 530 tests, 89.9% coverage on `faults` and `mutations`.

### Fixed in M2 development

- A `(tool, pre)` fault could not substitute a return value, so
  `ToolErrorFault(error_type="error_payload")` ran the real tool and discarded the
  envelope. Its unit test passed throughout; the end-to-end test caught it (D-57).
- `replace_args` could only deliver positional arguments, so `ArgumentTamperFault`
  never reached a tool called by keyword (D-57).
- `invoke_target` was unimplemented, so `DuplicateSideEffectFault` never duplicated
  anything (D-57).
- An exception in the engine's routing layer was reported as the agent's `error`,
  because the tool wrapper runs inside the agent's call stack — a library bug
  attributed to the agent under test, which `CLAUDE.md` forbids (D-58).

### Added

- **M1 — core engine (phase 01).** `Crossing` and the run-scoped context objects,
  seeded RNG and derived ids, secret redaction, the RFC 6902 subset, the trace
  recorder, targeting and triggering, the `Fault` base with `NoopFault`, the vanilla
  adapter, and the `ChaosEngine` run lifecycle (steps 1-7 and 14). `_post_run` is the
  seam M4 attaches metrics, assertions, probes and the judge to.
- `ChaosResult` now assembles and validates for real: identity, target, metrics,
  loop, artifacts, reproduce, randomness and `injected_faults` are populated, with
  `success=True` and `failure_mode="none"` pinned until M4 computes them from the
  probes and the assertions layer.
- Faults compose per D-13: value-replacing actions chain in registration order and
  each records its own `MutationLog`, while `raise`/`delay`/`invoke_target`/
  `resume_from_checkpoint` are first-wins and mark the rest `superseded`.
- 323 tests, 89.6% coverage. Targeting and trigger truth tables, `apply(diff(a, b),
  a) == b` as a hypothesis property, redaction as a property, RNG-stream
  independence, harness-attribution rules, and the resilience test that proves an
  exception inside a fault never reaches the agent under test.

### Fixed in M1 development

- A fault was offered crossings outside its `accepts` set whenever the target left
  `phase` unset, so a post-only fault fired at `pre`, spent its `max_fires`, and
  never reached the phase it was written for (D-55).
- `target.predicate` serialized as a boolean, which the report schema types as
  `string|null`; it now records the predicate's qualified name.
- The no-network test fixture blocked every socket family, including the `AF_UNIX`
  socketpair asyncio uses for its own self-pipe, which broke every async test while
  proving nothing. It now blocks only `AF_INET` and `AF_INET6`.
- `report.py` and `trace.py` imported `jsonschema` at module scope, so
  `import agent_loop_chaos` pulled it in eagerly and contradicted the documented
  lazy-import claim.

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
