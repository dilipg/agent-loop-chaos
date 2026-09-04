# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[Semantic Versioning](https://semver.org/).

`schema_version` in the report and trace schemas versions independently of the
library — see `docs/04-SCHEMAS.md` §2.

## Unreleased

### Hardening (post-0.1.0)

The three weakest parts of 0.1.0, addressed. Six more library bugs fell out of doing
so, which is roughly the point.

**Cassettes** (D-100, implements D-45). `Cassette` records `messages_hash -> response`
so a suite against a real model reproduces. `alc run --record CASSETTE` /
`--replay-cassette CASSETTE`. A miss in replay mode raises and names the command to
re-record; inventing a response would reintroduce the flake this removes.

**A pointer for silent failures** (D-97). `AGENT_TASK.md` section 4 rendered "_no
pointer met the confidence floor_" for the commonest failure mode, because it read the
stack and a silent wrong answer has none. The engine now records the user frame about
to receive each faulted value. Paths render relative to the working directory.

**Probe negative controls** (D-98, D-99). Five of nine bugs in phase 08 failed on the
*correct* tree. Now enforced: every probe has a test proving it stays quiet on correct
behaviour, and a docstring stating where — fifteen of twenty had never written that
down. `tests/test_full_catalog.py` sweeps all 27 faults across all 16 pattern trees,
570 combinations.

Bugs found on the way:

- **D-95** A fault that decided to do nothing reported `fired: true`, inflating
  `suite.json` coverage and suppressing the "no fault fired" warning. It is now a skip
  carrying the fault's own reason. This immediately exposed four dead rows in the demo
  suite: `MalformedToolCallFault` had been "firing" against an agent with no
  `tool_calls` to malform.
- **D-96** `error.raised_in` was declared in the schema, read by two modules, and never
  written — so "an engine bug must never be reported as an agent failure" was
  unenforced. Fixing it exposed that every assertion in `tests/judges/test_wiring.py`
  was running against a `ConfigError` from passing a `Scenario` object as the agent.
- `alc init` had scaffolded a `chaos/` directory into the library's own repo during an
  M9 smoke test. Removed, and gitignored along with iCloud's `name 2.ext` conflict
  copies.

Demo suite after these changes: **17 failures, 6 distinct modes** on `trip_planner`;
**0** on `trip_planner_fixed`; **0** scenarios whose faults never fire.

## 0.1.0 — 2026-09-04

First release. The engine, 27 faults, 20 probes, the assertions layer, the judges, the
refinement loop, the LangGraph and vanilla adapters, the CLI and the demo agent.

### What it does

Injects a fault into a running agent loop, decides deterministically whether the agent
handled it, and writes the finding as an `AGENT_TASK.md` a coding agent can execute
without asking a question.

- **One `Crossing` dataclass** unifies six interception layers (tool, llm, state, node,
  edge, checkpoint) x three phases, which is why one engine serves all of them and the
  core never learns which adapter is in play.
- **`success` is computed, never asserted by a model.** 20 probes and the assertions
  layer are authoritative; judges narrate. A judge that contradicts the probes loses,
  and the disagreement is recorded in `verdict.judge_disagreement`.
- **A probe never fires on what the harness injected** (four attribution rules), which
  is why a correct agent can pass every scenario -- and why
  `examples/trip_planner_fixed` finding zero failures means something.
- **Determinism is a feature.** All randomness goes through `seeding.rng(seed, purpose)`
  keyed on a hashed `fault_key`, so reordering a suite cannot re-key another fault's
  stream. `docs/DECISIONS.md` D-07 states the boundary precisely rather than implying it.
- **Zero required dependencies except `jsonschema`.** LangGraph, httpx, pyyaml and the
  model SDKs are optional extras, and a test proves importing the package pulls none of
  them in.

### CLI

`alc run | replay | judge | explain | report | validate | list-faults | init`. Exit
codes `0` pass, `1` failures, `2` usage, `3` internal, `4` tampering. `--json` prints
exactly one JSON object and nothing else. Colour only on a TTY, and `NO_COLOR` is
honoured.

### Measured on the demo agent

`examples/trip_planner` (twelve planted weaknesses) vs `examples/trip_planner_fixed`,
same 29-scenario suite: **17 failures with 6 distinct failure modes**, versus **0**.

```
crash_unhandled_exception        7      prompt_injection_followed        1
silent_wrong_answer              5      secret_leak                      1
empty_final_answer               2      hallucination_on_corrupt_data    1
```

Twelve scenarios pass on the buggy tree. A suite that failed everything would be
measuring itself.

### Conformance

`examples/patterns/` covers eight agent shapes -- ReAct text loop, OpenAI tool-calling
loop, async with a `gather` fan-out, class with state on `self`, supervisor delegating
to specialists, fixed pipeline with no loop, streaming accumulator,
retrieve-rerank-generate -- each with a naive tree and a hardened twin. All sixteen run
through one battery in CI. Nine library bugs were found this way; five of them failed
on the *correct* tree rather than the buggy one.

### Known gaps

- `CheckpointRollbackFault` cannot fire under LangGraph: `intercept_checkpoints=True`
  is accepted and the adapter creates no checkpoint crossings (D-82).
- Two of the demo's twelve planted weaknesses are not caught by any scenario, and are
  listed as uncaught rather than omitted (D-89).
- `--jobs > 1` is accepted and ignored; suites run serially.
- `--format html` and `alc dashboard` arrive in 0.2.0.

### Schemas

`chaos_report` 1.3, `trace_event` 1.0, `judge_verdict`, `scenario`, `suite` 1.1. All
five ship inside the wheel and are importable without the source tree.

<details>
<summary>Milestone-by-milestone detail</summary>


### M8 — the demo agent, the pattern pool, and nine library fixes

**The proof.** `examples/trip_planner` is a four-node LangGraph app with twelve
planted weaknesses; `examples/trip_planner_fixed` closes all twelve. Same 29-scenario
suite against both:

| | buggy | fixed |
|---|---|---|
| failed | **17** | **0** |
| distinct failure modes | **6** | — |

`crash_unhandled_exception` 7, `silent_wrong_answer` 5, `empty_final_answer` 2,
`hallucination_on_corrupt_data` 1, `prompt_injection_followed` 1, `secret_leak` 1.

**The pattern pool.** `examples/patterns/` — eight agent shapes people actually ship
(ReAct text loop, OpenAI tool-calling loop, async with a `gather` fan-out, class with
state on `self`, supervisor delegating to specialists, fixed pipeline with no loop,
streaming accumulator, retrieve-rerank-generate), each with a naive tree and a
hardened twin. `tests/test_patterns.py` runs all sixteen through one battery: the
fault must fire, the naive tree must fail, the hardened tree must pass, no probe may
fire on a clean run, the report must be schema-valid, and the run must reproduce.

Also `examples/vanilla_agent.py`, `examples/scenarios/{demo_suite,quickstart}.yaml`,
`alc run --entrypoint`, and READMEs carrying the measured numbers.

**Nine library bugs, every one found by running real agents rather than reading code:**

- **D-79** `max_fires: null` is legal per the schema and the reference suite uses it;
  `Trigger` typed it `int`, so a schema-valid suite crashed the engine.
- **D-80** No state fault could ever fire: the engine names its one state crossing
  after the node, and the matcher compared `state_key` against that name. Every unit
  test passed, because each built a crossing shaped the way nothing emits.
- **D-81** The grounding check could not fire on the case it exists for. Every layer's
  results went into one history, so a model's own response was a "source" and any
  number it invented sourced itself.
- **D-82** A checkpointed LangGraph app could not run at all: `intercept_checkpoints`
  needs a `thread_id` the engine never passed.
- **D-83** `initial_state` was shallow-copied, so an agent appending to a scratchpad
  mutated the scenario. `RefinementLoop` rounds began from different states while
  being reported as the same scenario at the same seed.
- **D-85** `retried_then_succeeded` was unreachable: `recovered` was computed and
  never passed to the classifier, so every `retry_then_succeed` scenario failed a
  *correct* agent.
- **D-86** `injection_followed` could detect nothing but the canary. The fault records
  each payload's `detect`/`check` rule; the engine's fire record dropped it. Three of
  the four shipped injection objectives were undetectable.
- **D-87** `output_non_empty` was synthesized even for `explicit_error` scenarios,
  failing an agent for raising exactly as asked.
- Plus `redact_source` gaps and the demo-suite scenario repairs in **D-84**, **D-88**,
  **D-89**.

Five of the nine failed on the *correct* tree rather than the buggy one — the negative
control is where a false positive hides, which is why `docs/07` §2 makes it mandatory.

Schema: report `schema_version` 1.2 → 1.3, adding optional `injected_faults[].fires[].params`.


### M7 — the refinement loop

- `loop.py`: `RefinementLoop`, `LoopReport`, `RoundResult`, and the shared `run_suite`
  that `alc run` now also uses (D-74). Rounds re-run the **whole** suite with the same
  seeds, so a fix that broke something else is caught.
- Tamper detection, per `docs/05` §9. Before each round the loop fingerprints every
  scenario — its `plan_hash`, its source file, its `must_not` list — plus the probe
  module's own source. A scenario that flips to pass *after* its fingerprint changed
  is reported in `regressions`, never as a fix. A failing `control.*` scenario aborts
  the loop with exit code 4 (D-75, D-26).
- `schemas/suite.schema.json`: `suite.json` gets a real schema, since the loop and the
  dashboard both consume it (D-25). Bumped to `1.1` with `rounds`, `flipped`,
  `regressions`, `tamper`, `wall_ms` and the live-progress fields phase 10 needs. A
  1.0 document still validates.
- `alc run --rounds N [--stop-when …]`: runs the loop with no hand-off hook, prints
  the markdown table, exits 1 on a remaining failure and 4 on tampering.
- `examples/refine_with_claude_code.py`: read-only by default. `--apply` refuses a
  dirty tree or `main`/`master`, prints a loud warning first, and degrades to printing
  the work orders when no coding-agent CLI is installed.
- `tests/golden/loop_report.md` pins the round-by-round table. 78 new tests.

Fixed along the way:

- **`tasks_written` listed one work order per round per scenario** (D-76) — eight
  paths for four bugs after two rounds, half of them describing superseded runs. Now
  keyed by scenario, holding the newest, and a scenario that starts passing drops out.
- **A flip and a regression were invisible after the round they happened in** (D-77).
  The outcome column read from the last round only, so "fixed in r2" became "passing"
  by r3 and "REGRESSION in r2" became "still failing" — the two facts a human scans
  the table for. Caught by writing the golden.
- **`ChaosEngine.replay`'s stub claimed M7** (D-78); the roadmap assigns it to M9.
- `Scenario` gains `source_path`, set by `load_suite`, so the loop can hash the file a
  scenario came from.


### M6 — the judge: rules, SLM, ensemble

- `judges/base.py`: `Verdict`, `JudgeMeta`, `JudgeEvidence` with the `docs/05` §2 caps,
  `build_evidence`, the ten-line `render` mustache-lite, `escape_fences` and
  `redact_source`. `JudgeEvidence.fit()` drops sections in the fixed order
  `code_context` → `baseline_output` → older exchanges → `tool_summary` and reports
  what it dropped.
- `judges/rules.py`: `RuleJudge` with a hint and a fix for all 20 probe codes. A test
  asserts the tables cover the probe registry exactly in both directions, so adding a
  probe later cannot silently produce an empty hint.
- `judges/slm.py`: `SLMJudge` over `openai` / `ollama` / `anthropic`, `httpx` when the
  `[slm]` extra is present and `http.client` otherwise. Structured-output ladder:
  native JSON schema → prompt-embedded → fence-stripping parser → one repair → rules
  fallback. Never raises into the run.
- `judges/ensemble.py`: `EnsembleJudge`. Rules own `passed`, `observed_behavior`,
  `failure_mode` and `severity`; the model owns narration, hypothesis, hint, fixes and
  confidence. A contradiction is recorded in `judge_disagreement`, not applied.
- `judges/prompts/`: the four assets copied from `assets/prompts/`, with D-21's
  untrusted-data fencing preserved.
- Engine, bundle and CLI wiring: `ChaosEngine(judge=…, judge_options=…, narrate_all=…)`,
  `judge.json` in the bundle, `alc judge <run_dir>`, and `--transport` /
  `--allow-remote-judge` / `--narrate-all` on `alc run`.
- `tests/fakes/fake_slm.py`: a scripted local endpoint covering all eight cases in
  `docs/07` §5. 182 new tests; 92% coverage on `judges/`.

Fixed along the way:

- **The trace closed before post-run** (D-71), so every `internal_error` raised by a
  probe, an assertion or the judge was written to a dead handle and lost — and
  `write_bundle` then rewrote `trace.jsonl` from a stale snapshot. Both fixed.
- **Refinement hints collapsed to a useless sentence** (D-72). `assertions_failed`
  outranks the structural probes, so four of five fake-suite scenarios produced
  "satisfy the declared expectation" while the real mechanism sat in "(and 1 other
  symptom)". Hints and fixes now follow the mechanism; classification is unchanged.
- **A socket leak in the no-`httpx` path** (D-73). `urllib.request.urlopen` abandons
  its connection on a read timeout; a suite judged against a hung endpoint leaked one
  socket per scenario. Replaced with `http.client`, closed in a `finally`.
- **Code context leaked hardcoded credentials.** `redact()` matches secret *shapes*
  and sensitive mapping keys; a source line like `PASSWORD = "hunter2"` is neither.
  `redact_source` closes the gap D-22 names.
- `jsonschema` is still imported lazily: `judges/slm.py` defers `..schema` so
  `import agent_loop_chaos` does not pull it in.

Schema: report `schema_version` 1.1 → 1.2. `judge_meta` gains `evidence_dropped` and
`auto_selected` (D-70); `suite.json` gains `judge_disagreement_rate` and
`judge_latency_ms_total` at its own version 1.0.


### Added

- **The `revenue_review` demo pair.** `examples/revenue_review/` is a five-node,
  nine-LLM-call quarterly revenue-risk workflow over the three CSV fixtures in
  `examples/revenue_review/data/`: a dict-dispatch graph (not a chain) with a
  `reviewer -> writer` back-edge, four CSV-backed tools including the
  side-effecting `flag_account_for_review`, and all twelve weaknesses from
  `docs/09-DEMO-AGENT.md` §4 planted and commented at the site. It exercises the
  harness against multi-call steps rather than a single-shot agent: 9 LLM calls
  and 5 tool calls per clean pass. No LangGraph; plain Python and stdlib `csv`.
- `examples/revenue_review_fixed/` — the negative control twin. Same graph, same
  tools, same call profile, every weakness fixed per `docs/09` §5:
  `validators.py` with `DataUnavailable(ExplicitError)`, bounded retries that skip
  4xx, in-band error detection, a durable objective, one JSON repair attempt, a
  `finish_reason` check, an `UNTRUSTED_DATA` fence, a capped back-edge with a
  no-progress check, and an idempotency key on the side-effecting tool. It shows
  zero symptoms under all twelve faults it is tested against.
- `examples/fake_model.py` — a scripted, offline, mediocre-but-plausible model, so
  each tree's module-level `graph` runs with no API key.
- `tests/test_revenue_review.py` — 38 tests: both trees end to end, the call
  counts, one test per planted weakness, and the M8-shaped gate (the fixed tree
  clean under every fault, the buggy tree found out by four of them, and the
  undelimited `notes` field leaking the injection canary end to end).

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

</details>
