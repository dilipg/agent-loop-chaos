# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

You are implementing the library specified in `docs/`. This file is binding.

## Repository status: a spec pack, not yet a codebase

There is **no code in this repository yet** — no `src/`, no `pyproject.toml`, no
`Makefile`, no `tests/`. What exists is a complete, normative specification plus the
eleven prompts that build it. Do not be surprised by missing modules; create them in
the phase that owns them.

The build is **phase-driven and sequential**. `prompts/NN-*.md` are the work orders,
`docs/08-ROADMAP.md` maps each to a milestone (M0–M10) and a checkable gate, and
`RUNBOOK.md` §4 lists the exact gate commands. One phase per context window; a phase
is not started before the previous gate passes. If you were handed a phase prompt,
implement **only** that phase.

### Where truth lives (precedence order)

When two documents disagree, higher wins:

1. `docs/DECISIONS.md` — 47 pre-seeded errata (D-01…D-47) from an adversarial review
   done *before* any code existed. Binding, and it overrides the prose of any doc it
   names. Read it in full first. Never edit an entry in place: supersede it with a
   new dated `D-nn` and mark the old one `SUPERSEDED BY D-nn`. **The next id you
   write is D-48.**
2. `docs/11-OUTCOMES-AND-ASSERTIONS.md` — the **sole** pass/fail authority. Nothing
   may compute `success`, `observed_behavior`, `failure_mode`, or `severity` by any
   other route.
3. `schemas/*.json` — the schemas are the product; code serves them.
4. `docs/02-API.md` — every public name is frozen. Additions allowed, renames never.
5. `SAFETY.md` — gates that must exist in code, not just in prose.
6. Everything else in `docs/`.

`README.md` is now the **library's** front door — install, integration, scenario
syntax, CLI, CI and pytest usage — not a build handover document. Its snippets and its
numbers are tested (`tests/test_readme_snippets.py`), so changing the fault catalog,
the probe list, the presets, the CLI surface or the demo suite's size will fail there
until the README is updated with it. `docs/00-VISION.md` holds positioning and the v0.1
non-goals.

## Commands

These land in phase 00 (`prompts/00-bootstrap.md`) and are the contract thereafter.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"              # add ,langgraph and/or ,slm for those phases

make check                           # lint + types + tests + schema parity (docs/07 §8 jobs 1-3)
make golden-update                   # ONLY way to refresh tests/golden/*.json; review the diff
python3 tools/verify_pack.py         # re-check spec self-consistency after editing docs/
python3 tools/verify_pack.py --strict   # a skipped check group is a failure (CI)
```

Tests:

```bash
pytest -q                            # default run: no network, excludes -m live
pytest tests/faults -q               # one layer (see docs/07-TESTING.md §1 for the layer map)
pytest tests/faults/test_tool.py::test_drop_key -q      # one test
pytest -m slow -q                    # demo-suite tests
pytest -m live -q                    # opt-in; needs a real model endpoint
pytest tests -q -k "not langgraph and not readme"       # prove the core has no framework dep
```

An autouse fixture monkeypatches `socket.socket` to raise, so any accidental
outbound call in the default run fails loudly. Coverage gate: 85% on
`src/agent_loop_chaos`.

CLI (`alc`, full surface in `docs/02-API.md` §10):

```bash
alc run <suite.yaml|scenario.yaml|module:attr> --judge rules --out .chaos
alc replay <run_dir>                 # re-run from plan.json; refuses if it cannot verify (D-31)
alc judge <run_dir>                  # re-judge without re-running the agent
alc explain <run_dir>                # narrative to stdout
alc report <run_dir> --format md|json|html
alc validate <report.json|suite.yaml>
alc list-faults [--json]
```

Exit codes: `0` all passed, `1` a scenario failed, `2` config/usage error, `3`
internal error. `--json` prints exactly one JSON object and nothing else.

## Architecture — the big picture

Read `docs/01-ARCHITECTURE.md` for the module map. The parts that only make sense
across several files:

### `Crossing` is the keystone

One dataclass unifies every interception point, which is why tool, LLM, node, edge,
state and checkpoint faults share a single engine instead of needing six. A crossing
carries `(layer, phase, name, step, call_index, args, kwargs, result, exception,
state, span_id)` where `layer` ∈ tool/llm/state/node/edge/checkpoint and `phase` ∈
pre/post/error. Each `Fault` declares the `(layer, phase)` pairs it accepts, and the
engine refuses a bad registration with `ConfigError` **at register time, not run
time**. Adapters (`adapters/vanilla.py`, `adapters/langgraph.py`) exist only to turn
their framework's hooks into crossings; the core never knows which one is in play.

### The pipeline

`Scenario → ChaosEngine.run → adapter → Crossing → targeting (Target × Trigger ×
seeded RNG) → Fault.apply on a deep copy → trace event with payload_before /
payload_after / json_patch → probes → metrics → assertions → judge → bundle`.

Two rules make the middle of that chain work:

- Metrics are computed **before** probes, and probes take `(trace, ctx)` (D-11).
- Probes and assertions are pure over the trace; the judge only ever reads their
  output.

### Three layers of authority (`docs/11` §1)

| Layer | Decides | Authority |
|---|---|---|
| `probes.py` (20 probes) | structural failures visible in the trace | authoritative |
| `assertions.py` (`Expect`) | did the agent meet the scenario's declared expectation | authoritative |
| `judges/` | narration, root cause, hints, ranked fixes | **advisory only** |

Probe false negatives are acceptable; false positives are bugs. If judge and probes
disagree, probes win and the disagreement is recorded in `verdict.judge_disagreement`.

### `HarnessFacts` — why `good_agent` can pass

The engine knows exactly what it injected and passes that to every probe and
assertion as `HarnessFacts`. Four attribution rules (`docs/11` §2, R1–R4) forbid
firing on the harness's own injection: event attribution, value attribution, budget
attribution (subtract `tokens_injected` / `delay_injected_ms` before any threshold),
and removal attribution (a dropped key's absence is expected afterwards; the finding
is a consumer reading it without a precondition check). This is the structural reason
`tests/fakes/good_agent` and `examples/trip_planner_fixed` — the two negative
controls, one per suite, not interchangeable — can pass every scenario. A probe that
fires on either has a false positive and the probe is wrong until proven otherwise.

### Identity and determinism

- `fault_key = sha256(canonical_json({type, params, target, trigger}))[:6]` keys RNG
  streams; `fault_id` (`f1`, `f2`, …) is a display label only (D-03). Reordering YAML
  must not re-key another fault's stream.
- `run_id = "run-" + sha256(f"{seed}|{scenario_id}|{plan_hash}|{attempt}")[:8]`;
  `attempt` is an explicit parameter, never derived from the filesystem (D-04).
- `seeding.rng(seed, purpose)` is a pure factory with **no module-level memo**; the
  only cache is `RunContext.rng_registry` (D-02).
- "Step" = one agent iteration in both adapters — node entry under LangGraph, one LLM
  call under vanilla. Tool calls increment `tool_calls`, not `steps` (D-05).
- `LimitExceeded` derives from `BaseException` so the agent's `except Exception`
  cannot swallow it; only the engine's outermost frame catches it (D-06).
- Determinism has a stated boundary (D-07): harness determinism is unconditional; a
  byte-identical `report.json` additionally needs a deterministic agent, a scripted
  or temperature-0 model, `--judge rules`, and a single-threaded graph.
  `tests/normalize.py` owns the strip list and nothing else may add to it.

### Output bundle

```
.chaos/
├── suite.json                      # what CI, an optimizer, or the dashboard polls
└── <scenario_id>/<run_id>/
    ├── report.json  trace.jsonl  plan.json  judge.json  baseline.diff
    └── AGENT_TASK.md               # failures only — this file is the product
```

`AGENT_TASK.md` is the whole pitch: if it isn't good enough for a coding agent to act
on without asking a question, phase 04 is not done.

### Safety gates that must exist in code

`ArgumentTamperFault`, `DuplicateSideEffectFault`, and `CheckpointRollbackFault`
perform real actions. Per `SAFETY.md` §1 / D-23, enforced at `register_fault` time:
they refuse a `side_effecting=True` tool without explicit `allow_side_effects:` opt-in;
glob targets never match a side-effecting tool; `--preset full` refuses to start if
any tool leaves `side_effecting` undeclared; and
`PromptInjectionFault(objective="call_forbidden_tool")` blocks and stubs the call —
the attempt is the finding.

## Before you write any code

0. Read `docs/DECISIONS.md` in full. It is pre-seeded errata from an adversarial
   review of this spec and is **binding**: where it contradicts another doc, it
   wins. Then read `docs/11-OUTCOMES-AND-ASSERTIONS.md`, which is the normative
   pass/fail authority, and `SAFETY.md`, which describes gates you must enforce in
   code.
1. Read `docs/01-ARCHITECTURE.md` and `docs/02-API.md` in full. They define names
   you must not rename.
2. Read `docs/04-SCHEMAS.md` and the JSON Schema files in `schemas/`. The schemas
   are the product. Code serves them.
3. Read the phase prompt you were given. Implement **only** that phase. Do not
   scaffold future phases "while you're in there".
4. If the spec is ambiguous or self-contradictory, do not guess silently: pick the
   reading that keeps the JSON output stable, implement it, and record the decision
   in `docs/DECISIONS.md` (create it if missing) as a dated one-liner.

## Non-negotiable design rules

- **Zero required third-party dependencies except `jsonschema`.** `langgraph`,
  `httpx`, `pyyaml`, `anthropic`, `openai` are optional extras. Importing
  `agent_loop_chaos` must never import them. Guard with local imports inside the
  adapter/judge that needs them, and raise a clear
  `MissingExtraError("install agent-loop-chaos[langgraph]")`.
- **`pyyaml` is a special case (D-27).** It is an optional *runtime* extra (`[yaml]`)
  **and** a member of `[dev]`. Consequence: YAML suite loading may not be treated as
  a feature the user opts into — every gate command in `RUNBOOK.md` and phases
  04/06/07/08 runs a `.yaml` suite after installing only `[dev]`, so `load_suite`
  must work there. JSON suite loading has no extra and must never require one.
- **Never mutate the user's objects in place.** Faults operate on deep copies and
  return new values. A fault that cannot copy an object must record
  `mutation_skipped_uncopyable` and pass the original through.
- **The library must never crash the run it is observing** unless a fault is
  explicitly designed to. Any internal error in the engine, a probe, or a judge is
  caught, recorded as an `internal_error` trace event with a traceback, and the run
  continues. An engine bug must never be reported as an agent failure.
- **Determinism is a feature.** All randomness goes through
  `agent_loop_chaos.seeding.rng(seed, purpose)`. No bare `random.*`, no
  `time.time()` inside decision logic, no set/dict iteration order dependence, no
  `uuid4` for anything that lands in a report (derive ids from the seed and a
  counter). Wall-clock values appear only in clearly-labelled timing fields.
- **A probe never fires on the harness's own injection.** Every probe and
  assertion receives `HarnessFacts` and must exclude what the engine itself caused
  (`docs/11` §2, rules R1–R4). A finding that the fault created rather than the
  agent is a bug, and it makes the `good_agent` control unpassable.
- **No clock in decision logic.** No probe, assertion, or classification rule may
  read wall-clock time, `ts`, or `ts_mono_ms`.
- **The side-effect gate is enforced in code**, not just documented: see `SAFETY.md`
  §1 and D-23.
- **Known secret shapes are redacted.** Every payload passes through `redact.py`
  before serialization. Default deny-list covers keys matching
  `(?i)(api[_-]?key|secret|token|password|authorization|bearer|cookie)` and values
  matching common key prefixes (`sk-`, `ghp_`, `AKIA`). Redacted values become
  `"<redacted:api_key>"`.
- **Additive schema evolution only.** `schema_version` is `MAJOR.MINOR`. Adding an
  optional field bumps MINOR. Removing or retyping a field bumps MAJOR and requires
  a note in `docs/DECISIONS.md`. Never silently change a field's meaning.
- **`success` is computed, never asserted by an LLM.** The boolean comes from
  deterministic probes, the assertions layer, and the scenario's
  `expected_behavior`, exactly as `docs/11-OUTCOMES-AND-ASSERTIONS.md` specifies. The SLM may
  contribute `failure_mode`, `root_cause_hypothesis`, `refinement_hint`, and
  narration only. If the judge and the probes disagree, the probes win and the
  disagreement is recorded in `verdict.judge_disagreement`.
- **Async and sync parity.** Every interception point supports both. If a wrapped
  callable is a coroutine function, the wrapper must be one too. Test both paths.

## Code style

- Python ≥3.10. `from __future__ import annotations` at the top of every module.
- Full type hints on all public functions. `py.typed` shipped. `mypy --strict` on
  `src/agent_loop_chaos` must pass.
- `ruff` for lint + format, line length 100. No `# type: ignore` without a reason
  comment.
- Public API is re-exported from `agent_loop_chaos/__init__.py` with an explicit
  `__all__`. Anything not in `__all__` is private and may change.
- Dataclasses (`@dataclass(frozen=True, slots=True)` where practical) for value
  objects. No Pydantic in the core; it is a heavy dependency and the schemas are
  the source of truth.
- Docstrings: one-line summary, then Args/Returns/Raises. Every fault class
  docstring states **what agent weakness it proves** and **what the graceful
  behaviour looks like**.
- Errors: one base `ChaosError`; subclasses `ConfigError`, `MissingExtraError`,
  `SchemaError`, `JudgeError`, `AdapterError`.

## Layout

```
agent-loop-chaos/
├── pyproject.toml
├── Makefile
├── src/agent_loop_chaos/…        (see docs/01-ARCHITECTURE.md for the module map)
├── tests/
├── examples/
├── schemas/                      (copied verbatim from this pack; also packaged as data files)
├── docs/
├── prompts/                      the eleven phase work orders + PROMPTING-GUIDE.md
├── assets/prompts/               judge_system.md, judge_user.md, narrator.md, refiner.md
│                                 (packaged into src/agent_loop_chaos/judges/prompts/)
└── tools/verify_pack.py          spec self-consistency checker; run after editing docs/
```

Never place code outside `src/`. Never import from `tests/` in library code.
`prompts/`, `assets/`, and `tools/` are pack infrastructure: read them, and do not
ship them as library code.

## Testing rules

- `pytest` + `pytest-asyncio`. No network in the default test run; the SLM judge is
  tested against a local fake HTTP server (`tests/fakes/fake_slm.py`), never a real
  endpoint. Tests needing a real model are marked `@pytest.mark.live` and excluded
  by default.
- Every fault class gets: a unit test on its mutation, a test that it fires exactly
  when its trigger says, a test that it records `payload_before`/`payload_after`/
  `json_patch` correctly, and an end-to-end test against a fake agent asserting the
  resulting `failure_mode`.
- Golden-file tests compare normalized `report.json` against
  `tests/golden/*.json`. Normalization strips `ts`, `duration_ms`, `wall_ms`,
  absolute paths, and the library version. Refresh goldens only with
  `make golden-update` and review the diff in the commit.
- A determinism test runs the same scenario twice with the same seed and asserts
  normalized equality; and runs it with two different seeds and asserts inequality
  for at least one probabilistic fault.
- Coverage gate: 85% on `src/agent_loop_chaos`, and 100% of fault classes must
  appear in at least one end-to-end test.

## Definition of done for any phase

Nothing is complete until all of these hold:

1. `make check` passes. It is exactly `docs/07-TESTING.md` §8 jobs 1–3:
   `ruff check`, `ruff format --check`, `mypy --strict`, `pytest -q`, **and** the
   schema job — validate every file in `schemas/examples/` and every golden report
   against the schemas, then assert dataclass↔schema field parity. That last check is
   the only thing enforcing "no report field outside the schema, no schema field
   nothing writes"; a `make check` without it is not `make check`.
2. New public API appears in `docs/02-API.md` — if you deviated from the spec,
   update the doc in the same commit and say so in the summary.
3. Any new report field is in `schemas/chaos_report.schema.json` **and** in a
   golden fixture under `schemas/examples/`.
4. The phase prompt's own acceptance checklist is ticked, item by item, in your
   final message.
5. `CHANGELOG.md` gets an entry under `## Unreleased`.

## Things that will get the work rejected

- Renaming anything defined in `docs/02-API.md` or a JSON Schema.
- Adding a required dependency.
- Making the engine raise into the agent under test on an internal error.
- Weakening or deleting a failing test to make the suite green. If a test is wrong,
  say so explicitly and explain why before changing it.
- Writing a report field that is not in the schema, or a schema field that no code
  ever writes.
- Letting a language model decide `success`, `observed_behavior`, or `failure_mode`.
- A probe that fires on `good_agent` or on a baseline run.
- `print()` in library code. Use the `logging` logger named `agent_loop_chaos`.
