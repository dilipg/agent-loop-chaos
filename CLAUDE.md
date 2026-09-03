# CLAUDE.md — build conventions for `agent-loop-chaos`

You are implementing the library specified in `docs/`. This file is binding.

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
└── docs/
```

Never place code outside `src/`. Never import from `tests/` in library code.

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

1. `make check` passes (`ruff check`, `ruff format --check`, `mypy --strict`,
   `pytest -q`).
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
- Using an LLM to decide `success`.
- Weakening or deleting a failing test to make the suite green. If a test is wrong,
  say so explicitly and explain why before changing it.
- Writing a report field that is not in the schema, or a schema field that no code
  ever writes.
- Letting a language model decide `success`, `observed_behavior`, or `failure_mode`.
- A probe that fires on `good_agent` or on a baseline run.
- `print()` in library code. Use the `logging` logger named `agent_loop_chaos`.
