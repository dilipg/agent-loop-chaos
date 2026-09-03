# Phase 04 — Probes, report pipeline, findings bundle (M4)

This is the most important phase in the build. Everything before it manufactures
signal; this phase turns signal into the artifact the whole library exists to
produce. Take your time and read the schema twice.

**Read first, and treat as binding:** `docs/DECISIONS.md` — pre-seeded errata from an adversarial review of this spec; it overrides any doc it contradicts.

**Read first, in full:** `docs/11-OUTCOMES-AND-ASSERTIONS.md` (all of it — it is
the pass/fail authority and defines this phase's hardest parts),
`docs/04-SCHEMAS.md`, `docs/07-TESTING.md` §3 (the probe table is normative), `schemas/chaos_report.schema.json`,
`schemas/examples/report_failing.json`, `schemas/examples/AGENT_TASK.example.md`,
`docs/01-ARCHITECTURE.md` §5 steps 8–12.

## Scope

`probes.py`, **`assertions.py`**, `metrics.py`, **`outcomes.py`** (the
classification rules), the full `report.py`, `bundle.py`, `scenarios.py`
(Scenario/Suite/matrix/presets/YAML), and wiring lifecycle steps 8–13 into
`ChaosEngine`. A stub `RuleJudge` good enough to fill the verdict is in scope; the
real judge module is phase 06.

## Build

### `assertions.py` (build this before the probes)

The declarative layer from `docs/11` §4: the `Expect` dataclass, every check in the
scenario syntax, the auto-synthesis table (§4.4), and `no_unsourced_numbers`
implemented exactly as §4.3 specifies — numeric extraction with units, a tolerance,
a declared unit-conversion table, derived-value matching, and `values_injected`
subtracted from the sourced set. Each evaluation emits an `assertion_result` trace
event and an `AssertionResult` record.

This is the only thing that can distinguish graceful degradation from a
plausible-sounding wrong answer, so it carries the same weight as the probes. Do not
reach for a heuristic on proper nouns or bare numbers; §4.3 lists what is
deliberately excluded and why.

### `outcomes.py`

`HarnessFacts` accumulation (if phase 01 left it partial), `classify_behavior` as the
**ordered rule list** in `docs/11` §6, `satisfies()` per §7, the `failure_mode`
precedence rules per §8, `dominant_symptom`, and `PROBE_PRECEDENCE`. One unit test
per rule, and a test that rule 16 (`indeterminate`) always carries a `spec_gap` log
event naming the pair that fell through.

Note the two corrections §§6–7 make: `completed_unaffected` is reachable **only**
when no fault had a data effect, and it no longer satisfies `graceful_degradation`.
Without both, the project's motivating example (tool returns `{}`, agent invents a
number) reports success.

### `probes.py`
All 20 probes from `docs/07-TESTING.md` §3 (that table is the normative list), each a
`Probe` subclass with `code`, `severity`, and `detect(trace, ctx) -> list[Symptom]`
where `ctx` carries metrics, limits, baseline, assertions and `HarnessFacts`. Rules:

- Pure over `(trace, ctx)`. No network, no model, and **no clock** — `ts` and
  `ts_mono_ms` are off limits in decision logic (that is why the old
  `latency_budget_exceeded` and the timing clause of `retry_storm` are gone).
- **Harness attribution is mandatory** (`docs/11` §2 R1–R4): no probe may fire on an
  event or value the engine itself caused. Eleven of the original rules did; that is
  the single biggest correction in this phase, and it is what makes `good_agent`
  passable.
- Every `Symptom` carries ≥1 evidence entry with a real `seq`. A test asserts no
  probe ever returns an evidence-free symptom.
- `run_probes(trace, ctx) -> list[Symptom]`, deterministically ordered by
  (severity desc, code, first evidence seq).
- A probe that raises is caught, logged as `internal_error`, and skipped — one bad
  probe must not lose the report.
- `classify_behavior(trace, symptoms) -> observed_behavior` — the table-driven pure
  function from `docs/04-SCHEMAS.md` §7. Exhaustively unit-tested.
- `FAILURE_MODE_TABLE[(observed_behavior, dominant_symptom_code)] -> failure_mode`
  with a documented fallback for unmapped pairs (`unknown`, plus a warning event so
  gaps surface).

### `metrics.py`
Counters from the trace, plus `delta_vs_baseline` including
`output_similarity` (`difflib.SequenceMatcher` on lowercased,
whitespace-collapsed text — deterministic, no embeddings) and the
new/missing tool-call sets.

### `report.py` (complete)
Every field in `docs/02-API.md` §5 populated for real. Also:

- `assemble(trace, plan, ctx, symptoms, verdict, baseline) -> ChaosResult` as a pure
  function, so `alc judge` and `alc report` can rebuild a result from disk.
- `extract_code_pointers(error, trace) -> list[CodePointer]`: user frames only
  (drop anything under the library's own path and under site-packages), innermost
  first, plus a pointer derived from the function that produced the faulted crossing
  even when no exception was raised — that is the case in the reference example.
- `extract_code_context(pointers) -> list[dict]` reading ±4 source lines, used by
  the judge in phase 06. Handle missing files gracefully.
- `validate()` in strict mode raises `SchemaError`; otherwise fills
  `schema_errors[]`. Default `strict_schema=True`.
- A test asserting **dataclass ↔ schema field parity in both directions**. This test
  is the guard that keeps the two from drifting; make its failure message list the
  offending names.

### `bundle.py`
Writes the run directory exactly as `docs/04-SCHEMAS.md` §9, and renders
`AGENT_TASK.md` **section for section** as
`schemas/examples/AGENT_TASK.example.md`: seven numbered sections, the summary
table, the fenced diff, the exact prompt, the raw response, the baseline contrast,
the probe table, the code-pointer table, the hypothesis block, the numbered task
with real commands, and the Rules list verbatim. Nothing in it may be invented at
render time — every value comes from the report. A golden test compares rendered
output for a fixture report against a stored expectation.

Also write `plan.json`, `judge.json`, `baseline.diff` (unified diff via `difflib`),
and `suite.json` at the suite level.

### `scenarios.py`
`Scenario`, `FaultSpec`, `JudgeSpec`, `ChaosSuite`, `SuiteResult`, `load_suite`,
`PRESETS` as **literal `FaultSpec` lists with explicit triggers** (D-14 — prose
presets cannot feed `plan_hash`), appended to `docs/03` §D in this commit. A preset
whose faults are not registered yet (`state_integrity`, `resume_safety` before phase
05) resolves lazily by `kind` string and records
`skipped: fault_not_registered`. `matrix` expansion uses the single canonical id
rule from D-15: `<base_id>-<last_key_segment>-<slug(value)>`, products ordered by
sorted key then declared value order, no brackets (they violate the id pattern).
`must_not` entries are validated against the registered probe codes and an unknown
code raises `ConfigError` with the nearest match (D-33). YAML via an
optional import with a JSON fallback; validate the loaded document against
`scenario.schema.json` before use and raise `ConfigError` with the JSON pointer on
failure. `defaults` merging: fault lists concatenate, everything else replaces.

Suite running: sequential by default, `jobs>1` via a thread pool with per-run
isolated contexts. Baseline runs once per (entrypoint, inputs) tuple and is shared
across scenarios — cache it, and record the shared baseline's `run_id` in each
report.

### Engine wiring
Lifecycle steps 8–12. `engine.run` now returns a fully populated `ChaosResult` and
writes a bundle when `write_bundle=True`. `alc run`, `alc report`, `alc validate`,
`alc explain` become real (judge-related flags still limited until phase 06).

## Tests

- Each probe: one positive and one negative fixture trace. Hand-write the fixtures;
  do not generate them from the engine, or a probe bug and an engine bug will cancel
  out.
- `classify_behavior`: a parametrized table covering every `observed_behavior` value.
- `satisfies()` matrix: all 5 expected × all passing/failing observed values.
- `good_agent` (from `tests/fakes/`) passes every preset it can construct — the
  false-positive guard. Any probe that fires on it is wrong until proven otherwise.
- Assertions: one test per check; `no_unsourced_numbers` gets a table including the
  cases that broke the old heuristic (`31C` in the answer vs `31` in the payload →
  sourced; a weekday name → not a finding; a legitimate sum → sourced when
  `allow_derived`; `24C` with nothing to source it → fails).
- A test that no probe reads the clock (assert on the module source, or monkeypatch
  `time` and `perf_counter` to raise during `run_probes`).
- Report parity test (dataclass ↔ schema).
- Golden report test for two scenarios, normalized per `docs/04` §3.
- `AGENT_TASK.md` golden test.
- Matrix expansion: `{faults.0.params.mutation_type: [a, b], seed: [1, 2]}` yields 4
  scenarios with the right ids and the right substituted values.
- Suite: baseline is computed once and reused; `fail_fast` stops early;
  `filter` globs work.
- Regenerate all three fixtures (`report_failing.json`, `trace_excerpt.jsonl`, and
  the `AGENT_TASK.md` golden) from a real fixture run and diff against the committed
  copies; differences must be explainable, and the committed copies are updated in
  this phase's commit (D-40). Pin `flatten()` to the format D-40 specifies, since it
  produces the golden-stable `exact_prompt`.
- `suite.json` is written at `schema_version "1.0"` here; phase 07 bumps it to 1.1
  (D-25). Do not add phase-07 fields now.

## Acceptance checklist

- [ ] `make check` green, coverage ≥85%
- [ ] all 20 probes from the `docs/07` §3 table implemented, with positive+negative fixture tests each
- [ ] `assertions.py` complete, including `no_unsourced_numbers` per `docs/11` §4.3 and the auto-synthesis table
- [ ] `classify_behavior`, `satisfies`, and the `failure_mode` precedence rules match `docs/11` §§6–8, one test per rule
- [ ] no probe fires on the harness's own injection (attribution test per R1–R4)
- [ ] no probe or assertion reads the clock
- [ ] no symptom without evidence (asserted)
- [ ] `good_agent` passes every preset; `naive_tool_agent` fails `tool_contract`
- [ ] dataclass ↔ schema parity test passes in both directions
- [ ] `AGENT_TASK.md` rendering matches the reference format section for section
- [ ] `alc run examples-less` — i.e. `alc run <suite.yaml>` against fake agents —
      produces a run directory with all seven artifacts
- [ ] `alc judge <run_dir>` and `alc report <run_dir>` work from disk with no re-run
- [ ] matrix expansion and all ten presets exercised by tests
- [ ] a probe that raises does not lose the report

## Verify

```bash
make check
pytest tests/probes tests/report -q
alc run tests/data/fake_suite.yaml --judge rules --out .chaos-smoke
cat .chaos-smoke/*/*/AGENT_TASK.md | head -80
alc validate .chaos-smoke/*/*/report.json
```

Then read one `AGENT_TASK.md` end to end as if you were the coding agent receiving
it. If you could not act on it without asking a question, the phase is not done.

## Out of scope

The SLM judge (phase 06) — `RuleJudge` here can be a simple table lookup. The
LangGraph adapter (phase 05). The demo agent (phase 08).
