# 04 — Schemas and output contracts

The schemas in `schemas/` are the product. Code exists to fill them. Copy them
verbatim into `src/agent_loop_chaos/schemas/` in phase 00 and ship them as package
data (`importlib.resources`), so a consumer can validate a report without cloning
the repo.

| File | Describes |
|---|---|
| `chaos_report.schema.json` | `report.json` — one run's full finding |
| `trace_event.schema.json` | one line of `trace.jsonl` |
| `judge_verdict.schema.json` | the `verdict` block, and the structured output the SLM must return |
| `scenario.schema.json` | suite/scenario YAML or JSON files |

Golden instances live in `schemas/examples/`:
`report_failing.json`, `trace_excerpt.jsonl`, `suite_demo.yaml`,
`AGENT_TASK.example.md`. Tests must validate the first three against their schemas
and treat `AGENT_TASK.example.md` as the format `bundle.py` renders.

## 1. Cross-file `$ref` resolution

`judge_verdict.schema.json` references `chaos_report.schema.json` and vice versa.
Use `jsonschema` ≥4.18 with a `referencing.Registry` built from all four packaged
schemas, keyed by `$id`. Never fetch over the network — build the registry from
local resources and pass `registry=` to the validator. Wrap it once:

```python
# agent_loop_chaos/schema.py
def validator_for(name: Literal["report", "trace", "verdict", "scenario"]) -> Validator: ...
def validate_obj(obj, name) -> list[str]:  # returns human-readable error strings
```

Error strings must include the JSON pointer and the failing value, truncated to
200 chars, because they surface in `schema_errors[]` and in CI logs.

## 2. Versioning

`schema_version` is `MAJOR.MINOR`, starting at `1.0`.

- **MINOR** bump: added an optional property, added an enum value, relaxed a
  constraint. Old consumers keep working.
- **MAJOR** bump: removed or renamed a property, tightened a type, changed a
  field's meaning. Requires an entry in `docs/DECISIONS.md` and a migration note in
  `CHANGELOG.md`.

Consumers must ignore unknown properties despite `additionalProperties: false` in
the schema — the strictness is for *producers*, so the library never emits a field
it has not documented. Say this explicitly in the README.

## 3. Determinism rules for serialization

A seeded run must produce a byte-identical `report.json` on re-run after
normalization. To make that true:

- `json.dumps(..., sort_keys=True, ensure_ascii=False, separators=(",", ": "))`
  for the canonical form used in hashing; the on-disk report is `indent=2` with
  sorted keys.
- `plan_hash = "sha256:" + sha256(canonical_json(plan))`. The plan dict contains
  only: seed, ordered fault specs (id, type, sorted params, target, trigger),
  limits, adapter, entrypoint string, expected_behavior, must_not (sorted). Nothing
  time- or path-dependent.
- `run_id = "run-" + sha256(f"{seed}|{scenario_id}|{plan_hash}|{attempt}")[:8]`.
  Never `uuid4`.
- Ids for spans and faults are counter-derived: `s1`, `s2`, … and `f1`, `f2`, …
- Non-deterministic fields, all confined to a known list so the normalizer can
  strip them: `started_at`, `finished_at`, `duration_ms`, `metrics.wall_ms`,
  `metrics.cost_usd`, every `duration_ms`/`latency_ms`, `ts`, `ts_mono_ms`,
  `judge_meta.latency_ms`, absolute paths under `artifacts`, `library_version`,
  `reproduce.env`. `tests/normalize.py` owns that list and nothing else may add to
  it without a test update.
- Floats are rounded to 6 decimals before serialization. Non-finite floats are
  encoded as `{"__float__": "NaN" | "Infinity" | "-Infinity"}` and every dump uses
  `allow_nan=False`, so a report is always valid JSON for a non-Python consumer
  (`docs/DECISIONS.md` D-09).

## 4. Redaction

Applied to every value before it enters a trace event or a report field.

- Key deny-list (case-insensitive regex):
  `api[_-]?key|secret|token|password|passwd|authorization|auth|bearer|cookie|
  session|credential|private[_-]?key|access[_-]?key`.
- Value patterns: `sk-[A-Za-z0-9]{16,}`, `ghp_[A-Za-z0-9]{20,}`, `AKIA[0-9A-Z]{16}`,
  `xox[bap]-[A-Za-z0-9-]+`, `-----BEGIN [A-Z ]*PRIVATE KEY-----`, JWTs
  (`eyJ[A-Za-z0-9_-]{10,}\.`), and anything the user passed in
  `ChaosEngine(redact_keys=…)`.
- Replacement: `"<redacted:api_key>"` (the matched reason, not the value).
- Redaction runs **before** truncation and **before** hashing, so a hash never
  leaks a secret's length distinctly enough to matter.
- Exception: the `secret_in_output` probe needs to know whether a *canary* secret
  reached the output. Use planted canaries (`ALC-CANARY-<run_id>`), never real
  values, and exempt only the canary from redaction.

## 5. Truncation

Large payloads are replaced by a stub, never silently cut:

```json
{"__truncated__": true, "bytes": 91234, "sha256": "…", "head": "first 512 chars…",
 "path": "payloads/seq-0019.json"}
```

Thresholds (configurable): 8 KB per payload in `standard`, 64 KB in `verbose`,
2 KB in `minimal`. When a payload is truncated in `verbose` mode, write the full
value to `payloads/seq-<seq>.json` inside the run dir and set `path`.
`payload_before`/`payload_after` in `injected_faults` get double the budget,
because those diffs are the whole point of the report.

## 6. The four report sections, and why each exists

A consumer reads `report.json` in this order. Keep every section populated even
when empty, so a coding agent never has to handle a missing key.

1. **Identity + reproduction** (`run_id`, `seed`, `plan_hash`, `reproduce`) — can
   the finding be re-created? Without this the report is an anecdote.
2. **What chaos was introduced** (`injected_faults[].fires[]`, `randomness`,
   `chaos_narrative`) — the before/after diff and the RNG audit. This is the
   section that answers your requirement *"what randomness/chaos was introduced"*.
3. **How the system behaved** (`symptoms`, `llm_exchanges`, `tool_calls`, `loop`,
   `error`, `agent_state_*`, `delta_vs_baseline`, `metrics`) — observed behaviour,
   with the exact prompt and raw response, plus the contrast against a clean run.
4. **What to do about it** (`failure_mode`, `root_cause_hypothesis`,
   `refinement_hint`, `suggested_fixes`, `code_pointers`) — the actionable layer.
   Everything here is advisory and must be labelled as such; only `symptoms` and
   `success` are authoritative.

## 7. `success` and `failure_mode` computation (deterministic)

> **This section is superseded by `docs/11-OUTCOMES-AND-ASSERTIONS.md`**, which is
> the normative pass/fail authority: it defines `HarnessFacts` and the attribution
> rules, the assertions layer, the `observed_behavior` rule list, the corrected
> `satisfies()` table, the `failure_mode` precedence rules, `dominant_symptom`, and
> `PROBE_PRECEDENCE`. The sketch below is kept only to show the shape.

```
symptoms          = run_probes(trace)
blocking          = [s for s in symptoms if s.code in scenario.must_not]
observed_behavior = classify_behavior(trace, symptoms)     # pure function, table-driven
success           = (not blocking) and satisfies(observed_behavior, expected_behavior)
failure_mode      = FAILURE_MODE_TABLE[(observed_behavior, dominant_symptom)]
severity          = max(severity of contributing symptoms, fault.severity_hint)
```

`satisfies()` table:

| expected | passing observed values |
|---|---|
| `graceful_degradation` | `graceful_degradation`, `explicit_error`, `retried_then_succeeded`, `aborted_with_message`, `completed_unaffected` |
| `explicit_error` | `explicit_error`, `aborted_with_message` |
| `retry_then_succeed` | `retried_then_succeeded`, `completed_unaffected` |
| `abort_with_message` | `aborted_with_message`, `explicit_error` |
| `ignore_and_continue` | `completed_unaffected`, `graceful_degradation` |

`classify_behavior` is rule-list-driven and unit-tested exhaustively — one test per
rule in `docs/11` §6. The SLM never touches it. When the model's `failure_mode`
differs, keep the rules' value and write the model's into
`verdict.judge_disagreement`.

Two corrections `docs/11` makes to the table above, both load-bearing:
`completed_unaffected` no longer satisfies `graceful_degradation` (it was letting an
agent that sailed past a corrupted payload report success), and every assertion must
pass for `success` to be true.

## 8. Suite-level output

A suite run writes `<out_dir>/suite.json`:

```json
{
  "schema_version": "1.1",
  "status": "running",
  "started_at": "…", "finished_at": null,
  "seed": 1337,
  "planned": ["tool.drop_required_key", "…"],
  "current": "tool.unit_swap",
  "round": 1, "rounds_planned": 1,
  "passed": 14, "failed": 7, "skipped": 1,
  "failure_modes": {"hallucination_on_corrupt_data": 3, "infinite_loop": 1},
  "coverage": {"ToolCorruptionFault": 7, "PromptInjectionFault": 4},
  "results": [{"scenario_id": "…", "run_id": "…", "success": false,
               "failure_mode": "…", "severity": "high", "report": "…/report.json",
               "agent_task": "…/AGENT_TASK.md"}],
  "worst": ["adversarial.injection_corpus[exfiltrate_secret]", "loop.pinned_tool_output"],
  "tasks_written": ["…/AGENT_TASK.md"]
}
```

`suite.json` is what a CI job, an optimizer, or the dashboard polls. Give it its own
schema in a later minor version; for now the shape above is fixed by a golden test.

Until phase 10 it is written once, at the end of the suite, with
`schema_version: "1.0"`, `status` absent, and `finished_at` set. Phase 10 makes it
live: written at suite start and updated after every scenario, atomically (temp file
+ `os.replace`), with the additive `1.1` fields shown above — `status`
(`running`/`completed`/`aborted`), `planned`, `current`, `round`, `rounds_planned`,
and a nullable `finished_at`. Additive only, so a v0.1 consumer keeps working.

## 9. Directory layout of outputs

```
.chaos/
├── suite.json
├── <scenario_id>/
│   └── <run_id>/
│       ├── report.json
│       ├── trace.jsonl
│       ├── plan.json
│       ├── judge.json
│       ├── baseline.diff
│       ├── AGENT_TASK.md          (failures only)
│       └── payloads/seq-*.json    (verbose truncation overflow)
└── latest -> <scenario_id>/<run_id>      (symlink; skip on platforms without symlinks)
```

Scenario ids are already constrained to filesystem-safe characters by
`scenario.schema.json`; matrix suffixes `[k=v]` are slugified to `-k-v`.

**Write ordering.** From phase 10 onward the run directory is created and
`trace.jsonl` opened at lifecycle step 2, `plan.json` written as soon as the plan is
frozen, and the sink flushed after every event — so a reader can follow a run while
it happens. `report.json`, `AGENT_TASK.md`, `judge.json` and `baseline.diff` are
still written at the end. A reader must therefore treat a run directory with no
`report.json` as *in progress*, not as broken, and must tolerate a partially written
final line in `trace.jsonl`.
