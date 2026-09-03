# Phase 01 — Core engine, targeting, tracing, vanilla adapter (M1)

The repo skeleton exists. Now build the machinery every later phase plugs into.
This is the phase that determines whether the rest of the library is pleasant or
painful, so favour clarity over cleverness.

**Read first, and treat as binding:** `docs/DECISIONS.md` — pre-seeded errata from an adversarial review of this spec; it overrides any doc it contradicts.

**Read first, in full:** `CLAUDE.md`, `docs/01-ARCHITECTURE.md`,
`docs/02-API.md` §§2–3 and §9 (vanilla part), `docs/04-SCHEMAS.md` §§3–5,
`docs/06-LANGGRAPH-ADAPTER.md` Part 2, `schemas/trace_event.schema.json`.

## Scope

`Crossing`, `RunContext`, targeting, triggering, seeding, redaction, JSON patch,
truncation, `TraceRecorder`, the `ChaosEngine` run lifecycle, and the vanilla
adapter. Exactly **one** trivial fault (`NoopFault`) so the plumbing is testable
end to end. No real faults, no probes, no judges, no report object yet — the run
returns a minimal `ChaosResult` with identity, trace path, metrics, and `success=True`.

## Build

### `seeding.py`
```python
def rng(seed: int, purpose: str) -> random.Random     # PURE FACTORY, no global cache (D-02)
def derive_id(seed: int, kind: str, n: int) -> str    # deterministic, no uuid
def canonical_json(obj: Any) -> str                  # sort_keys, no whitespace drift
def sha256_of(obj: Any) -> str                       # "sha256:<hex>"
```
`rng` seeds from `sha256(f"{seed}|{purpose}")` so purposes are independent.
**There is no module-level memo** — the only cache is `RunContext.rng_registry`
(D-02); a global one would make two runs in one process share a stream, break the
same-seed determinism test, and race under `jobs>1`. Purpose keys are
`f"{fault_key}:{purpose}"` where `fault_key` is the **hash** of the fault spec, not
the ordinal `fault_id` (D-03) — so inserting or reordering faults never re-keys
another fault's stream. Prove both with tests.

Also here: `sanitize_floats(obj)` replacing non-finite floats with
`{"__float__": "NaN"|"Infinity"|"-Infinity"}`, and every dump using
`allow_nan=False` (D-09). A report must be valid JSON for a `jq` or Go consumer.

### `redact.py`
Deny-list and value patterns exactly as `docs/04-SCHEMAS.md` §4. Recursive over
dict/list/tuple/set/dataclass/str. Canary exemption via
`redact(obj, allow=(canary,))`. Never mutates the input.

### `jsonpatch.py`
`diff(before, after) -> list[dict]` producing only `add`/`remove`/`replace` with
RFC-6901 pointers, plus `apply(before, patch) -> after` used only by tests. Handles
dicts, lists (index-based, no move detection), and scalars. On non-JSON-able input
return `[]` and signal `unrepresentable`.

### `context.py`
`Layer`, `Phase`, `Crossing`, `RunContext`, `Counters`, `Limits`, `StateView`, and
`FaultContext` **exactly as `docs/DECISIONS.md` D-01 specifies** (the architecture
doc referenced `FaultContext` without defining it; D-01 is the definition, and
phases 02/03/05 may not add fields to it ad hoc). Also `HarnessFacts` per
`docs/11-OUTCOMES-AND-ASSERTIONS.md` §2, accumulated by the engine as faults fire.

A step is **one agent iteration** — node entry under LangGraph, one LLM call under
vanilla; tool calls and crossings do not increment it (D-05).

### `targeting.py`
`Target`, `Trigger`, `matches(target, crossing) -> bool`,
`should_fire(trigger, crossing, ctx, fault_id) -> tuple[bool, str]` returning a
reason string on both branches (`"fired"`, `"probability_not_met"`,
`"max_fires_reached"`, `"cooldown"`, `"call_index_mismatch"`, …). Globs via
`fnmatch`. Dotted `state_key` matching with `*` segments.
RNG is consumed **only** when `probability < 1.0` — assert that in a test, because
it is what makes deterministic scenarios stable.

### `trace.py`
`EventKind` literal, `Event` dataclass, `TraceRecorder` with a `JsonlSink` and a
`MemorySink`, monotonic `seq` under a lock, `ts_mono_ms` from
`time.perf_counter()`, level filtering, truncation stubs per `docs/04` §5,
redaction on write. `TraceRecorder.load(path)` for the post-run pipeline. Every
event must validate against `trace_event.schema.json`; add a test-only strict mode
that validates every event as it is written and use it throughout the test suite.

### `faults/base.py`
`Fault` ABC, `FaultOutcome`, `MutationLog`, `FaultRecord`, `@register_fault`
decorator, `FAULT_REGISTRY`, `fault_from_dict`, `list_faults`. Plus `NoopFault`
(accepts everything, action `noop`, records a `MutationLog` with an empty patch) as
the plumbing test double.

### `engine.py`
- `__init__` per `docs/02-API.md` §2.
- `register_fault` with eager validation: `accepts` compatibility, glob sanity,
  trigger sanity. Assigns `f1`, `f2`, … Raises `ConfigError` with the offending
  value in the message.
- `plan()` → canonical dict; `plan_hash` per `docs/04` §3.
- `cross(crossing)` — the hot path: emit the layer event, evaluate every armed fault
  in registration order; **value-replacing actions chain** (each fault receives the
  previous one's output and records its own `MutationLog`), while
  `raise`/`delay`/`invoke_target`/`resume_from_checkpoint` are first-wins and mark
  the rest `fault_skipped: superseded` (D-13 — first-wins for everything would make
  the multi-mutation presets exercise only their first fault). Emit `fault_fired` +
  `mutation_applied` per firing, accumulate `HarnessFacts`, return the final value.
- `step()`, limit enforcement raising `LimitExceeded`, contextvar-scoped run state
  so nested/concurrent runs don't share counters.
- `run` / `arun` accepting the full signature from `docs/02-API.md` §2 (including
  `dry_run`, `attempt`, `expect`, `expected_errors`, `allow_side_effects`), with
  `run_id` per D-04 and lifecycle steps 1–7 and 14 from
  `docs/01-ARCHITECTURE.md` §5 (steps 8–12 land in phase 04; leave clearly-marked
  hooks, not TODO comments — call `_post_run(trace, plan)` which currently returns a
  minimal result).
- `dry_run=True` arms faults, records `fault_armed`, never applies — and still
  reports every fault in `injected_faults` with `fired: false` and
  `skipped_reason: "dry_run"` (D-12). A dry run must classify exactly as a baseline.
- **Limits:** `LimitExceeded` derives from `BaseException` so a broad
  `except Exception` in the agent cannot swallow it; it is raised at a crossing,
  caught only by the engine's outermost frame, and converted to `loop.limit_hit` —
  never surfaced in `error` (D-06). Timeouts are **cooperative**: a deadline compared
  at every crossing, plus `asyncio.wait_for` in `arun` (D-08). Do not reach for
  `signal.alarm`.
- **Fault composition:** value-replacing actions chain in registration order, each
  recording its own `MutationLog`; `raise`/`delay`/`invoke_target`/
  `resume_from_checkpoint` are first-wins and supersede the rest (D-13).
- **Markers:** `engine.validated()`, `engine.note()`, and `engine.bind_context()`
  (D-42), all inert outside a run.
- The run directory and `trace.jsonl` are opened at step 2 with a flush per event,
  so a reader can follow a run live; `plan.json` is written as soon as the plan is
  frozen.
- **Internal errors never propagate:** wrap every engine/fault call site so an
  exception inside the library becomes an `internal_error` event and the original
  value passes through untouched. Test it with a fault that raises on purpose.

### `adapters/base.py`, `adapters/vanilla.py`
`Adapter` protocol; `engine.tool`, `engine.llm`, `wrap_tools`, `wrap_callable`,
`intercept_tools`, `instrument_object` per `docs/06` Part 2. Sync and async
wrappers, `functools.wraps` + signature preservation, contextvar inertness outside a
run, message normalization for the `llm` layer (string → one user message; list of
dicts as-is; anything else → `messages_unavailable`).

### `report.py` (minimal for now)
`ChaosResult` dataclass with all fields from `docs/02-API.md` §5 present and
defaulted, `to_dict`, `to_json`, `validate()` via `schema.py`. Phase 01 fills only
identity, target, metrics, loop, artifacts, reproduce, randomness,
`injected_faults`, `chaos_narrative` (empty string), `success=True`,
`failure_mode="none"`, a stub `verdict`. It must already validate against the schema
— that is the point of doing it now.

## Tests to write

- Targeting truth table: ≥20 (Target, Crossing) pairs, parametrized.
- Trigger truth table: `on_call` scalar and list, `on_step`, `after_step`,
  `max_fires`, `cooldown_calls`, `probability` 0/0.5/1 with a fixed seed.
- `rng` independence: registering a second fault leaves the first's draws identical.
- Redaction property test (hypothesis) per `docs/07` §7.
- `apply(diff(a, b), a) == b` property test.
- Trace: every event validates; truncation stub shape; `load()` round-trips.
- Engine: `NoopFault` fires exactly once with `on_call=2` against a 3-call fake tool.
- Engine resilience: a fault whose `apply` raises produces `internal_error` and the
  original value; the run still completes and `success` is unaffected.
- Vanilla adapter: sync tool, async tool, dict-of-tools, decorated tool called
  outside a run (no interception, <5 µs overhead), `intercept_tools` scoping,
  class-based agent.
- Determinism: two runs, same seed, normalized reports equal.
- Limits: `max_steps` and `max_tool_calls` produce `limit_hit`, not an exception in
  the result's `error` field.

## Acceptance checklist

- [ ] `make check` green; coverage ≥85% on the modules touched
- [ ] a fake agent runs under `engine.run()` and writes a schema-valid `trace.jsonl`
- [ ] the minimal `report.json` validates against `chaos_report.schema.json`
- [ ] targeting and trigger truth tables pass, including the RNG-independence test
- [ ] an exception inside a fault never reaches the agent under test
- [ ] `dry_run=True` arms faults, applies none, and the trace proves it
- [ ] decorated tools are inert outside a run
- [ ] async and sync paths both tested for tool and llm layers
- [ ] `docs/02-API.md` updated if any signature changed, with the change listed in
      your final message

## Verify

```bash
make check
python - <<'PY'
from agent_loop_chaos import ChaosEngine
e = ChaosEngine(seed=1337, out_dir=".chaos-smoke")
@e.tool
def t(x): return {"a": 1, "x": x}
def agent(q, state): return t(q)
r = e.run(agent, inputs="hi", initial_state={})
print(r.summary_line()); print(r.validate())
PY
```

## Out of scope

Real faults, probes, judges, the LangGraph adapter, `AGENT_TASK.md`. `NoopFault` is
the only fault that exists at the end of this phase.
