# Phase 02 — Tool-execution faults and the mutation library (M2)

The engine can route crossings and record traces. Now give it something to inject.

**Read first, and treat as binding:** `docs/DECISIONS.md` — pre-seeded errata from an adversarial review of this spec; it overrides any doc it contradicts.

**Read first, in full:** `docs/03-FAULT-CATALOG.md` section A (all ten faults) and
the shared conventions above it; `docs/01-ARCHITECTURE.md` §4 (`Fault`,
`FaultOutcome`, `MutationLog`); `docs/07-TESTING.md` §§2–3.

## Safety gate — build this first

`ArgumentTamperFault`, `DuplicateSideEffectFault` (and `CheckpointRollbackFault` in
phase 05) perform **real operations the agent never requested**. Before writing any
of them, implement the gate from `SAFETY.md` §1 / D-23: `register_fault` refuses
them against a tool declared `side_effecting=True` unless the scenario lists that
tool in `allow_side_effects`; glob targets never match a side-effecting tool; and
`--preset full` refuses to start while any registered tool has `side_effecting`
undeclared. Tests: each refusal path, and a test that a glob target skips a
side-effecting tool with a recorded reason.

`DuplicateSideEffectFault` uses the new `invoke_target` action (D-10) — the
**adapter** performs the repeat invocations, not `apply()`, and each one is recorded
into `HarnessFacts.harness_invocation_seqs` so the `duplicate_side_effect` probe can
exclude the harness's own calls.

## Scope

`mutations.py` and every fault in catalog section A: `ToolCorruptionFault`,
`ToolErrorFault`, `ToolLatencyFault`, `ToolTimeoutFault`, `ArgumentTamperFault`,
`StaleDataFault`, `NonDeterminismFault`, `DuplicateSideEffectFault`,
`LoopTrapFault`, `RateLimitFault`. Nothing about LLMs, state, or probes.

## `mutations.py`

Every mutation is a pure function `(value, rng, **params) -> new_value` registered
in `MUTATIONS: dict[str, Mutation]`. Implement all names listed in catalog A1.
Requirements:

- Deep-copy first; never touch the input. A `hypothesis` test asserts input
  identity is preserved.
- Deterministic key selection: when `keys is None`, choose via
  `rng.sample(sorted(candidate_paths), count)` — sorted, so dict ordering cannot
  leak in.
- Walk nested containers to `depth`; a mutation applied to a list of records applies
  to **every** record unless `count` says otherwise, and the `note` reports
  `"… from 3 of 3 records"`.
- `unit_swap` needs a small unit table (`temp_c`↔`temp_f`, `km`↔`mi`, `kg`↔`lb`,
  `usd`↔`eur` at a fixed rate) and must change the *value* while leaving the *label*
  intact. This is the fault that catches agents that never sanity-check magnitudes.
- `nan_numbers` emits the canonical encoding from D-09:
  `{"__float__": "NaN"}` (or `"Infinity"` / `"-Infinity"`). There is **no**
  `unrepresentable_number` field — that was never in the schema, and a bare `NaN`
  would make `report.json` invalid JSON, break
  `apply(json_patch, before) == after`, and make golden equality unstable because
  `nan != nan`.
- `malformed_json_string` returns a `str` that is deliberately broken JSON; do not
  return a dict.
- Each mutation has a one-line docstring naming the real-world case it mimics.

`MUTATIONS` is also used by `ArgumentTamperFault` and (in phase 05) the state
faults, so keep it free of any tool-specific assumptions.

## Faults

For each: `kind`, `accepts`, `severity_hint`, eager param validation, `apply()`
returning a `FaultOutcome` with a populated `MutationLog` and a human `note`.
Specific requirements beyond the catalog:

- `ToolErrorFault` — build the exception from `exc_class` via a small allow-list
  (`RuntimeError`, `TimeoutError`, `ConnectionError`, `ValueError`, `KeyError`,
  and a library-provided `SimulatedToolError`); never `eval`. `"error_payload"`
  returns rather than raises, and records `in_band: true`.
- `ToolLatencyFault` — clamp to `limits.max_injected_delay_ms`, record both the
  requested and the applied delay, and use `asyncio.sleep` on async crossings.
  Accumulate into `metrics.injected_delay_ms` so the latency probe can subtract it.
- `StaleDataFault` — needs a per-tool result history on the run context. Add
  `ctx.history[name] -> list[result]` in the engine (a small addition; update
  `docs/01-ARCHITECTURE.md` §4 if you change the shape).
- `LoopTrapFault` — pins to a deep copy of the first observed result and keeps
  returning it. It must interact correctly with `Limits.max_steps`: the run ends with
  `limit_hit="max_steps"` and no exception in `error`.
- `DuplicateSideEffectFault` — calls the wrapped tool `times` times; the engine must
  emit a `tool_call_requested`/`returned` pair per real invocation so the
  duplicate-side-effect probe can count them in phase 04.
- `RateLimitFault` — stateful across calls via `ctx.counters`; returns a 429-shaped
  payload or raises, per `error_type` conventions shared with `ToolErrorFault`.

## Tests

For every fault, four tests (parametrize hard; this is 40 tests, keep them terse):

1. **unit** — `apply()` on a synthetic `Crossing` produces the expected
   `FaultOutcome.action` and value.
2. **trigger** — fires exactly on the configured call/step and no other.
3. **mutation log** — `payload_before`, `payload_after`, and
   `apply(json_patch, before) == after`; `note` is non-empty and mentions the field
   or count.
4. **end-to-end** — against a fake agent from `tests/fakes/`, produces a run whose
   trace contains `fault_fired` and whose result carries the fault in
   `injected_faults` with `fired: true`.

Plus:

- Every mutation name in `MUTATIONS` appears in at least one test (assert coverage
  programmatically: iterate `MUTATIONS` and fail on any name with no test id).
- `ToolCorruptionFault(mutation_type="random")` is reproducible under a fixed seed
  and selects different mutations under different seeds.
- Latency clamping test.
- A test that `NonDeterminismFault` yields different values across calls but the
  *sequence* is identical across runs with the same seed.

## Acceptance checklist

- [ ] `make check` green
- [ ] all 10 section-A faults implemented, registered in `FAULT_REGISTRY`, and
      constructible from a dict via `fault_from_dict`
- [ ] every name in catalog A1's mutation list exists in `MUTATIONS` and is tested
- [ ] `alc list-faults` prints all 10 with their params and one-line descriptions
- [ ] no mutation mutates its input (property test)
- [ ] `apply(diff) == after` holds for every fault's mutation log
- [ ] `LoopTrapFault` end-to-end run ends with `limit_hit="max_steps"` and
      `error is None`
- [ ] `metrics.injected_delay_ms` is populated by the latency faults
- [ ] docstrings state what weakness each fault proves and what graceful looks like

## Verify

```bash
make check
alc list-faults
pytest tests/faults -q --cov=agent_loop_chaos.faults --cov=agent_loop_chaos.mutations
```

## Out of scope

LLM faults, state faults, probes, judges. If a fault needs a probe to be meaningful,
note it in your final message; phase 04 will add it.
