# Contributing

`docs/` is the specification and it is authoritative. `docs/DECISIONS.md` outranks
every other document; `docs/11-OUTCOMES-AND-ASSERTIONS.md` is the sole pass/fail
authority. Read `CLAUDE.md` before opening a PR — it is the build contract, and it
applies to humans too.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make doctor          # confirms the environment resolves; explains it if not
make check          # ruff, mypy --strict, pytest, schema parity
```

## Adding a fault

1. Subclass `Fault` in the right family module (`faults/tool.py`, `faults/llm.py`,
   `faults/state.py`, `faults/injection.py`). Give it `kind`,
   `accepts` — the `(layer, phase)` pairs it handles — and `apply`.
2. Decorate with `@register_fault` so it lands in `FAULT_REGISTRY`.
3. Operate on a deep copy. Never mutate the caller's object. If a value cannot be
   copied, record `mutation_skipped_uncopyable` and pass the original through.
4. The docstring must state **what agent weakness the fault proves** and **what
   graceful behaviour looks like**.
5. Draw all randomness from `ctx.rng(purpose)`. No bare `random.*`, no clock.
6. Tests, all four: the mutation in isolation; that it fires exactly when its
   trigger says; that `payload_before` / `payload_after` / `json_patch` are correct
   and `apply_patch(before, patch) == after`; and an end-to-end run against a fake
   agent asserting the resulting `failure_mode`.
7. Add it to `docs/03-FAULT-CATALOG.md`. `tools/verify_pack.py` fails if a catalog
   fault is missing from `docs/02-API.md`.

A fault that performs a real action against the target — see `SAFETY.md` §1 — must
respect the D-23 gate: refuse a `side_effecting=True` tool without an explicit
`allow_side_effects` opt-in, and never match one via a glob.

## Adding a mutation

A pure function `(value, rng, **params) -> value`, registered as
`MUTATIONS["name"]`. Pure means: no I/O, no clock, no global RNG.

## Adding a probe

1. Subclass `Probe` with a `code` and `detect(trace, ctx)`, and append it to
   `PROBES`. `ctx` carries metrics, limits, the baseline reference, the assertions
   and `HarnessFacts`.
2. **A probe must never fire on what the harness itself injected.** Apply the four
   attribution rules in `docs/11-OUTCOMES-AND-ASSERTIONS.md` §2 (R1–R4). This is not
   advice: a probe that fires on `good_agent` or on `examples/trip_planner_fixed`
   has a false positive, and the probe is wrong until proven otherwise.
3. No clock. No `ts`, no `ts_mono_ms`, no `time.*` (D-47).
4. Positive **and** negative fixture tests, plus a row in `docs/07-TESTING.md` §3
   and an entry in `PROBE_PRECEDENCE` in `docs/11` §8. `tools/verify_pack.py`
   asserts those two stay in sync.

## Adding a judge or an adapter

A judge implements the `Judge` protocol and may only author `failure_mode`,
`root_cause_hypothesis`, `refinement_hint`, narration and ranked fixes. **A judge
never decides `success`.** If a judge contradicts the probes, the probes win and the
disagreement is recorded in `verdict.judge_disagreement`.

An adapter implements `instrument()`, `run()` and `state_of()`, and lives in
`adapters/<name>.py`. Its framework must be an optional extra, imported locally,
raising `MissingExtraError` when absent.

## Ground rules

- No new required dependency. `jsonschema` is the only one.
- Never weaken or delete a failing test to get a green suite. If a test is wrong,
  say so and explain why in a commit that touches only that test.
- Report fields and schema fields move together: no field the schema does not
  describe, no schema field nothing writes.
- `logging` with the `agent_loop_chaos` logger. No `print()` in library code.
