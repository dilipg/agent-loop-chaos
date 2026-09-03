# Phase 07 — The continuous refinement loop (M7)

Single runs produce findings. This phase closes the loop: run the suite, hand the
work orders out, re-run, and report what actually got fixed — including catching the
case where a "fix" was really a weakened test.

**Read first, and treat as binding:** `docs/DECISIONS.md` — pre-seeded errata from an adversarial review of this spec; it overrides any doc it contradicts.

**Read first, in full:** `docs/05-JUDGE-AND-LOOP.md` §§8–10, `docs/02-API.md` §8,
`docs/04-SCHEMAS.md` §8.

## Scope

`loop.py` (`RefinementLoop`, `LoopReport`), tamper detection, `suite.json`
enrichment, `alc run --rounds`, and `examples/refine_with_claude_code.py`. No new
faults, no new probes.

## Build

### `RefinementLoop`
Per `docs/02-API.md` §8. Behaviour:

- Round 1 runs the full suite (baseline once, shared).
- `on_findings(failures)` is called with the failing results. The library does
  nothing else with them — no subprocess, no source editing, ever.
- Subsequent rounds re-run **the whole suite**, not just the failures, so a fix that
  broke something else is caught. Same seeds every round: a scenario's `run_id`
  differs only by the attempt counter.
- `stop_when`: `all_pass`, `no_new_failures` (default — stop when a round surfaces
  no failure the previous round did not have), `rounds`.
- Classification per scenario per round: `pass`, `fail`, `flipped_to_pass`,
  `flipped_to_fail` (a regression), `still_failing`, `skipped`.

### Tamper detection (do not skip this)
Before each round, for every scenario compute and store:
`plan_hash`, `sha256` of the scenario file, and `sha256` of the sorted `must_not`
list. Then:

- A scenario that flipped to pass **and** whose plan hash or scenario-file hash
  changed is reported in `LoopReport.regressions` as
  `"<id>: passed after the scenario was modified (plan_hash changed)"`, never as a
  fix.
- The `control.dry_run` scenario failing at any point aborts the loop with a clear
  message: the harness itself has been altered.
- Additionally hash the installed library's own probe module set; if it changed
  mid-loop, say so. A coding agent editing the chaos library to make its agent pass
  is the failure mode to guard against, and it is not hypothetical.

### `LoopReport`
`to_dict()` and `markdown()`. The markdown is a round-by-round table:

```
| scenario | r1 | r2 | r3 | outcome |
|---|---|---|---|---|
| tool.drop_required_key | FAIL | PASS | PASS | fixed in r2 |
| loop.pinned_tool_output | FAIL | FAIL | FAIL | still failing (max_iterations_exhausted) |
| state.drop_location | PASS | FAIL | FAIL | REGRESSION in r2 |
```

Plus a header with totals, the failure-mode histogram per round, the judge
disagreement rate, total wall time, and the list of `AGENT_TASK.md` paths. This is
what a human reads after an unattended loop, so make it scannable.

### `suite.json` enrichment
Bump `schema_version` to **1.1** (D-25 — adding fields at 1.0 violates the additive
evolution rule) and add `rounds`, `flipped`, `regressions`, `tasks_written`,
`judge_disagreement_rate`, `wall_ms` and `tamper` (the hash record), alongside the
live fields phase 10 also needs (`status`, `planned`, `current`, `round`,
`rounds_planned`, nullable `finished_at`). `judge_disagreement_rate` is its own
top-level field — **not** inside `coverage`, whose namespace is fault kinds. Write
`schemas/suite.schema.json` in this phase, since the loop and the dashboard both
consume the file, and validate what you write against it.

### CLI
`alc run <suite> --rounds N [--stop-when …]` runs the loop with no `on_findings`
hook — it writes task files and prints the markdown. Exit code 1 if any scenario
still fails in the final round; exit 4 if tampering was detected.

### `examples/refine_with_claude_code.py`
Per `docs/09-DEMO-AGENT.md` §7. Default mode prints findings and task paths.
`--apply` shells out per task file, requires a clean git tree (check with
`git status --porcelain`), refuses to run on the default branch, and prints a loud
warning first. Between rounds it prints the markdown table. It must work even if no
coding-agent CLI is installed — in that case, print the tasks and exit 0 with a
message explaining what `--apply` would have done.

## Tests

- Scripted two-round loop with a fake `on_findings` that "fixes" a fake agent by
  swapping an implementation: assert `flipped_to_pass`, and that the final round
  re-ran everything.
- Regression detection: `on_findings` that breaks a different scenario; assert it
  appears in `regressions`.
- Tamper detection: `on_findings` that mutates the scenario file; assert the flip is
  reported as a regression, not a fix.
- Control-run abort: make `control.dry_run` fail; assert the loop stops with the
  right message and exit code.
- `stop_when="no_new_failures"` stops at the right round; `max_rounds` respected.
- `LoopReport.markdown()` golden test.
- Seeds are stable across rounds (same `plan_hash` each round for an unmodified
  scenario).
- `refine_with_claude_code.py` runs in default mode inside the test suite with a
  fake agent CLI on PATH, and does not touch git.

## Acceptance checklist

- [ ] `make check` green
- [ ] two-round loop reports flipped / still-failing / regressions correctly
- [ ] a scenario modified between rounds is reported as a regression, never as a fix
- [ ] control dry-run failure aborts the loop with exit code 4
- [ ] `LoopReport.markdown()` matches its golden and is readable at a glance
- [ ] `suite.json` carries rounds, tamper hashes, and the disagreement rate
- [ ] `alc run --rounds 2` works end to end on the fake suite
- [ ] `examples/refine_with_claude_code.py` is safe by default and loud about `--apply`

## Verify

```bash
make check
pytest tests/loop -q
alc run tests/data/fake_suite.yaml --rounds 2 --judge rules --out .chaos-loop
python examples/refine_with_claude_code.py --suite tests/data/fake_suite.yaml
```

## Out of scope

The demo agent (phase 08) and the CLI polish/release work (phase 09). Do not let the
loop edit source files under any flag.
