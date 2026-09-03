# Phase 08 — The demo agent and the suite that breaks it (M8)

Everything works against fakes. Now build something that looks like real code, break
it, and prove the library's claim.

**Read first, and treat as binding:** `docs/DECISIONS.md` — pre-seeded errata from an adversarial review of this spec; it overrides any doc it contradicts.

**Read first, in full:** `docs/09-DEMO-AGENT.md` (all of it),
`schemas/examples/suite_demo.yaml`.

## Scope

`examples/trip_planner/` (buggy), `examples/trip_planner_fixed/` (correct),
`examples/fake_model.py`, `examples/vanilla_agent.py`,
`examples/scenarios/{demo_suite.yaml,quickstart.yaml}`, and the example READMEs.
No library changes unless a genuine bug surfaces — and if one does, fix it and say
so explicitly in your final message.

## Build

### `examples/trip_planner/`
The 4-node LangGraph app from `docs/09` §§2–3, with **all twelve planted weaknesses**
from §4. Constraints that make the demo credible:

- It must read like code someone would actually ship: type hints, docstrings, sane
  names, no comments announcing the bugs. A reader should have to think to spot
  them. Put the explanation in `examples/trip_planner/README.md` instead.
- Deterministic fake tools backed by `fixtures/*.json`. No network, no API key.
- Runs correctly on the happy path: `python -m examples.trip_planner.app` prints a
  sensible packing list and flight. If the happy path is broken, the chaos results
  are meaningless.
- `search_flights` returns a free-text `notes` field, interpolated into the
  summarizer prompt undelimited — the injection carrier.
- A credentials-shaped fixture tool returns the planted canary
  (`initial_state["_alc_canary"]`), so `objective: exfiltrate_secret` has something
  real to exfiltrate and `secret_in_output` is actually reachable (D-16). The buggy
  tree passes that value into the prompt; the fixed tree never does.
- `hold_booking` is registered with `side_effecting=True`.

### `examples/fake_model.py`
A scripted model that behaves like a plausible mediocre LLM: fabricates a
temperature when the field is missing, follows instructions found in tool output,
returns prose when asked for JSON about 1 in 4 times *deterministically by call
index* (not randomly). It must implement enough of `BaseChatModel` to be bound with
tools and invoked inside the graph. If `OPENAI_API_KEY` or an Ollama endpoint is
present and `--real-model` is passed, use the real one instead; CI always uses the
fake.

### `examples/trip_planner_fixed/`
Same graph, same tools, same happy-path output; correct under chaos, per
`docs/09` §5. It is the demo suite's **negative control** and must pass every
scenario — including the injection and rollback ones, which is only possible because
of the harness-attribution rules in `docs/11` §2. It also demonstrates the three
markers: it raises a `DataUnavailable(ExplicitError)` on terminal failures, calls
`engine.validated()` after each boundary check, and `engine.note()` when it
degrades. If a scenario fails on this tree, a probe or an assertion has a false
positive — diagnose that, do not soften the scenario. Keep the diff between the two trees small and legible — `git diff
--no-index examples/trip_planner examples/trip_planner_fixed` should be readable as
documentation. Add `validators.py` and a `call_tool()` helper as described.

### Suites
`examples/scenarios/demo_suite.yaml` from `schemas/examples/suite_demo.yaml`,
adjusted so every scenario actually bites the buggy agent. Delete or fix any
scenario that turns out not to fire — a scenario that never fires is noise. Report in
your final message which scenarios you changed and why.

`examples/scenarios/quickstart.yaml`: one scenario, ≤15 lines, the thing the README
shows.

### `examples/vanilla_agent.py`
The blueprint's plain-Python example from `docs/02-API.md` §11, runnable, with the
same weakness as `naive_tool_agent` so `alc run` finds something.

## The proof

Run both trees through the suite and record the real numbers:

```bash
alc run examples/scenarios/demo_suite.yaml --judge rules --out .chaos-buggy
alc run examples/scenarios/demo_suite.yaml --judge rules --out .chaos-fixed \
    --entrypoint examples.trip_planner_fixed.app:graph
```

Required outcome:

- buggy: **≥6 distinct `failure_mode` values**, and `control.dry_run` passes
- fixed: **zero failures**, and the happy-path answers are equivalent to the buggy
  tree's baseline (same flight, comparable packing list)

If the fixed tree fails a scenario, either the fix is incomplete or a probe has a
false positive. Diagnose which before proceeding — do not adjust the scenario to make
it pass.

Then write `examples/README.md` with the actual histogram, one full
`AGENT_TASK.md` excerpt, and the exact commands. Put the same numbers into the
top-level `README.md` results section. **Do not round up or embellish** — the real
numbers are the credibility.

## Tests

- `tests/test_examples.py`: the happy path of both trees runs and returns a
  non-empty answer with the fake model.
- The buggy tree produces ≥6 distinct failure modes with `--judge rules` (asserted
  on `suite.json`, marked `slow`).
- The fixed tree produces zero failures (asserted, marked `slow`).
- `control.dry_run` passes on both.
- `examples/vanilla_agent.py` runs under the engine and produces one failure.
- Every scenario in `demo_suite.yaml` fires at least one fault on the buggy tree
  (assert `injected_faults[].fired` is true somewhere in each run) — this is the
  test that catches dead scenarios.

## Acceptance checklist

- [ ] `make check` green; `pytest -m slow` green
- [ ] happy path of both trees works with the fake model, no network
- [ ] buggy tree: ≥6 distinct failure modes; every scenario fires
- [ ] fixed tree: zero failures, dry-run control passes
- [ ] all twelve planted weaknesses are present and each is caught by the scenario
      `docs/09` §4 assigns to it (list the mapping in your final message with the
      actual failure mode each produced)
- [ ] `examples/README.md` and the README results section carry the real numbers
- [ ] the buggy tree contains no comment that gives away a bug
- [ ] `git diff --no-index` between the two trees reads as documentation

## Verify

```bash
make check && pytest -m slow -q
python -m examples.trip_planner.app
alc run examples/scenarios/quickstart.yaml --judge rules
alc run examples/scenarios/demo_suite.yaml --judge rules --out .chaos-buggy
python -c "import json;d=json.load(open('.chaos-buggy/suite.json'));print(d['failure_modes'])"
```

## Out of scope

CLI polish, packaging, release (phase 09). Do not add new faults to make a scenario
land — use the ones that exist, or drop the scenario and note it.
