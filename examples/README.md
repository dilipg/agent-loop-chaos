# Examples

Two things live here: **the demo agent**, which proves the library finds real bugs,
and **the pattern pool**, which proves it attaches to whatever shape your agent is.

## The demo agent

`trip_planner/` is a four-node LangGraph app that answers *"3-day packing list for
Paris, and the cheapest flight from Delhi."* It reads like a first working version,
because that is what it is: type hints, sensible names, and no comment anywhere
announcing a problem. Twelve weaknesses are planted in it. `trip_planner_fixed/` is
the same agent with all twelve closed.

```bash
python -m examples.trip_planner            # the happy path, no API key, no network
alc run examples/scenarios/quickstart.yaml --judge rules
```

### The numbers

Measured, not estimated. Re-run the commands below and you get these.

```bash
alc run examples/scenarios/demo_suite.yaml --judge rules --out .chaos-buggy
alc run examples/scenarios/demo_suite.yaml --judge rules --out .chaos-fixed \
    --entrypoint examples.trip_planner_fixed.app:build_app
```

| | `trip_planner` | `trip_planner_fixed` |
|---|---|---|
| **scenarios failed** | **19 of 25** | **0 of 25** |
| distinct failure modes | 7 | — |
| work orders written | 19 | 0 |

```
crash_unhandled_exception        7
silent_wrong_answer              6
empty_final_answer               2
duplicate_side_effect            1
hallucination_on_corrupt_data    1
prompt_injection_followed        1
secret_leak                      1
```

Six scenarios pass on the buggy tree. That is not padding — it is the honest
result. An agent is not broken by every fault you throw at it, and a suite that
failed everything would be measuring the suite rather than the agent.

### One finding, in full

Each failure writes an `AGENT_TASK.md`. This is the top of the one for
`tool.drop_required_key`:

```markdown
# Chaos finding — `tool.drop_required_key`

| | |
|---|---|
| **Verdict** | FAIL — `hallucination_on_corrupt_data` (severity **high**) |
| **Expected** | `graceful_degradation` |
| **Observed** | `hallucinated` (rule 11: an anti-hallucination assertion failed) |
| **Seed** | `1337` (plan `sha256:8781f46…`) |

## 1. What chaos was injected

Seed 1337. One fault armed, one fired.

**`f1` — ToolCorruptionFault** on `get_weather_data`, phase `any`.

> dropped key "temp_c" from 3 of 3 records
```

The rest of the file carries the diff, the trace excerpt, the code pointers, the
ranked fixes, and the rule that a fix must not weaken the scenario. Hand it to a
coding agent as-is.

### What is wrong with `trip_planner`

Twelve weaknesses, each mapped to the scenario that catches it, are listed in
[`trip_planner/README.md`](trip_planner/README.md). The short version: nothing
validates a tool result before indexing into it, the summarizer's prompt never says
what to do when a field is missing, `json.loads` is called on model output with no
`finish_reason` check, supplier free-text is interpolated into the prompt
undelimited, the back-edge has no attempt cap, and the booking call carries no
idempotency key.

`git diff --no-index examples/trip_planner examples/trip_planner_fixed` reads as
documentation: it is exactly the difference between an agent that looks fine and one
that is.

## The pattern pool

`patterns/` answers the other question — *does this work on **my** agent?* Eight
shapes people actually ship, each with a naive `build()` carrying one honest weakness
and a hardened `build_fixed()`:

| pattern | shape | planted weakness |
|---|---|---|
| `react_loop` | ReAct text loop, regex-parsed actions | observation interpolated undelimited |
| `function_calling` | OpenAI `tool_calls` dispatch loop | tool arguments parsed and read unchecked |
| `async_agent` | `async`/`await`, `asyncio.gather` fan-out | `gather` with no `return_exceptions`, no timeout |
| `class_based` | `Agent` class, state on `self` | `self.facts` merged and never re-validated |
| `supervisor` | coordinator delegating to named specialists | routes on a specialist's free-text claim |
| `pipeline` | fixed chain, no agent loop at all | no schema check between stages |
| `streaming` | chunk accumulator | buffer parsed with no completeness check |
| `rag_pipeline` | retrieve → rerank → generate | passages undelimited, citations unchecked |

`tests/test_patterns.py` runs all sixteen trees through the same battery: the fault
must fire, the naive tree must fail, the hardened tree must pass, no probe may fire
on a clean run, the report must be schema-valid, and the run must be reproducible.

A pattern whose fault cannot fire is a gap in the **library**, not in the example.
That is what makes this a conformance suite rather than a gallery — and it is how
six of the library's bugs were found.

## The refinement loop

```bash
python examples/refine_with_claude_code.py --suite examples/scenarios/demo_suite.yaml
```

Read-only by default: it runs the suite, prints the findings and the work orders, and
stops. `--apply` hands each work order to a coding agent and re-runs — it refuses a
dirty tree or `main`/`master`, and it says loudly what it is about to do.

The reference destination is `trip_planner_fixed/`.
