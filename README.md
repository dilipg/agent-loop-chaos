# agent-loop-chaos

> **Breaks your agent on purpose and hands the bug report to your coding agent.**
> Chaos engineering for agent loops, with output designed for machines, not dashboards.

**Status: pre-alpha.** The specification is complete and the repository skeleton is
in place; the engine, faults, probes and judges are being implemented milestone by
milestone. Nothing here does anything useful yet. See
[docs/08-ROADMAP.md](docs/08-ROADMAP.md).

## Why

Agent frameworks give you a happy path. Production gives you a corrupted tool
payload, a 3k-token context window, a model that returns prose where JSON was
promised, and a tool result carrying an instruction that says *ignore your previous
instructions*.

Infra chaos tools kill pods; they cannot see a corrupted JSON key. Eval and
observability tools score answers and draw traces for a human. Red-teaming targets
the model, not the loop — state, retries, tool contracts, iteration caps.

The thesis:

> If a failure is reproducible, classified, and accompanied by the exact prompt, the
> exact payload diff, and a file:line pointer, then fixing it is a mechanical task a
> coding agent can do unattended.

This library manufactures those artifacts: a schema-valid `report.json`, a
`trace.jsonl`, and — on failure — an `AGENT_TASK.md` work order.

## Install

```bash
pip install agent-loop-chaos
```

`jsonschema` is the only required dependency. Everything else is optional:

```bash
pip install "agent-loop-chaos[langgraph]"   # LangGraph adapter
pip install "agent-loop-chaos[slm]"         # local small-model judge
pip install "agent-loop-chaos[yaml]"        # YAML suite files (JSON needs no extra)
pip install "agent-loop-chaos[all]"
```

Importing `agent_loop_chaos` never imports `langgraph`, `httpx` or `pyyaml`. It
works fully offline with `--judge rules`.

## Quickstart

LangGraph, in five lines:

```python
from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import ToolCorruptionFault
from agent_loop_chaos.adapters.langgraph import instrument_graph

engine = ChaosEngine(seed=1337)
engine.register_fault(
    ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]), target_tool="get_weather_data"
)
result = engine.run(instrument_graph(app, engine), inputs={"query": "Pack list for Paris"})
print(result.to_json())
```

Plain Python:

```python
from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import ToolCorruptionFault

chaos = ChaosEngine()
chaos.register_fault(
    ToolCorruptionFault(mutation_type="empty_json"), target_tool="get_weather_data"
)


@chaos.tool
def get_weather_data(location: str) -> list[dict]: ...


@chaos.intercept_tools()
def weather_agent(user_query, state):
    data = get_weather_data(state["location"])
    return llm_generate(user_query, data)


result = chaos.run_with_state(
    weather_agent, query="Pack list for Paris", initial_state={"location": "Paris"}
)
print(result.to_json())
```

## What you get back

```
.chaos/
├── suite.json                      # what CI, an optimizer, or a dashboard polls
└── <scenario_id>/<run_id>/
    ├── report.json                 # schema-valid ChaosResult
    ├── trace.jsonl                 # every crossing, in order
    ├── plan.json                   # faults + seed + entrypoint → replay input
    ├── judge.json                  # raw judge request/response, for auditability
    ├── baseline.diff               # unfaulted vs faulted final answer
    └── AGENT_TASK.md               # the work order (failures only)
```

`AGENT_TASK.md` is the point of the whole library. If a coding agent would have to
ask a question before starting work on one, it isn't good enough.

```
$ alc run chaos/demo_suite.yaml --judge rules
tool.drop_required_key          FAIL  hallucination_on_corrupt_data   high
llm.prose_where_json_expected   FAIL  crash_unhandled_exception       critical
adversarial.injection_corpus    FAIL  prompt_injection_followed       critical
loop.pinned_tool_output         FAIL  infinite_loop                   high
control.dry_run                 pass  none                            info

5 scenarios, 1 passed, 4 failed — 4 work orders in .chaos/
```

## How pass/fail is decided

Three layers, and only two of them have authority
([docs/11-OUTCOMES-AND-ASSERTIONS.md](docs/11-OUTCOMES-AND-ASSERTIONS.md)):

| Layer | Decides | Authority |
|---|---|---|
| **Probes** | structural failures visible in the trace | authoritative |
| **Assertions** | whether the agent met the scenario's declared expectation | authoritative |
| **Judge** | narration, root cause, hints, ranked fixes | advisory only |

`success` is computed. A language model never decides it — the judge may only
contribute `failure_mode`, a root-cause hypothesis, a refinement hint and prose, and
if the judge contradicts the probes the probes win and the disagreement is recorded
in `verdict.judge_disagreement`.

A probe never fires on the harness's own injection. Every probe receives
`HarnessFacts` and excludes what the engine itself caused, which is why a
well-behaved agent passes every scenario instead of being punished for the fault it
handled correctly.

## Determinism, precisely

Two different guarantees, and it matters which one you are relying on:

- **Harness determinism — always.** The same seed and the same `plan_hash` give
  identical fault decisions, identical RNG draws and identical recorded fires, for a
  given crossing sequence.
- **Run determinism — conditional.** A byte-identical `report.json` additionally
  requires a deterministic agent, a deterministic model (a scripted fake, or a real
  model at temperature 0 whose provider is stable), `--judge rules`, and a
  single-threaded graph.

Model-authored fields are excluded from golden comparisons. No wall-clock value ever
reaches a probe, an assertion or a classification rule.

## Consuming a report

The schemas ship inside the package, so you can validate a report without cloning
this repo:

```python
from agent_loop_chaos.schema import validate_obj

errors = validate_obj(report_dict, "report")  # [] when valid
```

`schema_version` is `MAJOR.MINOR`. The schemas set `additionalProperties: false`,
but **that strictness is for producers, not consumers**: it guarantees this library
never emits a field it has not documented. A consumer should ignore properties it
does not recognise, so that a MINOR addition never breaks it.

## Safety

Three faults perform real actions the agent never requested — `ArgumentTamperFault`,
`DuplicateSideEffectFault` and `CheckpointRollbackFault`. Pointed at `charge_card` or
`delete_rows` those are a double charge and a delete with no predicate. There is a
gate enforced in code, and it is not optional. **Read [SAFETY.md](SAFETY.md) before
pointing this at an agent wired to real tools and real credentials.**

Run directories hold full prompts, tool payloads, state snapshots and source
excerpts. They are sensitive by default and `.gitignore`d.

## Documentation

| Document | What it covers |
|---|---|
| [docs/00-VISION.md](docs/00-VISION.md) | positioning, non-goals |
| [docs/01-ARCHITECTURE.md](docs/01-ARCHITECTURE.md) | module map, `Crossing`, run lifecycle |
| [docs/02-API.md](docs/02-API.md) | the public API; every name is frozen |
| [docs/03-FAULT-CATALOG.md](docs/03-FAULT-CATALOG.md) | all 26 faults, normative |
| [docs/04-SCHEMAS.md](docs/04-SCHEMAS.md) | output contracts and versioning |
| [docs/11-OUTCOMES-AND-ASSERTIONS.md](docs/11-OUTCOMES-AND-ASSERTIONS.md) | pass/fail authority |
| [docs/DECISIONS.md](docs/DECISIONS.md) | binding errata; outranks every other doc |
| [PACK.md](PACK.md) | the build handover pack this repo was implemented from |

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) explains how to add a fault and a
probe. Licensed under [Apache-2.0](LICENSE).
