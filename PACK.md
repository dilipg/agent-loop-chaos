# agent-loop-chaos — Build Handover Pack

This folder is **not the library**. It is the complete build specification for the
library, written so that a coding agent (Claude Code) can implement it end to end
with minimal extra instruction from you.

You are the product owner. Claude Code is the builder. This pack is the contract
between you.

---

## What is being built

`agent-loop-chaos` — a lightweight Python library that injects **semantic and
execution chaos** into agent loops (LangGraph first, plain Python second) and emits
**strictly-schema'd, machine-readable failure reports** that a coding agent can
consume directly to fix the agent under test.

Three things make it different from existing chaos / eval tooling:

1. **Faults are semantic, not network-level.** It corrupts tool payloads, shrinks
   context windows, drifts goals, seeds hallucinations, traps loops, and smuggles
   prompt injections. Not "drop 5% of packets".
2. **A small language model narrates and judges the run.** Every run produces a
   plain-language account of *what randomness was introduced* and *how the system
   behaved*, plus a classified `failure_mode` and a `refinement_hint`. Runs
   offline against a local SLM, and degrades to deterministic rules with no model
   at all.
3. **The output is an agent work order.** Every failing run writes an
   `AGENT_TASK.md` containing the repro command, the exact fault diff, the exact
   LLM prompt and response, stack frames with file:line code pointers, and the
   rule that the test must not be weakened. You paste it into Claude Code and it
   starts fixing.

---

## How to use this pack

**`RUNBOOK.md` has the exact commands, in order** — prerequisites, putting the spec
under version control before the agent touches anything, the per-phase loop, and the
gate commands to run yourself after each phase. Start there. The rest of this section
is the shape of it.

```
cd ~/ai-agents-lab/agent-loop-chaos
claude --model opus
```

Then, as the first message: `Read @prompts/00-bootstrap.md and execute that phase.
Follow @CLAUDE.md.`

Then run the phase prompts **in order**, one per session (or one per compact
window). Each prompt is self-contained and copy-pasteable.

| # | Prompt file | Produces |
|---|---|---|
| 00 | `prompts/00-bootstrap.md` | repo skeleton, packaging, CI, typing, test harness |
| 01 | `prompts/01-core-engine.md` | `ChaosEngine`, targeting, triggers, seeded RNG, trace recorder |
| 02 | `prompts/02-tool-faults.md` | tool-execution fault family + mutation library |
| 03 | `prompts/03-llm-faults.md` | LLM/prompt fault family incl. prompt injection |
| 04 | `prompts/04-report-and-probes.md` | deterministic probes, report/trace serialization, schema validation |
| 05 | `prompts/05-langgraph-adapter.md` | `instrument_graph`, state faults, checkpoint faults |
| 06 | `prompts/06-judge-slm.md` | judge protocol, rule judge, SLM judge, ensemble, narration |
| 07 | `prompts/07-refinement-loop.md` | baseline→chaos→judge→bundle loop, `AGENT_TASK.md` writer |
| 08 | `prompts/08-demo-agent.md` | deliberately-buggy demo agent + scenario suite that breaks it |
| 09 | `prompts/09-cli-and-release.md` | `alc` CLI, docs, examples, release checklist |
| 10 | `prompts/10-live-dashboard.md` | live trace dashboard (`alc dashboard`) + single-file HTML export |

Phases 00–09 are v0.1.0. Phase 10 (the dashboard) is v0.2.0 by default — run it
after 09, or before 09 if you want it in the first release, in which case add its CLI
to phase 09's checklists. It must not run before phase 04, since it reads the report
and trace contracts and must not shape them.

`prompts/PROMPTING-GUIDE.md` explains how to drive the sequence, what to do when a
phase fails its acceptance gate, and how to keep the agent from drifting.

**Read-order for the agent** is enforced by `CLAUDE.md`, which every Claude Code
session in this folder loads automatically.

---

## What is in this pack

```
agent-loop-chaos/
├── README.md                    ← you are here
├── RUNBOOK.md                   ← the command list: start here to run the build
├── CLAUDE.md                    ← conventions + guardrails, auto-loaded by Claude Code
├── SAFETY.md                    ← what this can break, and the gates that stop it
├── docs/
│   ├── 00-VISION.md             positioning, non-goals, success criteria
│   ├── 01-ARCHITECTURE.md       components, data flow, lifecycle, extension points
│   ├── 02-API.md                the exact public API to implement
│   ├── 03-FAULT-CATALOG.md      every injector: params, semantics, what it proves
│   ├── 04-SCHEMAS.md            report / trace / scenario / verdict contracts
│   ├── 05-JUDGE-AND-LOOP.md     SLM judge design + the continuous refinement loop
│   ├── 06-LANGGRAPH-ADAPTER.md  LangGraph and vanilla-Python integration
│   ├── 07-TESTING.md            test strategy, fixtures, determinism, CI gates
│   ├── 08-ROADMAP.md            milestones M0–M10 with acceptance gates
│   ├── 09-DEMO-AGENT.md         the buggy demo agent used to prove value
│   ├── 10-DASHBOARD.md          live trace dashboard: tailing, API, panes, export
│   ├── 11-OUTCOMES-AND-ASSERTIONS.md   ← the pass/fail authority. Read this one.
│   └── DECISIONS.md             binding errata from an adversarial review of the spec
├── schemas/
│   ├── chaos_report.schema.json
│   ├── trace_event.schema.json
│   ├── scenario.schema.json
│   ├── judge_verdict.schema.json
│   └── examples/                valid instances used as golden fixtures
├── assets/prompts/              prompt templates the library itself ships
│   ├── judge_system.md
│   ├── judge_user.md
│   ├── narrator.md
│   └── refiner.md
├── prompts/                     the build prompts for Claude Code
│   ├── PROMPTING-GUIDE.md
│   └── 00…10-*.md
└── tools/verify_pack.py         consistency check over this pack (see below)
```

`python3 tools/verify_pack.py` validates the schemas, validates every example
instance against them, and checks that names shared across the docs, schemas, and
prompts have not drifted (fault kinds, probe codes, `failure_mode` and fix-kind
enums, preset names, `must_not` codes, and every file path the prompts reference).
Run it after editing anything in this folder. It needs `jsonschema>=4.18` and
`pyyaml`; exit code 1 means something is inconsistent.

The `schemas/` and `assets/prompts/` folders are **not documentation** — they are
source files. Phase 00 moves them into the package tree verbatim.

---

## Before you start: read `docs/DECISIONS.md`

The spec was reviewed adversarially before any code existed, and ~45 findings were
folded back in. The structural ones are worth knowing up front:

- **`docs/11-OUTCOMES-AND-ASSERTIONS.md` is the pass/fail authority.** It defines
  harness attribution, the assertions layer, and the classification rules. The
  earlier sketch in `docs/04` §7 is superseded.
- **A probe never fires on the harness's own injection.** Eleven of the original
  probe rules did, which made the control fixture unpassable.
- **Assertions, not heuristics, decide whether the output behaved.** A deterministic
  rule can prove a failure; it cannot prove an answer is good.
- **`SAFETY.md` describes gates you must enforce in code**, not advice.

## Decisions already locked (do not re-litigate)

| Decision | Value |
|---|---|
| Language / package | Python ≥3.10, distribution `agent-loop-chaos`, import `agent_loop_chaos` |
| Primary target | LangGraph (`langgraph` ≥0.2) |
| Secondary target | plain Python functions / callables |
| Judge | pluggable protocol; default = ensemble of rules + local SLM over an OpenAI-compatible or Ollama endpoint; deterministic rules-only fallback |
| Core dependencies | stdlib only + `jsonschema`; everything else optional extras |
| Output contract | `report.json` (schema-validated), `trace.jsonl`, `AGENT_TASK.md` |
| Determinism | harness determinism always; byte-identical reports under the conditions in `docs/DECISIONS.md` D-07 |
| Pass/fail authority | deterministic probes **plus** a declarative assertions layer; the model only narrates (`docs/11`) |
| License | Apache-2.0 |
| CLI | `alc` |
| Live viewing | `alc dashboard` — stdlib-only local server that tails `trace.jsonl`; also a single-file HTML export. Read-only, localhost, no build step |

---

## Definition of done for the whole build

- `pip install -e ".[dev,langgraph,slm]"` then `make check` is green on a clean clone.
- `alc run examples/scenarios/demo_suite.yaml` finds **at least 6 distinct real
  failure modes** in the demo agent, with no false positives on the baseline run.
- Every `report.json` validates against `chaos_report.schema.json` in CI.
- Two identical seeded runs produce byte-identical `report.json` after
  timestamp/duration normalization — under `--judge rules`, with a deterministic
  agent and model, single-threaded (D-07 states the boundary; do not overclaim it).
- **No probe fires on `good_agent` or on a baseline run.** A finding the harness
  created rather than the agent is a bug (`docs/11` §2).
- `alc run --judge rules` works with no network and no model available.
- A human can paste one `AGENT_TASK.md` into a fresh Claude Code session and it
  fixes the demo agent's bug without further explanation.
- (after phase 10) `alc run … --dashboard` shows runs, events, and fault diffs live
  in a browser with only `jsonschema` installed, and `alc report --format html`
  produces one self-contained file with no external asset references.
