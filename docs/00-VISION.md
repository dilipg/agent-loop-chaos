# 00 — Vision, positioning, non-goals

## The gap

Agent frameworks give you a happy path. Production gives you a corrupted tool
payload, a 3k-token context window, a model that returns prose where JSON was
promised, and a tool result carrying an instruction that says *ignore your previous
instructions*. The existing tooling around this splits into three camps, and none
of them close the loop:

| Camp | Examples of what they do | What's missing |
|---|---|---|
| Infra chaos | kill pods, drop packets, add latency | knows nothing about agent semantics; a corrupted JSON key is invisible to it |
| Eval / observability | score final answers, show traces in a UI | describes what happened to a human; not a machine-actionable defect report |
| Red-teaming | adversarial prompts against a model | targets the model, not the *loop* — state, retries, tool contracts, iteration caps |

The gap is **State, Tracing, and Machine-Readable Evaluation**: a programmatic way
to break an agent loop on purpose, and get back a structured artifact that a coding
agent can act on without a human in the middle.

## The thesis

> If a failure is reproducible, classified, and accompanied by the exact prompt,
> the exact payload diff, and a file:line pointer, then fixing it is a mechanical
> task a coding agent can do unattended.

`agent-loop-chaos` exists to manufacture those artifacts.

## What it is

A small library with four moving parts:

1. **Injectors** — semantic faults aimed at tool execution, LLM/prompt handling,
   and graph state.
2. **A state-aware interceptor** — snapshots state before and after each fault, so
   every report can answer "what exactly changed, and what did the agent see".
3. **An SLM judge** — a small local model that narrates the injected chaos, names
   the failure mode, and proposes a refinement hint. Optional; rules-only mode is
   fully supported.
4. **A refinement loop** — baseline run → chaos matrix → probes → judge → an
   `AGENT_TASK.md` work order aimed at a coding agent.

## Positioning statement

> **agent-loop-chaos** breaks your agent on purpose and hands the bug report to
> your coding agent. Chaos engineering for agent loops, with output designed for
> machines, not dashboards.

## Who it is for

- Engineers shipping LangGraph agents who need regression tests for failure
  handling, not just for the happy path.
- Teams running continuous-refinement loops (DSPy, custom optimizers, Claude Code
  in CI) that need structured negative examples.
- Anyone who has ever watched an agent confidently invent a value because a tool
  returned `{}`.

## Success criteria for v0.1

- A LangGraph agent can be instrumented in **≤5 lines**.
- Nine or more fault types usable out of the box, covering tool, LLM/prompt, and
  state layers.
- Every run emits a schema-valid `report.json`, a `trace.jsonl`, and — on failure —
  an `AGENT_TASK.md`.
- Works fully offline (`--judge rules`) and with a 3B–7B local model
  (`--judge slm --model qwen2.5:7b-instruct`).
- The demo agent in `examples/` exhibits at least six distinct failure modes that
  the suite catches, and a coding agent handed the resulting task files fixes them.
- Two seeded runs are byte-identical after timestamp normalization.

## Non-goals (v0.1)

- No web UI, no dashboard, no hosted service. `report.json` and stdout only.
- No network/infra chaos (no proxy, no packet loss, no container killing).
- No model training, fine-tuning, or prompt optimization built in — the library
  produces the signal an optimizer consumes; it is not the optimizer.
- No framework support beyond LangGraph and plain Python callables. CrewAI,
  AutoGen, and OpenAI Agents SDK are post-v0.1 and must fit the same adapter
  protocol without changing the core.
- No JS/TS port in this build. The report schema is deliberately language-neutral
  so a port is possible later; do not build it now.
- No automatic patching of the agent under test. The library writes work orders; a
  coding agent applies fixes.

## Guiding principles

1. **The schema is the product.** Pretty output is a side effect.
2. **Deterministic by default, chaotic on request.** Chaos is seeded; the same seed
   always tells the same story.
3. **The observer never breaks the observed.** Engine failures are recorded, never
   raised into the agent.
4. **Rules decide, models explain.** Pass/fail is deterministic; the model supplies
   language and hypotheses.
5. **Small surface, deep output.** Few concepts, richly instrumented.
