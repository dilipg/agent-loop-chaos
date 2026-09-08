# Brief — running this on your service

You have the repo. This is the short version: how to get a chaos run against **your**
agent in about five minutes, and what to do when it does not work first time.

The long version is [README.md](README.md); the section that matters most when you get
stuck is [Onboarding a real repo](README.md#onboarding-a-real-repo).

---

## What it does

It breaks your agent on purpose — drops a key from a tool response, truncates a model
reply, skips a node, corrupts state — then decides deterministically whether your agent
handled it, and writes the finding as a work order a coding agent can act on.

It never needs a credential of its own, and with `--judge rules` it makes no network
call at all.

## Install

```bash
pip install "agent-loop-chaos[yaml,langgraph] @ git+https://github.com/dilipg/agent-loop-chaos@v0.3.1"
```

Drop `,langgraph` if you are not on LangGraph. From a clone: `pip install -e ".[dev,all]"`.

Requires Python ≥3.10. The only hard dependency is `jsonschema`.

## Five minutes

```bash
# 1. Whatever your service validates at import. Dummy values are fine — nothing dials out.
export SERVICE_DB_URI=mongodb://localhost/x SERVICE_API_KEY=dummy

# 2. Ask what it can see. Point at any function that runs one request end to end.
alc doctor your_package.service:answer --inputs "a realistic input"

# 3. It prints a suite. Save it verbatim.
alc doctor your_package.service:answer --inputs "a realistic input" | sed -n '/^version:/,$p' > chaos_suite.yaml

# 4. Run it, then open the page it leaves behind.
alc run chaos_suite.yaml --judge rules
open .chaos/index.html
```

Exit codes: `0` all passed, `1` a scenario failed, `2` your configuration, `3` a bug in
the library. **Exit 2 is never our fault and never yours to debug in this codebase** —
the message says what to fix.

## When `doctor` does not say what you hoped

| It says | What it means | What to do |
|---|---|---|
| `importing entrypoint … raised` | your service raised before anything ran | usually settings validated at import — export the variable it names |
| `No tool or llm seam was reached` | nothing for a payload fault to attach to | see the table below; node and state faults may still work |
| `Graphs compiled at import` | your graph is a module-level singleton | copy the `seams: {graph: …}` line it prints |
| model and tools listed | you are done | copy the suite and run it |

## How it attaches to your code

You should not have to write a harness. Four ways in, tried in this order:

| Your shape | What reaches it | You write |
|---|---|---|
| Model over HTTP — any hosted SDK, or a local server (Ollama, vLLM, LM Studio) | patches `httpx` | nothing, with `--intercept` |
| In-process LangChain model or `@tool`, including test fakes | patches `BaseChatModel` / `BaseTool` | nothing, with `--intercept` |
| A graph built and compiled inside the function that runs it | patches `StateGraph.compile` | nothing, with `--intercept` |
| A hand-rolled client, a repository function, or a graph compiled once at import | you name it | one line of `seams:` |

```yaml
defaults:
  intercept: true
  seams:
    llm:   ["summarizer=app.llm:LLMClient.chat"]   # target it as `llm: summarizer`
    tools: ["app.repositories:fetch_*"]            # a glob over the module's functions
    graph: ["app.graph.runner:_workflow"]          # a graph compiled once at import
```

A bad dotted path is an error before the run, not a mystery during it.

## If your project has tests, use them

A working test suite has already solved the expensive half — constructing your app
offline. A `chaos_engine` fixture is available as soon as the library is installed:

```python
@pytest.mark.chaos(intercept=True)
def test_it_degrades(chaos_engine, graph, seeded_db):   # graph, seeded_db are yours
    chaos_engine.register_fault(NodeSkipFault(), target_node="revalidation")
    result = chaos_engine.run(graph, inputs={...})
    assert result.success, result.failure_mode
```

## Four things worth knowing

**A fault that never fired proves nothing.** A run warning
`this scenario proves nothing` deserves as much attention as a failure: it ran, the
agent passed, and nothing was tested. `coverage` in `.chaos/suite.json` says which
fault types actually fired.

**A pass with a stub model is narrower than it looks.** If a canned reply stands in for
your model, an injection scenario proves your pipeline tolerated a hostile prompt — not
that the model resisted it. Record a cassette against the real endpoint before reading
`no_unsourced_numbers` as green:

```bash
alc run chaos_suite.yaml --record chaos/cassettes/tape.json      # calls out once
alc run chaos_suite.yaml --replay-cassette chaos/cassettes/tape.json   # offline forever
```

**Start with the faults that need no fixtures.** Node skips, state drops and retyped
channels fire on the first run and exercise the routing and reducer logic where a
multi-node pipeline is most likely to be wrong. Add payload faults once a model or tool
is wrapped.

**`.chaos/` holds prompts, payloads and state snapshots.** Add it to your `.gitignore`.
Credentials are redacted, but the directory is sensitive by design.

## Going deeper

- [README — Onboarding a real repo](README.md#onboarding-a-real-repo) — the eight things
  that block a cold start, with fixes
- [README — Writing a scenario](README.md#writing-a-scenario) — the whole suite surface,
  annotated
- [docs/03-FAULT-CATALOG.md](docs/03-FAULT-CATALOG.md) — every fault and what weakness it
  proves; `alc list-faults` prints the same list
- [SAFETY.md](SAFETY.md) — read before pointing anything at a tool that performs a real
  action
- [docs/11-OUTCOMES-AND-ASSERTIONS.md](docs/11-OUTCOMES-AND-ASSERTIONS.md) — how
  pass/fail is decided, which is never by a language model

Found a false positive — a probe firing on an agent that handled the fault correctly?
That is a bug in the probe until proven otherwise. There is an issue template for it.
