# How to drive the build

## The loop

```bash
cd ~/ai-agents-lab/agent-loop-chaos
claude
```

Then, for each phase in order:

1. Paste the contents of `prompts/NN-*.md` as your first message.
2. Let it work. It will read `CLAUDE.md` and the docs it needs on its own.
3. When it reports done, run the phase's verification commands yourself.
4. If green: `git add -A && git commit -m "MN: <phase name>"`, then `/clear` and
   move to the next phase.
5. If red: paste the failure output back with nothing else. Do not re-explain the
   phase; the context is already there.

One phase per context window. Do not run two phases in one session even if the
first was quick — later phases need room, and a compacted context loses the schema
details that matter most.

## When a phase stalls

| Symptom | Response |
|---|---|
| It's inventing names not in `docs/02-API.md` | "Re-read docs/02-API.md §N. Use the exact names. List what you renamed and revert it." |
| It's building the next phase too | "Out of scope for this phase. Revert the extra files and finish the acceptance checklist for this phase only." |
| Tests fail and it starts deleting them | "Stop. `CLAUDE.md` forbids weakening tests. Explain why the test is wrong, or fix the code." |
| It adds a dependency | "Core has no third-party deps except jsonschema. Move it to an optional extra with a local import and a MissingExtraError." |
| It can't find a LangGraph attribute | "Write the compat shim with a fallback chain and an AdapterError that names the installed version. Do not guess a single attribute path." |
| It asks you a design question | Answer from the docs if the answer is there. If it genuinely isn't: decide, tell it, and tell it to record the decision in `docs/DECISIONS.md`. |
| Output is drifting in quality late in a phase | Stop, commit what works, `/clear`, and re-enter with the phase prompt plus "already done: X, Y. Remaining: Z." |

## Rules of thumb

- **Never say "and also".** Mid-phase scope additions are how a build goes sideways.
  Write it in `docs/08-ROADMAP.md`'s backlog instead.
- **Run the verification yourself.** "Tests pass" from the agent is a hypothesis.
- **Commit at every green gate.** Nine small commits beat one large one when a
  later phase needs a bisect.
- **Keep `docs/` authoritative.** If you accept a deviation, make the agent update
  the doc in the same commit. A doc that lies is worse than no doc, because the next
  phase reads it.
- **Read the first `AGENT_TASK.md` the tool produces yourself, closely.** That file
  is the whole product. If it isn't good enough for you to act on, it isn't good
  enough for a coding agent, and phase 04 isn't done.

## Suggested model / effort settings

| Phase | Why |
|---|---|
| 00, 09 | mechanical; a fast model is fine |
| 01, 04, 05 | the hard ones — engine invariants, report/probe semantics, framework internals. Use your strongest model and let it think |
| 02, 03 | wide but shallow; many small classes. Good candidates for parallel subagents, one per fault family, if your setup supports it |
| 06, 07 | medium; the judge's failure handling is fiddlier than it looks |
| 08 | needs taste more than power — the demo has to look like real code |
| 10 | front-end taste plus care on the tailing logic; the single HTML file is long, so expect to commit it in two passes |

## After the build

Two follow-on prompts worth keeping:

- **Adversarial review:** "Read `docs/` and the implementation. Find every place the
  code and the spec disagree, every schema field nothing writes, and every documented
  guarantee no test enforces. Report as a list, most severe first. Do not fix
  anything yet."
- **Dogfood:** "Point the library at itself: write a scenario suite for
  `examples/refine_with_claude_code.py` and report what breaks."
