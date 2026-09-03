# Runbook — handing this pack to Claude Code

Exact commands, in order. Everything runs from
`~/ai-agents-lab/agent-loop-chaos`.

---

## 0. Prerequisites (once)

```bash
cd ~/ai-agents-lab/agent-loop-chaos
python3 -V            # need >= 3.10
git --version
claude --version
```

Optional, only for the SLM judge in phase 06 — the whole build works without it via
`--judge rules`:

```bash
# macOS
brew install ollama
ollama serve &
ollama pull qwen2.5:7b-instruct
```

---

## 1. Put the spec under version control *before* the agent touches anything

This folder is not a git repo yet. Making the pack commit #1 is what lets you diff
everything the agent writes from here on.

```bash
cd ~/ai-agents-lab/agent-loop-chaos
git init -b main
printf '%s\n' '.venv/' '__pycache__/' '*.pyc' '.env' '.DS_Store' '.chaos*/' 'dist/' \
  'build/' '*.egg-info/' '.coverage' 'htmlcov/' '.pytest_cache/' '.mypy_cache/' \
  '.ruff_cache/' > .gitignore
git add -A
git commit -m "spec: agent-loop-chaos build handover pack"
git tag spec-v1
```

Then work on a branch, so `main` always holds a known-good state:

```bash
git switch -c build/m0-bootstrap
```

---

## 2. The per-phase loop

Repeat this eleven times, once per phase, `00` through `10`. **One phase per
session.**

```bash
cd ~/ai-agents-lab/agent-loop-chaos
claude --model opus
```

Then, as the first message in the session:

```
Read @prompts/00-bootstrap.md and execute that phase. Follow @CLAUDE.md.
```

`CLAUDE.md` sends it to `docs/DECISIONS.md` (binding errata),
`docs/11-OUTCOMES-AND-ASSERTIONS.md` (the pass/fail authority) and `SAFETY.md`
before it writes anything, so you do not need to name them.

Or as a one-liner instead of the two steps above:

```bash
claude --model opus "Read @prompts/00-bootstrap.md and execute that phase. Follow @CLAUDE.md."
```

For phases **01, 04 and 05** — the hard ones — raise the reasoning depth first:

```
/effort high
```

When it reports done, **verify yourself** (never take "tests pass" on trust). Every
phase prompt has a `## Verify` block; the baseline is:

```bash
make check
```

Green → commit, clear, move on:

```bash
git add -A && git commit -m "M0: bootstrap repo skeleton, packaging, schemas, CI"
git switch -c build/m1-core-engine
```

```
/clear
```

Red → paste the failure output back into the same session with nothing else added.
The context is already there; re-explaining the phase makes it worse.

---

## 3. The eleven phases, with their commit messages

| # | First message | Commit on green |
|---|---|---|
| 00 | `Read @prompts/00-bootstrap.md and execute that phase. Follow @CLAUDE.md.` | `M0: repo skeleton, packaging, schemas, CI` |
| 01 | `Read @prompts/01-core-engine.md and execute that phase. Follow @CLAUDE.md.` | `M1: engine, targeting, tracing, vanilla adapter` |
| 02 | `Read @prompts/02-tool-faults.md and execute that phase. Follow @CLAUDE.md.` | `M2: tool-execution faults + mutation library` |
| 03 | `Read @prompts/03-llm-faults.md and execute that phase. Follow @CLAUDE.md.` | `M3: LLM/prompt faults + injection corpus` |
| 04 | `Read @prompts/04-report-and-probes.md and execute that phase. Follow @CLAUDE.md.` | `M4: assertions, outcomes, probes, report pipeline, bundle` |
| 05 | `Read @prompts/05-langgraph-adapter.md and execute that phase. Follow @CLAUDE.md.` | `M5: LangGraph adapter + state faults` |
| 06 | `Read @prompts/06-judge-slm.md and execute that phase. Follow @CLAUDE.md.` | `M6: judge — rules, SLM, ensemble` |
| 07 | `Read @prompts/07-refinement-loop.md and execute that phase. Follow @CLAUDE.md.` | `M7: refinement loop + tamper detection` |
| 08 | `Read @prompts/08-demo-agent.md and execute that phase. Follow @CLAUDE.md.` | `M8: demo agent + chaos suite` |
| 09 | `Read @prompts/09-cli-and-release.md and execute that phase. Follow @CLAUDE.md.` | `M9: CLI, docs, release 0.1.0` |
| 10 | `Read @prompts/10-live-dashboard.md and execute that phase. Follow @CLAUDE.md.` | `M10: live trace dashboard + HTML export` |

Phase 10 is v0.2.0 and may be run before 09 if you want the dashboard in the first
release — but never before 04.

---

## 4. Gate commands per phase

Run these yourself before committing. They are the short form; each prompt's
`## Verify` block has the full set.

```bash
# after 00
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]" && make check
alc --version

# after 01
make check
alc list-faults

# after 02 / 03
make check && pytest tests/faults -q
alc list-faults --json | python -c "import json,sys;print(len(json.load(sys.stdin)))"

# after 04
make check && pytest tests/probes tests/assertions tests/outcomes tests/report -q
alc run tests/data/fake_suite.yaml --judge rules --out .chaos-smoke
cat .chaos-smoke/*/*/AGENT_TASK.md      # read this one properly, end to end

# after 05
pip install -e ".[dev,langgraph]" && make check && pytest tests/adapters -q
pip uninstall -y langgraph langchain-core && pytest tests -q -k "not langgraph and not readme"
pip install -e ".[dev,langgraph]"

# after 06
pip install -e ".[dev,slm]" && make check && pytest tests/judges -q

# after 07
make check && pytest tests/loop -q
alc run tests/data/fake_suite.yaml --rounds 2 --judge rules --out .chaos-loop

# after 08
make check && pytest -m slow -q
python -m examples.trip_planner.app
alc run examples/scenarios/demo_suite.yaml --judge rules --out .chaos-buggy
python -c "import json;print(json.load(open('.chaos-buggy/suite.json'))['failure_modes'])"

# after 09
python -m build && twine check dist/*
python -m venv /tmp/v && /tmp/v/bin/pip install dist/*.whl && /tmp/v/bin/alc --help

# after 10
make check && pytest tests/dashboard -q
alc run examples/scenarios/demo_suite.yaml --judge rules --out .chaos-live --dashboard
# then open http://localhost:7717 while it runs
```

---

## 5. The two moments that decide whether this worked

1. **End of phase 04.** Open one `AGENT_TASK.md` and read it as if you were the
   coding agent receiving it. If you would have to ask a question before starting
   work, the phase is not done — say exactly what was missing and have it fixed
   before phase 05.
2. **End of phase 08.** Check the real failure-mode histogram against the twelve
   planted weaknesses in `docs/09-DEMO-AGENT.md` §4. Six or more distinct failure
   modes on the buggy tree and zero on the fixed tree is the bar. If the fixed tree
   fails something, a probe or an assertion has a false positive — diagnose it, do
   not adjust the scenario. `docs/11` §2 exists precisely so the fixed tree can
   pass.

---

## 6. Useful mid-build commands

```bash
claude --continue                 # resume the last session in this directory
claude --resume                   # pick from saved sessions
python3 tools/verify_pack.py      # re-check spec consistency after editing docs
                                  # (schemas, probe/precedence parity, judge output
                                  #  fields, canary regex, fencing, decision ids)
git diff spec-v1 --stat -- docs/  # what the agent changed in the spec
```

Inside a session:

```
/clear      start clean between phases
/compact    only if you must finish a phase that ran long
/effort high
```

---

## 7. Two follow-on prompts worth running after phase 09

Adversarial review:

```
Read @docs/ and the implementation. Find every place the code and the spec disagree,
every schema field nothing writes, and every documented guarantee no test enforces.
Report as a list, most severe first. Do not fix anything yet.
```

Dogfood:

```
Point the library at itself: write a chaos scenario suite for
examples/refine_with_claude_code.py and report what breaks.
```
