# FAQ

Every code block marked `<!-- test -->` is executed by `tests/test_docs_snippets.py`,
so if one is wrong CI says so rather than a reader discovering it.

## Why not just write evals?

An eval scores the answer your agent gave. This scores what your agent *does* when
the ground moves — when a tool drops a field it has always returned, when the model
answers with prose instead of JSON, when a retrieved document contains a sentence
addressed to the assistant.

The concrete gap: an agent that invents a temperature because `get_weather` returned
`{}` produces a *plausible* answer. Every eval you have scores it as fine, because it
reads fine. Nothing in your test suite returns `{}` from that tool.

Chaos and evals answer different questions and you want both. This one answers "does
it degrade honestly under conditions you did not think to write a test for".

## Why doesn't the model decide pass/fail?

Because then your test suite's verdict depends on a sampler.

`success` is computed by deterministic probes, the assertions layer, and the
scenario's `expected_behavior` — the rules in
[11-OUTCOMES-AND-ASSERTIONS.md](11-OUTCOMES-AND-ASSERTIONS.md), and nothing else. A
judge may write the narrative, the root-cause hypothesis, the refinement hint and the
ranked fixes. If it disagrees with the probes, the probes win and the disagreement is
recorded in `verdict.judge_disagreement` — a high rate there is the most useful signal
we get about whether the probes or the prompt need work.

The practical consequence: `--judge rules` needs no model at all, and a suite run with
it is byte-for-byte reproducible.

## How do I run this offline?

<!-- test -->
```python
from agent_loop_chaos import ChaosEngine

# `judge="rules"` makes no network call of any kind. The default, `judge=None`,
# probes for a local model endpoint once and falls back to rules when nothing
# answers -- which is fine, but it is a connection attempt.
engine = ChaosEngine(seed=1337, judge="rules", write_bundle=False)
assert engine.judge == "rules"
```

From the CLI: `alc run suite.yaml --judge rules`. The test suite proves this — an
autouse fixture makes `socket.socket` raise, and the rules path runs under it.

## How do I add a fault?

Subclass `Fault`, declare which `(layer, phase)` pairs it accepts, and return a
`FaultOutcome`. The engine refuses a bad registration at register time rather than
mid-run.

<!-- test -->
```python
from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import Fault, FaultOutcome, MutationLog


class TruncateStringsFault(Fault):
    """Cut every string in a tool result to `limit` characters.

    What agent weakness it proves: consuming a value without checking it is complete.
    What graceful behaviour looks like: detect the truncation and say the field was
    unusable, rather than parsing half a value.
    """

    kind = "TruncateStringsFault"
    accepts = frozenset({("tool", "post")})
    severity_hint = "medium"

    def apply(self, crossing, ctx):
        limit = int(self.params().get("limit", 8))
        before = crossing.result
        if not isinstance(before, dict):
            return FaultOutcome(action="noop", note="not a mapping; nothing to truncate")
        after = {k: (v[:limit] if isinstance(v, str) else v) for k, v in before.items()}
        return FaultOutcome(
            action="replace_result",
            value=after,
            note=f"truncated strings to {limit} chars",
            # The payload diff the report and AGENT_TASK.md render. Without it the
            # finding says what changed but cannot show it.
            mutation=MutationLog.of(before, after),
        )


engine = ChaosEngine(seed=1337, write_bundle=False)

@engine.tool(name="lookup", side_effecting=False)
def lookup() -> dict:
    return {"summary": "a long summary that will be cut short", "id": 7}

def agent(question=None) -> str:
    return lookup()["summary"]

engine.register_fault(TruncateStringsFault(limit=6), target_tool="lookup")
result = engine.run(agent, inputs={"question": "?"}, scenario_id="truncate")
assert result.final_output == "a long"
assert any(f["fired"] for f in result.injected_faults)
```

`accepts` is the important line. A fault registered against a layer it does not accept
raises `ConfigError` immediately, and the engine also checks it at every crossing — so
a `post`-only fault cannot burn its `max_fires` budget on the `pre` side and never
reach its own phase.

## My probe has a false positive. What now?

The probe is wrong until proven otherwise. That is not politeness, it is the design:
false negatives are acceptable and false positives are bugs, because a probe that
fires on correct behaviour makes the negative control unpassable and the whole suite
untrustworthy.

Check in this order:

1. **Did the harness cause it?** A probe must never fire on what the engine itself
   injected. The four attribution rules are in [11](11-OUTCOMES-AND-ASSERTIONS.md) §2:
   event attribution, value attribution, budget attribution (subtract
   `tokens_injected` and `delay_injected_ms` *before* any threshold), and removal
   attribution (a dropped key's absence is expected afterwards — the finding is a
   consumer reading it without a precondition check).
2. **Run it against a correct agent.** `examples/trip_planner_fixed` and the
   `build_fixed` half of every pattern in `examples/patterns/` exist for this. If a
   probe fires there, it is a false positive.
3. **Then fix the probe, not the scenario.** Softening the scenario hides the bug and
   leaves the next person to rediscover it.

Five of the nine library bugs found in phase 08 failed on the *correct* tree, not the
buggy one. The negative control is where false positives hide.

## How do I pin a scenario's seed?

Every scenario carries its own `seed`, and `--seed` overrides all of them:

```yaml
scenarios:
  - id: tool.drop_key
    seed: 1337          # this scenario always plans identically
    entrypoint: your_package.agent:build
```

The seed keys the RNG stream by `fault_key` — a hash of `(type, params, target,
trigger)` — not by position, so reordering your YAML does not re-key another fault's
stream. `run_id` derives from `(seed, scenario_id, plan_hash, attempt)`, so a re-run
with the same seed lands in a new directory rather than overwriting the old one.

Byte-identical reports additionally need a deterministic agent, a scripted or
temperature-0 model, `--judge rules`, and a single-threaded graph. That boundary is
stated in `docs/DECISIONS.md` D-07 rather than implied.

## LangGraph changed and the adapter broke. What do I do?

The adapter is deliberately small: it turns LangGraph's node and branch slots into
`Crossing` objects and does nothing else. When it breaks it is usually because a slot
moved.

<!-- test -->
```python
from agent_loop_chaos.adapters.langgraph import langgraph_version

# The adapter records the version it saw, so a report from a broken run says which
# LangGraph produced it rather than leaving you to guess.
version = langgraph_version()
assert isinstance(version, tuple)
```

Then: `instrument_graph` raises `AdapterError` rather than silently instrumenting
nothing, because a run with no interception looks exactly like a clean run. If you see
that error, the node names it lists are what LangGraph actually exposed.

The core has no framework dependency at all — `pytest tests -q -k "not langgraph"`
proves it — so you can drop to the vanilla adapter and wrap your tools directly while
waiting for a fix.

## Why did my scenario pass when I expected it to fail?

Usually one of three things:

1. **The fault never fired.** Check `injected_faults[].fired` and
   `skipped_reason` in `report.json`, or just read the warning: `alc run` prints
   `warning: <id>: no fault fired` when a scenario arms something and nothing lands.
   The commonest cause is a tool or llm name in the target that does not match what
   the agent registered.
2. **Your agent handled it.** That is a real result. A suite where everything fails is
   measuring the suite.
3. **The scenario's `expect` block does not check the thing you care about.** The
   auto-assertions are deliberately conservative — a bare number carries no claim
   strong enough to call fabricated — so a silent wrong answer often needs an explicit
   `output_matches` or `no_claim_about`.

## Can I use it in CI?

Yes, that is the intended home. `--json` prints exactly one JSON object to stdout and
nothing else, and the exit codes are stable: `0` all passed, `1` a scenario failed,
`2` config or usage error, `3` internal error, `4` tampering detected.

```bash
alc run chaos/suite.yaml --judge rules --json --quiet > result.json
```

`--judge rules` is the CI default for a reason: no endpoint, no network, reproducible.
