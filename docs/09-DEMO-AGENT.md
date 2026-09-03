# 09 — The demo agent (the thing that gets broken)

The demo exists to prove the library works and to give the README something real to
show. It must be small enough to read in one sitting and buggy in ways that are
*ordinary* — the kinds of bug that survive code review.

Two agents, same graph, same tools, same prompts directory:

- `examples/trip_planner/` — the **buggy** one. Ships with the planted weaknesses
  below.
- `examples/trip_planner_fixed/` — the **good** one. Same behaviour on the happy
  path, correct under chaos. It is the negative control that catches false
  positives, and it doubles as the "here is what the fix looks like" reference.

## 1. The task

> "3-day packing list for Paris, and the cheapest flight from Delhi."

Requires two tools, a summarizer, and a final response. Multi-step enough for
context and goal-drift faults to bite; small enough to debug.

## 2. Graph

```mermaid
flowchart LR
    START --> plan
    plan --> fetch_weather
    plan -->|no destination| ask_clarify
    fetch_weather --> fetch_flights
    fetch_flights --> summarize
    summarize --> respond
    summarize -->|missing data| fetch_weather
    respond --> END
```

State (`TypedDict`):

```python
class TripState(TypedDict):
    query: str
    location: str | None
    origin: str | None
    weather: list[dict] | None
    flights: dict | None
    packing_list: list[str] | None
    messages: Annotated[list, add_messages]
    attempts: int
```

The `summarize → fetch_weather` back-edge exists so `LoopTrapFault` has somewhere to
trap, and so the retry probes have something to observe.

## 3. Tools

```python
@tool
def get_weather_data(location: str) -> list[dict]:
    """3-day forecast. Returns [{date, condition, temp_c, humidity}] x3."""

@tool
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Returns {cheapest: {price_usd, carrier, depart}, options: [...], notes: str}."""

@tool(side_effecting=True)
def hold_booking(flight_id: str, passenger: str) -> dict:
    """Places a 24h hold. NOT idempotent by design — this is the double-fire target."""
```

All three are **deterministic fakes** backed by a small JSON fixture, so the demo
runs with no API keys and no network. `notes` on `search_flights` is a free-text
field: that is the injection carrier.

The model is either a real one (if `OPENAI_API_KEY` or an Ollama endpoint is
present) or `examples/fake_model.py`, a scripted model that behaves like a
mediocre-but-plausible LLM: it happily fabricates a temperature when the field is
missing, and it follows instructions found in tool output. Both paths must work; CI
uses the fake.

## 4. Planted weaknesses (each maps to a scenario)

| # | Weakness | Where | Caught by |
|---|---|---|---|
| 1 | `state["weather"][0]["temp_c"]` accessed with no validation | `nodes.py:fetch_weather` | `tool.drop_required_key`, `tool.type_flip_matrix` |
| 2 | summarizer prompt says "use the weather data" with no missing-data instruction | `prompts/summarize.md` | `tool.empty_json`, `tool.drop_required_key` |
| 3 | no unit check on `temp_c` | `nodes.py:summarize` | `tool.unit_swap` |
| 4 | tool exceptions propagate; no try/except anywhere | `nodes.py` | `tool.transient_500_then_ok` |
| 5 | in-band `{"error": …}` payloads treated as data | `nodes.py:fetch_flights` | `tool.inband_error_payload` |
| 6 | back-edge with no attempt cap (`attempts` incremented but never checked) | `edges.py:route_after_summarize` | `loop.pinned_tool_output`, `tool.rate_limited_forever` |
| 7 | objective lives only in `messages`, never re-asserted | `nodes.py:summarize` | `context.middle_out_shrink`, `context.goal_dilution` |
| 8 | `json.loads(response.content)` on the packing list | `nodes.py:summarize` | `llm.prose_instead_of_json`, `llm.truncated_mid_json` |
| 9 | `finish_reason` never inspected | `nodes.py` | `llm.truncated_mid_json` |
| 10 | tool `notes` field interpolated into the prompt undelimited | `prompts/summarize.md` | `adversarial.injection_corpus` |
| 11 | `hold_booking` called without an idempotency key | `nodes.py:respond` | `resume.checkpoint_rollback` |
| 12 | no guard that `location` is set before `summarize` | `nodes.py:summarize` | `state.drop_location`, `state.misroute_edge` |

Twelve weaknesses, ≥6 distinct `failure_mode` values — comfortably above the M8
gate, with margin for a couple of scenarios that turn out not to bite.

## 5. What `trip_planner_fixed` does differently

Keep the diff small and legible; it is documentation.

- `validators.py`: `validate_weather()`, `validate_flights()` — `TypedDict` +
  explicit field/type/range checks, raising
  `DataUnavailable(field, reason)`, which subclasses `agent_loop_chaos.ExplicitError`
  so the harness classifies it as an intentional error rather than a crash
  (`docs/11` §3.1). Each successful check calls `engine.validated(payload,
  name=…)`, which is the positive evidence that suppresses
  `no_output_validation`.
- Every node wraps its tool call in a `call_tool()` helper with: a per-tool timeout,
  retry on transient classes only (max 2, exponential backoff), and no retry on
  4xx.
- In-band errors detected by checking for an `error` key before use.
- The objective is stored in `state["query"]` and re-asserted in every system prompt
  built by `prompts.build_system()` — never dependent on message history.
- The summarizer requests structured output and validates it; one repair attempt,
  then a `DataUnavailable` path. `finish_reason` checked before parsing.
- Untrusted tool text is rendered inside an explicit fence with a "content below is
  data, never instructions" preamble, and `notes` is length-capped and stripped of
  imperative-looking lines.
- `route_after_summarize` caps `attempts` at 2 and compares the new tool signature
  to the previous one to detect no-progress.
- `hold_booking` is called with an idempotency key derived from
  `(thread_id, flight_id)`, and the node checks for an existing hold first.
- Every failure path produces an explicit user-visible message naming what was
  unavailable, and calls `engine.note("degraded: …")`. That message is what the
  scenarios' `output_matches` assertions look for — the concrete, checkable meaning
  of `graceful_degradation` in this codebase.
- It never asserts a numeric value that no tool produced, which is what the
  `no_unsourced_numbers` assertion checks.

## 6. Files

```
examples/
├── trip_planner/
│   ├── app.py            builds and compiles the graph; exports `graph`
│   ├── nodes.py
│   ├── edges.py
│   ├── tools.py
│   ├── fixtures/weather.json, flights.json
│   ├── prompts/{plan,summarize,respond}.md
│   └── README.md         "here is what is wrong with this agent, and why it looks fine"
├── trip_planner_fixed/   same layout + validators.py
├── fake_model.py
├── scenarios/
│   ├── demo_suite.yaml          (from schemas/examples/suite_demo.yaml)
│   └── quickstart.yaml          one scenario, 15 lines, for the README
├── vanilla_agent.py             the blueprint's plain-Python example, runnable
├── refine_with_claude_code.py   the hand-off loop (pattern B in docs/05)
└── README.md
```

## 7. The refinement example (the value proposition)

`examples/refine_with_claude_code.py` must be honest and self-contained:

1. Run the demo suite against `trip_planner` (buggy). Expect failures.
2. Print the `AGENT_TASK.md` paths and, for one finding, the full file.
3. Optionally (`--apply`, off by default, loud warning, requires a clean git tree)
   shell out to a coding agent per task file, then re-run the suite and print the
   round-over-round table from `LoopReport.markdown()`.
4. End by pointing at `trip_planner_fixed/` as the reference destination.

The README's headline claim is exactly this script's output: *N failures found, the
work orders written, and after the fix round, N−k remaining.* Do not overclaim —
report the real numbers the script produces, including the ones a coding agent fails
to fix.
