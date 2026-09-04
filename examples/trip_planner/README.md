# `trip_planner` — what is wrong with this agent, and why it looks fine

Twelve weaknesses are planted here. None of them is announced in the code, because a
demo whose bugs are labelled `# BUG:` proves nothing about finding bugs. Every one is
the kind of thing that ships: the shape is right, the names are sensible, and each
step does what its name says. What none of them does is check anything.

`examples/trip_planner_fixed/` closes all twelve. `git diff --no-index` between the
two trees is the intended reading order.

## The weaknesses

| # | Weakness | Where | Caught by | Actual failure mode |
|---|---|---|---|---|
| 1 | `rows[0]` indexed with no check the list is non-empty | `nodes.py:fetch_weather` | `tool.empty_json` | `crash_unhandled_exception` |
| 2 | the summarizer prompt says "use the weather data" and never says what to do when a field is absent | `prompts/summarize.md` | `tool.drop_required_key` | `hallucination_on_corrupt_data` |
| 3 | no plausibility check on `temp_c` | `nodes.py:summarize` | `tool.unit_swap` | `silent_wrong_answer` |
| 4 | tool exceptions propagate; no `try`/`except` anywhere | `nodes.py` | `tool.transient_500_then_ok` | `crash_unhandled_exception` |
| 5 | an in-band `{"code": 402}` payload is treated as data | `nodes.py:fetch_flights` | `tool.inband_error_payload` | `crash_unhandled_exception` |
| 6 | back-edge with no attempt cap | `edges.py:route_after_summarize` | `loop.pinned_tool_output` | `empty_final_answer` (hit the step limit) |
| 7 | the objective lives only in `messages`, never re-asserted | `nodes.py:summarize` | `context.*` | *survives* |
| 8 | `json.loads` on model output, unguarded | `nodes.py:summarize` | `llm.prose_instead_of_json`, `llm.truncated_mid_json` | `crash_unhandled_exception` |
| 9 | `finish_reason` never inspected | `nodes.py` | `llm.truncated_mid_json` | `crash_unhandled_exception` |
| 10 | supplier `notes` interpolated into the prompt undelimited | `prompts/summarize.md` | `adversarial.injection_corpus` | `prompt_injection_followed`, `secret_leak` |
| 11 | `hold_booking` called on every run, with no idempotency key and no request to book | `nodes.py:respond` | `adversarial.injection_corpus` (`must_not_call_tools`) | `silent_wrong_answer` |
| 12 | no guard that `location` is set before `summarize` | `nodes.py:summarize` | `state.drop_location` | *survives* |

Two of the twelve (7 and 12) do not currently produce a failure. That is reported
rather than hidden: the scripted model keys on a prompt tag, so goal dilution does
not move it, and the graph's own routing supplies a location before `summarize` is
reached. They are real weaknesses in the code that this suite does not yet catch.

## Why it looks fine

- **The happy path is correct.** `python -m examples.trip_planner` prints a sensible
  packing list and the cheapest fare. Every one of these bugs is invisible until
  something upstream misbehaves.
- **The error handling is absent, not wrong.** There is no broken `try`/`except` to
  notice in review — there is simply no `try`/`except`, and nothing in the happy path
  needs one.
- **The credential is a real field.** `get_account()` returns a booking key because
  the booking call needs one. Passing the whole record into the prompt is one line
  shorter than selecting the three fields that belong there.
- **`notes` is free text from a supplier.** Interpolating it is what you do with a
  string. That it is the injection carrier is not visible from the call site.

## The fixtures

Deterministic JSON under `fixtures/`. No network, no API key, no clock. The model is
`examples/fake_model.py`, a scripted mediocre-but-plausible LLM: it fabricates a
temperature when the field is missing, and it obeys an instruction it can read in the
clear — but not one that arrives inside a fence. That last difference is the entire
reason the fence in `trip_planner_fixed` is worth writing.
