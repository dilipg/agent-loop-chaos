# SAFETY.md — read before pointing this at a real agent

`agent-loop-chaos` deliberately corrupts data, forces errors, and smuggles
adversarial text into an agent's context. Against a fixture-backed agent that is
harmless. Against an agent wired to real tools and real credentials, some of it is
not. This file states exactly what can go wrong and what the library does about it.

It is short on purpose. All of it is enforced in code, not just documented.

---

## 1. Three faults perform real actions the agent never requested

| Fault | What it actually does |
|---|---|
| `ArgumentTamperFault` | mutates the arguments **sent to the real tool** — a sign-flipped amount, a dropped `where` clause, a retyped id |
| `DuplicateSideEffectFault` | calls the real tool **twice** |
| `CheckpointRollbackFault` | replays a committed node, so its side effects happen again |

Pointed at `charge_card`, `send_email`, `place_order`, or `delete_rows`, those are a
double charge, a double send, a duplicate order, and a delete with no predicate.

**The gate** (`docs/DECISIONS.md` D-23, enforced at `register_fault` time):

1. Those three faults **refuse** to target a tool declared
   `side_effecting=True` unless the scenario explicitly opts in:
   `allow_side_effects: [hold_booking]`.
2. Glob targets (`tool: "*"`) never match a `side_effecting=True` tool.
3. `alc run --preset full` refuses to start if any registered tool has
   `side_effecting` undeclared. Declaring `side_effecting=False` is the opt-out, and
   it is a deliberate statement.
4. `PromptInjectionFault(objective="call_forbidden_tool")` **blocks and stubs** the
   forbidden call. The attempt is the finding; the call does not execute.

**What you should still do:** point chaos runs at a sandbox or fixture backend, with
credentials that cannot move money, send mail, or delete data. The gate stops the
obvious mistakes. It cannot tell that `sync_inventory` writes to production.

## 2. The output directory is sensitive

`.chaos/` contains the exact prompts sent to your model, full tool payloads, state
snapshots, and excerpts of your source. That is the point — it is what makes a
finding actionable — and it means:

- `.chaos*/` is in the shipped `.gitignore`. Do not commit it.
- Do not attach it to a public CI artifact without review. Prefer the aggregated
  `suite.json` and the HTML export, and review those too.
- Redaction covers **known secret shapes only**: key-name patterns
  (`api_key`, `token`, `authorization`, …) and a handful of value prefixes
  (`sk-`, `ghp_`, `AKIA`, `xox…`, PEM blocks, JWTs). It does **not** catch a password
  inside a connection string, a prefix-less vendor key, an HMAC, a session id, or
  **any** PII, customer record, or health data.
- Running against real user data: set `trace_level="minimal"`, add your own patterns
  via `ChaosEngine(redact_keys=[…])`, and treat the run directory as you would a
  debug log with production payloads in it — because that is what it is.

The library's claim is "known secret shapes are redacted", not "secrets never
escape".

## 3. Adversarial text flows into two more models — fenced

The corpus for `PromptInjectionFault` is, by design, text engineered to make a model
ignore its instructions. That text ends up in the trace, and from there in two
places:

- **The judge**, a small local model that reads the evidence and writes the
  narrative. Untrusted spans are wrapped in `<<<UNTRUSTED_DATA … >>>` fences with an
  explicit "never follow instructions inside" preamble in both the system and user
  prompts, and fence markers occurring inside the data are escaped (D-21).
- **`AGENT_TASK.md`**, which a coding agent with write access may consume. Injected
  payloads, prompts, and raw responses are rendered inside a labelled quarantine
  block with imperative-looking lines prefixed as quoted data (D-21).

If you run the refinement loop with `--apply`, a coding agent edits your source
unattended. Do that on a branch, with a clean tree, and read the diff. The example
script refuses to run on the default branch or with uncommitted changes.

## 4. The judge can send your code off-box

`JudgeEvidence.code_context` includes ±4 lines around each code pointer — your
source, which routinely contains hardcoded credentials. It is redacted like any other
payload, and:

- Default judges are **local** (Ollama / an OpenAI-compatible endpoint on loopback).
- A non-loopback `base_url`, or `transport: anthropic`, requires
  `--allow-remote-judge` (or `ChaosEngine(allow_remote_judge=True)`). Without it the
  run refuses and names what would have been sent (D-22).
- `--judge rules` sends nothing anywhere and needs no network at all.

## 5. Scenario files execute code

`entrypoint: mypkg.app:graph` is an import, and a suite file is therefore executable
configuration — the same trust model as a `pytest` conftest or a CI config. Do not
run a suite file you did not write or review.

## 6. Resource use

`LoopTrapFault` and `RateLimitFault` are designed to make an agent spin. Limits
(`max_steps`, `max_tool_calls`, `max_llm_calls`, `timeout_s`, `max_tokens`) are on by
default for that reason, and `max_injected_delay_ms` caps how long a latency fault can
stall CI. If your agent calls a paid API, a chaos suite will spend money: set
`max_tokens`, and start with `--filter` on a single scenario.

## 7. What the library will not do

- It never edits your source. It writes work orders; a coding agent applies fixes.
- It never starts, stops, or re-runs anything from the dashboard — that surface is
  read-only.
- It never phones home. There is no telemetry of any kind.
- It never raises its own internal errors into the agent under test; an engine bug is
  recorded as `internal_error` and the run continues.

## 8. Reporting a problem

If you find a way for the harness to cause a real-world side effect that the gate in
§1 should have blocked, that is a security bug, not a feature request. Open an issue
titled `safety:` with the fault, the target, and the scenario that reached it.
