# Phase 10 — Live trace dashboard (M10)

Everything so far writes JSON for machines. This phase adds the one surface built
for a human: a local, dependency-free web view that tails `trace.jsonl` while a
suite is running and shows the injected chaos in the event stream as it happens.

**Read first, and treat as binding:** `docs/DECISIONS.md` — pre-seeded errata from an adversarial review of this spec; it overrides any doc it contradicts.

**Read first, in full:** `docs/10-DASHBOARD.md` (all of it — it is the spec, not an
outline), `docs/04-SCHEMAS.md` §§8–9, `schemas/trace_event.schema.json`,
`schemas/examples/trace_excerpt.jsonl`.

**Where this sits:** by default this is v0.2.0 work, run after phase 09. If you want
the dashboard in the first release instead, run this phase **before** phase 09 and
add `alc dashboard` and `--format html` to phase 09's CLI and README checklists.
Either way, do not start it before phase 04 — it reads the report and trace
contracts and must not shape them.

## Scope

`dashboard/` (server, watcher, api, export, one static HTML file), the two engine
changes that make live viewing possible, the `suite.json` 1.1 fields, and the CLI
surface. No new faults, no new probes, no report fields.

## Build

### 1. Engine changes first (small, and useful on their own)

- **Open the run directory at run start.** Split `bundle.py`: create
  `.chaos/<scenario_id>/<run_id>/` and open `trace.jsonl` at lifecycle step 2, write
  `plan.json` as soon as the plan is frozen, and leave `report.json`,
  `AGENT_TASK.md` and `judge.json` at the end as they are.
- **`JsonlSink` flushes per event** — one `write()` + `flush()` per line, no
  `fsync`. Add a test that reads the file from a second handle mid-run and sees the
  events already written.
- **`suite.json` written at suite start and updated after every scenario**, per
  `docs/10-DASHBOARD.md` §2: new optional fields `status`, `planned`, `current`,
  `round`, `rounds_planned`, and `finished_at` nullable. Bump its
  `schema_version` to `1.1`. Write atomically (temp file + `os.replace`) so a reader
  never sees half a document. Update the golden test rather than deleting it.

None of this may change the content of a finished `report.json`. Assert that with the
existing golden tests — they should pass untouched.

### 2. `dashboard/watcher.py`

`RunDirWatcher` per `docs/10` §4. The rules that matter, restated because they are
where this goes wrong:

- byte-offset tailing with a **trailing-fragment buffer**; advance the offset only
  past newline-terminated bytes;
- file smaller than the recorded offset → reset to 0 and emit `reset`;
- newline-terminated line that fails `json.loads` → emit
  `{"kind": "__unparseable__", "raw": …}`, bump `parse_errors`, keep going;
- a sparse `seq -> byte offset` index (one entry per 500 events) so
  `from=<seq>` seeks instead of re-reading;
- retention cap `--max-events` with a `truncated_head` marker;
- discovery scan of `out_dir` at the poll cadence, depth 2, skipping dotfiles and
  `payloads/`;
- run status: `running` until `run_finished` or `report.json`, then
  `completed`/`aborted`; `stale` after 120 s of silence with neither.

Importable and fully testable with no server running.

### 3. `dashboard/api.py`

The eight endpoints from `docs/10` §5, as pure functions over the watcher state plus
the filesystem. `Cache-Control: no-store` on everything.

Security, non-negotiable: `run_id` validated against `^[A-Za-z0-9._-]{1,64}$` **and**
resolved through the watcher's discovered map — never by joining request input onto a
path. Artifact names come from the literal allow-list
(`report.json`, `plan.json`, `judge.json`, `trace.jsonl`, `baseline.diff`,
`AGENT_TASK.md`). `payloads/` is not serveable.

### 4. `dashboard/server.py`

`ThreadingHTTPServer`, stdlib only. SSE per `docs/10` §5: heartbeat comment every
15 s, per-client bounded queue, `desync` frame instead of unbounded buffering, clean
handling of client disconnect (a broken pipe is normal, not an error to log at
warning level). `--no-sse` serves the polling path. Binds `127.0.0.1` unless
`--host` is passed, in which case print the warning naming what became reachable.

The server must never write into a run directory and must never import anything from
`faults/` or `judges/`. Keep the dependency arrow one-way.

### 5. `dashboard/static/index.html`

One file, all CSS and JS inlined, no external asset of any kind — that is a hard
constraint with a test behind it.

**Escaping is not optional (D-38).** The trace is *guaranteed* to contain
attacker-controlled markup: `ContextNoiseFault(html_boilerplate)`,
`PromptInjectionFault(placement="html_comment")` and `unicode_noise` put it there on
purpose, and the HTML export is designed to be shared. So: every payload-derived
value is rendered via `textContent` or an escaping helper — never `innerHTML`; `</`
is escaped as `<\/` inside the inlined JSON blob so a payload cannot break out of
`<script>`; zero-width and bidi characters render as visible placeholders. Build the three panes from `docs/10` §6:

- runs list with live counters, filter, failures-only toggle;
- virtualized event timeline with follow-mode, filters, and **fault rows rendered
  unmistakably** — that visual contrast is the reason the view exists;
- detail tabs: Event, **Diff** (before/after plus the RFC-6902 patch — the
  highest-value panel, make it genuinely good), Prompt (messages as a conversation,
  faulted messages marked), State (snapshot diff, `by: fault` vs `by: agent`),
  Verdict (symptoms with clickable evidence `seq` that jump the timeline), Task
  (rendered `AGENT_TASK.md` with copy-to-clipboard and a separate one-click copy of
  the `alc replay` command).

Theme-aware (full light palette on `:root`, dark under
`prefers-color-scheme`, `body` background painted explicitly). Keyboard: `j`/`k`,
`f` for next fault, `/` search, `g`/`G`, `?` help. Empty states that say what to do.
`localStorage` only for pane sizes and toggles, wrapped in try/catch.

Plain JS, no framework, no bundler. Keep it under ~1 500 lines; if it grows past
that, cut features rather than adding a build step.

### 6. `dashboard/export.py`

`alc report <out_dir> --format html [-o report.html]` → a single self-contained file:
the same page with the data inlined as a JSON blob (suite summary, every report,
every trace capped at `--max-trace-events`, default 5 000 per run, with an honest
truncation note rendered in the file). Run `redact()` over the blob before writing —
exported files get pasted into tickets.

### 7. CLI

`alc dashboard [--out] [--port 7717] [--host] [--open] [--poll-ms] [--max-events]
[--no-sse] [--once]`, plus `alc run <suite> --dashboard [--port] [--linger]` and the
`--format html` flag on `alc report`. Port-in-use behaviour, `--open` suppression
under `$CI`/`$SSH_CONNECTION`, and `--once` all per `docs/10` §7. Add the three
commands to `docs/02-API.md` §10 if anything you build differs from what is written
there.

## Tests

Everything in `docs/10-DASHBOARD.md` §10, specifically:

- watcher: append, partial line, corrupt line, shrink/reset, new run discovered
  mid-watch;
- `from=<seq>` seeks — assert the bytes read are far below the file size;
- API against a fixture `.chaos/` containing a finished run, a mid-run with no
  `report.json`, and an aborted run;
- path traversal: `../../etc/passwd`, `%2e%2e%2f`, and a symlink escape all rejected
  with nothing read;
- SSE ordering, heartbeat, and `desync` for a slow client;
- `--no-sse` returns the same events as the stream;
- no external asset URL in `index.html` or in an exported report;
- XSS fixture: a trace event carrying `<script>`, `</script>` and a bidi override
  renders as visible text, executes nothing, and does not break the inlined blob;
- export contains the data blob and the truncation note when capped;
- perf: 20 000-event fixture — `/api/runs` under 200 ms, first events page under
  300 ms;
- live end-to-end: run a fake-agent suite with `--dashboard`, poll `/api/suite`,
  assert `status` goes `running → completed` and `passed + failed` reaches the
  planned count;
- the suite's exit code is byte-identical with and without `--dashboard`;
- existing golden report tests pass **unchanged** after the bundle split.

## Acceptance checklist

- [ ] `make check` green; existing golden report tests unchanged
- [ ] run directory and `trace.jsonl` exist and grow *during* a run, not after
- [ ] `suite.json` is 1.1, written at start, updated per scenario, atomically
- [ ] `alc dashboard` runs with **no third-party dependency installed** (test it in a
      venv with only `jsonschema`)
- [ ] `index.html` and exported reports contain zero external asset references
- [ ] every payload-derived value is escaped (no `innerHTML`); the XSS fixture is inert
- [ ] the three path-traversal shapes are rejected; `payloads/` is unreachable
- [ ] SSE survives a slow client (`desync`, not memory growth) and a disconnect
- [ ] `--no-sse` and the polling fallback produce identical event streams
- [ ] fault events are visually unmistakable in the timeline; symptom evidence links
      jump to the right row
- [ ] the Task tab copies a working `AGENT_TASK.md` and `alc replay` command
- [ ] 20 000-event trace meets the perf budget
- [ ] `alc run --dashboard` never changes the suite's exit code, and a dashboard
      crash cannot fail a run (test it by raising inside the server thread)
- [ ] `docs/10-DASHBOARD.md` and `docs/02-API.md` §10 updated if you deviated

## Verify

```bash
make check
pytest tests/dashboard -q

# live: run a suite and watch it
alc run examples/scenarios/demo_suite.yaml --judge rules --out .chaos-live --dashboard &
sleep 2 && curl -s localhost:7717/api/suite | python -m json.tool | head -20
curl -s localhost:7717/api/runs | python -m json.tool | head -30
curl -sN localhost:7717/api/stream | head -5

# static export, and the no-deps check
alc report .chaos-live --format html -o /tmp/chaos.html
grep -cE 'https?://[^"]+\.(js|css|woff2?|png|svg)' /tmp/chaos.html   # must be 0
python -m venv /tmp/bare && /tmp/bare/bin/pip install -e . && /tmp/bare/bin/alc dashboard --once --out .chaos-live
```

Then open it in a browser during a real run of the demo suite and watch the
`loop.pinned_tool_output` scenario. If you cannot tell, at a glance, where the fault
fired and what changed, the timeline styling is not done.

## Out of scope

Run control from the browser (start/stop/re-run/re-judge), authentication, hosted
deployment, websockets, any JS framework or build step, and any new report or trace
field. The dashboard reads the schemas; it does not get to change them.
