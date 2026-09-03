# 10 — Live trace dashboard

`report.json` is for machines. This is the one surface built for a human watching a
suite run: a local, zero-dependency web view that tails `trace.jsonl` as it is
written and shows what the chaos engine is doing *while it is doing it*.

It is deliberately the last thing built. The schemas are the product; the dashboard
is a reader of them and must never become a reason to change them.

## 1. Constraints (these shape every decision below)

- **Stdlib only.** `http.server` + `json` + `pathlib` + `threading`. No Flask, no
  FastAPI, no npm, no build step, no CDN. Installing `agent-loop-chaos` must be
  enough to run `alc dashboard`.
- **One HTML file.** All CSS and JS inlined, shipped as package data at
  `dashboard/static/index.html`. No external font, script, or image. A test asserts
  the file contains no `http://` or `https://` asset reference.
- **Read-only.** The server never writes into a run directory, never mutates a
  report, never triggers a run. It opens files for reading and nothing else.
- **Localhost by default.** Bind `127.0.0.1`. `--host` is opt-in and prints a
  warning naming what becomes reachable.
- **Works on a finished directory too.** Pointing it at a `.chaos/` from last week
  must render everything with no live source attached.
- **Never blocks the run.** The engine does not know the dashboard exists. If the
  dashboard crashes, the suite keeps going. There is no coupling in that direction,
  ever.

## 2. What makes live viewing possible

Two small engine-side guarantees, both worth having regardless of the dashboard:

1. **The run directory is created at run start, not at the end.** `bundle.py`
   currently writes at step 12 of the lifecycle. Split it: create
   `.chaos/<scenario_id>/<run_id>/` and open `trace.jsonl` at step 2 (`run_started`),
   write `plan.json` immediately after the plan is frozen, and write
   `report.json` / `AGENT_TASK.md` / `judge.json` at the end as today.
2. **`JsonlSink` flushes after every event.** One `write()` + `flush()` per line,
   `os.fsync` never (too slow, not needed for a local reader). A partially written
   final line is expected and the reader handles it — see §4.

Plus one additive schema change: **`suite.json` is written at suite start and
updated after every scenario**, so a viewer knows the denominator.

```json
{
  "schema_version": "1.1",
  "status": "running",
  "started_at": "…", "finished_at": null,
  "seed": 1337,
  "planned": ["tool.drop_required_key", "tool.unit_swap", "…"],
  "current": "tool.unit_swap",
  "round": 1, "rounds_planned": 1,
  "passed": 3, "failed": 2, "skipped": 0,
  "results": [ … as before … ]
}
```

`status` ∈ `{"running", "completed", "aborted"}`. `planned`, `current`, `round`,
`rounds_planned` and `status` are new optional fields — a MINOR bump to `1.1` per
`docs/04-SCHEMAS.md` §2, so existing consumers keep working. Write it atomically
(temp file + `os.replace`) so a reader never sees a half-written JSON document.

## 3. Module layout

```
src/agent_loop_chaos/dashboard/
├── __init__.py
├── server.py        DashboardServer (ThreadingHTTPServer), routing, SSE hub
├── watcher.py       RunDirWatcher: discovers run dirs, tails trace.jsonl
├── api.py           the JSON endpoints, pure over the filesystem
├── export.py        static single-file HTML export (alc report --format html)
└── static/index.html
```

`watcher.py` and `api.py` must be importable and testable without starting a
server. `server.py` is a thin transport over them.

## 4. Tailing `trace.jsonl` correctly

This is the part that breaks if done casually. The rules:

- Track a per-file byte `offset`. Read from `offset` to EOF, split on `\n`, and
  **keep the trailing fragment in a buffer** — a line without a terminating newline
  is incomplete and must not be parsed until the rest arrives.
- Advance `offset` only past bytes that ended in a newline.
- If the file's size is smaller than the recorded offset, the directory was replaced:
  reset the offset to 0 and re-read, emitting a `reset` frame so the client clears.
- A line that fails `json.loads` after being newline-terminated is corrupt, not
  incomplete: count it in a `parse_errors` counter, emit it as a
  `{"kind": "__unparseable__", "raw": "…"}` frame, and continue. Never let one bad
  line kill the stream.
- Poll interval 250 ms default (`--poll-ms`), and only stat files whose directory
  mtime changed. Discovery of *new* run directories scans `out_dir` at the same
  cadence, depth 2, ignoring dotfiles and `payloads/`.
- Cap in-memory retention per run at `--max-events` (default 20 000, oldest
  dropped with a `truncated_head` marker). A `LoopTrapFault` scenario can produce a
  very long trace; the dashboard must degrade, not die.
- Never load a whole trace into memory to answer a `from=<seq>` query — seek by the
  offset index the watcher already keeps (`seq -> byte offset`, one entry per 500
  events, then scan forward).

Watch mode also needs to notice the *end* of a run: `report.json` appearing, or a
`run_finished` event, whichever comes first. Show a run as `running` until then, and
as `stale` if no event and no report for 120 s.

## 5. HTTP API

All JSON, all `GET`, all read-only. Every response carries
`Cache-Control: no-store`.

| Endpoint | Returns |
|---|---|
| `GET /` | the single-page app |
| `GET /api/suite` | `suite.json` if present, else a synthesized summary from the run dirs found |
| `GET /api/runs` | list of runs: `run_id`, `scenario_id`, `status`, `success`, `failure_mode`, `severity`, counts, `started_at`, `duration_ms`, artifact presence flags |
| `GET /api/run/<run_id>` | the run's `report.json` if written, else a partial: plan, counters so far, faults fired so far |
| `GET /api/run/<run_id>/events?from=<seq>&kinds=a,b&layers=c&limit=N` | trace events, `seq` ascending, plus `next_from` and `has_more` |
| `GET /api/run/<run_id>/artifact/<name>` | one of the fixed allow-list: `report.json`, `plan.json`, `judge.json`, `trace.jsonl`, `baseline.diff`, `AGENT_TASK.md`. Nothing else, ever |
| `GET /api/stream` | SSE: `event: trace` frames as lines arrive, plus `run_started`, `run_finished`, `suite`, `reset`, `heartbeat` |
| `GET /api/health` | version, out_dir, watcher stats, parse_errors |

Path safety: `run_id` is matched against `^[A-Za-z0-9._-]{1,64}$` and resolved by
lookup in the watcher's discovered map — never by joining user input onto a path.
The artifact name comes from a literal allow-list. A test attempts
`../../etc/passwd`, `%2e%2e%2f`, and a symlink escape, and asserts 400/404 with
nothing read.

SSE details: `Content-Type: text/event-stream`, `X-Accel-Buffering: no`, a
`heartbeat` comment every 15 s so proxies and browsers keep the connection, and a
per-client bounded queue — if a client falls behind by more than 5 000 events, drop
it with a `desync` frame telling the page to re-fetch from `next_from` rather than
buffering forever. `--no-sse` falls back to client polling of
`/api/run/<id>/events`, and the page auto-falls-back if the stream errors twice.

## 6. The page

Three panes, resizable, remembered in `localStorage` (a per-viewer convenience only
— never state that matters):

**Left — runs.** The suite as a list: scenario id, a status dot
(running / pass / fail / aborted), failure mode, severity, step count, wall time.
Filter box, and a "failures only" toggle. Live counter at the top:
`7/21 · 4 passed · 2 failed · 1 running`. New runs appear as they start.

**Centre — the timeline.** The selected run's events, newest at the bottom,
auto-scrolling with a "follow" toggle that switches off the moment the user scrolls
up (and a "jump to live" button to switch it back). One row per event:
`seq`, `+ms` since run start, kind, layer/name, and a one-line summary. Rows are
colour-coded by class — tool, llm, state/node/edge, fault, probe, judge, error —
and **fault rows are visually unmistakable**: full-width, accented, with the
fault's `note` as the summary. That contrast is the point of the whole view: you
should be able to see the injected chaos in the stream at a glance.

Filters: by kind group, by layer, by fault id, "faults and errors only", and a text
search over the summary. Filtering is client-side over what has been fetched, with a
server-side `kinds=`/`layers=` narrowing for long traces.

Virtualize the list. 20 000 rows must scroll at 60 fps; if a naive implementation
cannot, render a windowed slice. A perf test asserts first paint under 1 s for a
20 000-event trace.

**Right — detail.** Tabs for the selected event and the selected run:

- *Event* — the full payload, pretty-printed and collapsible.
- *Diff* — for `mutation_applied`, a side-by-side or unified before/after with the
  RFC-6902 patch listed beneath. This is the highest-value panel in the dashboard;
  make it good. Highlight removed keys, changed types, and changed magnitudes
  distinctly.
- *Prompt* — for `llm_request`/`llm_response`, the messages rendered as a
  conversation with role chips, plus `finish_reason` and token counts, and a marker
  on any message a fault touched.
- *State* — the latest `state_snapshot` with a diff against the previous one, and
  `by: fault` vs `by: agent` clearly distinguished.
- *Verdict* — symptoms with severity chips and clickable evidence `seq` links that
  jump the timeline; then `failure_mode`, the narrative, the refinement hint, the
  ranked fixes, the code pointers, and `judge_disagreement` when present.
- *Task* — the rendered `AGENT_TASK.md` with a copy-to-clipboard button and the
  `alc replay …` command as a separate one-click copy. This is how a human moves a
  finding to a coding agent, so it must be one click, not a file hunt.

Theme: define a full light palette on `:root`, override under
`@media (prefers-color-scheme: dark)`, and paint `body` explicitly. Keyboard:
`j`/`k` move rows, `f` next fault event, `/` focus search, `g`/`G` top/bottom,
`?` help. Empty states say what to do (`no runs in .chaos yet — start one with
alc run …`). Accessibility: real focus outlines, `aria-live="polite"` on the
counter, severity conveyed by text as well as colour.

## 7. CLI

```
alc dashboard [--out .chaos] [--port 7717] [--host 127.0.0.1] [--open]
              [--poll-ms 250] [--max-events 20000] [--no-sse] [--once]

alc run <suite> --dashboard [--port 7717]     # run the suite and serve alongside it
alc report <out_dir> --format html [-o report.html]
```

- `--open` opens a browser; suppressed when `$SSH_CONNECTION` or `$CI` is set.
- Port in use: try the next 10 ports and print the one chosen; `--port` given
  explicitly means fail instead of wandering.
- `--once` renders the current state and exits 0 — for a smoke test in CI.
- `alc run --dashboard` starts the server in a daemon thread, prints the URL before
  the first scenario, and keeps serving for `--linger` seconds (default 0, meaning
  exit with the run) after the suite finishes. The suite's exit code is unaffected by
  the dashboard.
- `alc report --format html` writes a **single self-contained file** with the data
  inlined as a JSON blob: the suite summary, every report, and every trace up to a
  size cap (`--max-trace-events`, default 5 000 per run, with an honest note in the
  file when truncated). Same page, same panes, no server, no network. This is the
  CI-artifact and share-a-finding path.

## 8. What the dashboard must not do

- No run control: no start, stop, re-run, or re-judge buttons. Read-only means
  read-only; the CLI is the control plane.
- No editing of scenarios or reports.
- No telemetry, no outbound request of any kind.
- No authentication theatre: it binds to localhost and says so. If `--host` is used,
  print the warning and leave the security to the operator's network — do not invent
  a token scheme that implies more safety than it delivers.
- No new report fields. If the dashboard wants something, it derives it, or the
  schema gets an additive field through the normal process — not a side channel.

## 9. Secrets

Payloads written to the trace are already redacted by the engine, so the dashboard
inherits that. Two additional rules:

- Never serve `payloads/*.json` (the verbose-mode overflow directory) — it is not on
  the artifact allow-list.
- The HTML export runs the same `redact()` pass over the inlined blob before writing,
  because an exported file gets attached to tickets and pasted into chats. Belt and
  braces, and cheap.

## 10. Test plan

- Watcher: appends arrive; a partial last line is buffered then parsed when
  completed; a corrupt line yields `__unparseable__` and the stream continues; a
  shrunk file resets; a new run directory is discovered mid-watch.
- `seq -> offset` index: `from=<seq>` returns the right slice without reading the
  whole file (assert bytes read).
- API: every endpoint against a fixture `.chaos/` (one finished run, one mid-run
  with no `report.json`, one aborted).
- Path traversal: the three attack shapes above are rejected.
- SSE: a client receives frames in order; a slow client gets `desync` rather than
  unbounded buffering; heartbeats arrive.
- `--no-sse` polling path returns the same events as the stream.
- Single-file guarantee: `index.html` and an exported report contain no external
  asset URL.
- Export: opening the exported file with the file:// scheme (checked by parsing, not
  by launching a browser) yields a document with the data blob present and the
  truncation note when capped.
- Perf: 20 000-event fixture, first `/api/runs` response under 200 ms, first events
  page under 300 ms.
- Live end-to-end: run a real fake-agent suite with `--dashboard`, poll
  `/api/suite` while it runs, and assert `status` goes `running → completed` and that
  `passed + failed` reaches the planned count.
- The suite's exit code is identical with and without `--dashboard`.
