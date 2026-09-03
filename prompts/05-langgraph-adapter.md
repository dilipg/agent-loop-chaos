# Phase 05 — LangGraph adapter, state / edge / checkpoint faults (M5)

The core works against plain Python. Now make it work against the primary target.
Expect this phase to be the fiddliest: LangGraph's internals move between versions,
so build the shims defensively.

**Read first, and treat as binding:** `docs/DECISIONS.md` — pre-seeded errata from an adversarial review of this spec; it overrides any doc it contradicts.

**Read first, in full:** `docs/06-LANGGRAPH-ADAPTER.md` Part 1 (all of it),
`docs/03-FAULT-CATALOG.md` section C, `docs/02-API.md` §9.

## Scope

`adapters/langgraph.py`, `adapters/_lc_messages.py`, `faults/state.py`, and the
compat test. Also the two README snippets from `docs/02-API.md` §11 as executable
tests.

## Build

### Version shims first
Before any wrapping logic, write the resolution layer:

```python
def _node_slots(graph) -> list[tuple[str, Callable[[], Any], Callable[[Any], None]]]
def _branch_slots(graph) -> list[tuple[str, str, Callable[[], Any], Callable[[Any], None]]]
def _is_compiled(graph) -> bool
def _lg_version() -> tuple[int, int, int]
```

Each tries a chain of attribute paths (pre-compile `builder.nodes[name].runnable`,
post-compile `compiled.nodes[name].bound`, and any fallback you find in the
installed version) and raises `AdapterError` naming the installed version and every
path it tried when none resolve. Never instrument zero nodes silently: if the slot
list is empty, that is an `AdapterError`.

### `instrument_graph`
Per `docs/06` §§1.2–1.7. Requirements that are easy to get wrong, so verify each:

- Node signature preservation — LangGraph inspects signatures to decide what to
  pass. Wrap with `functools.wraps` **and** set `__signature__`, or introspect once
  and dispatch. Test: a node declaring `(state, config)` and one declaring `(state)`
  both work after instrumentation.
- Async nodes get async wrappers.
- `Runnable` nodes are wrapped via a proxy, not converted to functions.
- Instrumenting twice is idempotent (check for an `__alc_instrumented__` marker).
- `nodes=[...]` restricts instrumentation; `intercept_state=False` skips state
  crossings; both tested.
- `recursion_limit` is set from `Limits.max_steps` and `GraphRecursionError` is
  caught and mapped to `limit_hit="max_steps"`, not to `error`.

### `instrument_model`, `wrap_langchain_tools`, `_lc_messages`
Per `docs/06` §§1.4–1.5. The proxy forwards unknown attributes, intercepts the six
invocation methods plus `bind_tools` (re-wrapping the result). Message
normalization must round-trip losslessly for the five message types, with unknown
fields preserved in `_extra`. Streaming is materialized, faulted once, re-yielded as
a single chunk, and the limitation is recorded in the trace as a `log` event.

### `faults/state.py`
The six section-C faults. `CheckpointRollbackFault` uses the new
`resume_from_checkpoint` action, and `DuplicateSideEffectFault` the `invoke_target`
action (D-10): the **adapter** performs the resume or the repeat invocation —
`apply()` cannot, and for an async node could not await — and every such invocation
is recorded into `HarnessFacts.harness_invocation_seqs` so the
`duplicate_side_effect` probe excludes the harness's own calls. The phase-02 safety
gate applies: `CheckpointRollbackFault` refuses a `side_effecting=True` target unless
the scenario lists it in `allow_side_effects`.

The `state_key_read_after_drop` probe replaces the old `state_key_lost`: once a fault
removes a key its absence is expected for the rest of the run (attribution rule R4),
and the finding is a **consumer reading it without a precondition check**. Wire
`HarnessFacts.state_keys_dropped_at` so the probe can tell the two apart. Enforce the reducer rule from `docs/06` §1.3:
`StateDropFault(mode="remove")` registered at a `post` phase raises `ConfigError`
whose message explains partial updates. Implement `state_get`/`state_set`/
`state_del` dispatching over dict, `TypedDict`, dataclass, and Pydantic model,
never mutating the caller's object. `EdgeMisrouteFault` validates `force_to`
against the graph's node set at registration time. `CheckpointRollbackFault`
records `fault_skipped: no_checkpointer` when no checkpointer exists.

### Real test graph
`tests/fakes/lg_agent.py` — a 4-node graph (`plan → fetch → summarize → respond`)
with a conditional back-edge, a `TypedDict` state with an `add_messages` reducer, two
tools, and a scripted `FakeChatModel` implementing enough of `BaseChatModel` to be
bound and invoked. Both a sync and an async variant. This is a test fixture, not the
demo agent — keep it under 120 lines.

## Tests

- Every section-C fault: unit, trigger, mutation-log, end-to-end on `lg_agent`.
- Sync and async graph runs both produce schema-valid reports.
- Node signature variants, `Runnable` nodes, double instrumentation, partial
  `nodes=` selection.
- Message round-trip property test.
- `bind_tools` after instrumentation keeps the model instrumented.
- Tool faults from phase 02 work unchanged through the LangGraph adapter — run three
  of them against `lg_agent` and assert the same `failure_mode` as the vanilla path.
  That equivalence is the point of the `Crossing` abstraction; if it does not hold,
  fix the adapter, not the fault.
- `tests/test_langgraph_compat.py` asserts each attribute path the shims rely on,
  with a message naming the version when one disappears.
- `tests/test_readme_snippets.py` executes both snippets from `docs/02-API.md` §11.
- Parallel branches: a graph with two concurrent nodes produces a trace whose
  `seq` is strictly monotonic and whose spans reconstruct into a tree — **and** fault
  placement is stable across runs even though the interleaving is not, because
  `call_index` counts per (name, branch) on a topology-derived branch key rather than
  by arrival order (D-32). Add that determinism test explicitly.

## Acceptance checklist

- [ ] `make check` green with `[langgraph]` installed **and** the core suite still
      green with it uninstalled
- [ ] `instrument_graph` works on both `StateGraph` and `CompiledStateGraph`
- [ ] all six state/edge/checkpoint faults implemented and end-to-end tested
- [ ] `StateDropFault(mode="remove")` at `post` raises `ConfigError` with the
      reducer explanation
- [ ] tool and LLM faults produce identical `failure_mode` via both adapters for
      three shared scenarios
- [ ] both README snippets pass as tests
- [ ] compat test documents the exact attribute paths used
- [ ] `GraphRecursionError` maps to `limit_hit`, never to `error`

## Verify

```bash
pip install -e ".[dev,langgraph]"
make check
pytest tests/adapters -q
python -m pytest tests/test_readme_snippets.py -q
pip uninstall -y langgraph langchain-core && pytest tests -q -k "not langgraph and not readme"
```

## Out of scope

The judge, the refinement loop, the demo agent. Do not build a second framework
adapter, however tempting the abstraction looks from here.
