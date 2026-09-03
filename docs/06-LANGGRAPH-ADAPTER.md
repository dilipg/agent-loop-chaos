# 06 — Adapters: LangGraph (primary) and vanilla Python

Both adapters implement the same `Adapter` protocol and produce the same
`Crossing` objects. Nothing above `adapters/` may contain a framework `import`.

```python
class Adapter(Protocol):
    name: str
    version: str
    def instrument(self, target, engine, ctx) -> Any: ...
    def run(self, instrumented, inputs, ctx) -> Any: ...
    def state_of(self, instrumented, ctx) -> Any: ...
```

`adapter="auto"` sniffs by duck-typing, never by importing langgraph
eagerly: if the target has `compile` and `add_node`, or `get_graph` and `invoke`,
and `"langgraph"` appears in `type(target).__module__`, choose the LangGraph
adapter; otherwise vanilla.

---

## Part 1 — LangGraph

### 1.1 Where to hook

LangGraph gives four distinct surfaces, and the library needs all four. Prefer
**pre-compile** instrumentation; support post-compile as a documented best-effort.

| Surface | Hook | Enables |
|---|---|---|
| Node functions | wrap each node callable | `node` crossings, `state` faults, step counting |
| Tools | wrap tool callables / `ToolNode` tools | all tool faults |
| Chat model | proxy the model object | all LLM/prompt faults |
| Conditional edges | wrap the branch path function | `EdgeMisrouteFault`, `edge_taken` events |
| Checkpointer | wrap `put` / `get_tuple` | `CheckpointRollbackFault`, checkpoint events |

```python
def instrument_graph(graph, engine, *, nodes=None, intercept_tools=True,
                     intercept_state=True, intercept_checkpoints=False): ...
```

Accepts either a `StateGraph` (not yet compiled — preferred) or a
`CompiledStateGraph`. Returns the same kind of object it was given, instrumented.
For a `StateGraph` it does not compile; the caller still calls `.compile(...)`, and
`instrument_graph` must tolerate being handed the builder and then seeing the
compile happen afterwards.

### 1.2 Node wrapping

```python
def _wrap_node(name: str, fn: Callable, engine, ctx) -> Callable:
    @functools.wraps(fn)
    def wrapper(state, config=None, **kw):
        engine.step()                                    # step_started
        state = engine.cross(Crossing(layer="node",  phase="pre",  name=name, state=state, ...))
        state = engine.cross(Crossing(layer="state", phase="pre",  name=name, state=state, ...))
        try:
            out = fn(state, config, **kw) if _takes_config(fn) else fn(state, **kw)
        except BaseException as exc:
            engine.cross(Crossing(layer="node", phase="error", name=name, exception=exc, ...))
            raise
        out = engine.cross(Crossing(layer="state", phase="post", name=name, result=out, state=state, ...))
        return engine.cross(Crossing(layer="node", phase="post", name=name, result=out, state=state, ...))
    return wrapper
```

Details that matter:

- **Signature preservation.** LangGraph inspects node signatures to decide whether
  to pass `config` / `store` / `writer`. Use `functools.wraps` **and** copy
  `__signature__` from the original, or introspect once and dispatch as above.
  Getting this wrong makes graphs fail to compile.
- **Async nodes** get an `async def` wrapper. Detect with
  `inspect.iscoroutinefunction` (and unwrap `functools.partial`).
- **Runnables.** A node may be a `Runnable` rather than a plain function. Wrap by
  building a `RunnableLambda` around the wrapper, or by wrapping `.invoke` /
  `.ainvoke` on a thin proxy. Keep the original object reachable at
  `wrapper.__alc_original__` for debugging.
- **Pre-compile access:** `builder.nodes[name]` holds a node spec whose callable
  lives at `.runnable` (LangGraph ≥0.2). Replace it in place.
  **Post-compile access:** `compiled.nodes[name].bound`. Both attribute paths are
  version-sensitive: resolve them through a small
  `_node_slots(graph) -> list[tuple[getter, setter]]` shim with a fallback chain,
  and raise `AdapterError` naming the installed langgraph version when no slot is
  found. Never silently instrument nothing.

### 1.3 State faults and the reducer problem

A LangGraph node returns a **partial update** that reducers merge into state. This
has a consequence the implementation must respect:

> You cannot remove a state key by returning a partial update. Removal must be
> applied to the state **entering** a node, not to the update leaving one.

Therefore:

- `StateDropFault(mode="remove")` is only valid at `(state, pre)` / `(node, pre)`.
  Registering it at `post` raises `ConfigError` with this explanation.
- `mode="null"` works at either boundary (setting `None` is expressible as an
  update), but note in the report that a reducer may reject `None`.
- `StateTypeFault` at `post` mutates the *update dict*; at `pre` it mutates the
  incoming state. Both are useful — the `pre` form exercises the consumer, the
  `post` form exercises the reducer.
- Record a `state_mutated` event with the JSON patch and `by: "fault"`, distinct
  from the agent's own updates (`by: "agent"`).
- State may not be a dict (dataclass, `TypedDict`, Pydantic model). Support all
  three via `state_get`/`state_set`/`state_del` helpers that dispatch on type and
  fall back to `dataclasses.replace` / `model_copy(update=…)`. Never mutate the
  caller's object.

### 1.4 Tool interception

Three shapes to handle:

1. **A `ToolNode`** — read its tool list (`.tools_by_name` in current versions),
   wrap each entry, rebuild.
2. **A model with bound tools** — `model.bind_tools([...])`. Wrap the tool objects
   *before* binding; document that instrumenting after `bind_tools` may miss them,
   and detect that case by comparing tool identity.
3. **Plain functions decorated with `@tool`** — wrap `tool.func` / `tool.coroutine`
   if present, else wrap `tool.invoke` / `tool.ainvoke` on a proxy.

`wrap_langchain_tools(tools, engine) -> list` is the public helper. Tool name comes
from `tool.name`, falling back to `fn.__name__`.

Args normalization: LangChain calls tools with a single dict of arguments. The
adapter presents that dict as `crossing.kwargs` (and leaves `args` empty) so
`ArgumentTamperFault` has one consistent shape to mutate regardless of framework.

### 1.5 Model interception

```python
def instrument_model(model, engine, *, name: str = "default"): ...
```

Returns a proxy object that forwards everything (`__getattr__`) but intercepts
`invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch`, and `bind_tools`
(re-wrapping the returned model so a bound copy stays instrumented).

Message normalization, both directions, in `adapters/_lc_messages.py`:

| LangChain | normalized |
|---|---|
| `SystemMessage` | `{"role": "system", "content": …}` |
| `HumanMessage` | `{"role": "user", …}` |
| `AIMessage` | `{"role": "assistant", "content": …, "tool_calls": [...]}` |
| `ToolMessage` | `{"role": "tool", "content": …, "tool_call_id": …, "name": …}` |
| a bare string | `{"role": "user", "content": …}` |

Round-tripping must be lossless for the fields above; unknown fields are carried in
`_extra` and restored. LLM faults operate only on the normalized form, which is
what makes them framework-agnostic and reusable by the vanilla adapter.

Streaming: for v0.1, `stream`/`astream` are instrumented by materializing the
stream, applying `post`-phase faults to the concatenated result, and re-yielding it
as a single chunk. Document the limitation. Do not attempt per-chunk faults.

### 1.6 Conditional edges

`builder.branches[source][name].path` is the decision callable. Wrap it: call the
original, emit `edge_taken` with `{from, to, forced: false}`, then let
`EdgeMisrouteFault` override the destination (`forced: true`). Validate that the
forced destination is a real node in the graph at **registration** time, so a typo
fails fast rather than producing a confusing runtime error.

### 1.7 Checkpointer

Wrap the checkpointer object: `put`, `put_writes`, `get_tuple`, `alist` and async
twins. Emit `checkpoint_written` / `checkpoint_restored`.
`CheckpointRollbackFault` re-invokes the graph from an earlier checkpoint using
`config={"configurable": {"thread_id": …, "checkpoint_id": <older>}}`, which is the
supported public path for time travel — prefer it over hand-editing the saver.
When no checkpointer is configured, record `fault_skipped` with reason
`no_checkpointer`; that is not a failure.

### 1.8 Steps, recursion, and limits

- The engine's `step` increments on node entry. Map `Limits.max_steps` onto
  LangGraph's own `recursion_limit` (`config={"recursion_limit": max_steps}`) so the
  framework aborts too, and catch `GraphRecursionError` as `limit_hit="max_steps"`
  rather than as an agent crash.
- Parallel branches: crossings interleave. Keep `span_id`/`parent_span_id` from a
  contextvar so the trace is reconstructable, and take a lock around `seq`.
- `interrupt()` / human-in-the-loop is out of scope for v0.1: if the graph
  interrupts, finish the run with `observed_behavior="indeterminate"` and a
  `harness` note. Do not attempt to auto-resume.

### 1.9 Version compatibility

Support `langgraph>=0.2,<0.7` initially. Add
`tests/test_langgraph_compat.py` that asserts the attribute paths the adapter
depends on still exist, with a clear failure message naming the version and the
missing path. Pin the tested version in CI and add a nightly job against
`langgraph@latest` so breakage is discovered by the library, not by users.

### 1.10 Minimal integration example (must work verbatim)

```python
from langgraph.graph import StateGraph
from agent_loop_chaos import ChaosEngine, Trigger
from agent_loop_chaos.faults import ToolCorruptionFault, ContextShrinkFault
from agent_loop_chaos.adapters.langgraph import instrument_graph, instrument_model, wrap_langchain_tools

engine = ChaosEngine(seed=1337)
engine.register_fault(ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
                      target_tool="get_weather_data", trigger=Trigger(on_call=1))
engine.register_fault(ContextShrinkFault(keep_ratio=0.4, strategy="middle_out"),
                      target_llm="default", trigger=Trigger(after_step=3, max_fires=None))

model = instrument_model(llm, engine)
tools = wrap_langchain_tools([get_weather_data, search_flights], engine)
builder = build_graph(model, tools)          # your existing code, unchanged
app = instrument_graph(builder, engine).compile()

result = engine.run(app, inputs={"query": "Pack list for Paris"},
                    initial_state={"location": "Paris"},
                    expected_behavior="graceful_degradation")
print(result.summary_line())
```

---

## Part 2 — Vanilla Python

For agents that are plain functions, classes, or hand-rolled while-loops.

### 2.1 Registration

```python
engine = ChaosEngine(seed=1337)

@engine.tool                        # name = fn.__name__
def get_weather_data(location: str) -> list[dict]: ...

@engine.tool(name="flights")
async def search_flights(origin: str, dest: str) -> dict: ...

@engine.llm                         # name = "default"
def llm_generate(prompt: str, data: Any) -> str: ...
```

`engine.tool` returns a wrapper that routes `(tool, pre)` → real call →
`(tool, post)` or `(tool, error)`. `engine.llm` does the same at the `llm` layer,
normalizing the outbound payload: if the callable's first argument is a list of
message dicts it is used as-is; if it is a string it becomes
`[{"role": "user", "content": …}]`; anything else is passed through with
`messages_unavailable=true` recorded, which disables `pre`-phase LLM faults for
that call with reason `unnormalizable_payload`. Say so in the report rather than
guessing.

### 2.2 Dict-of-tools and late binding

```python
tools = engine.wrap_tools({"get_weather_data": get_weather_data, "flights": search_flights})
```

Use this when tools are dispatched by name from a registry — the common
hand-rolled-loop shape.

### 2.3 `intercept_tools` (blueprint compatibility)

```python
@engine.intercept_tools()
def weather_agent(user_query, state): ...
```

Implementation: on entry, push a contextvar marking the engine active; module-level
functions already registered via `@engine.tool` consult that contextvar and pass
through untouched when it is unset. This keeps decorated tools inert outside a
chaos run — important, because users will leave the decorator on in production
code. Passing explicit names (`@engine.intercept_tools("get_weather_data")`)
restricts interception to those tools.

`@engine.tool` outside a run is a no-op wrapper with ~1 µs overhead; a test asserts
that.

### 2.4 Steps and state

- A step increments on each tool call and each LLM call. `Limits.max_steps` aborts
  by raising `LimitExceeded` inside the wrapper — which is recorded as
  `limit_hit`, not as an agent bug.
- State: whatever the caller passes as `initial_state` is tracked by the engine, and
  `state` faults apply to it at tool/LLM boundaries. If the agent keeps state in a
  local variable the engine cannot see, `state` faults record
  `fault_skipped: no_visible_state`. Document that graph/state faults are best
  served by the LangGraph adapter, and that vanilla agents can opt in by passing a
  mutable mapping as `initial_state` and using it as their working state.

### 2.5 Class-based agents

```python
agent = engine.instrument_object(agent, tools=["fetch", "rank"], llm_methods=["complete"])
```

Wraps named methods on an instance. Same crossing semantics; `name` is
`f"{Class}.{method}"` for nodes and the bare method name for tools.
