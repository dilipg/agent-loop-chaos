# 02 — Public API (normative)

Every name in this document is **frozen**. Implement these signatures exactly.
Additions are allowed; renames are not.

## 1. Top-level exports

```python
# agent_loop_chaos/__init__.py
__all__ = [
    "__version__",
    # engine
    "ChaosEngine", "ChaosResult", "Limits", "Crossing", "HarnessFacts",
    # targeting
    "Target", "Trigger",
    # scenarios
    "Scenario", "ChaosSuite", "SuiteResult", "load_suite",
    # loop
    "RefinementLoop", "LoopReport",
    # judges
    "Judge", "Verdict", "RuleJudge", "SLMJudge", "EnsembleJudge",
    # faults (re-exported for convenience)
    "Fault",
    # assertions
    "Expect", "AssertionResult",
    # errors
    "ChaosError", "ConfigError", "MissingExtraError", "SchemaError",
    "JudgeError", "AdapterError", "LimitExceeded", "ExplicitError",
    # enums
    "Severity", "ExpectedBehavior", "FailureMode",
]
```

## 2. `ChaosEngine`

```python
class ChaosEngine:
    def __init__(
        self,
        *,
        seed: int = 1337,
        out_dir: str | Path = ".chaos",
        trace_level: Literal["minimal", "standard", "verbose"] = "standard",
        limits: Limits | None = None,
        judge: Judge | str | None = None,      # "rules" | "slm" | "ensemble" | instance | None
        redact_keys: Sequence[str] = (),
        strict_schema: bool = True,
        dry_run: bool = False,
        write_bundle: bool = True,
        allow_remote_judge: bool = False,      # D-22: a non-loopback judge needs opt-in
        tags: Mapping[str, str] | None = None,
        judge_options: Mapping[str, Any] | None = None,   # model/base_url/transport for SLMJudge
        narrate_all: bool = False,             # narrate passing runs too; off by default
    ) -> None: ...
```

### Registering faults

```python
    def register_fault(
        self,
        fault: Fault,
        *,
        target: Target | None = None,
        trigger: Trigger | None = None,
        fault_id: str | None = None,          # default: "f{n}" in registration order
        # --- ergonomic shorthands, mutually exclusive with `target` ---
        target_tool: str | None = None,
        target_node: str | None = None,
        target_llm: str | None = None,
        target_state_key: str | None = None,
    ) -> str: ...                              # returns fault_id
```

`register_fault` validates eagerly: unknown target layer for the fault's `accepts`
set, impossible trigger (`on_call=0`), or `probability` outside `[0, 1]` raise
`ConfigError` immediately.

```python
    def clear_faults(self) -> None: ...
    def plan(self, *, entrypoint: str | None = None, adapter: str = "vanilla",
             expected_behavior: ExpectedBehavior = "graceful_degradation",
             must_not: Sequence[str] = ()) -> dict: ...   # canonical plan; plan_hash source
```

`plan()` still works bare; the keyword arguments exist because `plan_hash` is
computed over the entrypoint, adapter, `expected_behavior` and sorted `must_not`
as well as the fault list (`docs/04-SCHEMAS.md` §3), and the engine supplies them
at run time.

Three further methods exist for adapters and the post-run pipeline, and are stable
but secondary: `is_active()` (whether a run of this engine is live in the current
context — the check that keeps a decorated tool inert in production),
`route_sync()` / `route_async()` (drive the pre/post/error crossings for one call),
and `harness_facts()` (snapshot what the harness itself caused, for `docs/11` §2).
`ChaosEngine(strict_trace=True)` validates every trace event against the schema as
it is written; the test suite runs with it on.

### Instrumenting plain Python

```python
    def tool(self, fn: Callable | None = None, *, name: str | None = None,
             side_effecting: bool | None = None, schema: dict | None = None) -> Callable: ...
    def llm(self, fn: Callable | None = None, *, name: str = "default") -> Callable: ...
    def wrap_tools(self, tools: Mapping[str, Callable]) -> dict[str, Callable]: ...
    def wrap_callable(self, fn: Callable, *, layer: Layer, name: str) -> Callable: ...
    def intercept_tools(self, *tool_names: str) -> Callable: ...   # decorator for an agent fn
    def instrument_object(self, obj: Any, *, tools: Sequence[str] = (),
                          llm_methods: Sequence[str] = (),
                          nodes: Sequence[str] = ()) -> Any: ...   # wrap named methods on an instance
    def step(self) -> int: ...                                     # advance the step counter (adapters)
    def validated(self, value: Any = None, *, name: str | None = None) -> None: ...
    def note(self, message: str) -> None: ...
    def bind_context(self) -> AbstractContextManager[None]: ...    # D-42: enter inside your own thread
    def cross(self, crossing: Crossing) -> Any: ...                # route one crossing through the plan
```

Both `tool` and `llm` work bare (`@engine.tool`) and parameterized
(`@engine.tool(name="get_weather_data")`), sync and async.

`side_effecting` is tri-state (D-56): `True` and `False` are both deliberate
declarations, and omitting it leaves the tool **undeclared**.
`engine.require_declared_side_effects()` raises `ConfigError` while any registered
tool is undeclared, which is what `--preset full` calls before starting
(`SAFETY.md` §1 item 3). Undeclared is treated as not side-effecting everywhere
else.

`intercept_tools()` decorates the *agent entrypoint*; inside its dynamic scope,
module-level tool functions already registered by name are intercepted. It exists
for compatibility with the original blueprint API and is implemented on top of
`wrap_callable` + a contextvar.

### Running

```python
    def run(
        self,
        target: Callable | Any,                # agent fn, or a compiled LangGraph app
        *,
        inputs: Any = None,                    # positional/keyword payload for the agent
        initial_state: Mapping | None = None,
        scenario_id: str | None = None,
        expected_behavior: ExpectedBehavior = "graceful_degradation",
        must_not: Sequence[str] = (),           # probe codes that always fail the run
        expect: Expect | Mapping | None = None, # declarative assertions (docs/11 §4)
        expected_errors: Sequence[str] = (),    # exception classes meaning "explicit error"
        allow_side_effects: Sequence[str] = (), # tools the side-effect gate may target
        dry_run: bool | None = None,            # overrides the engine setting for this run
        attempt: int = 1,                       # part of run_id; the loop threads the round
        adapter: Literal["auto", "vanilla", "langgraph"] = "auto",
        baseline: ChaosResult | None = None,
        seed: int | None = None,                # overrides the engine seed for this run
    ) -> ChaosResult: ...

    async def arun(self, ...) -> ChaosResult: ...     # same signature

    # Compatibility alias for the original blueprint call shape.
    def run_with_state(self, target, *, query=None, initial_state=None, **kw) -> ChaosResult: ...

    def baseline(self, target, **kw) -> ChaosResult: ...
    # Convenience: run with the plan disabled (dry_run=True) and expected_behavior="ignore_and_continue".

    def replay(self, run_dir: str | Path, *, target: Callable | Any | None = None) -> ChaosResult: ...
    # Rebuilds the plan and seed from plan.json and re-runs. Same seed ⇒ same story.
```

### Limits

```python
@dataclass(frozen=True)
class Limits:
    max_steps: int = 25
    max_tool_calls: int = 50
    max_llm_calls: int = 25
    timeout_s: float = 120.0
    max_tokens: int | None = None
    max_injected_delay_ms: int = 2000
```

## 3. Targeting

```python
Layer = Literal["tool", "llm", "state", "node", "edge", "checkpoint"]
Phase = Literal["pre", "post", "error"]

Target(layer=None, tool=None, node=None, llm=None, state_key=None, phase=None, predicate=None)
Trigger(on_call=None, on_step=None, after_step=None, probability=1.0,
        max_fires=1, cooldown_calls=0, stop_after_step=None)
```

`on_call` accepts an `int` or a sequence of `int`, so one fault can fire on several
call indices (`on_call=[1, 3]`). `Trigger.call_indices()` normalizes it.

A `Target` with `phase=None` matches every phase on its layer, but a fault is only
ever offered a crossing whose `(layer, phase)` pair is in its `accepts` set — checked
at registration *and* at every crossing (D-55). `Target.implied_layer()` returns the
layer a name field implies.

`tool`, `node`, `llm` accept `fnmatch` globs. `state_key` is a dotted path with
`*` wildcards (`"messages.*.content"`).

## 4. Faults

All faults live in `agent_loop_chaos.faults` and are also reachable by string
`kind` through `FAULT_REGISTRY` (that is how YAML scenarios name them).

```python
from agent_loop_chaos.faults import (
    # tool execution
    ToolCorruptionFault, ToolErrorFault, ToolLatencyFault, ToolTimeoutFault,
    ArgumentTamperFault, StaleDataFault, NonDeterminismFault, DuplicateSideEffectFault,
    # loop / rate
    LoopTrapFault, RateLimitFault,
    # llm / prompt
    ContextShrinkFault, ContextNoiseFault, GoalDriftFault,
    LLMMalformedOutputFault, LLMRefusalFault, LLMEmptyFault, LLMTruncationFault,
    MalformedToolCallFault, HallucinationSeedFault,
    # adversarial
    PromptInjectionFault,
    # graph state
    StateDropFault, StateTypeFault, StateStaleFault,
    NodeSkipFault, EdgeMisrouteFault, CheckpointRollbackFault,
)
```

Constructor parameters, semantics, and expected graceful behaviour for each:
**`docs/03-FAULT-CATALOG.md`**. That catalog is normative too.

```python
def fault_from_dict(d: Mapping) -> Fault: ...     # {"type": "ToolCorruptionFault", "params": {...}}
def list_faults() -> list[FaultInfo]: ...         # powers `alc list-faults`
```

## 5. Result objects

```python
@dataclass
class ChaosResult:
    # identity
    schema_version: str
    run_id: str
    scenario_id: str | None
    library_version: str
    seed: int
    plan_hash: str
    started_at: str
    finished_at: str
    duration_ms: int
    target: TargetInfo                      # framework, entrypoint, adapter_version
    # outcome
    success: bool
    verdict: Verdict
    failure_mode: FailureMode
    severity: Severity
    symptoms: list[Symptom]
    # what chaos was introduced
    injected_faults: list[FaultRecord]
    chaos_narrative: str                    # SLM prose; deterministic template in rules mode
    randomness: RandomnessLog               # seed, per-purpose draw counts, decisions taken
    # what the agent saw and did
    agent_state_pre_fault: Any | None
    agent_state_post_fault: Any | None
    llm_exchanges: list[LLMExchange]
    tool_calls: list[ToolCallRecord]
    final_output: Any
    error: ErrorInfo | None
    # comparison + numbers
    baseline: BaselineRef | None
    delta_vs_baseline: BaselineDelta | None
    metrics: Metrics
    loop: LoopInfo
    # what to do about it
    root_cause_hypothesis: str | None
    refinement_hint: str | None
    suggested_fixes: list[SuggestedFix]
    code_pointers: list[CodePointer]
    reproduce: Reproduce
    artifacts: Artifacts
    schema_errors: list[str]

    def to_dict(self) -> dict: ...
    def to_json(self, *, indent: int | None = 2) -> str: ...
    @classmethod
    def from_dict(cls, d: Mapping) -> ChaosResult: ...
    def validate(self) -> list[str]: ...        # jsonschema errors, [] when valid
    def agent_task_markdown(self) -> str: ...   # the AGENT_TASK.md body
    def summary_line(self) -> str: ...          # one-line CLI output
```

`ChaosResult` also carries `attempt: int`, `dry_run: bool`, `tags: dict[str, str]`
and `assertions: list[AssertionResult]` (D-12, D-28). Pass/fail semantics:
`docs/11-OUTCOMES-AND-ASSERTIONS.md`.

Field-level types and allowed values: `schemas/chaos_report.schema.json` +
`docs/04-SCHEMAS.md`. The dataclass and the schema must not drift; a test asserts
that every dataclass field appears in the schema and vice versa.

## 6. Scenarios and suites

```python
@dataclass
class Scenario:
    id: str
    entrypoint: str | Callable                   # "module:attr" or a callable
    inputs: Any = None
    initial_state: Mapping | None = None
    faults: list[FaultSpec] = field(default_factory=list)
    seed: int = 1337
    expected_behavior: ExpectedBehavior = "graceful_degradation"
    must_not: list[str] = field(default_factory=list)
    expect: Expect | None = None
    expected_errors: list[str] = field(default_factory=list)
    allow_side_effects: list[str] = field(default_factory=list)
    objective_state_key: str = "query"
    dry_run: bool = False
    limits: Limits = Limits()
    judge: JudgeSpec | None = None
    tags: dict[str, str] = field(default_factory=dict)
    description: str | None = None
    matrix: dict[str, list] | None = None        # cartesian expansion over fault params/seeds

    def expand(self) -> list[Scenario]: ...      # applies `matrix`; ids get "[k=v]" suffixes

class ChaosSuite:
    def __init__(self, scenarios: Sequence[Scenario], *, engine: ChaosEngine | None = None): ...
    @classmethod
    def from_yaml(cls, path: str | Path) -> ChaosSuite: ...
    def run(self, *, jobs: int = 1, fail_fast: bool = False,
            baseline: bool = True, filter: str | None = None) -> SuiteResult: ...

@dataclass
class SuiteResult:
    results: list[ChaosResult]
    passed: int
    failed: int
    failure_modes: dict[str, int]
    coverage: dict[str, int]        # fault kind -> times exercised
    out_dir: Path
    def to_dict(self) -> dict: ...
    def worst(self, n: int = 5) -> list[ChaosResult]: ...
```

`load_suite(path)` is a module-level convenience for `ChaosSuite.from_yaml`.
YAML support requires the `[yaml]` extra; JSON suites work with no extras.

## 7. Judges

```python
class Judge(Protocol):
    name: str
    def judge(self, ev: JudgeEvidence) -> Verdict: ...

@dataclass
class Verdict:
    passed: bool                      # authority: probes + expected_behavior
    expected_behavior: ExpectedBehavior
    observed_behavior: str
    failure_mode: FailureMode
    severity: Severity
    confidence: float                 # 0..1
    narrative: str
    root_cause_hypothesis: str | None
    refinement_hint: str | None
    suggested_fixes: list[SuggestedFix]
    judge_meta: JudgeMeta             # kind, model, endpoint, temperature, prompt_hash,
                                      # latency_ms, tokens, attempts, fell_back_to_rules,
                                      # evidence_dropped, auto_selected (D-70)
    judge_disagreement: str | None    # set when the model contradicted the probes

class RuleJudge:
    def __init__(self, *, templates: Mapping[str, str] | None = None): ...

class SLMJudge:
    def __init__(
        self,
        *,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        transport: Literal["openai", "ollama", "anthropic"] = "openai",
        temperature: float = 0.0,
        max_tokens: int = 900,
        timeout_s: float = 60.0,
        retries: int = 2,
        prompt_dir: str | Path | None = None,
        offline_fallback: bool = True,     # fall back to RuleJudge instead of raising
        allow_remote_judge: bool = False,  # required for a non-loopback endpoint (D-22)
    ) -> None: ...

    def reachable(self) -> bool: ...                     # cached per instance, 1.5s timeout
    def narrate(self, ev: JudgeEvidence) -> str | None: ...          # narrator.md
    def suggest_fixes(self, ev, verdict=None) -> list[dict]: ...     # refiner.md
    last_call: dict[str, Any]                            # redacted, written to judge.json

class EnsembleJudge:
    def __init__(self, *, rules: RuleJudge | None = None, model_judge: Judge | None = None): ...

def select_judge(judge: Judge | str | None, **model_kwargs) -> Judge: ...
```

`EnsembleJudge` is the default when `judge="ensemble"` or `judge=None` and a model
endpoint is reachable; otherwise `RuleJudge`. Reachability is probed once per
process with a short timeout and cached. `select_judge` is what
`ChaosEngine(judge=…)` resolves through, and it records an automatic choice in
`judge_meta.auto_selected` (D-70).

`SLMJudge` raises `ConfigError` at construction for a non-loopback `base_url` or the
`anthropic` transport unless `allow_remote_judge=True`, naming what would be sent
(D-22). `base_url` is normalized per transport: `/v1` stripped for `ollama`, appended
for `openai` (D-34).

### The evidence projection

```python
@dataclass
class JudgeEvidence:                     # docs/05 section 2 for the full field list
    def to_prompt_dict(self) -> dict[str, Any]: ...   # flat, fence-escaped (D-21)
    def size_bytes(self) -> int: ...
    def fit(self, *, target=6144, hard_cap=12288) -> tuple[JudgeEvidence, list[str]]: ...

def build_evidence(result: ChaosResult, *, code_context=None) -> JudgeEvidence: ...
def render(template: str, values: Mapping[str, Any]) -> str: ...   # mustache-lite
def escape_fences(text: str) -> str: ...
def redact_source(text: str) -> str: ...  # string literals under a sensitive name
def judge_output_schema() -> dict[str, Any]:   # $defs/judge_output, refs inlined (D-20)
def normalize_base_url(base_url: str, transport: str) -> str: ...
```

`build_evidence` is pure over a `ChaosResult`, which is what lets `alc judge` rebuild
evidence from a stored `report.json` and re-judge it with a better model without
re-running the agent. `fit` drops sections in the fixed order `code_context` →
`baseline_output` → older exchanges → `tool_summary` and returns what it dropped;
the engine records that in `judge_meta.evidence_dropped`.

## 8. Refinement loop

```python
class RefinementLoop:
    def __init__(
        self,
        suite: ChaosSuite,
        *,
        engine: ChaosEngine | None = None,
        max_rounds: int = 3,
        stop_when: Literal["all_pass", "no_new_failures", "rounds"] = "no_new_failures",
        on_findings: Callable[[list[ChaosResult]], None] | None = None,
    ) -> None: ...

    def run(self) -> LoopReport: ...

@dataclass
class LoopReport:
    rounds: list[RoundResult]
    new_failures_per_round: list[int]
    fixed_between_rounds: list[list[str]]     # scenario ids that flipped to pass
    regressions: list[str]
    tasks_written: list[Path]                 # newest AGENT_TASK.md per scenario (D-76)
    out_dir: Path
    tamper: dict[str, Any]                    # plan/file/must_not hashes, probe fingerprint
    aborted: bool                             # a control scenario failed
    abort_reason: str | None
    wall_ms: int
    @property
    def flipped(self) -> list[dict]: ...      # [{scenario_id, round}]
    def to_dict(self) -> dict: ...
    def markdown(self) -> str: ...

@dataclass
class RoundResult:
    round: int
    results: list[ChaosResult]
    outcomes: dict[str, Outcome]              # pass | fail | flipped_to_pass |
                                              # flipped_to_fail | still_failing | skipped
    wall_ms: int

def run_suite(scenarios, *, out_dir, seed=None, attempt=1, ...) -> list[ChaosResult]: ...
def probes_fingerprint() -> str: ...
def scenario_fingerprint(scenario, plan_hash) -> dict: ...
```

`RefinementLoop` also accepts `out_dir`, `judge`, `judge_options` and `no_baseline`,
so it can be driven without constructing an engine first.

`run_suite` is the single-round path shared by `alc run` and one round of the loop
(D-74). Both must plan a scenario identically or the loop's "same seeds every round"
guarantee is not true.

A scenario whose id starts with `control.` is a control: it must pass in every round,
and its failure aborts the loop (D-75), which is exit code **4** from the CLI.

`on_findings` is the hand-off hook: the caller may shell out to a coding agent,
apply patches, and return; the loop then re-runs the suite and reports what flipped.
The loop never edits source itself.

## 9. Adapters

```python
class Adapter(Protocol):
    name: str
    version: str
    def instrument(self, target: Any, engine: ChaosEngine, ctx: RunContext) -> Any: ...
    def run(self, instrumented: Any, inputs: Any, ctx: RunContext) -> Any: ...
    def state_of(self, instrumented: Any, ctx: RunContext) -> Any: ...

# langgraph
def instrument_graph(
    graph: Any,                        # StateGraph or CompiledStateGraph
    engine: ChaosEngine,
    *,
    nodes: Sequence[str] | None = None,       # None = all
    intercept_tools: bool = True,
    intercept_state: bool = True,
    intercept_checkpoints: bool = False,
) -> Any: ...
```

Details and the exact LangGraph hook points: `docs/06-LANGGRAPH-ADAPTER.md`.

## 10. CLI

```
alc run <suite.yaml|scenario.yaml|module:attr>   [--seed N] [--jobs N] [--judge rules|slm|ensemble]
                                                 [--model M] [--base-url U] [--out DIR]
                                                 [--transport openai|ollama|anthropic]
                                                 [--allow-remote-judge] [--narrate-all]
                                                 [--trace-level L] [--filter GLOB] [--fail-fast]
                                                 [--no-baseline] [--dry-run] [--json]
                                                 [--rounds N] [--stop-when …]
                                                 [--dashboard] [--port N] [--linger S]
alc replay <run_dir> [--seed N]
alc judge <run_dir> [--judge …] [--model …] [--base-url U] [--transport T]
                    [--allow-remote-judge]         # re-judge without re-running the agent
alc explain <run_dir>                              # human-readable narrative to stdout
alc report <run_dir|out_dir> [--format md|json|html] [-o FILE] [--max-trace-events N]
alc validate <report.json|suite.yaml>
alc list-faults [--json]
alc init                                           # scaffold chaos/ dir + example scenario
alc dashboard [--out DIR] [--port 7717] [--host 127.0.0.1] [--open]
              [--poll-ms 250] [--max-events N] [--no-sse] [--once]
```

`dashboard`, `--dashboard`, and `--format html` arrive in phase 10; everything else
is v0.1. `alc dashboard` is stdlib-only and read-only — see
`docs/10-DASHBOARD.md`.

Exit codes: `0` all passed, `1` at least one scenario failed, `2` configuration or
usage error, `3` internal error. `--json` prints one JSON object to stdout and
nothing else, so CI can pipe it.

## 11. The 5-line integration promise

```python
from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import ToolCorruptionFault
from agent_loop_chaos.adapters.langgraph import instrument_graph

engine = ChaosEngine(seed=1337)
engine.register_fault(ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
                      target_tool="get_weather_data")
result = engine.run(instrument_graph(app, engine), inputs={"query": "Pack list for Paris"})
print(result.to_json())
```

Plain Python, matching the original blueprint:

```python
from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import ToolCorruptionFault

chaos = ChaosEngine()
chaos.register_fault(ToolCorruptionFault(mutation_type="empty_json"),
                     target_tool="get_weather_data")

@chaos.tool
def get_weather_data(location: str) -> list[dict]: ...

@chaos.intercept_tools()
def weather_agent(user_query, state):
    data = get_weather_data(state["location"])
    return llm_generate(user_query, data)

result = chaos.run_with_state(weather_agent, query="Pack list for Paris",
                              initial_state={"location": "Paris"})
print(result.to_json())
```

Both snippets must work verbatim by the end of phase 05. Add them to
`tests/test_readme_snippets.py` and run them in CI.
