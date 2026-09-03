"""Catalog section B: LLM and prompt faults.

Tool faults break the data an agent receives. These break the reasoning substrate:
the context it reasons over and the responses it trusts. Every `pre` fault's effect
is asserted on **the messages the model actually received**, not on the outcome
object, because that is the only thing that proves the plumbing carried it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine, Limits
from agent_loop_chaos.context import Counters, Crossing, FaultContext, RunContext
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.faults._messages import flatten, total_tokens
from agent_loop_chaos.trace import TraceRecorder


def context(seed: int = 1337, *, key: str = "llm001") -> FaultContext:
    """Build a `FaultContext` for calling `apply()` directly.

    Args:
        seed: The run seed.
        key: The fault's spec hash.

    Returns:
        The context.
    """
    ctx = RunContext(
        run_id="run-3f9a12c4",
        seed=seed,
        started_at="1970-01-01T00:00:00Z",
        limits=Limits(),
        trace=TraceRecorder("run-3f9a12c4"),
        counters=Counters(),
        canary="ALC-CANARY-run-3f9a12c4",
    )
    return FaultContext(
        fault_id="f1",
        fault_key=key,
        run=ctx,
        limits=ctx.limits,
        counters=ctx.counters,
        canary=ctx.canary,
    )


def llm_pre(messages: list[dict[str, Any]], *, step: int = 2, call_index: int = 1) -> Crossing:
    """Build an `(llm, pre)` crossing carrying normalized messages.

    Args:
        messages: The normalized conversation.
        step: The agent iteration.
        call_index: Which LLM call this is.

    Returns:
        The crossing.
    """
    return Crossing(
        layer="llm",
        phase="pre",
        name="default",
        messages=messages,
        step=step,
        call_index=call_index,
    )


def conversation(turns: int = 8) -> list[dict[str, Any]]:
    """Build a system-plus-turns conversation.

    Args:
        turns: How many non-system messages to include.

    Returns:
        Normalized messages, ending on a user turn.
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": "Plan a trip. Always cite sources."}]
    for index in range(turns - 1):
        role = "user" if index % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"{role} turn {index} " + "x" * 200})
    out.append({"role": "user", "content": "What should I pack? " + "y" * 200})
    return out


def engine(tmp_path: Path, **kw: Any) -> ChaosEngine:
    """Build an engine writing into a temp directory.

    Args:
        tmp_path: pytest's temp directory.
        **kw: Constructor overrides.

    Returns:
        The engine.
    """
    kw.setdefault("seed", 1337)
    kw.setdefault("out_dir", tmp_path / ".chaos")
    kw.setdefault("strict_trace", True)
    return ChaosEngine(**kw)


# ============================================================= B1 ContextShrinkFault


def test_shrink_returns_replace_messages() -> None:
    """1/4 unit: it rewrites the outgoing conversation."""
    from agent_loop_chaos.faults.llm import ContextShrinkFault

    outcome = ContextShrinkFault(keep_ratio=0.4).apply(llm_pre(conversation()), context())
    assert outcome.action == "replace_messages"


def test_shrink_middle_out_keeps_the_system_message_and_the_last_user_turn() -> None:
    """The two things an agent needs to stay grounded, so they are never dropped."""
    from agent_loop_chaos.faults.llm import ContextShrinkFault

    original = conversation()
    outcome = ContextShrinkFault(keep_ratio=0.3).apply(llm_pre(original), context())
    kept = outcome.value
    assert kept[0]["role"] == "system"
    assert kept[-1]["content"] == original[-1]["content"]


def test_shrink_actually_reduces_the_token_count() -> None:
    """A shrink that shrank nothing would prove nothing."""
    from agent_loop_chaos.faults.llm import ContextShrinkFault

    original = conversation()
    outcome = ContextShrinkFault(keep_ratio=0.3).apply(llm_pre(original), context())
    assert total_tokens(outcome.value) < total_tokens(original)


def test_shrink_records_how_much_it_removed() -> None:
    """The report must be able to say how much context the agent lost."""
    from agent_loop_chaos.faults.llm import ContextShrinkFault

    outcome = ContextShrinkFault(keep_ratio=0.3).apply(llm_pre(conversation()), context())
    assert outcome.params["messages_removed"] > 0
    assert outcome.params["tokens_removed"] > 0


def test_shrink_never_empties_the_conversation() -> None:
    """The floor is `[system, last_user]`, and reaching it is recorded.

    An empty message list is not a degraded context, it is a broken request — the
    agent would fail for a reason that says nothing about its handling of context
    loss.
    """
    from agent_loop_chaos.faults.llm import ContextShrinkFault

    outcome = ContextShrinkFault(keep_tokens=1).apply(llm_pre(conversation()), context())
    assert len(outcome.value) >= 1
    assert outcome.params["floor_reached"] is True
    assert outcome.value[0]["role"] == "system"


def test_shrink_drop_tool_results_keeps_everything_except_tool_messages() -> None:
    """The strategy that models a context window losing its evidence."""
    from agent_loop_chaos.faults.llm import ContextShrinkFault

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "tool", "content": '{"temp_c": 21}'},
        {"role": "assistant", "content": "a"},
    ]
    outcome = ContextShrinkFault(strategy="drop_tool_results").apply(llm_pre(messages), context())
    assert [m["role"] for m in outcome.value] == ["system", "user", "assistant"]


def test_shrink_drop_system_removes_the_system_message() -> None:
    """The sharpest form of context loss: the instructions themselves go."""
    from agent_loop_chaos.faults.llm import ContextShrinkFault

    outcome = ContextShrinkFault(strategy="drop_system").apply(llm_pre(conversation()), context())
    assert all(m["role"] != "system" for m in outcome.value)


@pytest.mark.parametrize("strategy", ["tail", "head", "middle_out"])
def test_shrink_strategies_all_reduce(strategy: str) -> None:
    """Every documented strategy has to actually do something."""
    from agent_loop_chaos.faults.llm import ContextShrinkFault

    original = conversation()
    outcome = ContextShrinkFault(strategy=strategy, keep_ratio=0.3).apply(
        llm_pre(original), context()
    )
    assert len(outcome.value) < len(original)


def test_shrink_rejects_an_unknown_strategy() -> None:
    """Eager validation with the offending value."""
    from agent_loop_chaos.faults.llm import ContextShrinkFault

    with pytest.raises(ConfigError, match="squeeze"):
        ContextShrinkFault(strategy="squeeze")


def test_shrink_echoes_the_estimator_it_used() -> None:
    """D-43: the report never implies a precision it did not have."""
    from agent_loop_chaos.faults.llm import ContextShrinkFault

    outcome = ContextShrinkFault(token_estimator="chars4").apply(llm_pre(conversation()), context())
    assert outcome.params["token_estimator"] == "chars4"


def test_shrink_end_to_end_reaches_the_model(tmp_path: Path) -> None:
    """4/4 end-to-end, asserted on what the model actually received."""
    from agent_loop_chaos.faults.llm import ContextShrinkFault

    eng = engine(tmp_path)
    eng.register_fault(ContextShrinkFault(keep_ratio=0.3), target_llm="default")
    received: list[Any] = []

    @eng.llm
    def chat(messages: list[dict[str, Any]]) -> str:
        received.append(messages)
        return "ok"

    original = conversation()
    result = eng.run(lambda: chat(original))
    assert received, "the model must have been called"
    assert len(received[0]) < len(original), "the model saw a shortened conversation"
    assert result.injected_faults[0]["fired"] is True


# ============================================================== B2 ContextNoiseFault


def test_noise_inserts_text_into_the_conversation() -> None:
    """1/4 unit."""
    from agent_loop_chaos.faults.llm import ContextNoiseFault

    original = conversation(4)
    outcome = ContextNoiseFault(noise_type="gibberish", tokens=50).apply(
        llm_pre(original), context()
    )
    assert outcome.action == "replace_messages"
    assert len(outcome.value) == len(original) + 1


def test_noise_gibberish_is_identical_across_two_same_seed_runs() -> None:
    """Generated from the seed, not random each call, or goldens would never match."""
    from agent_loop_chaos.faults.llm import ContextNoiseFault

    def noise(seed: int) -> str:
        outcome = ContextNoiseFault(noise_type="gibberish", tokens=60).apply(
            llm_pre(conversation(4)), context(seed)
        )
        return flatten(outcome.value)

    assert noise(5) == noise(5)


def test_noise_gibberish_differs_across_seeds() -> None:
    """Otherwise "generated" would be a fixed string wearing a costume."""
    from agent_loop_chaos.faults.llm import ContextNoiseFault

    def noise(seed: int) -> str:
        outcome = ContextNoiseFault(noise_type="gibberish", tokens=60).apply(
            llm_pre(conversation(4)), context(seed)
        )
        return flatten(outcome.value)

    assert len({noise(s) for s in range(6)}) > 1


@pytest.mark.parametrize(
    "noise_type",
    [
        "gibberish",
        "unrelated_transcript",
        "repeated_block",
        "conflicting_instruction",
        "stale_conversation",
        "html_boilerplate",
    ],
)
def test_every_noise_type_produces_content(noise_type: str) -> None:
    """All six documented types must be reachable."""
    from agent_loop_chaos.faults.llm import ContextNoiseFault

    outcome = ContextNoiseFault(noise_type=noise_type, tokens=80).apply(
        llm_pre(conversation(4)), context()
    )
    inserted = [m for m in outcome.value if m not in conversation(4)]
    assert any(str(m.get("content", "")).strip() for m in inserted)


def test_conflicting_instruction_is_a_plausible_but_wrong_directive() -> None:
    """Catalog B2: it tests instruction-precedence handling, so it must read as an
    instruction rather than as noise."""
    from agent_loop_chaos.faults.llm import ContextNoiseFault

    outcome = ContextNoiseFault(noise_type="conflicting_instruction").apply(
        llm_pre(conversation(4)), context()
    )
    text = flatten(outcome.value).lower()
    assert any(word in text for word in ("respond", "reply", "answer", "always", "only"))


def test_noise_records_the_inserted_text_so_a_probe_can_find_it() -> None:
    """R2: the probe needs to know this text came from the harness, not the agent."""
    from agent_loop_chaos.faults.llm import ContextNoiseFault

    outcome = ContextNoiseFault(noise_type="gibberish", tokens=40).apply(
        llm_pre(conversation(4)), context()
    )
    assert outcome.params["inserted_text"]
    assert outcome.mutation is not None


def test_noise_position_is_honoured() -> None:
    """Where the distraction lands changes what it dilutes."""
    from agent_loop_chaos.faults.llm import ContextNoiseFault

    outcome = ContextNoiseFault(noise_type="gibberish", position="start", tokens=40).apply(
        llm_pre(conversation(4)), context()
    )
    assert outcome.value[0]["role"] != "system", "the noise went first"


def test_noise_end_to_end_reaches_the_model(tmp_path: Path) -> None:
    """4/4 end-to-end, asserted on the received messages."""
    from agent_loop_chaos.faults.llm import ContextNoiseFault

    eng = engine(tmp_path)
    eng.register_fault(
        ContextNoiseFault(noise_type="conflicting_instruction"), target_llm="default"
    )
    received: list[Any] = []

    @eng.llm
    def chat(messages: list[dict[str, Any]]) -> str:
        received.append(messages)
        return "ok"

    eng.run(lambda: chat(conversation(4)))
    assert len(received[0]) == len(conversation(4)) + 1


# ================================================================ B3 GoalDriftFault


def test_goal_drift_replace_rewrites_the_objective() -> None:
    """1/4 unit: the classic multi-step degradation."""
    from agent_loop_chaos.faults.llm import GoalDriftFault

    outcome = GoalDriftFault(mode="replace", replacement="Summarize the weather.").apply(
        llm_pre(conversation(4)), context()
    )
    assert "Summarize the weather." in flatten(outcome.value)


def test_goal_drift_dilute_broadens_the_objective() -> None:
    """`dilute` appends qualifiers rather than replacing the goal outright."""
    from agent_loop_chaos.faults.llm import GoalDriftFault

    original = conversation(4)
    outcome = GoalDriftFault(mode="dilute").apply(llm_pre(original), context())
    text = flatten(outcome.value)
    assert len(text) > len(flatten(original))


def test_goal_drift_paraphrase_weaken_turns_imperatives_into_suggestions() -> None:
    """ "Always cite sources" becoming "you might cite sources" is the whole point."""
    from agent_loop_chaos.faults.llm import GoalDriftFault

    messages = [
        {"role": "system", "content": "Always cite sources. Must include prices."},
        {"role": "user", "content": "go"},
    ]
    outcome = GoalDriftFault(mode="paraphrase_weaken").apply(llm_pre(messages), context())
    weakened = outcome.value[0]["content"].lower()
    assert "always" not in weakened or "might" in weakened
    assert "must" not in weakened or "could" in weakened


def test_goal_drift_drop_constraint_removes_matching_sentences() -> None:
    """A constraint that vanishes is harder to notice than one that changes."""
    from agent_loop_chaos.faults.llm import GoalDriftFault

    messages = [
        {"role": "system", "content": "Plan a trip. Never exceed the budget. Be terse."},
        {"role": "user", "content": "go"},
    ]
    outcome = GoalDriftFault(mode="drop_constraint", constraint_pattern="budget").apply(
        llm_pre(messages), context()
    )
    assert "budget" not in outcome.value[0]["content"]
    assert "Plan a trip." in outcome.value[0]["content"]


def test_goal_drift_is_a_pure_string_transform_with_no_model_call() -> None:
    """Deterministic by construction: a model call here would make runs unreproducible."""
    from agent_loop_chaos.faults.llm import GoalDriftFault

    first = GoalDriftFault(mode="dilute").apply(llm_pre(conversation(4)), context(1))
    second = GoalDriftFault(mode="dilute").apply(llm_pre(conversation(4)), context(2))
    assert first.value == second.value, "no seed dependence, so no model in the loop"


def test_goal_drift_records_only_the_affected_message() -> None:
    """A diff of the whole conversation would bury the one change that matters."""
    from agent_loop_chaos.faults.llm import GoalDriftFault

    outcome = GoalDriftFault(mode="replace", replacement="Do something else.").apply(
        llm_pre(conversation(4)), context()
    )
    log = outcome.mutation
    assert log is not None
    assert isinstance(log.payload_before, dict), "the affected message, not the list"
    assert log.payload_before.get("role")


def test_goal_drift_rejects_an_unknown_mode() -> None:
    """Eager validation."""
    from agent_loop_chaos.faults.llm import GoalDriftFault

    with pytest.raises(ConfigError, match="wobble"):
        GoalDriftFault(mode="wobble")


def llm_post(result: Any, *, call_index: int = 1) -> Crossing:
    """Build an `(llm, post)` crossing carrying the model's response.

    Args:
        result: What the model returned.
        call_index: Which LLM call this is.

    Returns:
        The crossing.
    """
    return Crossing(layer="llm", phase="post", name="default", result=result, call_index=call_index)


# ======================================================= B4 LLMMalformedOutputFault


@pytest.mark.parametrize(
    "mode",
    [
        "prose_instead_of_json",
        "trailing_commentary",
        "markdown_fenced",
        "broken_json",
        "wrong_enum_value",
        "double_encoded",
    ],
)
def test_malformed_output_string_modes_return_a_string(mode: str) -> None:
    """Catalog B4: every mode returns a string body except the three schema modes.

    Returning a dict would hand the agent a parsed object it never had to parse,
    which is the exact step this fault exists to test.
    """
    from agent_loop_chaos.faults.llm import LLMMalformedOutputFault

    outcome = LLMMalformedOutputFault(mode=mode).apply(
        llm_post('{"city": "Paris", "temp_c": 21}'), context()
    )
    assert isinstance(outcome.value, str)


def test_broken_json_does_not_parse() -> None:
    """A "broken" body that still parses would prove nothing."""
    from agent_loop_chaos.faults.llm import LLMMalformedOutputFault

    outcome = LLMMalformedOutputFault(mode="broken_json").apply(
        llm_post('{"city": "Paris"}'), context()
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(outcome.value)


def test_markdown_fenced_wraps_valid_json_in_a_code_fence() -> None:
    """The commonest real failure: correct JSON the agent cannot `json.loads`."""
    from agent_loop_chaos.faults.llm import LLMMalformedOutputFault

    outcome = LLMMalformedOutputFault(mode="markdown_fenced").apply(
        llm_post('{"city": "Paris"}'), context()
    )
    assert outcome.value.startswith("```")
    with pytest.raises(json.JSONDecodeError):
        json.loads(outcome.value)


def test_prose_instead_of_json_contains_no_json_at_all() -> None:
    """The model answered in English when a contract promised JSON."""
    from agent_loop_chaos.faults.llm import LLMMalformedOutputFault

    outcome = LLMMalformedOutputFault(mode="prose_instead_of_json").apply(
        llm_post('{"city": "Paris"}'), context()
    )
    assert "{" not in outcome.value


def test_extra_fields_breaks_the_observed_response_rather_than_inventing_one() -> None:
    """Catalog B4: parse the real response, then break it.

    Deriving from what the model actually returned keeps the fault realistic — a
    synthetic object would not resemble the agent's own contract.
    """
    from agent_loop_chaos.faults.llm import LLMMalformedOutputFault

    outcome = LLMMalformedOutputFault(mode="extra_fields").apply(
        llm_post('{"city": "Paris", "temp_c": 21}'), context()
    )
    parsed = json.loads(outcome.value)
    assert "city" in parsed
    assert len(parsed) > 2


def test_missing_required_removes_a_field_the_response_had() -> None:
    """The schema-shaped counterpart to `drop_key`."""
    from agent_loop_chaos.faults.llm import LLMMalformedOutputFault

    outcome = LLMMalformedOutputFault(mode="missing_required").apply(
        llm_post('{"city": "Paris", "temp_c": 21}'), context()
    )
    assert len(json.loads(outcome.value)) < 2


def test_wrong_schema_uses_a_declared_schema_when_one_is_given() -> None:
    """A declared schema makes the violation precise rather than approximate."""
    from agent_loop_chaos.faults.llm import LLMMalformedOutputFault

    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    outcome = LLMMalformedOutputFault(mode="wrong_schema", schema=schema).apply(
        llm_post('{"city": "Paris"}'), context()
    )
    assert "summary" not in json.loads(outcome.value)


def test_malformed_output_rejects_an_unknown_mode() -> None:
    """Eager validation."""
    from agent_loop_chaos.faults.llm import LLMMalformedOutputFault

    with pytest.raises(ConfigError, match="scrambled"):
        LLMMalformedOutputFault(mode="scrambled")


def test_malformed_output_end_to_end_reaches_the_agent(tmp_path: Path) -> None:
    """4/4 end-to-end: the agent receives the broken body, not the real one."""
    from agent_loop_chaos.faults.llm import LLMMalformedOutputFault

    eng = engine(tmp_path)
    eng.register_fault(LLMMalformedOutputFault(mode="markdown_fenced"), target_llm="default")

    @eng.llm
    def chat(prompt: str) -> str:
        return '{"ok": true}'

    def agent() -> str:
        raw = chat("go")
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            return "detected a malformed body"
        return "parsed fine"

    assert eng.run(agent).final_output == "detected a malformed body"


# =============================================================== B5 LLMRefusalFault


@pytest.mark.parametrize("style", ["policy", "capability", "clarifying_question"])
def test_refusal_produces_a_non_answer_in_each_style(style: str) -> None:
    """Three shapes of "no", because agents mishandle them differently."""
    from agent_loop_chaos.faults.llm import LLMRefusalFault

    outcome = LLMRefusalFault(style=style).apply(llm_post("the real answer"), context())
    assert isinstance(outcome.value, str)
    assert outcome.value != "the real answer"
    assert len(outcome.value) > 10


def test_clarifying_question_actually_asks_something() -> None:
    """This is the style that sends a naive agent into a re-ask loop."""
    from agent_loop_chaos.faults.llm import LLMRefusalFault

    outcome = LLMRefusalFault(style="clarifying_question").apply(llm_post("x"), context())
    assert outcome.value.rstrip().endswith("?")


def test_refusal_honours_explicit_text() -> None:
    """A scenario can supply the exact wording it wants to test against."""
    from agent_loop_chaos.faults.llm import LLMRefusalFault

    outcome = LLMRefusalFault(text="I can't help with that.").apply(llm_post("x"), context())
    assert outcome.value == "I can't help with that."


# ================================================================= B6 LLMEmptyFault


@pytest.mark.parametrize(
    ("mode", "check"),
    [
        ("empty_string", lambda v: v == ""),
        ("whitespace", lambda v: isinstance(v, str) and v.strip() == "" and v != ""),
        ("null", lambda v: v is None),
        ("empty_tool_calls", lambda v: isinstance(v, dict) and v.get("tool_calls") == []),
    ],
)
def test_empty_modes(mode: str, check: Any) -> None:
    """Four flavours of nothing, each of which breaks a different guard."""
    from agent_loop_chaos.faults.llm import LLMEmptyFault

    outcome = LLMEmptyFault(mode=mode).apply(llm_post("real answer"), context())
    assert check(outcome.value), f"{mode} produced {outcome.value!r}"


def test_empty_end_to_end_is_visible_to_the_agent(tmp_path: Path) -> None:
    """4/4 end-to-end."""
    from agent_loop_chaos.faults.llm import LLMEmptyFault

    eng = engine(tmp_path)
    eng.register_fault(LLMEmptyFault(mode="empty_string"), target_llm="default")

    @eng.llm
    def chat(prompt: str) -> str:
        return "a real answer"

    def agent() -> str:
        return chat("go") or "guarded against an empty response"

    assert eng.run(agent).final_output == "guarded against an empty response"


# ============================================================ B7 LLMTruncationFault


def test_truncation_cuts_the_response() -> None:
    """1/4 unit: simulates hitting `max_tokens`."""
    from agent_loop_chaos.faults.llm import LLMTruncationFault

    body = "sentence one. sentence two. sentence three. sentence four."
    outcome = LLMTruncationFault(at_ratio=0.5, cut="chars").apply(llm_post(body), context())
    assert len(outcome.value) < len(body)
    assert body.startswith(outcome.value)


def test_mid_json_leaves_the_body_genuinely_unparseable() -> None:
    """The high-value case, and the one easiest to get subtly wrong.

    A cut at a structurally safe point would still parse, and the fault would prove
    nothing about `finish_reason` being ignored.
    """
    from agent_loop_chaos.faults.llm import LLMTruncationFault

    body = json.dumps({"days": [{"d": 1, "t": 21}, {"d": 2, "t": 22}], "note": "sunny all week"})
    outcome = LLMTruncationFault(cut="mid_json").apply(llm_post(body), context())
    with pytest.raises(json.JSONDecodeError):
        json.loads(outcome.value)


def test_truncation_sets_the_finish_reason() -> None:
    """An agent that checks `finish_reason` should survive this; it must be set."""
    from agent_loop_chaos.faults.llm import LLMTruncationFault

    outcome = LLMTruncationFault().apply(llm_post("a" * 200), context())
    assert outcome.params["finish_reason"] == "length"


def test_mid_sentence_cuts_inside_a_word() -> None:
    """A cut at a word boundary reads as a short answer rather than a broken one."""
    from agent_loop_chaos.faults.llm import LLMTruncationFault

    body = "The forecast for Paris is warm and settled throughout the week ahead."
    outcome = LLMTruncationFault(at_ratio=0.5, cut="mid_sentence").apply(llm_post(body), context())
    assert not outcome.value.endswith(" ")
    assert body.startswith(outcome.value)


# ========================================================= B8 MalformedToolCallFault


def registry_context() -> FaultContext:
    """A context whose tool registry holds a realistic inventory.

    Returns:
        The context.
    """
    from agent_loop_chaos.context import ToolInfo

    ctx = context()
    ctx.tool_registry = {
        "get_weather_data": ToolInfo(name="get_weather_data", side_effecting=False),
        "search_flights": ToolInfo(name="search_flights", side_effecting=False),
    }
    return ctx


def test_unknown_tool_invents_a_plausible_neighbour() -> None:
    """Catalog B8: a random string is an easier case than the real one.

    `get_weather_forecast` next to `get_weather_data` is what a model actually does,
    and it is the case a naive dispatcher fails on.
    """
    from agent_loop_chaos.faults.llm import MalformedToolCallFault

    outcome = MalformedToolCallFault(mode="unknown_tool").apply(
        llm_post({"tool_calls": [{"id": "c1", "name": "get_weather_data", "arguments": {}}]}),
        registry_context(),
    )
    name = outcome.value["tool_calls"][0]["name"]
    assert name not in {"get_weather_data", "search_flights"}
    assert "weather" in name or "flight" in name, "it must be plausible, not random"


@pytest.mark.parametrize(
    "mode",
    [
        "missing_arg",
        "extra_arg",
        "wrong_type",
        "duplicate_call_id",
        "two_calls_same_tool",
        "args_as_string",
        "null_args",
    ],
)
def test_every_malformed_tool_call_mode_changes_the_call(mode: str) -> None:
    """All eight documented modes must be reachable and must alter the call."""
    from agent_loop_chaos.faults.llm import MalformedToolCallFault

    original = {
        "tool_calls": [{"id": "c1", "name": "get_weather_data", "arguments": {"location": "Paris"}}]
    }
    outcome = MalformedToolCallFault(mode=mode).apply(llm_post(original), registry_context())
    assert outcome.value != original


def test_args_as_string_hands_the_dispatcher_a_string() -> None:
    """The dispatcher expected a mapping; a model handed it JSON text."""
    from agent_loop_chaos.faults.llm import MalformedToolCallFault

    outcome = MalformedToolCallFault(mode="args_as_string").apply(
        llm_post({"tool_calls": [{"id": "c1", "name": "get_weather_data", "arguments": {"a": 1}}]}),
        registry_context(),
    )
    assert isinstance(outcome.value["tool_calls"][0]["arguments"], str)


def test_two_calls_same_tool_duplicates_the_call() -> None:
    """A dispatcher without dedup runs the same work twice."""
    from agent_loop_chaos.faults.llm import MalformedToolCallFault

    outcome = MalformedToolCallFault(mode="two_calls_same_tool").apply(
        llm_post({"tool_calls": [{"id": "c1", "name": "get_weather_data", "arguments": {}}]}),
        registry_context(),
    )
    assert len(outcome.value["tool_calls"]) == 2


def test_malformed_tool_call_is_a_noop_without_tool_calls() -> None:
    """A plain text response has no call to malform."""
    from agent_loop_chaos.faults.llm import MalformedToolCallFault

    outcome = MalformedToolCallFault().apply(llm_post("just text"), registry_context())
    assert outcome.action == "noop"


# ========================================================= B9 HallucinationSeedFault


def test_contradict_tool_output_reads_the_real_tool_result() -> None:
    """Catalog B9: this is what makes the fabricated-value probe's job real.

    The forced answer contradicts a value the tool actually returned, so a
    verification step comparing claims to evidence has something to catch.
    """
    from agent_loop_chaos.faults.llm import HallucinationSeedFault

    ctx = context()
    ctx.history["get_weather_data"] = [{"temp_c": 21, "city": "Paris"}]
    outcome = HallucinationSeedFault(mode="contradict_tool_output").apply(
        llm_post("It is 21C in Paris."), ctx
    )
    assert "21" not in str(outcome.value)
    assert outcome.params["contradicted_field"] == "temp_c"
    assert outcome.params["evidence_value"] == 21


def test_contradict_tool_output_falls_back_without_history() -> None:
    """With no tool evidence there is nothing to contradict; say so rather than guess."""
    from agent_loop_chaos.faults.llm import HallucinationSeedFault

    outcome = HallucinationSeedFault(mode="contradict_tool_output").apply(
        llm_post("an answer"), context()
    )
    assert outcome.params.get("fell_back") is True


@pytest.mark.parametrize(
    "mode", ["invent_value", "invent_citation", "invent_tool", "confident_wrong_number"]
)
def test_every_hallucination_mode_produces_a_confident_wrong_answer(mode: str) -> None:
    """All five modes must be reachable and must replace the answer."""
    from agent_loop_chaos.faults.llm import HallucinationSeedFault

    outcome = HallucinationSeedFault(mode=mode).apply(llm_post("real answer"), context())
    assert outcome.action == "replace_result"
    assert outcome.value != "real answer"


def test_invent_citation_looks_like_a_citation() -> None:
    """A fabricated source is only a useful test if it reads like a real one."""
    from agent_loop_chaos.faults.llm import HallucinationSeedFault

    outcome = HallucinationSeedFault(mode="invent_citation").apply(llm_post("x"), context())
    text = str(outcome.value).lower()
    assert any(token in text for token in ("http", "et al", "report", "20"))
