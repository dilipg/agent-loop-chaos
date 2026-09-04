"""`MalformedToolCallFault` against the OpenAI wire shape.

The fault read `call["name"]` and `call["arguments"]` at the top level of each entry
in `tool_calls`. The OpenAI shape nests them:

    {"id": "call_1", "type": "function",
     "function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}}

So every mode wrote a stray top-level key beside `function`, and a dispatcher reading
`call["function"]["name"]` never saw any of it. The fault reported that it had fired
and changed nothing the agent reads -- the worst kind of no-op, because it reads as
coverage in `suite.json` rather than as an error.

Found by a subagent building `examples/patterns/function_calling`, which had to
declare a different fault to get a real finding.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_loop_chaos.context import Crossing, FaultContext
from agent_loop_chaos.faults import MalformedToolCallFault

MODES = ["unknown_tool", "missing_arg", "wrong_type", "null_args", "args_as_string"]


def _openai_response(arguments: str = '{"city": "Paris"}') -> dict[str, Any]:
    """A response in the shape OpenAI, Groq, vLLM and LM Studio all emit."""
    return {
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": arguments},
            }
        ],
    }


def _legacy_response() -> dict[str, Any]:
    """The flat shape, which some clients and every older example use."""
    return {
        "content": None,
        "tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}],
    }


def _apply(mode: str, result: Any) -> Any:
    """Run the fault over a response and return what it produced."""
    from agent_loop_chaos.context import Counters, Limits, RunContext
    from agent_loop_chaos.trace import TraceRecorder

    run = RunContext(
        run_id="run-tc",
        seed=1337,
        started_at="1970-01-01T00:00:00Z",
        limits=Limits(),
        trace=TraceRecorder("run-tc"),
        counters=Counters(),
    )
    ctx = FaultContext(
        fault_id="f1",
        fault_key="tc0001",
        run=run,
        limits=run.limits,
        counters=run.counters,
        tool_registry={"get_weather": None, "search_flights": None},
    )
    crossing = Crossing(layer="llm", phase="post", name="default", result=result)
    return MalformedToolCallFault(mode=mode).apply(crossing, ctx)


class TestTheNestedShapeIsReached:
    @pytest.mark.parametrize("mode", MODES)
    def test_the_call_a_dispatcher_reads_actually_changes(self, mode: str) -> None:
        before = _openai_response()
        outcome = _apply(mode, before)
        assert outcome.action == "replace_result", f"{mode} produced {outcome.action}"
        call = outcome.value["tool_calls"][0]
        assert "function" in call, f"{mode} dropped the nested shape"
        # What a dispatcher reads is `call["function"]`. If that is untouched, the
        # fault changed nothing the agent can see.
        assert call["function"] != before["tool_calls"][0]["function"], (
            f"{mode} left call['function'] identical -- a silent no-op"
        )

    @pytest.mark.parametrize("mode", MODES)
    def test_no_stray_top_level_keys_are_written(self, mode: str) -> None:
        """A top-level `name` beside `function` is not a shape any client emits."""
        call = _apply(mode, _openai_response()).value["tool_calls"][0]
        assert "name" not in call, f"{mode} wrote a stray top-level name"
        assert "arguments" not in call, f"{mode} wrote stray top-level arguments"

    def test_unknown_tool_renames_the_nested_function(self) -> None:
        call = _apply("unknown_tool", _openai_response()).value["tool_calls"][0]
        assert call["function"]["name"] != "get_weather"

    def test_missing_arg_drops_a_key_from_the_encoded_arguments(self) -> None:
        response = _openai_response('{"city": "Paris", "units": "c"}')
        call = _apply("missing_arg", response).value["tool_calls"][0]
        # `function.arguments` is a JSON *string* on the wire, so the fault has to
        # decode, mutate and re-encode rather than writing a dict into it.
        assert isinstance(call["function"]["arguments"], str)
        assert len(json.loads(call["function"]["arguments"])) == 1

    def test_null_args_is_representable_on_the_wire(self) -> None:
        call = _apply("null_args", _openai_response()).value["tool_calls"][0]
        assert call["function"]["arguments"] in (None, "null", ""), call["function"]

    def test_args_as_string_becomes_args_as_object_when_already_a_string(self) -> None:
        """On the wire `arguments` is already a string; re-stringifying changes nothing.

        The equivalent break is to hand back an *object* where the contract says
        string -- which is what a client's `json.loads` will choke on.
        """
        call = _apply("args_as_string", _openai_response()).value["tool_calls"][0]
        assert not isinstance(call["function"]["arguments"], str)

    def test_the_mutation_log_records_a_real_patch(self) -> None:
        outcome = _apply("unknown_tool", _openai_response())
        assert outcome.mutation is not None
        assert outcome.mutation.json_patch, "no patch means nothing to show in the report"


class TestTheFlatShapeStillWorks:
    """The older flat shape must keep working; this is an addition, not a swap."""

    @pytest.mark.parametrize("mode", MODES)
    def test_the_flat_call_still_changes(self, mode: str) -> None:
        before = _legacy_response()
        call = _apply(mode, before).value["tool_calls"][0]
        assert call != before["tool_calls"][0], f"{mode} no longer touches the flat shape"

    def test_unknown_tool_renames_the_flat_name(self) -> None:
        call = _apply("unknown_tool", _legacy_response()).value["tool_calls"][0]
        assert call["name"] != "get_weather"


class TestNoCallToMalform:
    def test_it_noops_when_the_response_requested_nothing(self) -> None:
        outcome = _apply("unknown_tool", {"content": "just prose"})
        assert outcome.action == "noop"
