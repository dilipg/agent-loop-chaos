"""`truncated_output_used` has to be reachable at all.

The probe gates on `finish_reason == "length"` in the `llm_response` event's payload.
The engine wrote that payload as `{"result": result}` and nothing else, so the key was
never present and the probe could never fire -- not from `LLMTruncationFault`, whose
`set_finish_reason` param went into the fire record instead, and not from a **real**
model whose response genuinely was cut off.

An agent that consumed a `cut="mid_sentence"` body without checking anything scored
`success=True, failure_mode=graceful_degradation, symptoms=[]`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import LLMTruncationFault
from agent_loop_chaos.targeting import Target, Trigger


def _agent(engine: ChaosEngine, reply: Any) -> Any:
    """An agent that consumes whatever the model returns, unchecked."""

    def model(prompt: Any) -> Any:
        return reply

    chat = engine.llm(model, name="default")

    def run(question: Any = None) -> str:
        return f"The summary is: {chat('summarise the trip')}"

    return run


class TestTheEngineRecordsFinishReason:
    def test_a_real_truncated_response_is_visible_in_the_trace(self, tmp_path: Path) -> None:
        """No fault involved: a real model said it ran out of room."""
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        result = engine.run(
            _agent(engine, {"content": "The trip to Par", "finish_reason": "length"}),
            inputs={"question": "?"},
            scenario_id="real-truncation",
            expected_behavior="ignore_and_continue",
        )
        events = [
            json.loads(line) for line in Path(result.artifacts["trace"]).read_text().splitlines()
        ]
        responses = [e for e in events if e["kind"] == "llm_response"]
        assert responses, "no llm_response event at all"
        assert any((e.get("payload") or {}).get("finish_reason") == "length" for e in responses), (
            "the engine dropped finish_reason, so the probe can never see it"
        )

    def test_an_ordinary_response_records_stop(self, tmp_path: Path) -> None:
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        result = engine.run(
            _agent(engine, {"content": "A complete answer.", "finish_reason": "stop"}),
            inputs={"question": "?"},
            scenario_id="complete",
            expected_behavior="ignore_and_continue",
        )
        events = [
            json.loads(line) for line in Path(result.artifacts["trace"]).read_text().splitlines()
        ]
        reasons = {
            (e.get("payload") or {}).get("finish_reason")
            for e in events
            if e["kind"] == "llm_response"
        }
        assert reasons == {"stop"}

    def test_a_bare_string_reply_records_nothing(self, tmp_path: Path) -> None:
        """Most agents' models return a string. Inventing a reason would be a lie."""
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        result = engine.run(
            _agent(engine, "a plain string reply"),
            inputs={"question": "?"},
            scenario_id="plain",
            expected_behavior="ignore_and_continue",
        )
        events = [
            json.loads(line) for line in Path(result.artifacts["trace"]).read_text().splitlines()
        ]
        for event in events:
            if event["kind"] == "llm_response":
                assert "finish_reason" not in (event.get("payload") or {})


class TestTheProbeFires:
    @staticmethod
    def _run(tmp_path: Path, **fault_params: Any) -> Any:
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        engine.register_fault(
            LLMTruncationFault(cut="mid_sentence", at_ratio=0.4, **fault_params),
            target=Target(llm="default"),
            trigger=Trigger(on_call=1),
        )
        return engine.run(
            _agent(
                engine, {"content": "The trip to Paris is warm and dry.", "finish_reason": "stop"}
            ),
            inputs={"question": "?"},
            scenario_id="truncation",
            expected_behavior="graceful_degradation",
        )

    def test_consuming_a_truncated_body_is_caught(self, tmp_path: Path) -> None:
        result = self._run(tmp_path)
        codes = [s["code"] for s in result.symptoms]
        assert "truncated_output_used" in codes, (
            f"the probe still cannot fire; symptoms were {codes}"
        )

    def test_the_fault_sets_the_finish_reason_it_recorded(self, tmp_path: Path) -> None:
        """The fault's `set_finish_reason` has to reach the trace, not just the report."""
        result = self._run(tmp_path)
        events = [
            json.loads(line) for line in Path(result.artifacts["trace"]).read_text().splitlines()
        ]
        assert any(
            (e.get("payload") or {}).get("finish_reason") == "length"
            for e in events
            if e["kind"] == "llm_response"
        )

    def test_it_classifies_as_truncated_output_used(self, tmp_path: Path) -> None:
        assert self._run(tmp_path).failure_mode == "truncated_output_used"

    def test_an_agent_that_re_requests_is_not_caught(self, tmp_path: Path) -> None:
        """A continuation request is the graceful path; firing on it is a false positive."""
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", strict_schema=False)
        engine.register_fault(
            LLMTruncationFault(cut="mid_sentence", at_ratio=0.4),
            target=Target(llm="default"),
            trigger=Trigger(on_call=1),
        )

        def model(prompt: Any) -> Any:
            return {"content": "The trip to Paris is warm and dry.", "finish_reason": "stop"}

        chat = engine.llm(model, name="default")

        def careful(question: Any = None) -> str:
            first = chat("summarise the trip")
            body = first.get("content", "") if isinstance(first, dict) else str(first)
            if not body.rstrip().endswith("."):
                # Cut short: ask again rather than using half an answer.
                first = chat("summarise the trip, briefly")
            return f"The summary is: {first}"

        result = engine.run(
            careful,
            inputs={"question": "?"},
            scenario_id="re-request",
            expected_behavior="graceful_degradation",
        )
        assert "truncated_output_used" not in [s["code"] for s in result.symptoms]
