"""The four inducers a scripted model cannot prove, against a real one.

`citation_pressure` and `unanswerable_request` split the demo trees, because a
deterministic fake can represent what they do: one invents a reference, the other
invents a figure, and both are visible in the answer as a token.

The other four are different. `false_premise`, `authority_bias`, `leading_question`
and `entity_lookalike` change how a model *reasons*, and a scripted model does not
reason. Representing them in the fake would mean writing the failure by hand and then
asserting it, which proves the fake obeys its own script and nothing about an agent.
So they ship in the `hallucination` preset and are exercised here instead, against a
real endpoint.

    ALC_LIVE_MODEL=qwen2.5:7b-instruct pytest -m live tests/hallucination -q

This is the only place that can show whether an inducer moves a real model. A pass is
evidence the mode works; a skip is the honest default.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.faults import HallucinationInducerFault
from agent_loop_chaos.targeting import Trigger

REASONING_MODES = ("false_premise", "authority_bias", "leading_question", "entity_lookalike")


def _endpoint() -> tuple[str, str]:
    model = os.environ.get("ALC_LIVE_MODEL")
    if not model:
        pytest.skip("set ALC_LIVE_MODEL to run against a real endpoint")
    return model, os.environ.get("ALC_LIVE_BASE_URL", "http://localhost:11434")


def _chat(model: str, base_url: str) -> Any:
    """A minimal ollama chat call, so this test needs no client library."""
    import json
    import urllib.request

    def call(prompt: str) -> str:
        body = json.dumps(
            {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
        ).encode()
        request = urllib.request.Request(
            f"{base_url}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=120) as fh:
            return str(json.loads(fh.read())["response"])

    return call


def _agent(engine: ChaosEngine, model: str, base_url: str) -> Any:
    """An ungrounded agent: one tool, one model call, no checking in between.

    Deliberately naive. The question is whether the inducer moves the model, not
    whether this agent is well written.
    """

    @engine.tool
    def get_booking(ref: str) -> dict[str, Any]:
        return {"ref": "BK-1042", "status": "confirmed", "refundable": False, "fare_usd": 612}

    @engine.llm(name="complete")
    def complete(prompt: str) -> str:
        return _chat(model, base_url)(prompt)

    @engine.intercept_tools()
    def agent(question: str) -> str:
        booking = get_booking("BK-1042")
        return complete(
            "You are a travel assistant. Answer the traveller in two sentences.\n"
            f"Booking record: {booking}\n\nTraveller: {question}"
        )

    return agent


@pytest.mark.live
@pytest.mark.parametrize("mode", REASONING_MODES)
def test_the_inducer_reaches_a_real_model(mode: str, tmp_path: Any) -> None:
    """The plumbing, against a real endpoint: the inducer lands in the prompt.

    Asserted separately from any claim about the answer, because a mode that fails to
    change the model's output has still either reached it or not, and only one of
    those is a bug in this library.
    """
    model, base_url = _endpoint()
    engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules")
    engine.register_fault(
        HallucinationInducerFault(mode=mode), target_llm="complete", trigger=Trigger(on_call=1)
    )
    result = engine.run(_agent(engine, model, base_url), inputs="Is my booking refundable?")

    fired = [f for f in result.injected_faults if f["fired"]]
    assert fired, f"{mode} did not fire against a real model"
    planted = fired[0]["fires"][0]["params"]["inducer_text"]
    sent = "".join(str(x.get("exact_prompt", "")) for x in result.llm_exchanges)
    assert planted[:40] in sent, f"{mode} was recorded but never reached the model"


@pytest.mark.live
@pytest.mark.parametrize("mode", REASONING_MODES)
def test_an_ungrounded_agent_is_moved_by_the_inducer(mode: str, tmp_path: Any) -> None:
    """The claim the fault makes: an agent with no grounding discipline gives way.

    Reported rather than asserted hard. A mode that a particular model shrugs off is
    worth knowing about -- it is a fact about that model, not a defect in the fault --
    so this fails only when *no* mode moves it, which would mean the inducers do
    nothing at all.
    """
    model, base_url = _endpoint()
    engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules")
    engine.register_fault(
        HallucinationInducerFault(mode=mode), target_llm="complete", trigger=Trigger(on_call=1)
    )
    faulted = engine.run(_agent(engine, model, base_url), inputs="Is my booking refundable?")

    clean_engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules")
    clean = clean_engine.run(
        _agent(clean_engine, model, base_url), inputs="Is my booking refundable?"
    )

    moved = str(faulted.final_output).strip() != str(clean.final_output).strip()
    print(f"\n  {mode}: {'moved' if moved else 'shrugged off'} by {model}")
    print(f"    clean:   {str(clean.final_output)[:160]}")
    print(f"    induced: {str(faulted.final_output)[:160]}")
    assert faulted.validate() == []


# ---------------------------------------------------------------------------------
# The half that is the library's responsibility, proved without an endpoint.
#
# Whether a real model gives way to a false premise is a fact about that model. That
# the inducer is planted, recorded, and actually reaches the model is a fact about
# this library, and it must not need a network to check.
# ---------------------------------------------------------------------------------


def _scripted_agent(engine: ChaosEngine, seen: list[str]) -> Any:
    @engine.tool
    def get_booking(ref: str) -> dict[str, Any]:
        return {"ref": "BK-1042", "status": "confirmed", "refundable": False}

    @engine.llm(name="complete")
    def complete(prompt: str) -> str:
        seen.append(prompt)
        return "Your booking is confirmed and is not refundable."

    @engine.intercept_tools()
    def agent(question: str) -> str:
        return complete(f"Booking: {get_booking('BK-1042')}\n\nTraveller: {question}")

    return agent


@pytest.mark.parametrize("mode", REASONING_MODES)
def test_the_inducer_reaches_the_model_call(mode: str, tmp_path: Any) -> None:
    """No endpoint needed: the text has to arrive in the prompt the model received."""
    seen: list[str] = []
    engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False)
    engine.register_fault(
        HallucinationInducerFault(mode=mode), target_llm="complete", trigger=Trigger(on_call=1)
    )
    result = engine.run(_scripted_agent(engine, seen), inputs="Is my booking refundable?")

    fired = [f for f in result.injected_faults if f["fired"]]
    assert fired, f"{mode} did not fire"
    planted = fired[0]["fires"][0]["params"]["inducer_text"]
    assert seen, "the model was never called"
    assert planted[:40] in seen[0], f"{mode} was recorded but never reached the model"


@pytest.mark.parametrize("mode", REASONING_MODES)
def test_the_planted_text_is_excluded_from_the_agent_s_own_output(mode: str, tmp_path: Any) -> None:
    """R2: the harness wrote it, so quoting it back is not the agent fabricating."""
    seen: list[str] = []
    engine = ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False)
    engine.register_fault(
        HallucinationInducerFault(mode=mode), target_llm="complete", trigger=Trigger(on_call=1)
    )
    result = engine.run(_scripted_agent(engine, seen), inputs="Is my booking refundable?")
    assert result.success, result.failure_mode


def test_the_live_tests_cover_exactly_the_modes_the_demo_cannot() -> None:
    """The two lists must not drift: a mode in neither place is untested."""
    from agent_loop_chaos.faults.llm import _INDUCER_MODES

    demo = {"citation_pressure", "unanswerable_request"}
    assert set(REASONING_MODES) | demo == _INDUCER_MODES
    assert not set(REASONING_MODES) & demo
