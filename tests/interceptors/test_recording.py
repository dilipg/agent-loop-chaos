"""Record a run once, replay it offline forever.

This is the chosen answer to "how does a colleague supply fake data": they do not write
any. They point the library at their app, it records what the app really did, and every
later run replays that with faults injected on top. The recording is the fixture.

The property that matters is the last test in each class: with a cassette, the real call
never happens again, so the suite runs with no credential and no network -- and the
faults still bite, because interception sits outside the cassette rather than inside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine, Target
from agent_loop_chaos.cassettes import Cassette
from agent_loop_chaos.faults import LLMTruncationFault, ToolCorruptionFault
from tests.fakes import homegrown

httpx = pytest.importorskip("httpx")


def _engine(tmp_path: Path, **kw: Any) -> ChaosEngine:
    return ChaosEngine(seed=1337, out_dir=tmp_path, judge="rules", write_bundle=False, **kw)


class TestRecordingASeam:
    def test_the_real_call_happens_once_and_then_never_again(self, tmp_path: Path) -> None:
        calls: list[str] = []
        original = homegrown.fetch_weather

        def counted(city: str) -> dict[str, Any]:
            calls.append(city)
            return original(city)

        homegrown.fetch_weather = counted  # type: ignore[assignment]
        try:
            tape = tmp_path / "c.json"
            record = Cassette(tape, mode="record")
            engine = _engine(
                tmp_path,
                seams={"tools": ["tests.fakes.homegrown:fetch_weather"]},
                cassette=record,
            )
            engine.run(lambda q: homegrown.fetch_weather("Paris"), inputs="q")
            record.save()
            assert calls == ["Paris"], "the recording run did not reach the real call"

            replayed: list[Any] = []
            engine2 = _engine(
                tmp_path,
                seams={"tools": ["tests.fakes.homegrown:fetch_weather"]},
                cassette=Cassette(tape, mode="replay"),
            )
            engine2.run(lambda q: replayed.append(homegrown.fetch_weather("Paris")), inputs="q")

            assert calls == ["Paris"], "replay reached the real call again"
            assert replayed == [{"temp_c": 21, "city": "Paris"}]
        finally:
            homegrown.fetch_weather = original  # type: ignore[assignment]

    def test_a_fault_still_bites_on_replay(self, tmp_path: Path) -> None:
        """Interception sits outside the cassette, so a replayed payload is faultable.
        If it sat inside, a recorded run could never be tested -- which would make the
        whole feature pointless."""
        tape = tmp_path / "c.json"
        record = Cassette(tape, mode="record")
        seams = {"tools": ["tests.fakes.homegrown:fetch_weather"]}
        _engine(tmp_path, seams=seams, cassette=record).run(
            lambda q: homegrown.fetch_weather("Paris"), inputs="q"
        )
        record.save()

        engine = _engine(tmp_path, seams=seams, cassette=Cassette(tape, mode="replay"))
        engine.register_fault(
            ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
            target=Target(layer="tool", phase="post"),
        )
        seen: list[Any] = []
        result = engine.run(lambda q: seen.append(homegrown.fetch_weather("Paris")), inputs="q")

        assert result.injected_faults[0]["fired"] is True
        assert seen and "temp_c" not in seen[0], "the replayed payload was not faulted"


class TestRecordingAModelOverHttp:
    def _server(self, calls: list[int]) -> Any:
        def handler(request: Any) -> Any:
            calls.append(1)
            return httpx.Response(
                200,
                json={
                    "model": "gpt-4o",
                    "choices": [{"message": {"role": "assistant", "content": "Pack layers."}}],
                },
            )

        return httpx.MockTransport(handler)

    def _agent(self, calls: list[int], seen: list[str]) -> Any:
        def agent(question: str) -> str:
            with httpx.Client(transport=self._server(calls)) as client:
                reply = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": question}],
                    },
                )
            seen.append(str(reply.json()["choices"][0]["message"]["content"]))
            return seen[-1]

        return agent

    def test_the_endpoint_is_reached_once_and_then_replayed(self, tmp_path: Path) -> None:
        calls: list[int] = []
        seen: list[str] = []
        tape = tmp_path / "c.json"

        record = Cassette(tape, mode="record")
        _engine(tmp_path, intercept=True, cassette=record).run(
            self._agent(calls, seen), inputs="what should I pack?"
        )
        record.save()
        assert calls, "the recording run never reached the endpoint"
        before = len(calls)

        engine = _engine(tmp_path, intercept=True, cassette=Cassette(tape, mode="replay"))
        engine.register_fault(LLMTruncationFault(), target=Target(layer="llm", phase="post"))
        engine.run(self._agent(calls, seen), inputs="what should I pack?")

        assert len(calls) == before, "replay called the endpoint again"
        assert seen[-1] != "Pack layers.", "the replayed reply was not faulted"
