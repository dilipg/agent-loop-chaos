"""LLM record/replay cassettes (D-45).

Without these, no scenario against a real model is reproducible and users blame the
library for flake. The interception point already makes it cheap: the engine sees
every model call at the normalized-message boundary, so a cassette is a map from a
hash of what was sent to what came back.

The design constraint that matters: a cassette must not silently paper over a miss.
An unrecorded prompt in replay mode is either an error or a recorded gap, never a
guess -- because a guess is exactly the flake this exists to remove.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.cassettes import Cassette, CassetteMiss, messages_hash


class TestTheHash:
    """The key is over what was sent, and nothing else."""

    def test_identical_messages_hash_identically(self) -> None:
        a = [{"role": "user", "content": "hello"}]
        b = [{"role": "user", "content": "hello"}]
        assert messages_hash(a) == messages_hash(b)

    def test_different_content_hashes_differently(self) -> None:
        a = [{"role": "user", "content": "hello"}]
        b = [{"role": "user", "content": "goodbye"}]
        assert messages_hash(a) != messages_hash(b)

    def test_a_string_prompt_works_too(self) -> None:
        assert messages_hash("hello") == messages_hash("hello")
        assert messages_hash("hello") != messages_hash("goodbye")

    def test_key_order_does_not_matter(self) -> None:
        a = [{"role": "user", "content": "hi"}]
        b = [{"content": "hi", "role": "user"}]
        assert messages_hash(a) == messages_hash(b)

    def test_the_model_name_is_part_of_the_key(self) -> None:
        """Two models given the same prompt are two different recordings."""
        messages = [{"role": "user", "content": "hi"}]
        assert messages_hash(messages, model="a") != messages_hash(messages, model="b")

    def test_it_survives_an_unserializable_message(self) -> None:
        class Opaque:
            def __repr__(self) -> str:
                return "<opaque>"

        assert messages_hash([{"role": "user", "content": Opaque()}])


class TestRecordThenReplay:
    def test_a_recorded_call_replays_byte_for_byte(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        recorder = Cassette(path, mode="record")
        recorder.record("hello", {"content": "world", "finish_reason": "stop"})
        recorder.save()

        player = Cassette(path, mode="replay")
        assert player.play("hello") == {"content": "world", "finish_reason": "stop"}

    def test_the_file_is_human_readable_and_stable(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        recorder = Cassette(path, mode="record")
        recorder.record("b", "second")
        recorder.record("a", "first")
        recorder.save()
        # Sorted keys, so a re-record produces a reviewable diff rather than a
        # reshuffle. A cassette is committed; it has to diff cleanly.
        document = json.loads(path.read_text())
        assert list(document["entries"]) == sorted(document["entries"])
        assert document["schema_version"]

    def test_a_miss_in_replay_mode_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        Cassette(path, mode="record").save()
        with pytest.raises(CassetteMiss, match="not recorded"):
            Cassette(path, mode="replay").play("never seen")

    def test_the_miss_names_what_to_do(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        Cassette(path, mode="record").save()
        with pytest.raises(CassetteMiss) as excinfo:
            Cassette(path, mode="replay").play("unseen prompt")
        message = str(excinfo.value)
        assert "--record" in message, "a miss should say how to fix itself"
        assert str(path) in message

    def test_auto_mode_records_a_miss_and_replays_a_hit(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        cassette = Cassette(path, mode="auto")
        calls: list[str] = []

        def live(prompt: Any) -> str:
            calls.append(str(prompt))
            return f"answer to {prompt}"

        assert cassette.play("q1", live=live) == "answer to q1"
        assert cassette.play("q1", live=live) == "answer to q1"
        assert calls == ["q1"], "the second call should come from the cassette"

    def test_replay_mode_never_calls_the_model(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        recorder = Cassette(path, mode="record")
        recorder.record("q", "recorded")
        recorder.save()

        def explode(_prompt: Any) -> str:
            raise AssertionError("replay mode reached the live model")

        assert Cassette(path, mode="replay").play("q", live=explode) == "recorded"

    def test_a_repeated_prompt_replays_its_responses_in_order(self, tmp_path: Path) -> None:
        """An agent that asks the same thing twice usually gets two answers."""
        path = tmp_path / "c.json"
        recorder = Cassette(path, mode="record")
        recorder.record("same", "first")
        recorder.record("same", "second")
        recorder.save()

        player = Cassette(path, mode="replay")
        assert player.play("same") == "first"
        assert player.play("same") == "second"
        # Past the end, the last response repeats rather than raising: an extra call
        # is a difference in the agent, and it is reported as divergence, not as a
        # crash that hides every later finding.
        assert player.play("same") == "second"

    def test_a_missing_file_in_replay_mode_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(CassetteMiss, match="no cassette"):
            Cassette(tmp_path / "absent.json", mode="replay").play("q")


class TestWrappingAModel:
    """The point of all this: one call wraps a model and makes a suite reproducible."""

    def test_a_wrapped_model_records_then_replays(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        live_calls: list[int] = []

        def flaky_model(prompt: Any) -> dict[str, Any]:
            live_calls.append(1)
            # A real model varies. That is the flake cassettes exist to remove.
            return {"content": f"reply #{len(live_calls)}", "finish_reason": "stop"}

        recorded = Cassette(path, mode="record").wrap(flaky_model)
        first = recorded("what is the weather?")
        Cassette.saved_for(recorded).save()

        replayed = Cassette(path, mode="replay").wrap(flaky_model)
        assert replayed("what is the weather?") == first
        assert len(live_calls) == 1, "replay reached the live model"

    def test_it_works_through_the_engine(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        counter: list[int] = []

        def model(prompt: Any) -> str:
            counter.append(1)
            return f"call {len(counter)}"

        def build(engine: ChaosEngine, cassette: Cassette) -> Any:
            chat = engine.llm(cassette.wrap(model), name="default")

            def agent(question: Any = None) -> str:
                return chat("summarise")

            return agent

        recorder = Cassette(path, mode="record")
        engine = ChaosEngine(seed=1337, out_dir=tmp_path, write_bundle=False, judge="rules")
        first = engine.run(
            build(engine, recorder),
            inputs={"question": "?"},
            scenario_id="cassette",
            expected_behavior="ignore_and_continue",
        )
        recorder.save()

        player = Cassette(path, mode="replay")
        engine2 = ChaosEngine(seed=1337, out_dir=tmp_path, write_bundle=False, judge="rules")
        again = engine2.run(
            build(engine2, player),
            inputs={"question": "?"},
            scenario_id="cassette",
            expected_behavior="ignore_and_continue",
        )
        assert again.final_output == first.final_output
        assert len(counter) == 1, "the replayed run called the live model"


class TestTheCliFlags:
    def test_record_and_replay_are_accepted(self) -> None:
        from agent_loop_chaos.cli import build_parser

        args = build_parser().parse_args(["run", "s.yaml", "--record", "c.json"])
        assert args.record == "c.json"
        args = build_parser().parse_args(["run", "s.yaml", "--replay-cassette", "c.json"])
        assert args.replay_cassette == "c.json"

    def test_they_are_mutually_exclusive(self) -> None:
        from agent_loop_chaos.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["run", "s.yaml", "--record", "c.json", "--replay-cassette", "c.json"]
            )
