"""Recording any crossing, not just a prompt.

The chosen answer to "how does a colleague supply fake data" is: they do not write any.
They run their app once for real, and every later run replays it offline and
deterministically -- the VCR.py lesson. That needs the cassette to key on more than a
message list, because a repository read and an HTTP tool call are the payloads that make
the data-shape faults bite.

Two rules carry over unchanged and are tested here: a miss in replay mode raises rather
than inventing a response, and a cassette is redacted on write -- it gets committed, so
a credential in one is worse than a credential in a report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loop_chaos.cassettes import Cassette, CassetteMiss, crossing_key


class TestTheKey:
    def test_the_same_call_keys_the_same(self) -> None:
        a = crossing_key("tool", "fetch", {"city": "Paris"})
        b = crossing_key("tool", "fetch", {"city": "Paris"})
        assert a == b

    def test_a_different_payload_keys_differently(self) -> None:
        assert crossing_key("tool", "fetch", {"city": "Paris"}) != crossing_key(
            "tool", "fetch", {"city": "Berlin"}
        )

    def test_the_layer_and_the_name_are_part_of_it(self) -> None:
        """Two seams that happen to take the same argument are not the same call."""
        assert crossing_key("tool", "a", {"x": 1}) != crossing_key("tool", "b", {"x": 1})
        assert crossing_key("tool", "a", {"x": 1}) != crossing_key("llm", "a", {"x": 1})

    def test_key_order_does_not_matter(self) -> None:
        """A dict is canonicalized, so an unrelated reordering is not a cassette miss."""
        assert crossing_key("tool", "f", {"a": 1, "b": 2}) == crossing_key(
            "tool", "f", {"b": 2, "a": 1}
        )

    def test_an_unserializable_payload_still_keys(self) -> None:
        """A real repository takes objects. Refusing to key one would refuse to record
        the very calls this exists for."""
        assert crossing_key("tool", "f", object())


class TestRecordAndReplay:
    def test_a_recorded_value_replays_without_calling_the_original(self, tmp_path: Path) -> None:
        cassette = Cassette(tmp_path / "c.json", mode="record")
        key = crossing_key("tool", "fetch", {"city": "Paris"})
        calls: list[int] = []
        first = cassette.play_key(key, live=lambda: (calls.append(1), {"temp_c": 21})[1])
        cassette.save()

        replay = Cassette(tmp_path / "c.json", mode="replay")
        again = replay.play_key(key, live=lambda: (calls.append(1), {"temp_c": 99})[1])

        assert first == {"temp_c": 21}
        assert again == {"temp_c": 21}, "replay called the original instead of the tape"
        assert calls == [1], "the original was called during replay"

    def test_a_miss_in_replay_mode_raises_rather_than_guessing(self, tmp_path: Path) -> None:
        (tmp_path / "c.json").write_text(
            json.dumps({"schema_version": "1.0", "entries": {}}), encoding="utf-8"
        )
        cassette = Cassette(tmp_path / "c.json", mode="replay")
        with pytest.raises(CassetteMiss) as caught:
            cassette.play_key(crossing_key("tool", "fetch", {"city": "Paris"}), live=lambda: {})
        assert "record" in str(caught.value).lower(), "the error must say how to fix itself"

    def test_repeated_calls_replay_in_order(self, tmp_path: Path) -> None:
        cassette = Cassette(tmp_path / "c.json", mode="record")
        key = crossing_key("llm", "gpt-4o", [{"role": "user", "content": "hi"}])
        answers = iter(["first", "second"])
        cassette.play_key(key, live=lambda: next(answers))
        cassette.play_key(key, live=lambda: next(answers))
        cassette.save()

        replay = Cassette(tmp_path / "c.json", mode="replay")
        assert [replay.play_key(key), replay.play_key(key)] == ["first", "second"]

    @pytest.mark.asyncio
    async def test_the_async_path_records_and_replays(self, tmp_path: Path) -> None:
        cassette = Cassette(tmp_path / "c.json", mode="record")
        key = crossing_key("llm", "gpt-4o", "hi")

        async def live() -> str:
            return "hello"

        assert await cassette.aplay_key(key, live=live) == "hello"
        cassette.save()

        replay = Cassette(tmp_path / "c.json", mode="replay")
        assert await replay.aplay_key(key, live=live) == "hello"


class TestItIsRedactedOnWrite:
    def test_a_credential_never_reaches_the_file(self, tmp_path: Path) -> None:
        """A cassette is committed. This matters more here than in a report."""
        cassette = Cassette(tmp_path / "c.json", mode="record")
        cassette.play_key(
            crossing_key("tool", "login", {}),
            live=lambda: {
                "authorization": "Bearer sk-proj-abcdefghijklmnopqrstuvwxyz012345",
                "user": "dana",
            },
        )
        written = cassette.save().read_text(encoding="utf-8")
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz012345" not in written
        assert "redacted" in written
        assert "dana" in written, "redaction must not eat the payload it was protecting"
