"""Record and replay model responses, so a suite against a real model reproduces.

Without this, no scenario that talks to a real model is reproducible, and the flake
gets blamed on the library rather than on the sampler (D-45). The interception point
makes it cheap: the engine already sees every model call at the normalized-message
boundary, so a cassette is a map from a hash of what was sent to what came back.

One rule shapes the whole design: **a miss is never a guess.** In replay mode an
unrecorded prompt raises, because silently inventing a response is exactly the
non-determinism cassettes exist to remove. The error says how to fix itself.

    cassette = Cassette("cassettes/demo.json", mode="record")
    chat = engine.llm(cassette.wrap(my_model), name="default")
    ...
    cassette.save()

Then commit the file and run with `mode="replay"` forever after.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

from .errors import ChaosError

__all__ = ["Cassette", "CassetteMiss", "CassetteMode", "crossing_key", "messages_hash"]

CassetteMode = Literal["record", "replay", "auto"]

SCHEMA_VERSION = "1.0"
_KEY_CHARS = 16


class CassetteMiss(ChaosError):  # noqa: N818 - it is a miss, and reads better than …Error
    """A prompt was not in the cassette and replay mode may not invent one."""


def messages_hash(messages: Any, *, model: str | None = None) -> str:
    """Key a model call by exactly what was sent.

    Args:
        messages: A prompt string, or the normalized message list.
        model: The model identity. Two models given the same prompt are two
            different recordings, so it is part of the key.

    Returns:
        A short hex digest. Short because a cassette is meant to be read and
        diffed by a person, and 16 hex characters is far past collision risk for
        the number of calls one suite makes.
    """
    payload = {"model": model, "messages": _canonical(messages)}
    blob = json.dumps(payload, sort_keys=True, default=repr)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:_KEY_CHARS]


def crossing_key(layer: str, name: str, payload: Any, *, model: str | None = None) -> str:
    """Key any crossing by what was sent to it.

    A prompt is not the only payload worth recording: a repository read and an HTTP
    tool call are the ones that make the data-shape faults bite, and neither is a
    message list. The layer and the name are part of the key, so two seams that happen
    to take the same argument are not mistaken for the same call.

    Args:
        layer: The crossing layer -- `llm`, `tool`.
        name: The seam's name, as a target would write it.
        payload: What was sent. Canonicalized, so an unrelated key reordering is not
            a cassette miss; an object the encoder cannot handle falls back to `repr`,
            because a real repository takes objects and refusing to key one would
            refuse to record the calls this exists for.
        model: Folded in when the caller tracks a model identity separately.

    Returns:
        A short hex digest, in the same shape as `messages_hash`.
    """
    blob = json.dumps(
        {"layer": layer, "name": name, "model": model, "payload": _canonical(payload)},
        sort_keys=True,
        default=repr,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:_KEY_CHARS]


def _canonical(messages: Any) -> Any:
    """Reduce a prompt to the parts that decide the response.

    Args:
        messages: A string, a mapping, or a sequence of them.

    Returns:
        A JSON-friendly projection. Anything unserializable becomes its `repr`, so
        an exotic message object keys stably instead of raising.
    """
    if isinstance(messages, str):
        return messages
    if isinstance(messages, Mapping):
        return {
            str(k): _canonical(v) for k, v in sorted(messages.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        return [_canonical(m) for m in messages]
    if isinstance(messages, (int, float, bool)) or messages is None:
        return messages
    return repr(messages)


class Cassette:
    """A recorded conversation, keyed by prompt hash.

    Attributes:
        path: Where the cassette lives. Commit it.
        mode: `record` calls the model and stores every response; `replay` never
            calls it; `auto` records a miss and replays a hit, which is what you
            want while writing a scenario.
    """

    #: `wrap` remembers which cassette produced a wrapper, so a caller holding only
    #: the wrapped callable can still save. Keyed by the wrapper's id.
    _by_wrapper: ClassVar[dict[int, Cassette]] = {}

    def __init__(
        self, path: str | Path, *, mode: CassetteMode = "auto", model: str | None = None
    ) -> None:
        """Open a cassette.

        Args:
            path: The cassette file.
            mode: `record`, `replay`, or `auto`.
            model: Model identity folded into every key.
        """
        self.path = Path(path)
        self.mode: CassetteMode = mode
        self.model = model
        self._entries: dict[str, list[Any]] = {}
        self._played: dict[str, int] = {}
        if mode != "record" and self.path.is_file():
            document = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = {k: list(v) for k, v in (document.get("entries") or {}).items()}

    # -- recording -------------------------------------------------------------

    def record(self, messages: Any, response: Any) -> None:
        """Store one response.

        Args:
            messages: What was sent.
            response: What came back.
        """
        self._entries.setdefault(messages_hash(messages, model=self.model), []).append(response)

    def save(self) -> Path:
        """Write the cassette to disk.

        Keys are sorted so a re-record produces a reviewable diff rather than a
        reshuffle -- a cassette is committed, so it has to diff cleanly.

        Returns:
            The path written.
        """
        from .redact import redact

        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            # A cassette is committed, which makes a credential in one worse than a
            # credential in a report. Redaction happens on the way to disk, never on
            # the value the run itself used.
            "entries": {k: redact(self._entries[k]) for k in sorted(self._entries)},
        }
        self.path.write_text(
            json.dumps(document, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return self.path

    # -- playback --------------------------------------------------------------

    def play_key(self, key: str, *, live: Callable[[], Any] | None = None) -> Any:
        """Return the recorded value for a precomputed key.

        Args:
            key: From `crossing_key` or `messages_hash`.
            live: The real call, invoked only in `record` and `auto` modes.

        Returns:
            The recorded value, or what `live` produced.

        Raises:
            CassetteMiss: In `replay` mode when this key was never recorded. Inventing
                a value here would reintroduce exactly the non-determinism a cassette
                removes.
        """
        # `record` stores *every* response, which is its documented contract: a model
        # asked the same thing twice may answer differently, and a cassette that keeps
        # only the first reply cannot reproduce the run it recorded. `auto` replays a
        # hit, which is what makes it useful while a scenario is being written.
        recorded = self._entries.get(key)
        if recorded and self.mode != "record":
            index = self._played.get(key, 0)
            self._played[key] = index + 1
            return recorded[min(index, len(recorded) - 1)]
        if self.mode == "replay":
            raise CassetteMiss(self._miss(key))
        if live is None:
            raise CassetteMiss(f"{key} is not recorded and no live call was supplied")
        value = live()
        self._entries.setdefault(key, []).append(value)
        return value

    async def aplay_key(self, key: str, *, live: Callable[[], Any] | None = None) -> Any:
        """Async twin of `play_key`.

        Args:
            key: From `crossing_key` or `messages_hash`.
            live: The real call, awaited only in `record` and `auto` modes.

        Returns:
            The recorded value, or what `live` produced.

        Raises:
            CassetteMiss: As `play_key`.
        """
        recorded = self._entries.get(key)
        if recorded and self.mode != "record":
            index = self._played.get(key, 0)
            self._played[key] = index + 1
            return recorded[min(index, len(recorded) - 1)]
        if self.mode == "replay":
            raise CassetteMiss(self._miss(key))
        if live is None:
            raise CassetteMiss(f"{key} is not recorded and no live call was supplied")
        value = await live()
        self._entries.setdefault(key, []).append(value)
        return value

    def _miss(self, key: str) -> str:
        """Explain a replay miss, and how to fix it.

        Args:
            key: The key that was not found.

        Returns:
            The message.
        """
        if not self.path.is_file():
            return f"no cassette at {self.path}. Record one first: alc run <suite> --record"
        return (
            f"{key} is not recorded in {self.path}. The agent did something the "
            "cassette has never seen -- either it changed, or the cassette is stale. "
            "Re-record with: alc run <suite> --record"
        )

    def play(self, messages: Any, *, live: Callable[[Any], Any] | None = None) -> Any:
        """Return the recorded response for a prompt.

        Args:
            messages: What is being sent.
            live: The real model, called only in `record` and `auto` modes.

        Returns:
            The response.

        Raises:
            CassetteMiss: In `replay` mode when the prompt was never recorded, or
                when the cassette file does not exist. Inventing a response here
                would reintroduce exactly the non-determinism this removes.
        """
        # Past the end of a recording the last response repeats: an extra call is a
        # difference in the agent, and it belongs in a divergence report rather than as
        # a crash that hides every later finding.
        return self.play_key(
            messages_hash(messages, model=self.model),
            live=None if live is None else lambda: live(messages),
        )

    def wrap(self, model: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a model callable so every call goes through the cassette.

        Args:
            model: The real model callable.

        Returns:
            A callable with the same shape. In `replay` mode it never reaches
            `model`.
        """
        import functools

        @functools.wraps(model)
        def wrapper(messages: Any, *args: Any, **kwargs: Any) -> Any:
            return self.play(messages, live=lambda m: model(m, *args, **kwargs))

        Cassette._by_wrapper[id(wrapper)] = self
        return wrapper

    @classmethod
    def saved_for(cls, wrapper: Callable[..., Any]) -> Cassette:
        """Recover the cassette behind a wrapped model.

        Lets a caller who only kept the wrapped callable still save.

        Args:
            wrapper: A callable returned by `wrap`.

        Returns:
            Its cassette.

        Raises:
            KeyError: When the callable did not come from `wrap`.
        """
        return cls._by_wrapper[id(wrapper)]

    def __len__(self) -> int:
        """How many distinct prompts are recorded.

        Returns:
            The entry count.
        """
        return len(self._entries)
