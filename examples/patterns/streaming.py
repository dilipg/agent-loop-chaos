"""A streaming agent: chunks in, a buffer out, parsed once the stream ends.

This is the shape behind every typing-cursor UI. The model call hands back a
sequence of text chunks, the agent appends them to a buffer as they arrive (a real
one would paint each one), and the buffer is parsed as structured output only after
the sequence is exhausted.

**Where the fault bites.** The library has no per-chunk interception point yet, so
the seam is the call that *returns* the chunk sequence, not the iteration over it.
`_stream` therefore materialises its chunks eagerly, and `_accumulate` concatenates
whatever the sequence yields, so a `(llm, post)` fault that rewrites the returned
value is still seen by the accumulator. The cost of that is honest and worth
stating: `LLMTruncationFault` re-types the chunk list into the JSON rendering of
that list before cutting it, so the truncated buffer carries a stray ``["`` prefix
that a genuinely truncated stream would not have. The *failure* is the real one --
a partial body consumed as a whole answer -- but the trace excerpt looks more
synthetic than it should until a per-chunk seam exists.

**Planted weakness in `build`.** The buffer is parsed the moment the sequence ends,
and nothing asks *why* it ended. There is no completeness check: no inspection of a
finish reason, no test that the JSON object ever closed, no minimum length. A stream
the token limit cut in half is fed straight to `json.loads`.

**What `build_fixed` does instead.** It treats "the sequence ended" and "the answer
finished" as different facts. The buffer has to parse *and* carry both declared
keys before it counts as an answer; otherwise the agent records a `degraded:` note,
asks once more for a shorter complete reply, and if that also arrives incomplete it
says plainly that no headline and no confidence rating were produced. It never
fills either one in from the fragment it holds.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from . import PatternSpec, register

__all__ = [
    "MAX_CHUNKS",
    "PATTERN",
    "RELEASE_NOTES",
    "SUMMARIZE_PROMPT",
    "build",
    "build_fixed",
]

#: Hard cap on accumulator iterations. A faulted stream must not be able to hang the
#: test suite, and the cap is high enough that no honest body reaches it.
MAX_CHUNKS = 4096

#: How much text one chunk carries. Small enough that a mid-body cut lands inside a
#: chunk boundary rather than neatly between two.
_CHUNK_CHARS = 18

RELEASE_NOTES = """- Incremental sync now resumes from the last acknowledged cursor.
- Fixed a crash when a workspace had no default project.
- The export job retries once on a 503."""

SUMMARIZE_PROMPT = """[SUMMARIZE]
Summarise the release below for a changelog reader.
Reply with a JSON object and nothing else:
{{"headline": str, "confidence": "low" | "medium" | "high"}}
NOTES:
{notes}"""

_RETRY_SUFFIX = "\nYour previous reply was cut off. Answer again, complete and shorter."

_NOTES_BLOCK = re.compile(r"NOTES:\n(.*)\Z", re.S)

#: What the scripted model says when the notes block reached it empty. A mediocre
#: model does not admit the gap: it reaches for a plausible-sounding release it
#: remembers instead, which is what makes a context fault worth catching here.
_FABRICATED = "Release 4.2 ships incremental sync and two crash fixes."

_REQUIRED_KEYS = frozenset({"headline", "confidence"})


def _prompt_text(messages: Any) -> str:
    """Flatten whatever the agent sent into one searchable string.

    Args:
        messages: A string, or a list of message mappings.

    Returns:
        The prompt as text.
    """
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        return "\n".join(
            str(m.get("content", m)) if isinstance(m, dict) else str(m) for m in messages
        )
    return str(messages)


def _script(prompt: str, call: int) -> str:
    """Produce the body the scripted model would have generated.

    Args:
        prompt: The rendered prompt.
        call: 1-based index of this call, so a re-ask reads differently.

    Returns:
        A JSON body as text.
    """
    match = _NOTES_BLOCK.search(prompt)
    notes = (match.group(1) if match else "").strip()
    if not notes:
        return json.dumps({"headline": _FABRICATED, "confidence": "high"})

    first = notes.splitlines()[0].strip().lstrip("- ").rstrip(".")
    if call > 1:
        return json.dumps({"headline": first, "confidence": "medium"})
    detail = f"{first}, alongside two smaller fixes in the same release"
    return json.dumps({"headline": detail, "confidence": "medium"})


def _stream(messages: Any, call: int) -> list[str]:
    """Render one model response as the chunk sequence a stream would yield.

    Args:
        messages: The prompt.
        call: 1-based index of this call.

    Returns:
        The fully-formed chunk sequence, materialised so a `(llm, post)` fault can
        rewrite it before the agent ever iterates.
    """
    body = _script(_prompt_text(messages), call)
    return [body[i : i + _CHUNK_CHARS] for i in range(0, len(body), _CHUNK_CHARS)]


def _model() -> Callable[[Any], list[str]]:
    """Build a scripted streaming model with its own call counter.

    Returns:
        A callable taking messages and returning chunks.
    """
    calls = [0]

    def stream(messages: Any) -> list[str]:
        """Stream one response.

        Args:
            messages: The prompt.

        Returns:
            The chunk sequence.
        """
        calls[0] += 1
        return _stream(messages, calls[0])

    return stream


def _accumulate(stream: Any) -> str:
    """Concatenate a chunk sequence into a buffer.

    Args:
        stream: Whatever the model call returned -- a chunk sequence, or the string
            a `(llm, post)` fault replaced it with.

    Returns:
        The accumulated text, capped at `MAX_CHUNKS` iterations.
    """
    buffer: list[str] = []
    for index, chunk in enumerate(stream):
        if index >= MAX_CHUNKS:
            break
        buffer.append(str(chunk))
    return "".join(buffer)


def _complete(buffer: str) -> dict[str, Any] | None:
    """Decide whether a buffer holds a finished answer.

    Args:
        buffer: The accumulated text.

    Returns:
        The parsed object when it parses and carries every declared key, else
        `None`.
    """
    try:
        parsed = json.loads(buffer)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict) or not parsed.keys() >= _REQUIRED_KEYS:
        return None
    return parsed


def build(engine: Any) -> Callable[..., str]:
    """Build the naive streaming summariser.

    Args:
        engine: The `ChaosEngine`, or `None` to run uninstrumented.

    Returns:
        The agent callable.
    """
    raw = _model()
    model = raw if engine is None else engine.llm(raw, name="writer")

    def agent(notes: str | None = None) -> str:
        """Summarise a release from a streamed response.

        Args:
            notes: The release notes.

        Returns:
            The headline and its confidence rating.

        Raises:
            ValueError: When the accumulated buffer is not JSON.
            KeyError: When the parsed object is missing a field.
        """
        prompt = SUMMARIZE_PROMPT.format(notes=notes or RELEASE_NOTES)
        buffer = _accumulate(model([{"role": "user", "content": prompt}]))
        answer = json.loads(buffer)
        return f"{answer['headline']} (confidence: {answer['confidence']})"

    return agent


def build_fixed(engine: Any) -> Callable[..., str]:
    """Build the hardened twin, which checks the stream actually finished.

    Args:
        engine: The `ChaosEngine`, or `None` to run uninstrumented.

    Returns:
        The agent callable.
    """
    raw = _model()
    model = raw if engine is None else engine.llm(raw, name="writer")

    def agent(notes: str | None = None) -> str:
        """Summarise a release, refusing to read a fragment as an answer.

        Args:
            notes: The release notes.

        Returns:
            The headline and its confidence rating, or a message naming what the
            cut-off stream failed to deliver.
        """
        prompt = SUMMARIZE_PROMPT.format(notes=notes or RELEASE_NOTES)
        for attempt in range(2):
            text = prompt if attempt == 0 else prompt + _RETRY_SUFFIX
            buffer = _accumulate(model([{"role": "user", "content": text}]))
            answer = _complete(buffer)
            if answer is not None:
                if engine is not None:
                    engine.validated({"buffered_chars": len(buffer)}, name="stream_complete")
                return f"{answer['headline']} (confidence: {answer['confidence']})"
            if engine is not None:
                engine.note(
                    f"degraded: the stream ended after {len(buffer)} characters with the "
                    "JSON object still open, so the response was not treated as an answer"
                )
        return (
            "The release summary is unavailable: the model stream was cut off before its "
            "JSON object closed, so neither a headline nor a confidence rating was produced."
        )

    return agent


PATTERN = register(
    PatternSpec(
        name="streaming",
        description=(
            "A token stream accumulated into a buffer and parsed once it ends -- the shape "
            "behind every typing-cursor UI."
        ),
        build=build,
        build_fixed=build_fixed,
        weakness=(
            "the accumulated buffer is parsed with no completeness check, so a stream the "
            "token limit cut short is consumed as though it were a whole answer"
        ),
        faults=(
            {
                "type": "LLMTruncationFault",
                "params": {"cut": "mid_json", "at_ratio": 0.55},
                "target": {"llm": "writer"},
                "trigger": {"on_call": 1},
            },
        ),
        inputs={"notes": RELEASE_NOTES},
        llm_names=("writer",),
        expected_behavior="graceful_degradation",
    )
)
