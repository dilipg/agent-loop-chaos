"""Fault a model or tool that never touches the network.

The transport strategy covers every hosted SDK and every local *server*, because both
speak HTTP. It cannot see a model that answers in-process: a `transformers` pipeline, an
MLX model, a hand-rolled class, or the fake models a test suite already owns. All of
them inherit `BaseChatModel.generate`, so one patch on the base class reaches every one
at once -- the Langfuse lesson, attached at the framework's own abstraction rather than
per subclass.

That the fakes are covered is the point rather than a side effect. A repository with a
working test suite has already solved offline construction, and this makes its existing
fixtures usable as chaos fixtures with nothing new written.

`BaseTool.run` is the same argument for the tool layer: a repo whose tools are
`@tool`-decorated gets the data-shape faults with no wrapping at all.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any

from ..adapters._lc_messages import from_langchain, to_langchain
from ..context import Layer
from .base import Attachment

__all__ = ["LangChainStrategy"]

logger = logging.getLogger("agent_loop_chaos")


def _reply_text(result: Any) -> str | None:
    """Read the assistant text out of an `LLMResult`.

    Args:
        result: What `generate` returned.

    Returns:
        The text of the first generation, or `None` when the shape is unfamiliar.
    """
    generations = getattr(result, "generations", None)
    if not generations or not generations[0]:
        return None
    text = getattr(generations[0][0], "text", None)
    return text if isinstance(text, str) else None


def _with_reply_text(result: Any, text: str) -> Any:
    """Write mutated assistant text back into an `LLMResult`.

    The generation is mutated in place because `LLMResult` is the object the caller
    already holds; a copy would be discarded. This is the engine's own output, not the
    user's input, so the no-in-place-mutation rule does not apply.

    Args:
        result: What `generate` returned.
        text: The mutated text.

    Returns:
        The same result, carrying the new text.
    """
    generation = result.generations[0][0]
    generation.text = text
    message = getattr(generation, "message", None)
    if message is not None:
        try:
            message.content = text
        except Exception:  # pragma: no cover - a frozen message shape
            logger.debug("could not write mutated text onto the message object")
    return result


def _model_name(model: Any) -> str:
    """Name a model for targeting and the trace.

    Args:
        model: The `BaseChatModel` instance.

    Returns:
        Its configured model id when it has one, else the class name -- so a fake
        model is targetable as `GenericFakeChatModel`.
    """
    for attr in ("model_name", "model", "model_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


class LangChainStrategy:
    """Patch `BaseChatModel` and `BaseTool` so in-process calls become crossings."""

    name = "langchain-core"

    def __init__(self) -> None:
        """Build the strategy. Nothing is patched until `attach`."""
        self._originals: list[tuple[Any, str, Any]] = []

    def available(self) -> str | None:
        """Report whether `langchain_core` is importable.

        Returns:
            `None` when it is, else the reason it is not.
        """
        return None if find_spec("langchain_core") is not None else "not installed"

    def attach(self, engine: Any, seen: dict[str, int]) -> list[Attachment]:
        """Patch the model and tool base classes.

        Args:
            engine: The engine to route crossings through.
            seen: Shared call tally.

        Returns:
            One `Attachment` per patched method.
        """
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.tools import BaseTool

        points: list[Attachment] = []
        plan: tuple[tuple[Any, str, Layer, bool], ...] = (
            (BaseChatModel, "generate", "llm", False),
            (BaseChatModel, "agenerate", "llm", True),
            (BaseTool, "run", "tool", False),
            (BaseTool, "arun", "tool", True),
        )
        for cls, method, layer, is_async in plan:
            original = getattr(cls, method)
            self._originals.append((cls, method, original))
            target = f"{cls.__name__}.{method}"
            build = _model_wrapper if layer == "llm" else _tool_wrapper
            setattr(cls, method, build(engine, seen, original, target, is_async=is_async))
            points.append(Attachment(layer=layer, strategy=self.name, target=target))
        return points

    def detach(self) -> None:
        """Restore every patched method. Safe to call twice."""
        while self._originals:
            cls, method, original = self._originals.pop()
            setattr(cls, method, original)


def _model_wrapper(
    engine: Any, seen: dict[str, int], original: Any, target: str, *, is_async: bool
) -> Any:
    """Build a replacement for `BaseChatModel.generate` or `agenerate`.

    `generate` takes a *batch* of message lists. Only the first is presented to the
    fault: a fault mutating one prompt of a batch is a clear finding, whereas mutating
    all of them makes the report ambiguous about which call was affected.

    Args:
        engine: The engine.
        seen: Shared call tally.
        original: The method being replaced.
        target: The tally key.
        is_async: Whether to build a coroutine wrapper.

    Returns:
        The wrapper.
    """

    def payload_of(batch: Any) -> Any:
        if not isinstance(batch, list) or not batch:
            return None
        return from_langchain(batch[0])

    if is_async:

        async def agenerate(model: Any, messages: Any, *args: Any, **kwargs: Any) -> Any:
            if not engine.is_active():
                return await original(model, messages, *args, **kwargs)
            payload = payload_of(messages)
            if payload is None:
                return await original(model, messages, *args, **kwargs)
            seen[target] = seen.get(target, 0) + 1
            held: dict[str, Any] = {}

            async def call(mutated: Any) -> Any:
                batch = [to_langchain(mutated), *messages[1:]]
                held["result"] = await original(model, batch, *args, **kwargs)
                return _reply_text(held["result"])

            wrapped = engine.llm(call, name=_model_name(model))
            return _apply(held, await wrapped(payload))

        return agenerate

    def generate(model: Any, messages: Any, *args: Any, **kwargs: Any) -> Any:
        if not engine.is_active():
            return original(model, messages, *args, **kwargs)
        payload = payload_of(messages)
        if payload is None:
            return original(model, messages, *args, **kwargs)
        seen[target] = seen.get(target, 0) + 1
        held: dict[str, Any] = {}

        def call(mutated: Any) -> Any:
            batch = [to_langchain(mutated), *messages[1:]]
            held["result"] = original(model, batch, *args, **kwargs)
            return _reply_text(held["result"])

        return _apply(held, engine.llm(call, name=_model_name(model))(payload))

    return generate


def _apply(held: Mapping[str, Any], text: Any) -> Any:
    """Write a mutated reply back into the result the caller will read.

    Args:
        held: The result captured during the call.
        text: What the engine returned, mutated or not.

    Returns:
        The `LLMResult`, carrying the mutated text when it changed.
    """
    result = held.get("result")
    if result is None:
        return result
    if not isinstance(text, str) or text == _reply_text(result):
        return result
    return _with_reply_text(result, text)


def _tool_wrapper(
    engine: Any, seen: dict[str, int], original: Any, target: str, *, is_async: bool
) -> Any:
    """Build a replacement for `BaseTool.run` or `arun`.

    Args:
        engine: The engine.
        seen: Shared call tally.
        original: The method being replaced.
        target: The tally key.
        is_async: Whether to build a coroutine wrapper.

    Returns:
        The wrapper.
    """

    if is_async:

        async def arun(tool: Any, tool_input: Any = None, *args: Any, **kwargs: Any) -> Any:
            if not engine.is_active():
                return await original(tool, tool_input, *args, **kwargs)
            seen[target] = seen.get(target, 0) + 1

            async def call(payload: Any) -> Any:
                return await original(tool, payload, *args, **kwargs)

            wrapped = engine.tool(call, name=getattr(tool, "name", type(tool).__name__))
            return await wrapped(tool_input)

        return arun

    def run(tool: Any, tool_input: Any = None, *args: Any, **kwargs: Any) -> Any:
        if not engine.is_active():
            return original(tool, tool_input, *args, **kwargs)
        seen[target] = seen.get(target, 0) + 1

        def call(payload: Any) -> Any:
            return original(tool, payload, *args, **kwargs)

        wrapped = engine.tool(call, name=getattr(tool, "name", type(tool).__name__))
        return wrapped(tool_input)

    return run
