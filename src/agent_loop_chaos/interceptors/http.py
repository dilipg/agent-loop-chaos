"""Fault a model call at the transport, so no agent has to be wrapped.

Every hosted SDK (openai, anthropic, azure, bedrock) and every local server (Ollama,
vLLM, LM Studio, llama.cpp) reaches its endpoint through `httpx`. One patch on
`Client.send` therefore reaches all of them, and a local server is not a special case:
it is the same HTTP call to a different host. This is the mechanism
[AgentChaos](https://arxiv.org/abs/2608.06790) validates -- agnosticity comes from the
shared transport rather than from a plugin per framework.

The translation is the work. A fault expects a message list on the way in and assistant
text on the way out, so a provider shape reads those out of the wire format and writes
the mutated values back. An endpoint no shape recognizes is still useful: it is one of
the agent's tools, and its JSON body goes through the tool layer instead.

What this cannot do is mutate a stream. A streaming call is passed through untouched
and the fault records `unsupported_streaming`, because a fault that silently does
nothing is the failure this library exists to detect (D-64).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Literal

from ..context import Layer
from .base import Attachment

__all__ = ["HttpxStrategy", "Shape", "shape_for"]

# Where one HTTP call is routed. `skip` produces no crossing at all.
Route = Literal["llm", "tool", "skip"]

logger = logging.getLogger("agent_loop_chaos")


@dataclass(frozen=True, slots=True)
class Shape:
    """How one provider's wire format carries a prompt and a reply.

    Attributes:
        name: The shape's name, for the trace.
        suffixes: URL path endings this shape claims.
        read: Pull the message list out of a request body.
        write: Put a mutated message list back into a request body.
        read_reply: Pull the assistant text out of a response body.
        write_reply: Put mutated assistant text back into a response body.
    """

    name: str
    suffixes: tuple[str, ...]
    read: Callable[[Mapping[str, Any]], Any]
    write: Callable[[Mapping[str, Any], Any], dict[str, Any]]
    read_reply: Callable[[Mapping[str, Any]], str | None]
    write_reply: Callable[[Mapping[str, Any], str], dict[str, Any]]


def _openai_read(body: Mapping[str, Any]) -> Any:
    return body.get("messages")


def _openai_write(body: Mapping[str, Any], messages: Any) -> dict[str, Any]:
    return {**body, "messages": messages}


def _openai_read_reply(body: Mapping[str, Any]) -> str | None:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    text = message.get("content") if isinstance(message, Mapping) else None
    return text if isinstance(text, str) else None


def _openai_write_reply(body: Mapping[str, Any], text: str) -> dict[str, Any]:
    out = json.loads(json.dumps(body))
    out["choices"][0]["message"]["content"] = text
    return dict(out)


def _anthropic_read_reply(body: Mapping[str, Any]) -> str | None:
    blocks = body.get("content")
    if not isinstance(blocks, list):
        return None
    for block in blocks:
        if isinstance(block, Mapping) and block.get("type") == "text":
            text = block.get("text")
            return text if isinstance(text, str) else None
    return None


def _anthropic_write_reply(body: Mapping[str, Any], text: str) -> dict[str, Any]:
    out = json.loads(json.dumps(body))
    for block in out.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            block["text"] = text
            break
    return dict(out)


_SHAPES: tuple[Shape, ...] = (
    Shape(
        name="openai",
        # Covers Azure OpenAI, Ollama's compatibility endpoint, vLLM, LM Studio,
        # llama.cpp's server and everything else that speaks the same dialect.
        suffixes=("/chat/completions",),
        read=_openai_read,
        write=_openai_write,
        read_reply=_openai_read_reply,
        write_reply=_openai_write_reply,
    ),
    Shape(
        name="anthropic",
        suffixes=("/v1/messages", "/v1/messages?beta=true"),
        read=_openai_read,  # the request carries `messages` in the same place
        write=_openai_write,
        read_reply=_anthropic_read_reply,
        write_reply=_anthropic_write_reply,
    ),
)


def shape_for(path: str) -> Shape | None:
    """Find the provider shape for a URL path.

    Args:
        path: The request's URL path.

    Returns:
        The matching shape, or `None` when this is not a recognized model endpoint --
        in which case the call is one of the agent's tools, not its model.
    """
    for shape in _SHAPES:
        if any(path.endswith(suffix) for suffix in shape.suffixes):
            return shape
    return None


def _request_body(request: Any) -> dict[str, Any] | None:
    """Read a request's JSON body.

    Args:
        request: The `httpx.Request`.

    Returns:
        The decoded object, or `None` when the body is absent, not JSON, or not an
        object -- none of which this strategy can translate.
    """
    try:
        content = request.content
    except Exception:  # pragma: no cover - a stream has no readable content
        return None
    if not content:
        return None
    try:
        body = json.loads(content)
    except (ValueError, TypeError):
        return None
    return body if isinstance(body, dict) else None


def _response_body(response: Any) -> dict[str, Any] | None:
    """Read a response's JSON body without consuming a stream.

    Args:
        response: The `httpx.Response`.

    Returns:
        The decoded object, or `None` when it is unreadable or not an object.
    """
    if getattr(response, "is_stream_consumed", False) is False and not hasattr(
        response, "_content"
    ):
        return None
    try:
        body = response.json()
    except Exception:
        return None
    return body if isinstance(body, dict) else None


def _rebuild_request(request: Any, body: Mapping[str, Any]) -> Any:
    """Build a new request carrying a mutated body.

    `Content-Length` is dropped so httpx recomputes it; a stale one makes the server
    read a truncated body, which would look like a fault of its own.

    Args:
        request: The original request.
        body: The mutated JSON body.

    Returns:
        A new `httpx.Request`.
    """
    import httpx

    headers = {k: v for k, v in request.headers.items() if k.lower() != "content-length"}
    return httpx.Request(
        request.method,
        request.url,
        headers=headers,
        content=json.dumps(body).encode("utf-8"),
        extensions=dict(request.extensions),
    )


def _rebuild_response(response: Any, body: Mapping[str, Any]) -> Any:
    """Build a new response carrying a mutated body.

    Args:
        response: The original response.
        body: The mutated JSON body.

    Returns:
        A new `httpx.Response`, or the original when it cannot be rebuilt.
    """
    import httpx

    skip = {"content-length", "content-encoding"}
    headers = {k: v for k, v in response.headers.items() if k.lower() not in skip}
    try:
        return httpx.Response(
            response.status_code,
            headers=headers,
            content=json.dumps(body).encode("utf-8"),
            request=response.request,
        )
    except Exception:  # pragma: no cover - defensive; never break the observed run
        return response


def _capture(response: Any) -> dict[str, Any]:
    """Reduce a response to something a cassette can hold.

    Args:
        response: The `httpx.Response`.

    Returns:
        Status and decoded body. Headers are deliberately dropped: they carry the
        credentials this run was authenticated with, and a cassette is committed.
    """
    return {"status": response.status_code, "json": _response_body(response)}


def _restore(recorded: Mapping[str, Any], request: Any) -> Any:
    """Rebuild a response from a cassette entry.

    Args:
        recorded: What `_capture` stored.
        request: The request being answered, so the response knows its own.

    Returns:
        An `httpx.Response`.
    """
    import httpx

    body = recorded.get("json")
    return httpx.Response(
        int(recorded.get("status", 200)),
        json=body if body is not None else {},
        request=request,
    )


def _send(
    engine: Any, original: Any, layer: Layer, name: str, client: Any, request: Any, kwargs: Any
) -> Any:
    """Make one real request, or replay one from the cassette.

    Args:
        engine: The engine, which may carry a cassette.
        original: The unpatched `send`.
        layer: The crossing layer, part of the key.
        name: The model id or tool name, part of the key.
        client: The client instance.
        request: The outgoing request.
        kwargs: The keyword arguments to `send`.

    Returns:
        The response, whether it came from the network or the tape.
    """
    cassette = getattr(engine, "cassette", None)
    if cassette is None:
        return original(client, request, **kwargs)
    key = _tape_key(layer, name, request)
    recorded = cassette.play_key(key, live=lambda: _capture(original(client, request, **kwargs)))
    return _restore(recorded, request) if isinstance(recorded, Mapping) else recorded


async def _asend(
    engine: Any, original: Any, layer: Layer, name: str, client: Any, request: Any, kwargs: Any
) -> Any:
    """Async twin of `_send`.

    Args:
        engine: The engine, which may carry a cassette.
        original: The unpatched `send`.
        layer: The crossing layer, part of the key.
        name: The model id or tool name, part of the key.
        client: The client instance.
        request: The outgoing request.
        kwargs: The keyword arguments to `send`.

    Returns:
        The response, whether it came from the network or the tape.
    """
    cassette = getattr(engine, "cassette", None)
    if cassette is None:
        return await original(client, request, **kwargs)
    key = _tape_key(layer, name, request)

    async def live() -> Any:
        return _capture(await original(client, request, **kwargs))

    recorded = await cassette.aplay_key(key, live=live)
    return _restore(recorded, request) if isinstance(recorded, Mapping) else recorded


def _tape_key(layer: Layer, name: str, request: Any) -> str:
    """Key an HTTP call for the cassette.

    The method and path are part of it, so two endpoints that happen to take the same
    body are not mistaken for each other.

    Args:
        layer: The crossing layer.
        name: The model id or tool name.
        request: The outgoing request.

    Returns:
        The key.
    """
    from ..cassettes import crossing_key

    payload = {
        "method": request.method,
        "path": request.url.path,
        "body": _request_body(request),
    }
    return crossing_key(layer, name, payload)


def _is_streaming(body: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> bool:
    """Report whether this call streams, at either layer it can be requested.

    Args:
        body: The decoded request body, when there is one.
        kwargs: The keyword arguments to `send`.

    Returns:
        True when the response arrives incrementally.
    """
    return bool(kwargs.get("stream")) or bool((body or {}).get("stream"))


class HttpxStrategy:
    """Patch `httpx` so a model call nobody wrapped still goes through the engine."""

    name = "httpx"

    def __init__(self) -> None:
        """Build the strategy. Nothing is patched until `attach`."""
        self._originals: list[tuple[Any, Any]] = []

    def available(self) -> str | None:
        """Report whether `httpx` is importable.

        Returns:
            `None` when it is, else the reason it is not.
        """
        return None if find_spec("httpx") is not None else "not installed"

    def attach(self, engine: Any, seen: dict[str, int]) -> list[Attachment]:
        """Patch the sync and async clients.

        Args:
            engine: The engine to route crossings through.
            seen: Shared call tally.

        Returns:
            One `Attachment` per client and layer, since the URL decides which layer
            a given call belongs to and that is not knowable in advance.
        """
        import httpx

        points: list[Attachment] = []
        for cls in (httpx.Client, httpx.AsyncClient):
            original = cls.send
            self._originals.append((cls, original))
            base = f"httpx.{cls.__name__}.send"
            layers: tuple[Layer, ...] = ("llm", "tool")
            keys: dict[Layer, str] = {layer: f"{base} ({layer})" for layer in layers}
            wrapper = (
                _async_wrapper(engine, seen, original, keys)
                if cls is httpx.AsyncClient
                else _sync_wrapper(engine, seen, original, keys)
            )
            cls.send = wrapper  # type: ignore[method-assign]
            points += [
                Attachment(layer=layer, strategy=self.name, target=key)
                for layer, key in keys.items()
            ]
        return points

    def detach(self) -> None:
        """Restore both clients. Safe to call twice."""
        while self._originals:
            cls, original = self._originals.pop()
            cls.send = original


def _plan(request: Any, kwargs: Mapping[str, Any]) -> tuple[Route, Shape | None, Any]:
    """Decide how to treat one HTTP call.

    Args:
        request: The outgoing request.
        kwargs: The keyword arguments to `send`.

    Returns:
        ``(route, shape, body)``. A recognized model endpoint is `llm` with a shape;
        an unrecognized one is one of the agent's tools; a streaming model call is
        `skip`, and passes through with no crossing at all.

        A streaming call must not fall through to the tool layer. It is a *model*
        call, so a tool-layer fault mutating its request would be the library
        inventing a finding -- exactly the false positive that makes a probe wrong.
    """
    body = _request_body(request)
    shape = shape_for(request.url.path) if body is not None else None
    if shape is None:
        return "tool", None, body
    if not _is_streaming(body, kwargs):
        return "llm", shape, body
    logger.warning(
        "streaming model call to %s passed through untouched: a fault cannot mutate an "
        "incremental response, so no llm crossing was produced and any llm fault "
        "aimed at this call did not fire",
        request.url.path,
    )
    return "skip", None, None


def _sync_wrapper(
    engine: Any, seen: dict[str, int], original: Any, keys: Mapping[Layer, str]
) -> Any:
    """Build the replacement for `httpx.Client.send`.

    Args:
        engine: The engine to route crossings through.
        seen: Shared call tally.
        original: The method being replaced.
        keys: Per-layer tally keys.

    Returns:
        The wrapper.
    """

    def send(client: Any, request: Any, **kwargs: Any) -> Any:
        if not engine.is_active():
            return original(client, request, **kwargs)
        route, shape, body = _plan(request, kwargs)
        if route == "skip":
            return original(client, request, **kwargs)
        if shape is None or body is None:
            return _tool_call_sync(engine, seen, original, keys, client, request, kwargs, body)
        seen[keys["llm"]] = seen.get(keys["llm"], 0) + 1
        model = str(body.get("model") or "default")
        held: dict[str, Any] = {}

        def call(messages: Any) -> Any:
            outgoing = _rebuild_request(request, shape.write(body, messages))
            held["response"] = _send(engine, original, "llm", model, client, outgoing, kwargs)
            held["body"] = _response_body(held["response"])
            return shape.read_reply(held["body"]) if held["body"] is not None else None

        text = engine.llm(call, name=model)(shape.read(body))
        return _apply_reply(shape, held, text)

    return send


def _async_wrapper(
    engine: Any, seen: dict[str, int], original: Any, keys: Mapping[Layer, str]
) -> Any:
    """Build the replacement for `httpx.AsyncClient.send`.

    Args:
        engine: The engine to route crossings through.
        seen: Shared call tally.
        original: The method being replaced.
        keys: Per-layer tally keys.

    Returns:
        The coroutine wrapper. Async parity is structural: the replacement for a
        coroutine method is itself one.
    """

    async def send(client: Any, request: Any, **kwargs: Any) -> Any:
        if not engine.is_active():
            return await original(client, request, **kwargs)
        route, shape, body = _plan(request, kwargs)
        if route == "skip":
            return await original(client, request, **kwargs)
        if shape is None or body is None:
            return await _tool_call_async(
                engine, seen, original, keys, client, request, kwargs, body
            )
        seen[keys["llm"]] = seen.get(keys["llm"], 0) + 1
        model = str(body.get("model") or "default")
        held: dict[str, Any] = {}

        async def call(messages: Any) -> Any:
            held["response"] = await _asend(
                engine,
                original,
                "llm",
                model,
                client,
                _rebuild_request(request, shape.write(body, messages)),
                kwargs,
            )
            held["body"] = _response_body(held["response"])
            return shape.read_reply(held["body"]) if held["body"] is not None else None

        wrapped = engine.llm(call, name=model)
        text = await wrapped(shape.read(body))
        return _apply_reply(shape, held, text)

    return send


def _apply_reply(shape: Shape, held: Mapping[str, Any], text: Any) -> Any:
    """Write a mutated reply back into the response the caller will see.

    Args:
        shape: The provider shape.
        held: The response and its decoded body, captured during the call.
        text: What the engine returned, mutated or not.

    Returns:
        The response, rebuilt only when the text actually changed.
    """
    response, body = held.get("response"), held.get("body")
    if response is None:
        return response
    if body is None or not isinstance(text, str) or text == shape.read_reply(body):
        return response
    return _rebuild_response(response, shape.write_reply(body, text))


def _tool_name(request: Any) -> str:
    """Name an HTTP tool call for targeting and the trace.

    Args:
        request: The outgoing request.

    Returns:
        ``"GET /v1/current"`` -- globbable, and readable in a report.
    """
    return f"{request.method} {request.url.path}"


def _tool_call_sync(
    engine: Any,
    seen: dict[str, int],
    original: Any,
    keys: Mapping[Layer, str],
    client: Any,
    request: Any,
    kwargs: Mapping[str, Any],
    body: Any,
) -> Any:
    """Route a non-model HTTP call through the tool layer.

    Args:
        engine: The engine.
        seen: Shared call tally.
        original: The method being replaced.
        keys: Per-layer tally keys.
        client: The client instance.
        request: The outgoing request.
        kwargs: The keyword arguments to `send`.
        body: The decoded request body, when there is one.

    Returns:
        The response, rebuilt when a fault changed the payload.
    """
    seen[keys["tool"]] = seen.get(keys["tool"], 0) + 1
    held: dict[str, Any] = {}

    def call(payload: Any = None) -> Any:
        outgoing = request if payload is None else _rebuild_request(request, payload)
        held["response"] = _send(
            engine, original, "tool", _tool_name(request), client, outgoing, kwargs
        )
        held["body"] = _response_body(held["response"])
        return held["body"]

    returned = engine.tool(call, name=_tool_name(request))(body)
    return _apply_tool_reply(held, returned)


async def _tool_call_async(
    engine: Any,
    seen: dict[str, int],
    original: Any,
    keys: Mapping[Layer, str],
    client: Any,
    request: Any,
    kwargs: Mapping[str, Any],
    body: Any,
) -> Any:
    """Async twin of `_tool_call_sync`.

    Args:
        engine: The engine.
        seen: Shared call tally.
        original: The method being replaced.
        keys: Per-layer tally keys.
        client: The client instance.
        request: The outgoing request.
        kwargs: The keyword arguments to `send`.
        body: The decoded request body, when there is one.

    Returns:
        The response, rebuilt when a fault changed the payload.
    """
    seen[keys["tool"]] = seen.get(keys["tool"], 0) + 1
    held: dict[str, Any] = {}

    async def call(payload: Any = None) -> Any:
        outgoing = request if payload is None else _rebuild_request(request, payload)
        held["response"] = await _asend(
            engine, original, "tool", _tool_name(request), client, outgoing, kwargs
        )
        held["body"] = _response_body(held["response"])
        return held["body"]

    returned = await engine.tool(call, name=_tool_name(request))(body)
    return _apply_tool_reply(held, returned)


def _apply_tool_reply(held: Mapping[str, Any], returned: Any) -> Any:
    """Write a mutated tool payload back into the response.

    Args:
        held: The response and its decoded body, captured during the call.
        returned: What the engine returned, mutated or not.

    Returns:
        The response, rebuilt only when the payload actually changed.
    """
    response, body = held.get("response"), held.get("body")
    if response is None:
        return response
    if not isinstance(returned, dict) or returned == body:
        return response
    return _rebuild_response(response, returned)
