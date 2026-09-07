"""An agent with no knowledge of this library, for testing interception.

No import from `agent_loop_chaos`, no decorator, no builder, no engine parameter --
this is the shape of a colleague's repository. The transport is a `MockTransport` so
the tests need no socket, which is the only concession to being a fixture.
"""

from __future__ import annotations

from typing import Any

import httpx


def _model(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gpt-4o",
            "choices": [
                {"message": {"role": "assistant", "content": "Pack light layers."}},
            ],
        },
    )


def _weather(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"temp_c": 21, "city": "Paris"})


def _route(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/chat/completions"):
        return _model(request)
    return _weather(request)


def answer(question: str) -> str:
    """Answer a question using a tool and a model, wrapping neither.

    Args:
        question: The user's question.

    Returns:
        The model's reply.
    """
    with httpx.Client(transport=httpx.MockTransport(_route)) as client:
        data = client.get("https://weather.example/v1/current").json()
        reply = client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": f"{question} {data}"}],
            },
        )
    return str(reply.json()["choices"][0]["message"]["content"])


def no_seams(question: str) -> str:
    """An agent that makes no model or tool call at all.

    Args:
        question: The user's question.

    Returns:
        A canned answer, reached without any interceptable call.
    """
    return f"I cannot help with {question}"


def build(engine: Any = None) -> Any:
    """Builder form, for a scenario `entrypoint`.

    Args:
        engine: Ignored. The point of this fake is that it wraps nothing.

    Returns:
        The agent callable.
    """
    return answer
