"""A programmable fake model endpoint (`docs/07-TESTING.md` §5).

Speaks both wire shapes the judge supports -- OpenAI-compatible
`/v1/chat/completions` and Ollama `/api/chat` -- plus the two reachability paths.
Responses are scripted, so every branch of the structured-output ladder is reachable
without a real model and without the network.

Never point this at anything real. It exists so the default test run can exercise
the SLM path with sockets otherwise blocked.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["VALID_OUTPUT", "FakeSLM", "Reply"]

# A schema-valid `judge_output` object, used as the happy path and as the base for
# the malformed variants.
VALID_OUTPUT: dict[str, Any] = {
    "observed_behavior": "crashed",
    "failure_mode": "crash_unhandled_exception",
    "severity": "high",
    "confidence": 0.8,
    "narrative": "A key was dropped from the tool result and the caller indexed it directly.",
    "root_cause_hypothesis": "The consumer assumes every documented field is present.",
    "refinement_hint": "Check the key exists before reading it.",
    "suggested_fixes": [
        {
            "kind": "input_validation",
            "description": "Validate the tool result before indexing it.",
            "target": "app/nodes.py:34",
            "confidence": 0.7,
            "patch_sketch": None,
        }
    ],
}


@dataclass
class Reply:
    """One scripted response.

    Attributes:
        body: The assistant message content the endpoint should return.
        status: HTTP status. Anything but 200 skips `body`.
        delay_s: Seconds to sleep before responding, for timeout tests.
        raw: Return this exact bytes payload as the whole HTTP body, bypassing the
            transport envelope. For malformed-envelope tests.
    """

    body: str = ""
    status: int = 200
    delay_s: float = 0.0
    raw: bytes | None = None


@dataclass
class FakeSLM:
    """A threaded fake endpoint with a scripted reply queue.

    Replies are consumed in order; the last one repeats once the queue is drained,
    so a test that only cares about the first response need not script the rest.
    """

    replies: Sequence[Reply] = field(default_factory=lambda: [Reply(json.dumps(VALID_OUTPUT))])
    _server: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    requests: list[dict[str, Any]] = field(default_factory=list, init=False)
    _index: int = field(default=0, init=False)
    # A delayed handler must be interruptible. Without this, `shutdown()` returns
    # while a handler is still sleeping, and the wake-up writes to a closed socket --
    # surfacing as an unraisable exception that fails whichever test runs next.
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    @property
    def base_url(self) -> str:
        """The root URL the judge should be pointed at.

        Returns:
            ``http://127.0.0.1:<port>``.

        Raises:
            RuntimeError: When the server is not running.
        """
        if self._server is None:
            raise RuntimeError("FakeSLM is not started")
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def _next(self) -> Reply:
        """Pop the next scripted reply, repeating the last one forever.

        Returns:
            The reply to send.
        """
        if not self.replies:
            return Reply(json.dumps(VALID_OUTPUT))
        reply = self.replies[min(self._index, len(self.replies) - 1)]
        self._index += 1
        return reply

    def __enter__(self) -> FakeSLM:
        """Start the server on an ephemeral port.

        Returns:
            Self.
        """
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                """Silence the default stderr access log."""

            def _send(self, status: int, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                """Answer the two reachability probes."""
                if self.path.rstrip("/").endswith(("/models", "/api/tags")):
                    self._send(200, b'{"data": [], "models": []}')
                else:
                    self._send(404, b"{}")

            def do_POST(self) -> None:
                """Answer a chat completion with the next scripted reply."""
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                try:
                    request = json.loads(raw or b"{}")
                except ValueError:  # pragma: no cover - malformed client
                    request = {}
                fake.requests.append({"path": self.path, "json": request})

                reply = fake._next()
                if reply.delay_s and fake._stop.wait(reply.delay_s):
                    # Shutting down; the client has long since timed out.
                    self.close_connection = True
                    return
                if reply.status != 200:
                    self._send(reply.status, b'{"error": "scripted failure"}')
                    return
                if reply.raw is not None:
                    self._send(200, reply.raw)
                    return
                if self.path.startswith("/api/"):
                    envelope: dict[str, Any] = {
                        "message": {"role": "assistant", "content": reply.body},
                        "done": True,
                        "prompt_eval_count": 11,
                        "eval_count": 22,
                    }
                else:
                    envelope = {
                        "choices": [{"message": {"role": "assistant", "content": reply.body}}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 22},
                    }
                self._send(200, json.dumps(envelope).encode("utf-8"))

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # Not daemon threads: `server_close()` then joins the handlers instead of
        # abandoning them, so their sockets are closed before the test ends. An
        # abandoned handler leaks a socket and surfaces as a ResourceWarning that
        # fails whichever test runs next.
        self._server.daemon_threads = False
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop the server, releasing any handler still inside a scripted delay."""
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def replies(*bodies: str) -> Iterator[Reply]:
    """Build a reply sequence from message bodies.

    Args:
        bodies: Assistant message contents, in order.

    Yields:
        One `Reply` per body.
    """
    for body in bodies:
        yield Reply(body)
