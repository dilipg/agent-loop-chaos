"""A stdlib HTTP transport over `watcher.py` and `api.py`.

`ThreadingHTTPServer`, `json`, `pathlib`, `threading`. No framework, no build step,
no asset fetched from anywhere. Everything with a decision in it lives in the two
modules underneath; this file only turns a request line into one of their calls.

Read-only, in the strong sense: there is no handler for a method other than `GET`,
and nothing here opens a file for writing.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from . import api
from .watcher import RunDirWatcher

__all__ = ["DashboardServer"]

log = logging.getLogger("agent_loop_chaos")

_STATIC = Path(__file__).parent / "static" / "index.html"
_PORT_TRIES = 10


@dataclass(slots=True, eq=False)
class _Client:
    """One SSE subscriber.

    Attributes:
        q: Its bounded frame queue.
        desynced: Set once it fell too far behind; nothing more is queued after.
    """

    q: queue.Queue[dict[str, Any]] = field(default_factory=queue.Queue)
    desynced: bool = False


class _Hub:
    """Fan-out to SSE clients, with a bound per client.

    A viewer that stops reading must not be able to grow the server's memory without
    limit, so a client that falls behind is dropped with a `desync` frame telling the
    page to re-fetch from `next_from` (`docs/10` §5).
    """

    def __init__(self, maxsize: int) -> None:
        """Initialise.

        Args:
            maxsize: Frames a client may fall behind by before it is dropped.
        """
        self._maxsize = maxsize
        self._clients: set[_Client] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> _Client:
        """Register a client.

        Returns:
            The new client.
        """
        client = _Client(q=queue.Queue(maxsize=self._maxsize))
        with self._lock:
            self._clients.add(client)
        return client

    def unsubscribe(self, client: _Client) -> None:
        """Deregister a client.

        Args:
            client: The client to drop.
        """
        with self._lock:
            self._clients.discard(client)

    def publish(self, frames: list[dict[str, Any]]) -> None:
        """Hand frames to every live client.

        Args:
            frames: The frames from one poll.
        """
        if not frames:
            return
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            if client.desynced:
                continue
            for frame in frames:
                try:
                    client.q.put_nowait(frame)
                except queue.Full:
                    client.desynced = True
                    _drain(client.q)
                    client.q.put_nowait({"kind": "desync"})
                    break


def _drain(q: queue.Queue[dict[str, Any]]) -> None:
    """Empty a queue without blocking.

    Args:
        q: The queue.
    """
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


class DashboardServer:
    """Serves a `.chaos` directory, live.

    The engine does not know this exists. Nothing here can slow a run down or fail
    one: the server only ever reads files that are already on disk.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 7717,
        poll_ms: int = 250,
        max_events: int = 20_000,
        sse: bool = True,
        strict_port: bool = False,
        client_queue: int = 5_000,
    ) -> None:
        """Initialise.

        Args:
            out_dir: The `.chaos` directory to serve.
            host: Bind address. Anything but loopback is the operator's call.
            port: Preferred port; `0` picks a free one.
            poll_ms: Filesystem poll cadence.
            max_events: Per-run in-memory retention.
            sse: Whether `/api/stream` is available.
            strict_port: Fail rather than trying the next 10 ports.
            client_queue: Frames a stream client may fall behind by.
        """
        self.watcher = RunDirWatcher(out_dir, max_events=max_events)
        self.host = host
        self.port = port
        self.poll_s = max(poll_ms, 1) / 1000
        self.sse = sse
        self.strict_port = strict_port or port == 0
        self.hub = _Hub(client_queue)
        self._httpd: ThreadingHTTPServer | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._suite_mtime: float | None = None

    @property
    def url(self) -> str:
        """Where the page is.

        Returns:
            The base URL, with the port actually bound.
        """
        host = "127.0.0.1" if self.host in ("", "0.0.0.0") else self.host
        return f"http://{host}:{self.port}"

    def start(self) -> None:
        """Bind, then serve and poll in daemon threads.

        Raises:
            OSError: The port was taken and `--port` was explicit.
        """
        self._httpd = self._bind()
        self.port = self._httpd.server_address[1]
        self.watcher.poll()
        for target in (self._httpd.serve_forever, self._poll_forever):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        """Stop serving. Safe to call twice."""
        self._stop.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        for thread in self._threads:
            thread.join(timeout=2)
        self._threads.clear()

    def _bind(self) -> ThreadingHTTPServer:
        """Bind the listening socket, wandering unless told not to.

        Returns:
            The bound server.

        Raises:
            OSError: Nothing free, or the port was explicit and taken.
        """
        handler = _make_handler(self)
        last: OSError | None = None
        tries = 1 if self.strict_port else _PORT_TRIES
        for offset in range(tries):
            try:
                return ThreadingHTTPServer((self.host, self.port + offset), handler)
            except OSError as exc:
                last = exc
        raise last  # type: ignore[misc]  # tries >= 1, so last is set

    def _poll_forever(self) -> None:
        """Poll the filesystem and publish what changed."""
        while not self._stop.is_set():
            try:
                frames = self.watcher.poll() + self._suite_frames()
                self.hub.publish(frames)
            except Exception:  # pragma: no cover - the dashboard never dies
                log.debug("dashboard poll failed", exc_info=True)
            self._stop.wait(self.poll_s)

    def _suite_frames(self) -> list[dict[str, Any]]:
        """Notice `suite.json` changing.

        Returns:
            A `suite` frame when the file's mtime moved, else nothing.
        """
        path = self.watcher.out_dir / "suite.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return []
        if mtime == self._suite_mtime:
            return []
        self._suite_mtime = mtime
        return [{"kind": "suite", "suite": api.suite(self.watcher)}]


def _make_handler(server: DashboardServer) -> type[BaseHTTPRequestHandler]:
    """Build the request handler bound to one server.

    Args:
        server: The owning server.

    Returns:
        A handler class.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "alc-dashboard"

        def log_message(self, fmt: str, *args: Any) -> None:
            """Route access logs to the library logger instead of stderr.

            Args:
                fmt: Format string.
                *args: Its arguments.
            """
            log.debug("dashboard %s", fmt % args)

        # do_GET is BaseHTTPRequestHandler's dispatch name, not ours to choose.
        def do_GET(self) -> None:
            """Route one request."""
            parsed = urlparse(self.path)
            # The path is split *before* unquoting, so a %2F cannot invent a segment
            # boundary or erase one. Segments are then compared literally.
            parts = [unquote(p) for p in parsed.path.split("/") if p]
            query = parse_qs(parsed.query)
            try:
                self._route(parts, query)
            except BrokenPipeError:  # pragma: no cover - the viewer closed the tab
                pass
            except Exception:  # pragma: no cover - never take the server down
                log.debug("dashboard request failed", exc_info=True)
                self._send(500, b'{"error":"internal"}', "application/json")

        def _route(self, parts: list[str], query: dict[str, list[str]]) -> None:
            """Dispatch to an endpoint.

            Args:
                parts: Decoded path segments.
                query: Parsed query string.
            """
            w = server.watcher
            if not parts:
                return self._static()
            if parts == ["api", "suite"]:
                return self._json(api.suite(w))
            if parts == ["api", "runs"]:
                return self._json(api.runs(w))
            if parts == ["api", "tasks.md"]:
                body, ctype = api.tasks(w)
                return self._send(200, body.encode(), ctype, filename="chaos-findings.md")
            if parts == ["api", "glossary"]:
                return self._json(api.glossary(w))
            if parts == ["api", "health"]:
                return self._json(api.health(w))
            if parts == ["api", "stream"]:
                return self._stream(query)
            if len(parts) >= 3 and parts[:2] == ["api", "run"]:
                return self._run(parts[2], parts[3:], query)
            return self._send(404, b'{"error":"not found"}', "application/json")

        def _run(self, run_id: str, rest: list[str], query: dict[str, list[str]]) -> None:
            """Handle the `/api/run/<id>...` family.

            Args:
                run_id: The requested run.
                rest: Remaining segments.
                query: Parsed query string.
            """
            if not rest:
                found = api.run(server.watcher, run_id)
                if found is None:
                    return self._send(404, b'{"error":"no such run"}', "application/json")
                return self._json(found)
            if rest == ["events"]:
                return self._json(
                    api.events(
                        server.watcher,
                        run_id,
                        from_seq=_int(query, "from", 0),
                        limit=_int(query, "limit", 500),
                        kinds=_csv(query, "kinds"),
                        layers=_csv(query, "layers"),
                    )
                )
            if len(rest) == 2 and rest[0] == "artifact":
                body, ctype = api.artifact(server.watcher, run_id, rest[1])
                if body is None or ctype is None:
                    return self._send(404, b'{"error":"no such artifact"}', "application/json")
                # The filename comes from the allow-listed name, never from the query:
                # the flag only decides *whether* to attach, never what to call it.
                name = rest[1] if "download" in query else None
                return self._send(200, body, ctype, filename=name)
            return self._send(404, b'{"error":"not found"}', "application/json")

        def _static(self) -> None:
            """Serve the single page."""
            try:
                body = _STATIC.read_bytes()
            except OSError:  # pragma: no cover - packaging failure
                return self._send(500, b"index.html missing", "text/plain; charset=utf-8")
            self._send(200, body, "text/html; charset=utf-8")

        def _stream(self, query: dict[str, list[str]]) -> None:
            """Hold an SSE connection open until the client goes away.

            Args:
                query: Parsed query string; `heartbeat_s` is for tests.
            """
            if not server.sse:
                return self._send(404, b'{"error":"sse disabled"}', "application/json")
            heartbeat = float(query.get("heartbeat_s", ["15"])[0])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            client = server.hub.subscribe()
            try:
                self._frame({"kind": "hello", "suite": api.suite(server.watcher)})
                while not server._stop.is_set():
                    try:
                        frame = client.q.get(timeout=heartbeat)
                    except queue.Empty:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        continue
                    self._frame(frame)
                    if frame.get("kind") == "desync":
                        return
            except Exception:
                # The viewer closed the tab, or the socket went away. Either way the
                # stream is over; it is never a reason to take the server down.
                log.debug("dashboard stream ended", exc_info=True)
            finally:
                server.hub.unsubscribe(client)

        def _frame(self, frame: dict[str, Any]) -> None:
            """Write one SSE frame.

            Args:
                frame: The frame.
            """
            payload = json.dumps(frame, separators=(",", ":"), default=str)
            self.wfile.write(f"event: {frame['kind']}\ndata: {payload}\n\n".encode())
            self.wfile.flush()

        def _json(self, document: Any) -> None:
            """Send a JSON body.

            Args:
                document: What to serialize.
            """
            body = json.dumps(document, default=str).encode()
            self._send(200, body, "application/json")

        def _send(self, status: int, body: bytes, ctype: str, filename: str | None = None) -> None:
            """Send a complete response.

            Args:
                status: HTTP status.
                body: The bytes.
                ctype: Content type.
                filename: When given, offer the body as a download under this name.
                    Callers pass a name from the artifact allow-list; nothing from the
                    request ever reaches this header.
            """
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _refuse(self) -> None:
            """Refuse a mutating method. Read-only means read-only."""
            self._send(405, b'{"error":"read-only"}', "application/json")

        do_POST = do_PUT = do_DELETE = do_PATCH = _refuse  # noqa: N815

    return Handler


def _int(query: dict[str, list[str]], key: str, default: int) -> int:
    """Read one integer query parameter.

    Args:
        query: Parsed query string.
        key: Parameter name.
        default: Value when absent or unparseable.

    Returns:
        The integer.
    """
    try:
        return int(query[key][0])
    except (KeyError, IndexError, ValueError):
        return default


def _csv(query: dict[str, list[str]], key: str) -> list[str] | None:
    """Read one comma-separated query parameter.

    Args:
        query: Parsed query string.
        key: Parameter name.

    Returns:
        The values, or `None` when absent.
    """
    raw = query.get(key, [""])[0]
    values = [v for v in raw.split(",") if v]
    return values or None
