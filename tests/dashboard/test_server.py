"""The HTTP transport: routing, headers, path safety over the wire, and SSE.

`server.py` is deliberately thin -- everything with a decision in it lives in
`api.py` or `watcher.py` and is tested there. What is tested here is what only
appears once a socket is involved: status codes, `Cache-Control`, port fallback,
and the stream's ordering / heartbeat / desync behaviour.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos.dashboard.server import DashboardServer


def _write_run(root: Path, scenario: str, run_id: str, *, finished: bool = True) -> Path:
    run = root / scenario / run_id
    run.mkdir(parents=True)
    lines = [{"seq": 1, "kind": "run_started", "layer": "engine"}]
    if finished:
        lines.append({"seq": 2, "kind": "run_finished", "layer": "engine"})
    (run / "trace.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (run / "plan.json").write_text(json.dumps({"seed": 1337, "faults": []}))
    if finished:
        (run / "report.json").write_text(
            json.dumps({"run_id": run_id, "scenario_id": scenario, "success": True})
        )
    (run / "payloads").mkdir()
    (run / "payloads" / "secret.json").write_text('{"api_key": "sk-nope"}')
    return run


@pytest.fixture
def chaos(tmp_path: Path) -> Path:
    root = tmp_path / ".chaos"
    _write_run(root, "tool.drop_key", "run-done")
    (root / "suite.json").write_text(
        json.dumps({"schema_version": "1.1", "status": "running", "planned": ["a"]})
    )
    return root


@pytest.fixture
def server(chaos: Path, loopback: None) -> Any:
    srv = DashboardServer(chaos, port=0, poll_ms=25)
    srv.start()
    yield srv
    srv.stop()


def _get(url: str) -> tuple[int, bytes, dict[str, str]]:
    try:
        with urllib.request.urlopen(url, timeout=5) as fh:
            return fh.status, fh.read(), dict(fh.headers)
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, exc.read(), dict(exc.headers)


class TestRouting:
    def test_it_serves_the_page_at_root(self, server: Any) -> None:
        status, body, headers = _get(server.url + "/")
        assert status == 200
        assert b"<title>" in body
        assert headers["Content-Type"].startswith("text/html")

    def test_the_page_has_no_external_asset(self, server: Any) -> None:
        _, body, _ = _get(server.url + "/")
        text = body.decode()
        assert "http://" not in text and "https://" not in text

    @pytest.mark.parametrize(
        "path", ["/api/suite", "/api/runs", "/api/health", "/api/run/run-done"]
    )
    def test_every_response_is_no_store(self, server: Any, path: str) -> None:
        status, _, headers = _get(server.url + path)
        assert status == 200
        assert headers["Cache-Control"] == "no-store"

    def test_it_serves_the_run_list(self, server: Any) -> None:
        _, body, _ = _get(server.url + "/api/runs")
        rows = json.loads(body)
        assert [r["run_id"] for r in rows] == ["run-done"]

    def test_it_serves_a_page_of_events(self, server: Any) -> None:
        _, body, _ = _get(server.url + "/api/run/run-done/events?from=0&limit=1")
        page = json.loads(body)
        assert [e["seq"] for e in page["events"]] == [1]
        assert page["has_more"] is True
        assert page["next_from"] == 2

    def test_it_serves_an_allow_listed_artifact(self, server: Any) -> None:
        status, body, headers = _get(server.url + "/api/run/run-done/artifact/plan.json")
        assert status == 200
        assert json.loads(body)["seed"] == 1337
        assert headers["Content-Type"] == "application/json"

    def test_an_unknown_path_is_404(self, server: Any) -> None:
        assert _get(server.url + "/api/nope")[0] == 404

    def test_a_write_method_is_refused(self, server: Any) -> None:
        req = urllib.request.Request(server.url + "/api/suite", method="POST", data=b"{}")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        with exc.value:
            assert exc.value.code == 405


class TestPathSafety:
    """`docs/10` §5: the three attack shapes, over the wire."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/run/run-done/artifact/../../../etc/passwd",
            "/api/run/run-done/artifact/%2e%2e%2f%2e%2e%2fsuite.json",
            "/api/run/run-done/artifact/payloads/secret.json",
            "/api/run/run-done/artifact/payloads%2Fsecret.json",
            "/api/run/..%2f..%2fetc/artifact/report.json",
            "/api/run/../../etc/passwd",
        ],
    )
    def test_it_refuses_the_attack_shapes(self, server: Any, path: str) -> None:
        status, body, _ = _get(server.url + path)
        assert status in (400, 404)
        assert b"sk-nope" not in body and b"root:" not in body

    def test_a_symlink_escape_is_refused(self, server: Any, chaos: Path, tmp_path: Path) -> None:
        secret = tmp_path / "outside.json"
        secret.write_text('{"api_key": "sk-nope"}')
        (chaos / "tool.drop_key" / "run-done" / "judge.json").symlink_to(secret)
        status, body, _ = _get(server.url + "/api/run/run-done/artifact/judge.json")
        assert status == 404
        assert b"sk-nope" not in body


class TestStream:
    def test_frames_arrive_in_order(self, server: Any, chaos: Path) -> None:
        frames = _stream(server, count=2, then=lambda: _append(chaos, [3, 4]))
        assert [f.get("seq") for f in frames if f.get("seq")] == [3, 4]

    def test_a_heartbeat_keeps_the_connection(self, server: Any) -> None:
        with urllib.request.urlopen(server.url + "/api/stream?heartbeat_s=0.1", timeout=5) as fh:
            assert fh.headers["Content-Type"].startswith("text/event-stream")
            assert fh.headers["X-Accel-Buffering"] == "no"
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if fh.readline().startswith(b":"):
                    return
        pytest.fail("no heartbeat within 5s")

    def test_a_slow_client_is_dropped_with_desync(self, chaos: Path, loopback: None) -> None:
        srv = DashboardServer(chaos, port=0, poll_ms=25, client_queue=4)
        srv.start()
        try:
            with urllib.request.urlopen(srv.url + "/api/stream", timeout=10) as fh:
                fh.readline()  # the hello frame, then stop reading: the queue fills
                _append(chaos, list(range(3, 90)))
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    line = fh.readline()
                    if b"desync" in line:
                        return
                    if not line:
                        break
            pytest.fail("no desync frame")
        finally:
            srv.stop()

    def test_no_sse_refuses_the_stream(self, chaos: Path, loopback: None) -> None:
        srv = DashboardServer(chaos, port=0, poll_ms=25, sse=False)
        srv.start()
        try:
            assert _get(srv.url + "/api/stream")[0] == 404
        finally:
            srv.stop()

    def test_polling_returns_what_the_stream_would_have(self, server: Any, chaos: Path) -> None:
        _append(chaos, [3, 4])
        time.sleep(0.3)
        _, body, _ = _get(server.url + "/api/run/run-done/events?from=0")
        assert [e["seq"] for e in json.loads(body)["events"]] == [1, 2, 3, 4]


class TestBinding:
    def test_it_falls_back_to_the_next_free_port(self, chaos: Path, loopback: None) -> None:
        taken = socket.socket()
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        srv = DashboardServer(chaos, port=port, poll_ms=25)
        try:
            srv.start()
            assert srv.port != port
        finally:
            srv.stop()
            taken.close()

    def test_an_explicit_port_fails_rather_than_wandering(
        self, chaos: Path, loopback: None
    ) -> None:
        taken = socket.socket()
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        srv = DashboardServer(chaos, port=port, poll_ms=25, strict_port=True)
        try:
            with pytest.raises(OSError):
                srv.start()
        finally:
            taken.close()


def _append(chaos: Path, seqs: list[int]) -> None:
    trace = chaos / "tool.drop_key" / "run-done" / "trace.jsonl"
    with trace.open("a", encoding="utf-8") as fh:
        for seq in seqs:
            fh.write(json.dumps({"seq": seq, "kind": "tool_call", "layer": "tool"}) + "\n")
            fh.flush()


def _stream(server: Any, *, count: int, then: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with urllib.request.urlopen(server.url + "/api/stream", timeout=10) as fh:
        threading.Timer(0.2, then).start()
        deadline = time.monotonic() + 10
        while len(out) < count and time.monotonic() < deadline:
            line = fh.readline()
            if line.startswith(b"data: "):
                frame = json.loads(line[6:])
                if frame.get("kind") == "trace":
                    out.append(frame["event"])
    return out


class TestIsolation:
    """`docs/10` §1: the dashboard can never take a run with it."""

    def test_a_poll_that_raises_does_not_kill_the_server(
        self, chaos: Path, loopback: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv = DashboardServer(chaos, port=0, poll_ms=25)
        boom = {"n": 0}

        def explode() -> list[dict[str, Any]]:
            boom["n"] += 1
            raise RuntimeError("watcher exploded")

        srv.start()
        try:
            monkeypatch.setattr(srv.watcher, "poll", explode)
            deadline = time.monotonic() + 5
            while boom["n"] < 3 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert boom["n"] >= 3, "the poll thread died on the first exception"
            assert _get(srv.url + "/api/health")[0] == 200
        finally:
            srv.stop()

    def test_the_dashboard_imports_nothing_third_party(self) -> None:
        import ast
        import sys

        allowed = set(sys.stdlib_module_names) | {"agent_loop_chaos"}
        pkg = Path(__file__).resolve().parents[2] / "src" / "agent_loop_chaos" / "dashboard"
        for module in sorted(pkg.glob("*.py")):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name.split(".")[0] in allowed, f"{module.name}: {alias.name}"
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    assert node.module.split(".")[0] in allowed, f"{module.name}: {node.module}"


class TestStreamMatchesPolling:
    def test_both_paths_yield_the_same_events(self, chaos: Path, loopback: None) -> None:
        srv = DashboardServer(chaos, port=0, poll_ms=25)
        srv.start()
        try:
            streamed = _stream(srv, count=3, then=lambda: _append(chaos, [3, 4, 5]))
            with urllib.request.urlopen(srv.url + "/api/run/run-done/events?from=3") as fh:
                polled = json.loads(fh.read())["events"]
            assert [e["seq"] for e in streamed] == [e["seq"] for e in polled] == [3, 4, 5]
        finally:
            srv.stop()
