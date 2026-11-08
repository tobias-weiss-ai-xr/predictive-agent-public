"""Regression tests for Docker-native remediation (container_restart action).

These lock in the fixes that make restart-in-Docker-mode actually work:

  * ``DockerSocketClient.request`` must decode ``Transfer-Encoding: chunked``
    response bodies. Docker's ``/containers/json`` list endpoint uses chunked
    encoding (no ``Content-Length``), so before the fix the body came back
    empty and container lookup always failed.
  * ``ContainerRestartAction._get_container_id`` must match the *bare* container
    name, because the reconcile loop passes the pod key as ``ns/name``
    (e.g. ``docker/predagent_sick``).
  * ``discover_docker_containers`` must read ``RestartCount`` from the top
    level of ``docker inspect`` (it is ``null`` under ``State`` in modern
    Docker), otherwise crash-looping containers look healthy.
"""

import json
import socket
import threading
import uuid

import pytest

from predictive_agent.actions.container_restart import (
    ContainerRestartAction,
    DockerSocketClient,
)
from predictive_agent.remediator import ActionContext


def _chunked_body(payload: str) -> bytes:
    """Encode a string as a single HTTP chunk followed by the terminator."""
    data = payload.encode("utf-8")
    return b"%x\r\n" % len(data) + data + b"\r\n0\r\n\r\n"


class _ChunkedUnixServer:
    """Tiny HTTP/1.1 server over a unix socket that returns chunked bodies.

    Used to exercise the real ``DockerSocketClient`` chunked decoder without a
    Docker daemon.
    """

    def __init__(self, handler):
        self._handler = handler
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.path = f"/tmp/pa-chunk-{uuid.uuid4().hex[:8]}.sock"
        self.sock.bind(self.path)
        self.sock.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            try:
                buf = b""
                while b"\r\n\r\n" not in buf:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                first_line = buf.split(b"\r\n", 1)[0].decode()
                method, path, _ = first_line.split(" ", 2)
                status, headers, body = self._handler(method, path)
                resp = f"HTTP/1.1 {status}\r\n".encode()
                for key, value in headers.items():
                    resp += f"{key}: {value}\r\n".encode()
                resp += b"\r\n"
                conn.sendall(resp + body)
            finally:
                conn.close()

    def stop(self):
        try:
            self.sock.close()
        except OSError:
            pass


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"),
                    reason="AF_UNIX sockets unavailable (e.g. Windows)")
def test_docker_socket_client_decodes_chunked():
    """The real client must parse Transfer-Encoding: chunked bodies."""
    containers = [
        {"Id": "abc123", "Names": ["/predagent_sick"]},
        {"Id": "def456", "Names": ["/predagent-demo-nginx"]},
    ]

    def handler(method, path):
        if method == "GET" and path.startswith("/containers/json"):
            return (
                200,
                {"Transfer-Encoding": "chunked", "Content-Type": "application/json"},
                _chunked_body(json.dumps(containers)),
            )
        return 404, {}, b""

    server = _ChunkedUnixServer(handler)
    try:
        client = DockerSocketClient(server.path)
        status, headers, body = client.request("GET", "/containers/json?all=1")
        assert status == 200
        assert headers.get("Transfer-Encoding") == "chunked"
        assert json.loads(body) == containers
    finally:
        server.stop()


class _FakeChunkedClient:
    """Mock DockerSocketClient.

    Returns the *decoded* container list (as the real DockerSocketClient does
    after chunk decoding) so the action logic under test sees a normal body.
    The chunked decoder itself is exercised separately against a real unix
    socket in test_docker_socket_client_decodes_chunked.
    """

    def __init__(self, containers, restarted):
        self._containers = containers
        self._restarted = restarted

    def request(self, method, path, data=None, headers=None, timeout=30):
        if method == "GET" and path.startswith("/containers/json"):
            return 200, {}, json.dumps(self._containers)
        if method == "POST" and "/restart" in path:
            self._restarted.append(path)
            return 204, {}, ""
        return 404, {}, ""


def test_container_restart_resolves_bare_name_from_pod_key():
    """pod_key 'docker/predagent_sick' must resolve to the bare container."""
    containers = [{"Id": "abc123", "Names": ["/predagent_sick"]}]
    action = ContainerRestartAction(_FakeChunkedClient(containers, []))
    assert action._get_container_id("docker/predagent_sick", "docker") == "abc123"


def test_container_restart_executes_on_disposable_container():
    """A real restart is issued against the resolved container id."""
    containers = [{"Id": "abc123", "Names": ["/predagent_sick"]}]
    restarted = []
    action = ContainerRestartAction(_FakeChunkedClient(containers, restarted))
    ctx = ActionContext(namespace="docker", dry_run=False)
    result = action.execute("docker/predagent_sick", ctx)
    assert result.success is True
    assert restarted and "/containers/abc123/restart" in restarted[0]


def test_discover_docker_containers_reads_top_level_restart_count(monkeypatch):
    """RestartCount lives at the top level of `docker inspect`, not State."""
    from predictive_agent import collector

    ps_line = json.dumps({
        "ID": "abc123",
        "Names": "predagent_sick",
        "Image": "alpine",
        "Status": "Restarting",
        "Labels": "",
    })
    monkeypatch.setattr(
        collector,
        "run_cmd",
        lambda cmd, timeout=10: (0, ps_line + "\n", "") if "ps" in cmd else (0, "{}", ""),
    )
    monkeypatch.setattr(
        collector,
        "_get_container_inspect",
        lambda cid: {
            "Name": "/predagent_sick",
            "State": {"Restarting": True, "RestartCount": None, "Status": "restarting"},
            "Config": {"Image": "alpine"},
            "RestartCount": 11,  # modern Docker: top-level, State.RestartCount is None
        },
    )
    monkeypatch.setenv("OPERATOR_RUNTIME", "docker")
    monkeypatch.setattr(collector, "_RUNTIME_CACHE", None)

    result = collector.discover_docker_containers()
    assert "abc123" in result
    assert result["abc123"]["restart_count"] == 11
    assert result["abc123"]["status"] == "restarting"
