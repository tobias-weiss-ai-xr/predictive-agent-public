"""Second round of edge tests for the container_restart action (PA-S3).

test_container_restart.py, test_container_restart_edges.py and
test_docker_remediation.py already cover the CLI/socket happy paths, the
Content-Length and restart-count-window edges, cooldown/rate-limit gates and
bare-name resolution. This file targets the remaining UNCOVERED lines of
predictive_agent/actions/container_restart.py (from the PA-S3 coverage sweep):

  * ``DockerSocketClient._get_connection``  — socket-not-found / permission-
    denied / connect-OSError branches (lines 65, 72-77).
  * ``DockerSocketClient.request``           — custom headers, request body
    serialisation, and the close-error swallow in the ``finally`` (118,
    122-123, 136, 226-227).
  * chunked decoder                          — empty size line skipped, non-hex
    size line terminating the decode, and EOF inside a chunk body (186,
    189-190, 198).
  * ``ContainerRestartAction``               — ``socket_path`` constructor
    branch, non-string ``state``/``status`` on pod_state, client-less
    container lookup, compose-style name matching, socket dry-run, and the
    socket restart error handlers: FileNotFoundError / PermissionError /
    generic (258, 292-294, 302-304, 326, 359-367, 382, 408, 443-462).

All docker socket / CLI interactions use fakes or a local AF_UNIX test
server — nothing ever touches a real docker daemon; no sleeps; no network.
"""

import json
import os
import socket
import tempfile
import threading
import uuid
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from predictive_agent.actions import container_restart as container_restart_module
from predictive_agent.actions.container_restart import (
    ContainerRestartAction,
    DockerSocketClient,
)
from predictive_agent.remediator import ActionContext, RemediationResult

# ── shared helpers / fixtures (pattern reused from test_container_restart_edges.py) ──


def _make_pod_state(
    name: str = "predagent_sick",
    namespace: str = "monitoring",
    restart_count: int = 0,
    running: bool = True,
    state: str = "Running",
    status: str = "running",
) -> MagicMock:
    """Build a mock pod/container state object with the attrs the action reads."""
    mock = MagicMock()
    mock.name = name
    mock.namespace = namespace
    mock.restart_count = restart_count
    mock.running = running
    mock.state = state
    mock.status = status
    return mock


@pytest.fixture(autouse=True)
def _reset_module_globals():
    """Reset container_restart module globals before/after every test
    (RISK_THRESHOLD / REMEDIATION_PROTECTED_NS are env-derived at import)."""
    saved_threshold = container_restart_module.RISK_THRESHOLD
    saved_ns = set(container_restart_module.REMEDIATION_PROTECTED_NS)
    yield
    container_restart_module.RISK_THRESHOLD = saved_threshold
    container_restart_module.REMEDIATION_PROTECTED_NS.clear()
    container_restart_module.REMEDIATION_PROTECTED_NS.update(saved_ns)


class _RawUnixServer:
    """Tiny raw HTTP/1.1 server over a unix socket with a scripted verbatim
    response (reused from test_container_restart_edges.py)."""

    def __init__(self, response_bytes: bytes = b"") -> None:
        self._response_bytes = response_bytes
        self.requests: List[bytes] = []
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.path = f"/tmp/pa-edges2-{uuid.uuid4().hex[:8]}.sock"
        self.sock.bind(self.path)
        self.sock.listen(4)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
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
                self.requests.append(buf)
                conn.sendall(self._response_bytes)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def stop(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class _FakeRestartClient:
    """Scripted DockerSocketClient stand-in for the socket execution path
    (reused from test_container_restart_edges.py)."""

    def __init__(
        self,
        containers: Optional[List[Dict]] = None,
        get_status: int = 200,
        restart_status: int = 204,
    ) -> None:
        self._containers = containers or []
        self._get_status = get_status
        self._restart_status = restart_status
        self.calls: List[tuple] = []

    def request(self, method: str, path: str, data: Optional[bytes] = None,
                headers: Optional[dict] = None, timeout: int = 30):
        self.calls.append((method, path))
        if method == "GET" and path.startswith("/containers/json"):
            return self._get_status, {}, json.dumps(self._containers)
        if method == "POST" and "/restart" in path:
            return self._restart_status, {}, ""
        return 404, {}, ""


class _RaisingRestartClient:
    """Socket stand-in whose container LISTING succeeds but whose restart POST
    raises a scripted exception (FileNotFoundError / PermissionError) — the
    mirror of _HangingClient in edges.py, used to drive the _execute_via_socket
    error handlers where _get_container_id would otherwise swallow the error."""

    CONTAINER_ID = "abc123"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls: List[tuple] = []

    def request(self, method: str, path: str, *args, **kwargs):
        self.calls.append((method, path))
        if method == "GET" and path.startswith("/containers/json"):
            return 200, {}, json.dumps(
                [{"Id": self.CONTAINER_ID, "Names": ["/predagent_sick"]}]
            )
        raise self._exc


class _FeedSocket:
    """Deterministic fake socket for driving the DockerSocketClient decoder
    through branches a raw server would hang on (EOF inside a chunk).

    ``recv`` returns up to ``n`` bytes from the scripted response buffer; once
    the buffer is exhausted it returns ``b""`` exactly once (simulating the
    peer closing), and every later ``recv`` raises TimeoutError so the decoder
    read loops terminate instead of spinning. ``close`` can be scripted to
    raise (close-swallow test).
    """

    def __init__(self, response: bytes, close_error: Optional[Exception] = None) -> None:
        self._data = bytearray(response)
        self._eof_delivered = False
        self._close_error = close_error
        self.timeout: Optional[float] = None

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def sendall(self, data: bytes) -> None:
        del data  # we only need to persist what the client reads back

    def recv(self, n: int) -> bytes:
        if self._data:
            out = bytes(self._data[:n])
            del self._data[:n]
            return out
        if not self._eof_delivered:
            self._eof_delivered = True
            return b""
        raise TimeoutError("connection closed (simulated EOF)")

    def close(self) -> None:
        if self._close_error is not None:
            raise self._close_error


class _FedClient(DockerSocketClient):
    """DockerSocketClient whose _get_connection returns a scripted feed socket
    instead of touching a real AF_UNIX socket — deterministic decoder tests."""

    def __init__(self, response: bytes, close_error: Optional[Exception] = None) -> None:
        super().__init__("/tmp/pa-edges2-feed.sock")
        self._feed = _FeedSocket(response, close_error)

    def _get_connection(self):
        return self._feed, "unix"


# ── 1. _get_connection input validation: missing / unreadable / refused ────


def test_get_connection_missing_socket_raises_file_not_found():
    """A Docker socket path that does not exist surfaces as FileNotFoundError
    with the path in the message (not a bare OSError)."""
    client = DockerSocketClient("/nonexistent/pa-s3-docker.sock")
    with pytest.raises(FileNotFoundError) as excinfo:
        client._get_connection()
    assert "Docker socket not found" in str(excinfo.value)
    assert "/nonexistent/pa-s3-docker.sock" in str(excinfo.value)


def test_get_connection_permission_denied_wraps_connect_error():
    """An existing socket the process may not connect to raises PermissionError
    with a descriptive message (lines 72-75)."""
    fd, path = tempfile.mkstemp(prefix="pa-s3-", suffix=".sock")
    os.close(fd)
    try:
        client = DockerSocketClient(path)
        with patch("socket.socket") as mock_sock_cls:
            mock_sock_cls.return_value.connect.side_effect = PermissionError("denied")
            with pytest.raises(PermissionError) as excinfo:
                client._get_connection()
        assert "Permission denied accessing Docker socket" in str(excinfo.value)
    finally:
        os.unlink(path)


def test_get_connection_other_os_error_wraps_message():
    """A generic connect failure (e.g. connection refused) is rewrapped with
    context as OSError — only PermissionError keeps its own type (lines 76-77)."""
    fd, path = tempfile.mkstemp(prefix="pa-s3-", suffix=".sock")
    os.close(fd)
    try:
        client = DockerSocketClient(path)
        with patch("socket.socket") as mock_sock_cls:
            mock_sock_cls.return_value.connect.side_effect = OSError("connection refused")
            with pytest.raises(OSError) as excinfo:
                client._get_connection()
        assert "Failed to connect to Docker socket" in str(excinfo.value)
        assert not isinstance(excinfo.value, PermissionError)
    finally:
        os.unlink(path)


# ── 2. request(): custom headers + body serialisation ──────────────────────


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"),
                    reason="AF_UNIX sockets unavailable (e.g. Windows)")
def test_request_forwards_custom_headers_and_body():
    """A request with a JSON body and extra headers must set Content-Type and
    Content-Length, forward the custom headers verbatim, and append the body
    (lines 117-118, 121-123, 135-136)."""
    payload = b'{"DesiredState": "restart"}'
    resp = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
    server = _RawUnixServer(response_bytes=resp)
    try:
        client = DockerSocketClient(server.path)
        status, headers, body = client.request(
            "POST",
            "/containers/create",
            data=payload,
            headers={"X-PA-Sweep": "edges2"},
        )
        assert status == 200
        assert body == "ok"
        req = server.requests[0]
        assert b"X-PA-Sweep: edges2\r\n" in req
        assert b"Content-Type: application/json\r\n" in req
        assert f"Content-Length: {len(payload)}".encode() in req
        assert b"\r\n\r\n" + payload in req or payload in req
    finally:
        server.stop()


# ── 3. chunked decoder: malformed / truncated streams ──────────────────────


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"),
                    reason="AF_UNIX sockets unavailable (e.g. Windows)")
def test_chunked_empty_size_line_is_skipped():
    """A spurious empty line between chunks is skipped, not fatal; the decoded
    body is the concatenation of the real chunks (line 186)."""
    resp = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"\r\n"          # empty size line -> skipped
        b"5\r\nhello\r\n"
        b"0\r\n\r\n"
    )
    server = _RawUnixServer(response_bytes=resp)
    try:
        client = DockerSocketClient(server.path)
        status, headers, body = client.request("GET", "/containers/json?all=1")
        assert status == 200
        assert body == "hello"
    finally:
        server.stop()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"),
                    reason="AF_UNIX sockets unavailable (e.g. Windows)")
def test_chunked_non_hex_size_line_stops_decoding():
    """A chunk size that is not hex stops the decode; body accumulated so far
    is preserved (lines 189-190)."""
    resp = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nhello\r\n"
        b"ZZZ\r\n"       # non-hex size -> decoder stops
        b"0\r\n\r\n"
    )
    server = _RawUnixServer(response_bytes=resp)
    try:
        client = DockerSocketClient(server.path)
        status, headers, body = client.request("GET", "/containers/json?all=1")
        assert status == 200
        assert body == "hello"
    finally:
        server.stop()


def test_chunked_eof_mid_chunk_terminates_with_timeout():
    """EOF inside a chunk body: the decoder records the partial chunk and
    terminates as TimeoutError instead of spinning forever on the closed
    socket (line 198)."""
    feed = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nhel"      # header says 5 bytes, only 3 delivered then EOF
    )
    client = _FedClient(feed)
    with pytest.raises(TimeoutError) as excinfo:
        client.request("GET", "/containers/json?all=1", timeout=30)
    assert "timed out" in str(excinfo.value)


def test_request_swallows_socket_close_error():
    """A failure while closing the socket must not propagate — the response is
    still returned (finally-block swallow, lines 226-227)."""
    resp = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
    client = _FedClient(resp, close_error=OSError("close boom"))
    status, headers, body = client.request("GET", "/containers/json?all=1")
    assert status == 200
    assert body == "ok"


# ── 4. ContainerRestartAction: constructor / lookup / dry-run edges ────────


def test_init_with_socket_path_creates_docker_client():
    """Passing only socket_path lazily builds a DockerSocketClient (line 258)."""
    action = ContainerRestartAction(socket_path="/tmp/pa-s3-docker.sock")
    assert isinstance(action.docker_client, DockerSocketClient)
    assert action.docker_client.socket_path == "/tmp/pa-s3-docker.sock"


def test_should_execute_tolerates_non_string_state_and_status():
    """A pod_state whose state/status attributes are non-strings (ints) must
    not raise — the AttributeError/TypeError guards keep is_running unchanged
    (lines 292-294, 302-304)."""
    pod = _make_pod_state(restart_count=0, running=True, state=123, status=456)
    action = ContainerRestartAction()
    assert action.should_execute(pod, None, 0.9) is False


def test_get_container_id_without_client_returns_none():
    """No DockerSocketClient injected: the lookup swallows the RuntimeError
    and returns None (lines 325-326 and the 366-367 catch-all)."""
    action = ContainerRestartAction()
    assert action._get_container_id("predagent_sick", "monitoring") is None


@pytest.mark.parametrize(
    "names,expected",
    [
        (["/monitoring_predagent_sick_1"], "c1"),            # ns_target_ prefix
        (["/monitoring-predagent_sick-1"], "c2"),            # ns-target- prefix
        (["/rotating_pre_predagent_sick_1"], "c3"),          # _target_ infix
        (["/abc-predagent_sick-def"], "c4"),                 # -target- infix
    ],
    ids=["underscore-prefix", "dash-prefix", "underscore-infix", "dash-infix"],
)
def test_get_container_id_matches_compose_style_names(names: List[str], expected: str):
    """Compose-style container names ({ns}_{svc}_{idx}) resolve through the
    namespace/target token matcher (lines 359-365)."""
    client = _FakeRestartClient(containers=[{"Id": expected, "Names": names}])
    action = ContainerRestartAction(client)
    assert action._get_container_id("predagent_sick", "monitoring") == expected


def test_execute_socket_dry_run_returns_result_without_calls():
    """dry_run on the socket path short-circuits before any daemon call and
    returns the synthetic 'Would restart' result (line 382)."""
    client = _FakeRestartClient(
        containers=[{"Id": "abc123", "Names": ["/predagent_sick"]}]
    )
    action = ContainerRestartAction(client)
    ctx = ActionContext(namespace="docker", dry_run=True)
    result = action.execute("predagent_sick", ctx)
    assert isinstance(result, RemediationResult)
    assert result.success is True
    assert result.dry_run is True
    assert "Would restart" in result.message
    assert result.command == "docker restart predagent_sick"
    assert client.calls == []


# ── 5. Socket restart error handlers (FileNotFoundError / PermissionError / generic) ──


def test_execute_via_socket_client_gone_returns_error_result():
    """If the injected client vanishes between lookup and restart, the
    RuntimeError surfaces as a generic 'Error restarting container' result,
    not a traceback (lines 406-408, 461-462)."""
    action = ContainerRestartAction()  # docker_client is None
    ctx = ActionContext(namespace="docker", dry_run=False)
    with patch.object(action, "_get_container_id", return_value="abc123"):
        result = action._execute_via_socket("predagent_sick", ctx)
    assert result.success is False
    assert "Error restarting container" in result.message


def test_execute_socket_file_not_found_during_restart():
    """The daemon socket disappears between listing and restart: reported as
    'Docker socket not available', the exception does not propagate (lines
    443-451)."""
    client = _RaisingRestartClient(FileNotFoundError("socket disappeared"))
    action = ContainerRestartAction(client)
    ctx = ActionContext(namespace="docker", dry_run=False)
    result = action.execute("predagent_sick", ctx)
    assert result.success is False
    assert "Docker socket not available" in result.message
    posts = [path for method, path in client.calls if method == "POST"]
    assert posts == ["/containers/abc123/restart?t=1"]


def test_execute_socket_permission_denied_during_restart():
    """Permission dropped between listing and restart: reported as
    'Permission denied for Docker socket' (lines 452-460)."""
    client = _RaisingRestartClient(PermissionError("access denied"))
    action = ContainerRestartAction(client)
    ctx = ActionContext(namespace="docker", dry_run=False)
    result = action.execute("predagent_sick", ctx)
    assert result.success is False
    assert "Permission denied for Docker socket" in result.message
