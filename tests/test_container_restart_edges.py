"""Edge-case tests for the container_restart action (PA-R6).

test_container_restart.py covers the CLI happy paths and should_execute
decision matrix; test_docker_remediation.py covers the chunked-body socket
decoder and bare-name resolution. This file adds ONLY the missing edge cases:

  1. Content-Length body loop against truncated / oversized / non-numeric /
     negative / malformed HTTP responses from the docker socket.
  2. Timeout enforcement on a hung docker socket call (real client + action).
  3. Restart-count window logic at the exact limit boundary (limit-1, limit,
     limit+1), asserted against the module constant.
  4. Rate-limit / cooldown rejection paths for *this* action as driven through
     the RemediationManager (real ContainerRestartAction, never executed).
  5. risk_tier propagation: the action's "safe" tier bypasses the P6 review
     gate at "moderate" and returns an acted result.
  6. docker CLI returning rc!=0 with empty/invalid output (fallback messages).
  7. Containers already exited / not-found during restart via the socket path.

All docker socket / CLI interactions use fakes or a local AF_UNIX test server —
nothing ever touches a real docker daemon. Module globals are reset via the
autouse fixture.
"""

import json
import socket
import threading
import time
import uuid
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from predictive_agent.actions import container_restart as container_restart_module
from predictive_agent.actions.container_restart import (
    MIN_RESTART_COUNT,
    ContainerRestartAction,
    DockerSocketClient,
)
from predictive_agent.remediator import ActionContext, RemediationResult

# ── shared helpers / fixtures ─────────────────────────────────────────────

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
    """Reset container_restart module globals before/after every test.

    RISK_THRESHOLD and REMEDIATION_PROTECTED_NS are computed at import time
    from the environment; tests that need to exercise a different threshold /
    protected set mutate the globals directly, so this fixture guarantees a
    clean state for every test (and for the rest of the suite).
    """
    saved_threshold = container_restart_module.RISK_THRESHOLD
    saved_ns = set(container_restart_module.REMEDIATION_PROTECTED_NS)
    yield
    container_restart_module.RISK_THRESHOLD = saved_threshold
    container_restart_module.REMEDIATION_PROTECTED_NS.clear()
    container_restart_module.REMEDIATION_PROTECTED_NS.update(saved_ns)


class _RawUnixServer:
    """Tiny raw HTTP/1.1 server over a unix socket with scripted responses.

    ``response_bytes`` is sent verbatim after the request is consumed.
    ``hang=True`` never responds (client must time out).
    ``hold_open_after_send`` sends the bytes but keeps the connection open
    (used to simulate a body shorter than Content-Length that never finishes).
    """

    def __init__(
        self,
        response_bytes: bytes = b"",
        hang: bool = False,
        hold_open_after_send: bool = False,
    ) -> None:
        self._response_bytes = response_bytes
        self._hang = hang
        self._hold_open = hold_open_after_send
        self.requests: List[bytes] = []
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.path = f"/tmp/pa-edges-{uuid.uuid4().hex[:8]}.sock"
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
                if self._hang:
                    # Never answer — the client must notice via its timeout.
                    time.sleep(5)
                else:
                    conn.sendall(self._response_bytes)
                    if not self._hold_open:
                        conn.close()
                    else:
                        time.sleep(5)
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
    """Scripted DockerSocketClient stand-in for the socket execution path."""

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


class _HangingClient:
    """DockerSocketClient stand-in: container listing works, the restart
    POST hangs (times out) — exercising the real TimeoutError handler in
    execute(), not the container-not-found fallback."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def request(self, method: str, path: str, *args, **kwargs):
        if method == "GET" and path.startswith("/containers/json"):
            body = json.dumps(
                [{"Id": "abc123def456", "Names": ["/predagent_sick"]}]
            ).encode()
            return 200, {"Content-Type": "application/json"}, body
        raise self._exc


# ── 1. Content-Length body loop: truncated / oversized / malformed ────────


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"),
                    reason="AF_UNIX sockets unavailable (e.g. Windows)")
def test_content_length_truncated_body_returns_partial():
    """Content-Length advertises 100 bytes but the daemon sends 10 and closes:
    the client must return the partial body instead of looping forever."""
    resp = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: 100\r\n\r\n"
        b"0123456789"
    )
    server = _RawUnixServer(response_bytes=resp)
    try:
        client = DockerSocketClient(server.path)
        status, headers, body = client.request("GET", "/containers/json?all=1")
        assert status == 200
        assert headers.get("Content-Length") == "100"
        # Loop terminated on EOF, partial body preserved.
        assert body == "0123456789"
        assert len(body) == 10
    finally:
        server.stop()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"),
                    reason="AF_UNIX sockets unavailable (e.g. Windows)")
def test_content_length_oversized_returns_exact_length():
    """Content-Length: 5 but the daemon sends 20 bytes: only the advertised
    length is consumed, the body buffer is not over-read."""
    resp = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: 5\r\n\r\n"
        b"0123456789ABCDEFGHIJ"
    )
    server = _RawUnixServer(response_bytes=resp)
    try:
        client = DockerSocketClient(server.path)
        status, headers, body = client.request("GET", "/containers/json?all=1")
        assert status == 200
        assert body == "01234"
        assert len(body) == 5
    finally:
        server.stop()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"),
                    reason="AF_UNIX sockets unavailable (e.g. Windows)")
def test_content_length_non_numeric_raises_connection_error():
    """A garbage Content-Length header must surface as ConnectionError, not an
    unhandled ValueError or a hang."""
    resp = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: abc\r\n\r\n"
        b"{}"
    )
    server = _RawUnixServer(response_bytes=resp)
    try:
        client = DockerSocketClient(server.path)
        with pytest.raises(ConnectionError) as excinfo:
            client.request("GET", "/containers/json?all=1")
        assert "Docker API request failed" in str(excinfo.value)
    finally:
        server.stop()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"),
                    reason="AF_UNIX sockets unavailable (e.g. Windows)")
def test_content_length_negative_returns_empty_body():
    """A malformed negative Content-Length must not loop; the body is empty."""
    resp = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: -1\r\n\r\n"
    )
    server = _RawUnixServer(response_bytes=resp)
    try:
        client = DockerSocketClient(server.path)
        status, headers, body = client.request("GET", "/containers/json?all=1")
        assert status == 200
        assert body == ""
    finally:
        server.stop()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"),
                    reason="AF_UNIX sockets unavailable (e.g. Windows)")
def test_malformed_status_line_raises_connection_error():
    """A response that is not a parseable HTTP status line must surface as a
    ConnectionError (ValueError is swallowed), not an unhandled traceback."""
    server = _RawUnixServer(response_bytes=b"GARBAGE\r\n\r\n")
    try:
        client = DockerSocketClient(server.path)
        with pytest.raises(ConnectionError) as excinfo:
            client.request("GET", "/containers/json?all=1")
        assert "Docker API request failed" in str(excinfo.value)
    finally:
        server.stop()


# ── 2. Timeout enforcement on a hung docker call ──────────────────────────


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"),
                    reason="AF_UNIX sockets unavailable (e.g. Windows)")
def test_hung_socket_raises_timeout_error():
    """A daemon that accepts but never answers must hit the request timeout."""
    server = _RawUnixServer(hang=True)
    try:
        client = DockerSocketClient(server.path)
        with pytest.raises(TimeoutError) as excinfo:
            client.request("GET", "/containers/json?all=1", timeout=0.5)
        assert "timed out" in str(excinfo.value)
    finally:
        server.stop()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"),
                    reason="AF_UNIX sockets unavailable (e.g. Windows)")
def test_partial_body_then_hang_raises_timeout_error():
    """Content-Length: 100, only 10 bytes arrive and the connection stays open:
    the body loop must be interrupted by the timeout instead of blocking
    forever."""
    resp = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: 100\r\n\r\n"
        b"0123456789"
    )
    server = _RawUnixServer(response_bytes=resp, hold_open_after_send=True)
    try:
        client = DockerSocketClient(server.path)
        with pytest.raises(TimeoutError) as excinfo:
            client.request("GET", "/containers/json?all=1", timeout=0.5)
        assert "timed out" in str(excinfo.value)
    finally:
        server.stop()


def test_execute_socket_timeout_returns_failed_result():
    """A hung restart call on the socket path surfaces as a failed
    RemediationResult with a 'timed out' message (never a hang)."""
    action = ContainerRestartAction(_HangingClient(TimeoutError("timed out after 30s")))
    ctx = ActionContext(namespace="docker", dry_run=False)
    result = action.execute("predagent_sick", ctx)
    assert isinstance(result, RemediationResult)
    assert result.success is False
    assert result.dry_run is False
    assert "timed out" in result.message
    assert result.command == "docker restart predagent_sick"


# ── 7. Containers already exited / not-found during restart ───────────────


def test_execute_not_found_when_container_missing():
    """The container is not in the daemon's list: restart fails with
    'not found' and no restart call is issued."""
    client = _FakeRestartClient(containers=[])
    action = ContainerRestartAction(client)
    ctx = ActionContext(namespace="docker", dry_run=False)
    result = action.execute("predagent_sick", ctx)
    assert result.success is False
    assert "not found" in result.message
    assert not any(m == "POST" for m, _ in client.calls)


def test_execute_already_exited_restart_404():
    """The container resolved but was already removed/gone by restart time:
    Docker answers 404 and the action reports the HTTP failure."""
    containers = [{"Id": "abc123", "Names": ["/predagent_sick"]}]
    client = _FakeRestartClient(containers=containers, restart_status=404)
    action = ContainerRestartAction(client)
    ctx = ActionContext(namespace="docker", dry_run=False)
    result = action.execute("predagent_sick", ctx)
    assert result.success is False
    assert "HTTP 404" in result.message
    assert any(m == "POST" and "/restart" in p for m, p in client.calls)


def test_execute_restart_http_500_failure():
    """A non-2xx restart response other than 404 (e.g. daemon error) is a
    failure and does not raise."""
    containers = [{"Id": "abc123", "Names": ["/predagent_sick"]}]
    client = _FakeRestartClient(containers=containers, restart_status=500)
    action = ContainerRestartAction(client)
    ctx = ActionContext(namespace="docker", dry_run=False)
    result = action.execute("predagent_sick", ctx)
    assert result.success is False
    assert "HTTP 500" in result.message


def test_execute_container_list_failure_returns_not_found():
    """Container listing itself fails (GET != 200 even after retry): treat as
    'not found' instead of raising."""
    client = _FakeRestartClient(containers=[{"Id": "abc123", "Names": ["/predagent_sick"]}],
                                get_status=500)
    action = ContainerRestartAction(client)
    ctx = ActionContext(namespace="docker", dry_run=False)
    result = action.execute("predagent_sick", ctx)
    assert result.success is False
    assert "not found" in result.message
    # Retried the list once when the first GET failed.
    gets = [p for m, p in client.calls if m == "GET"]
    assert len(gets) == 2


# ── 6. docker CLI rc != 0 / empty output ──────────────────────────────────


@patch("predictive_agent.actions.container_restart.subprocess.run")
def test_cli_nonzero_rc_with_empty_output_uses_fallback(mock_run):
    """rc != 0 with EMPTY stderr (real docker prints nothing on some errors):
    falls back to the generic failure message, not a blank message."""
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
    action = ContainerRestartAction()
    ctx = ActionContext(namespace="monitoring", dry_run=False)
    result = action.execute("missing-container", ctx)
    assert result.success is False
    assert result.message == "docker restart failed for missing-container"


@patch("predictive_agent.actions.container_restart.subprocess.run")
def test_cli_nonzero_rc_ignores_stdout(mock_run):
    """rc != 0: stdout content is ignored (only stderr is trusted), so the
    fallback failure message is used."""
    mock_run.return_value = MagicMock(returncode=2, stdout="restarted ok", stderr="")
    action = ContainerRestartAction()
    ctx = ActionContext(namespace="monitoring", dry_run=False)
    result = action.execute("app-1", ctx)
    assert result.success is False
    assert result.message == "docker restart failed for app-1"


@patch("predictive_agent.actions.container_restart.subprocess.run")
def test_cli_success_with_blank_stdout_uses_fallback(mock_run):
    """rc == 0 but stdout is only whitespace: falls back to the generic success
    message instead of an empty one."""
    mock_run.return_value = MagicMock(returncode=0, stdout="   \n\t", stderr="")
    action = ContainerRestartAction()
    ctx = ActionContext(namespace="monitoring", dry_run=False)
    result = action.execute("app-1", ctx)
    assert result.success is True
    assert result.message == "container app-1 restarted"


# ── 3. Restart-count window at the exact limit boundary ───────────────────


@pytest.mark.parametrize(
    "restart_count,expected",
    [
        (MIN_RESTART_COUNT - 1, False),  # limit - 1: below threshold
        (MIN_RESTART_COUNT, True),        # limit: exactly at threshold
        (MIN_RESTART_COUNT + 1, True),    # limit + 1: above threshold
    ],
    ids=["below-limit", "at-limit", "above-limit"],
)
def test_restart_count_limit_boundary(restart_count: int, expected: bool):
    """Restart-count window logic at exactly the limit boundary, asserted
    against the module constant (so it tracks MIN_RESTART_COUNT automatically)."""
    action = ContainerRestartAction()
    pod = _make_pod_state(restart_count=restart_count, running=True)
    assert action.should_execute(pod, None, 0.9) is expected


# ── 4. Rate-limit / cooldown rejection paths for this action ──────────────


def test_cooldown_rejects_container_restart_through_manager():
    """With a cooldown already active for container_restart:<target>, the real
    action driven through the RemediationManager is skipped — execute() is
    never reached (no docker CLI/socket needed)."""
    from predictive_agent.remediator import RemediationManager, SafetyPolicy

    action = ContainerRestartAction()
    manager = RemediationManager(
        dry_run=False,
        safety_policy=SafetyPolicy(cooldown_seconds=300,
                                   protected_namespaces={"kube-system"}),
        review_tier="none",
    )
    manager.register_action(action)
    # Seed a cooldown for this exact (action,target) signature.
    manager.context.cooldowns["container_restart:predagent_sick"] = time.time()

    pod = _make_pod_state(restart_count=9, running=True)
    results = manager.evaluate(pod, None, 0.9)

    assert results
    result = results[0]
    assert result.action == "container_restart"
    assert result.success is False
    assert "cooldown" in result.message
    assert "skipped" in result.message


def test_rate_limit_rejects_container_restart_through_manager():
    """Per-minute rate limit exhausted: the real container_restart action is
    skipped with the rate-limit reason instead of executing."""
    from predictive_agent.remediator import RemediationManager, SafetyPolicy

    action = ContainerRestartAction()
    manager = RemediationManager(
        dry_run=False,
        safety_policy=SafetyPolicy(max_per_minute=1, cooldown_seconds=0,
                                   protected_namespaces={"kube-system"}),
        review_tier="none",
    )
    manager.register_action(action)
    # One action already recorded in the current minute -> at the limit.
    manager.context.rate_limiter["minute"] = [time.time() - 1.0]

    pod = _make_pod_state(restart_count=9, running=True)
    results = manager.evaluate(pod, None, 0.9)

    assert results
    result = results[0]
    assert result.action == "container_restart"
    assert result.success is False
    assert "rate limit" in result.message


# ── 5. risk_tier propagation into the returned action result ──────────────


def test_safe_risk_tier_propagates_through_manager_review_gate():
    """container_restart is a 'safe' tier action: with the P6 review gate at
    'moderate' it must NOT be deferred (no review_required) — the returned
    result proves the action's risk_tier propagated into the decision."""
    from predictive_agent.remediator import RemediationManager, SafetyPolicy

    action = ContainerRestartAction()
    # Risk tier is declared on the class and propagated to the instance.
    assert action.risk_tier == "safe"
    assert ContainerRestartAction.risk_tier == "safe"

    manager = RemediationManager(
        dry_run=False,
        safety_policy=SafetyPolicy(cooldown_seconds=0, max_per_minute=100,
                                   max_per_hour=1000,
                                   protected_namespaces={"kube-system"}),
        review_tier="moderate",
        action_timeout_s=5,
    )
    manager.register_action(action)

    mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="ok", stderr=""))
    with patch("predictive_agent.actions.container_restart.subprocess.run", mock_run):
        pod = _make_pod_state(name="app-1", restart_count=6, running=True)
        results = manager.evaluate(pod, None, 0.9)

    assert results
    result = results[0]
    assert result.action == "container_restart"
    assert result.outcome == "acted"
    assert result.success is True
    assert "review_required" not in result.outcome
    assert "review_required" not in result.message
    mock_run.assert_called_once()


# ── Module-global reset (env-driven thresholds / protected namespaces) ────


def test_risk_threshold_global_respected_and_reset():
    """Raising RISK_THRESHOLD to 0.9 stops a 0.85-risk restart; the autouse
    fixture restores the default afterwards."""
    saved = container_restart_module.RISK_THRESHOLD
    container_restart_module.RISK_THRESHOLD = 0.9
    try:
        action = ContainerRestartAction()
        pod = _make_pod_state(restart_count=9, running=True)
        assert action.should_execute(pod, None, 0.85) is False
        assert action.should_execute(pod, None, 0.95) is True
    finally:
        container_restart_module.RISK_THRESHOLD = saved
    # Default restored by the direct reset (and the fixture doubles it).
    assert saved == container_restart_module.RISK_THRESHOLD


def test_protected_namespace_global_extend_and_reset():
    """Adding a namespace to REMEDIATION_PROTECTED_NS blocks restarts there;
    the autouse fixture restores the original set afterwards."""
    saved = set(container_restart_module.REMEDIATION_PROTECTED_NS)
    container_restart_module.REMEDIATION_PROTECTED_NS.add("monitoring")
    try:
        action = ContainerRestartAction()
        pod = _make_pod_state(namespace="monitoring", restart_count=9, running=True)
        assert action.should_execute(pod, None, 0.9) is False
    finally:
        container_restart_module.REMEDIATION_PROTECTED_NS.clear()
        container_restart_module.REMEDIATION_PROTECTED_NS.update(saved)
    # Original globals preserved after the test.
    assert saved == container_restart_module.REMEDIATION_PROTECTED_NS
    assert "monitoring" not in container_restart_module.REMEDIATION_PROTECTED_NS
