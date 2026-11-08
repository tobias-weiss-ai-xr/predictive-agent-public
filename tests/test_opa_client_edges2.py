"""
Round-2 edge-case tests for predictive_agent/opa_client.py.

Complements tests/test_opa_client_edges.py (no coverage duplication) by
exercising the remaining UNCOVERED branches of the stdlib-only OPA client:

- TLS context hardening with a CA cert + client cert/key (mTLS)
- ssl.SSLError / http.client.HTTPException transport failure branches
- exponential backoff on retry in the transport-exception path
- connection close raising while being torn down (finally guard)
- transient-5xx retries exhausting and raising OPASecurityError
- a successful retry re-closing an OPEN circuit breaker
- rate-limit token-bucket refill after the window elapses
- circuit breaker auto-reset once the 30s cool-off window passes
- _get_auth_headers() bearer-token injection branch
- the check_pod_security / check_llm_usage / check_cache_config wrappers

The transport boundary patched is the one the client really uses:
http.client.HTTPConnection / HTTPSConnection (stdlib — no requests).
No network is touched and time.sleep is always patched out.
"""

import http.client
import json
import logging
import ssl
from typing import Dict, Optional

import pytest

from predictive_agent.opa_client import (
    OPAClient,
    OPASecurityError,
)

# Keep transport-failure / circuit-breaker log spam out of pytest output.
logging.getLogger("predictive_agent.opa_client").setLevel(logging.ERROR)


# ────────────────────────────────────────────────────────────────────────
# HTTP transport stand-in (same pattern as test_opa_client_edges.py)
# ────────────────────────────────────────────────────────────────────────

class _FakeResponse:
    """Stand-in for http.client.HTTPResponse."""

    def __init__(self, status: int = 200, reason: str = "OK", body: bytes = b"{}"):
        self.status = status
        self.reason = reason
        self.body = body

    def read(self) -> bytes:
        return self.body


class _FakeConnection:
    """Stand-in for http.client.HTTPConnection / HTTPSConnection."""

    instances = []
    response = None
    response_sequence = []
    exc_on_request = None

    def __init__(
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.closed = False
        _FakeConnection.instances.append(self)

    def request(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.method = method
        self.path = path
        self.body = body
        self.headers = headers
        if _FakeConnection.exc_on_request is not None:
            raise _FakeConnection.exc_on_request

    def getresponse(self) -> "_FakeResponse":
        if _FakeConnection.response_sequence:
            return _FakeConnection.response_sequence.pop(0)
        return _FakeConnection.response

    def close(self) -> None:
        self.closed = True


class _CloseRaisesConnection(_FakeConnection):
    """Transport whose teardown (close) raises — exercises the finally guard."""

    def close(self) -> None:
        self.closed = True
        raise OSError("close failed")


@pytest.fixture
def fake_http(monkeypatch):
    """Replace http.client connections with a controllable fake."""
    _FakeConnection.instances = []
    _FakeConnection.response = _FakeResponse()
    _FakeConnection.response_sequence = []
    _FakeConnection.exc_on_request = None
    monkeypatch.setattr(http.client, "HTTPConnection", _FakeConnection)
    monkeypatch.setattr(http.client, "HTTPSConnection", _FakeConnection)
    return _FakeConnection


class _FakeSSLContext:
    """SSLContext stand-in that records hardening calls (no filesystem I/O)."""

    def __init__(self, protocol):
        self.minimum_version = None
        self.check_hostname = None
        self.verify_mode = None
        self.loaded_ca: Optional[str] = None
        self.loaded_chain: Optional[tuple] = None

    def load_verify_locations(self, cafile=None, capath=None, cadata=None):
        self.loaded_ca = cafile

    def load_cert_chain(self, certfile, keyfile=None, password=None):
        self.loaded_chain = (certfile, keyfile)


# ────────────────────────────────────────────────────────────────────────
# TLS context hardening: CA pinning + mutual TLS (lines 98, 103)
# ────────────────────────────────────────────────────────────────────────

def test_ssl_context_loads_ca_and_client_chain(monkeypatch):
    """_build_ssl_context() wires CA verification and mTLS cert/key in."""
    import predictive_agent.opa_client as oc

    monkeypatch.setattr(oc.ssl, "SSLContext", _FakeSSLContext)
    client = OPAClient(
        use_tls=True,
        ca_cert="/etc/opa/ca.pem",
        client_cert="/etc/opa/client.pem",
        client_key="/etc/opa/client.key",
    )

    ctx = client._build_ssl_context()

    assert ctx.loaded_ca == "/etc/opa/ca.pem"          # CA-pinned verification
    assert ctx.loaded_chain == (                       # mutual TLS
        "/etc/opa/client.pem",
        "/etc/opa/client.key",
    )
    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_3


def test_ssl_context_without_ca_still_requires_cert(monkeypatch):
    """Without a pinned CA the context keeps hostname checking + CERT_REQUIRED."""
    import predictive_agent.opa_client as oc

    monkeypatch.setattr(oc.ssl, "SSLContext", _FakeSSLContext)
    client = OPAClient(use_tls=True)

    ctx = client._build_ssl_context()

    assert ctx.loaded_ca is None
    assert ctx.check_hostname is True
    assert ctx.verify_mode == ssl.CERT_REQUIRED


# ────────────────────────────────────────────────────────────────────────
# Transport-exception branches (lines 150, 154) + backoff (line 158)
# ────────────────────────────────────────────────────────────────────────

def test_ssl_error_path_logs_tls_error_and_raises(fake_http, caplog):
    """An ssl.SSLError during request is logged as a TLS error, then raised."""
    fake_http.exc_on_request = ssl.SSLError("certificate verify failed")
    client = OPAClient(use_tls=True, max_retries=1, timeout=0.1)

    with caplog.at_level(logging.ERROR, logger="predictive_agent.opa_client"):
        with pytest.raises(OPASecurityError):
            client.evaluate_policy("pkg", {})

    assert any("TLS error" in r.message for r in caplog.records)
    assert client._failure_count == 1


def test_http_exception_path_logs_request_failed(fake_http, caplog):
    """A bare http.client.HTTPException hits the generic log branch."""
    fake_http.exc_on_request = http.client.HTTPException("broken pipe")
    client = OPAClient(use_tls=False, max_retries=1, timeout=0.1)

    with caplog.at_level(logging.ERROR, logger="predictive_agent.opa_client"):
        with pytest.raises(OPASecurityError):
            client.evaluate_policy("pkg", {})

    assert any("Request failed" in r.message for r in caplog.records)


def test_transport_exception_retries_with_exponential_backoff(fake_http, monkeypatch):
    """Transport failures back off 1s, 2s between attempts (patched out)."""
    import predictive_agent.opa_client as oc

    sleep_calls = []
    monkeypatch.setattr(oc.time, "sleep", sleep_calls.append)
    fake_http.exc_on_request = ConnectionRefusedError("connection refused")
    client = OPAClient(use_tls=False, max_retries=3, timeout=0.05)

    with pytest.raises(OPASecurityError) as excinfo:
        client.evaluate_policy("pkg", {})

    assert sleep_calls == [1.0, 2.0]  # min(2**(attempt-1), 5)
    assert "after 3 attempts" in str(excinfo.value)
    assert len(fake_http.instances) == 3


# ────────────────────────────────────────────────────────────────────────
# Connection teardown guard (lines 163–164)
# ────────────────────────────────────────────────────────────────────────

def test_connection_close_exception_is_swallowed(fake_http, monkeypatch):
    """An exception from conn.close() must not mask a successful response."""
    monkeypatch.setattr(http.client, "HTTPSConnection", _CloseRaisesConnection)
    fake_http.response = _FakeResponse(200, "OK", b'{"result": {"allow": true}}')
    client = OPAClient(use_tls=True, max_retries=1)

    result = client.evaluate_policy("pkg", {})

    assert result == {"result": {"allow": True}}
    assert fake_http.instances[-1].closed is True


# ────────────────────────────────────────────────────────────────────────
# Retryable 5xx exhausting retries (eventual OPASecurityError)
# ────────────────────────────────────────────────────────────────────────

def test_transient_5xx_exhaustion_raises_security_error(fake_http, monkeypatch):
    """Persistent 5xx across all retries surfaces as OPASecurityError."""
    import predictive_agent.opa_client as oc

    monkeypatch.setattr(oc.time, "sleep", lambda _s: None)
    fake_http.response = _FakeResponse(503, "Service Unavailable", b"{}")
    client = OPAClient(use_tls=False, max_retries=2)

    with pytest.raises(OPASecurityError) as excinfo:
        client.evaluate_policy("pkg", {})

    assert "after 2 attempts" in str(excinfo.value)
    assert "503" in str(excinfo.value)
    assert len(fake_http.instances) == 2


# ────────────────────────────────────────────────────────────────────────
# A successful retry re-closes an OPEN circuit breaker (lines 183–184)
# ────────────────────────────────────────────────────────────────────────

def test_success_after_retry_resets_open_circuit_breaker(fake_http, monkeypatch):
    """A 5th failure mid-request opens the breaker; the retry success re-closes it."""
    import predictive_agent.opa_client as oc

    monkeypatch.setattr(oc.time, "sleep", lambda _s: None)
    client = OPAClient(use_tls=False, max_retries=1, timeout=0.05)

    # Four transport failures → breaker still closed but one away from tripping.
    fake_http.exc_on_request = ConnectionRefusedError("connection refused")
    for _ in range(4):
        with pytest.raises(OPASecurityError):
            client.evaluate_policy("pkg", {})
    assert client._failure_count == 4
    assert client._circuit_closed is True

    # Attempt 1: transient 5xx trips the breaker; attempt 2: success re-closes it.
    client.max_retries = 2
    fake_http.exc_on_request = None
    fake_http.response_sequence = [
        _FakeResponse(500, "Internal Server Error", b"{}"),
        _FakeResponse(200, "OK", b'{"result": {"allow": true}}'),
    ]

    result = client.evaluate_policy("pkg", {})

    assert result == {"result": {"allow": True}}
    assert client._circuit_closed is True
    assert client._failure_count == 0


# ────────────────────────────────────────────────────────────────────────
# Rate-limit token refill after the window elapses (lines 205–206)
# ────────────────────────────────────────────────────────────────────────

def test_rate_limit_bucket_refills_after_window(fake_http, monkeypatch):
    """Tokens are restored once more than rate_window seconds have passed."""
    import predictive_agent.opa_client as oc

    now = [1000.0]
    monkeypatch.setattr(oc.time, "time", lambda: now[0])
    fake_http.response = _FakeResponse(200, "OK", b"{}")
    client = OPAClient(use_tls=False, rate_limit=2, rate_window=60, max_retries=1)

    client.evaluate_policy("pkg", {})
    client.evaluate_policy("pkg", {})
    assert client._request_tokens == 0

    with pytest.raises(OPASecurityError) as excinfo:
        client.evaluate_policy("pkg", {})  # exhausted
    assert "Rate limit exceeded" in str(excinfo.value)

    now[0] += 61  # new window → bucket refilled

    assert client.evaluate_policy("pkg", {}) == {}
    assert client._request_tokens == 1


# ────────────────────────────────────────────────────────────────────────
# Circuit breaker auto-reset after the cool-off window (lines 220–222)
# ────────────────────────────────────────────────────────────────────────

def test_circuit_breaker_auto_resets_after_cooloff(fake_http, monkeypatch, caplog):
    """Once the 30s reset window elapses the breaker re-closes on next request."""
    import predictive_agent.opa_client as oc

    now = [1000.0]
    monkeypatch.setattr(oc.time, "time", lambda: now[0])
    fake_http.exc_on_request = ConnectionRefusedError("connection refused")
    client = OPAClient(use_tls=False, max_retries=1, timeout=0.05)

    for _ in range(5):
        with pytest.raises(OPASecurityError):
            client.evaluate_policy("pkg", {})

    assert client._circuit_closed is False
    assert client._failure_count == 5
    assert client._circuit_reset_time == now[0] + 30  # open for 30s

    # Still inside the window → requests fast-fail.
    with pytest.raises(OPASecurityError) as excinfo:
        client.evaluate_policy("pkg", {})
    assert "Circuit breaker open" in str(excinfo.value)

    # Cool-off passes → breaker re-closes and the request goes through.
    now[0] += 31
    fake_http.exc_on_request = None
    fake_http.response = _FakeResponse(200, "OK", b"{}")
    with caplog.at_level(logging.INFO, logger="predictive_agent.opa_client"):
        result = client.evaluate_policy("pkg", {})

    assert result == {}
    assert client._circuit_closed is True
    assert client._failure_count == 0
    assert any("Circuit breaker reset" in r.message for r in caplog.records)


# ────────────────────────────────────────────────────────────────────────
# _get_auth_headers() bearer-token branch (lines 227–230)
# ────────────────────────────────────────────────────────────────────────

def test_get_auth_headers_injects_bearer_token():
    """_get_auth_headers() adds the Authorization header only when token set."""
    with_token = OPAClient(use_tls=False, auth_token="super-secret")
    headers = with_token._get_auth_headers()
    assert headers["Authorization"] == "Bearer super-secret"
    assert headers["Content-Type"] == "application/json"
    # Leave a valid dict even if no Accept key is expected by callers — the
    # contract is: application/json content type, Authorization on top.
    assert isinstance(headers, dict)

    no_token = OPAClient(use_tls=False)
    assert "Authorization" not in no_token._get_auth_headers()
    assert no_token._get_auth_headers()["Content-Type"] == "application/json"


# ────────────────────────────────────────────────────────────────────────
# Public policy wrappers (lines 257, 264, 271)
# ────────────────────────────────────────────────────────────────────────

def test_check_pod_security_wraps_input(fake_http):
    """check_pod_security() queries the pod policy with an 'input.kind' body."""
    fake_http.response = _FakeResponse(200, "OK", b'{"result": {"allow": true}}')
    client = OPAClient(use_tls=False, max_retries=1)

    result = client.check_pod_security({"metadata": {"name": "web"}})

    assert result == {"result": {"allow": True}}
    conn = fake_http.instances[-1]
    assert conn.path == "/v1/data/predictive_agent.pod"
    assert json.loads(conn.body.decode()) == {"input": {"kind": {"metadata": {"name": "web"}}}}


def test_check_llm_usage_forwards_payload(fake_http):
    """check_llm_usage() queries the LLM policy with the raw request."""
    fake_http.response = _FakeResponse(200, "OK", b'{"result": {"approved": true}}')
    client = OPAClient(use_tls=False, max_retries=1)

    result = client.check_llm_usage({"prompt": "summarize", "model": "gpt-x"})

    assert result == {"result": {"approved": True}}
    conn = fake_http.instances[-1]
    assert conn.path == "/v1/data/predictive_agent.llm"
    assert json.loads(conn.body.decode()) == {"input": {"prompt": "summarize", "model": "gpt-x"}}


def test_check_cache_config_wraps_cache_config(fake_http):
    """check_cache_config() queries the cache policy with an input wrapper."""
    fake_http.response = _FakeResponse(200, "OK", b'{"result": {"ok": true}}')
    client = OPAClient(use_tls=False, max_retries=1)

    result = client.check_cache_config({"ttl": 60, "max_size": 100})

    assert result == {"result": {"ok": True}}
    conn = fake_http.instances[-1]
    assert conn.path == "/v1/data/predictive_agent.cache"
    assert json.loads(conn.body.decode()) == {
        "input": {"cache_config": {"ttl": 60, "max_size": 100}}
    }
