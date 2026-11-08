"""
Edge-case tests for the OPA HTTP client (predictive_agent/opa_client.py) and
the OPA response cache (predictive_agent/opa_cache.py).

Covers:
- opa_client: successful query → parsed policy result; connection refused /
  timeout / non-200 / malformed bodies follow the documented fail-open
  fallbacks (health_check → False, get_status → {"error": ...}); URL and
  timeout configuration honored; retry on transient 5xx; rate limiting;
  circuit breaker; health_check semantics.
- opa_cache: cache hit/miss, TTL expiry (time is frozen — never slept),
  invalidation + missing-entry paths, policy-version invalidation,
  oldest-entry eviction, cleanup, and a thread-safety smoke test.

The transport boundary patched here is the one the client actually uses:
``http.client.HTTPConnection`` / ``HTTPSConnection`` (stdlib — the module has
no ``requests`` dependency). No network is ever touched.
"""

import datetime as _dt
import http.client
import json
import logging
import threading
from typing import Dict, Optional

import pytest

from predictive_agent.opa_cache import (
    OPACache,
    OPAPolicyCache,
    get_opa_policy_cache,
    reset_opa_policy_cache,
)
from predictive_agent.opa_client import (
    OPAClient,
    OPASecurityError,
    get_opa_client,
    reset_opa_client,
)

# Keep connection-error / circuit-breaker log spam out of pytest output.
logging.getLogger("predictive_agent.opa_client").setLevel(logging.ERROR)


# ────────────────────────────────────────────────────────────────────────
# HTTP transport stand-in (patches http.client, the real boundary)
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
    """Stand-in for http.client.HTTPConnection / HTTPSConnection.

    Configured through class attributes (reset by the ``fake_http`` fixture):

    - ``response``: the _FakeResponse returned by ``getresponse()``
    - ``response_sequence``: if non-empty, popped per request (retry tests)
    - ``exc_on_request``: exception raised in ``request()`` — this matches the
      real ``http.client`` code path, where the socket connect (and thus
      connection-refused / timeout) surfaces inside ``HTTPConnection.request``
    """

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
        self.method: Optional[str] = None
        self.path: Optional[str] = None
        self.body: Optional[bytes] = None
        self.headers: Optional[Dict[str, str]] = None
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


# ────────────────────────────────────────────────────────────────────────
# Frozen clock for cache TTL tests (never sleep)
# ────────────────────────────────────────────────────────────────────────

class _FrozenDatetime(_dt.datetime):
    """datetime stand-in with a manually advanceable ``now()``."""

    current = _dt.datetime(2024, 6, 1, 12, 0, 0)

    @classmethod
    def now(cls):
        return cls.current


@pytest.fixture
def frozen_time(monkeypatch):
    """Patch opa_cache.datetime so TTL logic runs against a controlled clock."""
    import predictive_agent.opa_cache as oc

    _FrozenDatetime.current = _dt.datetime(2024, 6, 1, 12, 0, 0)
    monkeypatch.setattr(oc, "datetime", _FrozenDatetime)
    return _FrozenDatetime


@pytest.fixture(autouse=True)
def _reset_module_globals():
    """Reset OPA client/cache singletons before and after every test."""
    reset_opa_client()
    reset_opa_policy_cache()
    yield
    reset_opa_client()
    reset_opa_policy_cache()


# ────────────────────────────────────────────────────────────────────────
# opa_client — successful queries
# ────────────────────────────────────────────────────────────────────────

def test_successful_query_returns_parsed_policy_result(fake_http):
    """A 200 response body is parsed as JSON and returned as-is."""
    fake_http.response = _FakeResponse(200, "OK", json.dumps({"result": {"allow": True}}).encode())
    client = OPAClient(host="opa.test", port=8181, use_tls=False, max_retries=1)

    result = client.evaluate_policy("predictive_agent.pod", {"kind": {"metadata": {"name": "p"}}})

    assert result == {"result": {"allow": True}}
    conn = fake_http.instances[-1]
    assert conn.host == "opa.test"
    assert conn.port == 8181
    assert conn.method == "POST"
    assert conn.path == "/v1/data/predictive_agent.pod"
    # The input is wrapped under the "input" key as the OPA API expects.
    assert json.loads(conn.body.decode()) == {"input": {"kind": {"metadata": {"name": "p"}}}}


def test_auth_token_and_content_headers_sent(fake_http):
    """Bearer token + JSON content headers reach the wire."""
    fake_http.response = _FakeResponse(200, "OK", b"{}")
    client = OPAClient(use_tls=False, auth_token="secret-token", max_retries=1)

    client.evaluate_policy("pkg", {})

    headers = fake_http.instances[-1].headers
    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"


def test_health_check_true_on_200(fake_http):
    """health_check() returns True when OPA answers."""
    fake_http.response = _FakeResponse(200, "OK", b"{}")
    client = OPAClient(use_tls=False, max_retries=1)
    assert client.health_check() is True


def test_get_status_parses_success(fake_http):
    """get_status() returns the parsed status body on success."""
    fake_http.response = _FakeResponse(200, "OK", b'{"status": "ok"}')
    client = OPAClient(use_tls=False, max_retries=1)
    assert client.get_status() == {"status": "ok"}


def test_empty_body_returns_empty_dict(fake_http):
    """An empty 200 body is tolerated as {}."""
    fake_http.response = _FakeResponse(200, "OK", b"")
    client = OPAClient(use_tls=False, max_retries=1)
    assert client.evaluate_policy("pkg", {}) == {}


# ────────────────────────────────────────────────────────────────────────
# opa_client — failure / fallback semantics (no sleeping, no network)
# ────────────────────────────────────────────────────────────────────────

def test_connection_refused_health_check_returns_false(fake_http):
    """Connection refused → health_check() returns False without raising."""
    fake_http.exc_on_request = ConnectionRefusedError("connection refused")
    client = OPAClient(host="127.0.0.1", port=1, use_tls=False, max_retries=1, timeout=0.1)
    assert client.health_check() is False


def test_timeout_health_check_returns_false(fake_http):
    """Timeout → health_check() returns False without raising."""
    fake_http.exc_on_request = TimeoutError("timed out")
    client = OPAClient(host="127.0.0.1", port=1, use_tls=False, max_retries=1, timeout=0.05)
    assert client.health_check() is False


def test_connection_refused_get_status_returns_error_dict(fake_http):
    """Connection refused → get_status() returns {'error': ...} without raising."""
    fake_http.exc_on_request = ConnectionRefusedError("connection refused")
    client = OPAClient(use_tls=False, max_retries=1, timeout=0.1)
    status = client.get_status()
    assert isinstance(status, dict)
    assert "error" in status


def test_evaluate_policy_raises_on_connection_refused(fake_http):
    """evaluate_policy() surfaces connection failure as OPASecurityError (fail-open upstream)."""
    fake_http.exc_on_request = ConnectionRefusedError("connection refused")
    client = OPAClient(use_tls=False, max_retries=1, timeout=0.1)
    with pytest.raises(OPASecurityError):
        client.evaluate_policy("predictive_agent.pod", {"kind": {}})


def test_non_200_forbidden_raises_security_error(fake_http):
    """A 4xx response from OPA → OPASecurityError, never a silent default."""
    fake_http.response = _FakeResponse(403, "Forbidden", b'{"code": "unauthorized"}')
    client = OPAClient(use_tls=False, max_retries=1)
    with pytest.raises(OPASecurityError) as excinfo:
        client.evaluate_policy("pkg", {})
    assert "403" in str(excinfo.value)


def test_non_200_health_check_returns_false(fake_http):
    """A 4xx response → health_check() returns False (documented fallback)."""
    fake_http.response = _FakeResponse(403, "Forbidden", b"{}")
    client = OPAClient(use_tls=False, max_retries=1)
    assert client.health_check() is False


def test_5xx_get_status_returns_error_dict(fake_http):
    """A 5xx response (retries exhausted) → get_status() returns {'error': ...}."""
    fake_http.response = _FakeResponse(503, "Service Unavailable", b"{}")
    client = OPAClient(use_tls=False, max_retries=1)
    status = client.get_status()
    assert isinstance(status, dict)
    assert "error" in status


def test_malformed_json_raises_documented_security_error(fake_http):
    """Malformed JSON follows the real code path: OPASecurityError + health False."""
    fake_http.response = _FakeResponse(200, "OK", b"<html>not json at all</html>")
    client = OPAClient(use_tls=False, max_retries=1)
    with pytest.raises(OPASecurityError) as excinfo:
        client.evaluate_policy("pkg", {})
    assert "Malformed OPA response" in str(excinfo.value)
    # health_check tolerates the malformed body by failing closed to False.
    assert client.health_check() is False


# ────────────────────────────────────────────────────────────────────────
# opa_client — retry, URL/timeout config, rate limiting, circuit breaker
# ────────────────────────────────────────────────────────────────────────

def test_retry_recovers_from_transient_5xx(fake_http, monkeypatch):
    """Transient 5xx are retried (backoff patched out) and recover on 200."""
    import predictive_agent.opa_client as oc

    monkeypatch.setattr(oc.time, "sleep", lambda _s: None)
    fake_http.response_sequence = [
        _FakeResponse(500, "Internal Server Error", b"{}"),
        _FakeResponse(500, "Internal Server Error", b"{}"),
        _FakeResponse(200, "OK", b'{"result": {"allow": true}}'),
    ]
    client = OPAClient(use_tls=False, max_retries=3)

    result = client.evaluate_policy("pkg", {})

    assert result == {"result": {"allow": True}}
    assert len(fake_http.instances) == 3  # two failures, then success


def test_url_and_timeout_configuration_honored(fake_http):
    """The configured host/port/timeout are passed to the transport verbatim."""
    client = OPAClient(host="opa.internal", port=9191, use_tls=False, timeout=2.5, max_retries=1)
    assert client.base_url == "http://opa.internal:9191"

    client.evaluate_policy("pkg", {})

    conn = fake_http.instances[-1]
    assert conn.host == "opa.internal"
    assert conn.port == 9191
    assert conn.timeout == 2.5


def test_tls_configuration_uses_https_connection(fake_http):
    """use_tls=True builds an https base URL and passes context/timeout through."""
    client = OPAClient(host="secure.opa", port=8443, use_tls=True, timeout=3.0, max_retries=1)
    assert client.base_url == "https://secure.opa:8443"

    client.evaluate_policy("pkg", {})

    conn = fake_http.instances[-1]
    assert conn.host == "secure.opa"
    assert conn.port == 8443
    assert conn.timeout == 3.0


def test_rate_limit_exceeded(fake_http):
    """Once the token budget is exhausted, requests raise OPASecurityError."""
    fake_http.response = _FakeResponse(200, "OK", b"{}")
    client = OPAClient(use_tls=False, rate_limit=2, rate_window=60, max_retries=1)

    client.evaluate_policy("pkg", {})
    client.evaluate_policy("pkg", {})

    with pytest.raises(OPASecurityError) as excinfo:
        client.evaluate_policy("pkg", {})
    assert "Rate limit exceeded" in str(excinfo.value)


def test_circuit_breaker_opens_after_repeated_failures(fake_http):
    """5 consecutive transport failures trip the breaker → fast-fail open."""
    fake_http.exc_on_request = ConnectionRefusedError("connection refused")
    client = OPAClient(use_tls=False, max_retries=1, timeout=0.05)

    for _ in range(5):
        with pytest.raises(OPASecurityError):
            client.evaluate_policy("pkg", {})

    assert client._circuit_closed is False

    with pytest.raises(OPASecurityError) as excinfo:
        client.evaluate_policy("pkg", {})
    assert "Circuit breaker open" in str(excinfo.value)


def test_get_opa_client_singleton_reset():
    """get_opa_client() returns one instance until reset_opa_client() is called."""
    a = get_opa_client()
    b = get_opa_client()
    assert a is b

    reset_opa_client()
    c = get_opa_client()
    assert a is not c


# ────────────────────────────────────────────────────────────────────────
# opa_cache — hit/miss, TTL expiry, invalidation
# ────────────────────────────────────────────────────────────────────────

def test_cache_hit_and_miss():
    """set()/get() round-trip hits; differing input or package is a miss."""
    cache = OPACache()
    cache.set("pkg.a", {"x": 1}, {"result": {"allow": True}})

    hit, value = cache.get("pkg.a", {"x": 1})
    assert hit is True
    assert value == {"result": {"allow": True}}

    # Same package, different input → miss.
    assert cache.get("pkg.a", {"x": 2}) == (False, None)
    # Same input, different package → miss.
    assert cache.get("pkg.b", {"x": 1}) == (False, None)


def test_cache_ttl_expiry(frozen_time):
    """Entries expire once their TTL passes (clock frozen, never slept)."""
    cache = OPACache()
    cache.set("predictive_agent.pod", {"id": 1}, {"result": {"allow": True}}, ttl=60)

    hit, value = cache.get("predictive_agent.pod", {"id": 1})
    assert hit is True
    assert value == {"result": {"allow": True}}

    frozen_time.current += _dt.timedelta(seconds=61)

    assert cache.get("predictive_agent.pod", {"id": 1}) == (False, None)


def test_entry_missing_returns_miss_without_error():
    """An entry that was never stored returns (False, None), never raises."""
    cache = OPACache()
    assert cache.get("nonexistent.pkg", {}) == (False, None)


def test_invalidate_policy_counts_and_clears(frozen_time):
    """invalidate_policy() clears only the named policy and returns the count."""
    cache = OPACache()
    cache.set("pkg.one", {"a": 1}, {"ok": 1})
    cache.set("pkg.one", {"a": 2}, {"ok": 2})
    cache.set("pkg.two", {"a": 1}, {"ok": 3})

    assert cache.invalidate_policy("pkg.one") == 2
    # Re-running on an already-empty policy yields 0.
    assert cache.invalidate_policy("pkg.one") == 0

    assert cache.get("pkg.one", {"a": 1}) == (False, None)
    assert cache.get("pkg.one", {"a": 2}) == (False, None)
    # Unrelated policy survives.
    assert cache.get("pkg.two", {"a": 1})[0] is True


def test_update_policy_version_invalidates_on_change(frozen_time):
    """Same version keeps entries; a new version invalidates the policy."""
    cache = OPACache()
    cache.update_policy_version("pkg", "v2")
    cache.set("pkg", {"k": 1}, {"v": 1})

    cache.update_policy_version("pkg", "v2")  # unchanged → cache kept
    assert cache.get("pkg", {"k": 1})[0] is True

    cache.update_policy_version("pkg", "v3")  # change → invalidated
    assert cache.get("pkg", {"k": 1}) == (False, None)


def test_max_size_eviction_removes_oldest(frozen_time):
    """When max_size is hit, the oldest entry is evicted first."""
    cache = OPACache(max_size=2)
    cache.set("pkg", {"k": 1}, {"v": 1})
    frozen_time.current += _dt.timedelta(seconds=1)
    cache.set("pkg", {"k": 2}, {"v": 2})
    frozen_time.current += _dt.timedelta(seconds=1)
    cache.set("pkg", {"k": 3}, {"v": 3})

    stats = cache.get_stats()
    assert stats["size"] == 2
    assert stats["evictions"] == 1

    assert cache.get("pkg", {"k": 1}) == (False, None)  # oldest evicted
    assert cache.get("pkg", {"k": 3})[0] is True        # newest survives


def test_cleanup_removes_only_expired_entries(frozen_time):
    """cleanup() returns the count of expired entries and keeps fresh ones."""
    cache = OPACache()
    cache.set("pkg", {"k": 1}, {"v": 1}, ttl=10)
    frozen_time.current += _dt.timedelta(seconds=11)
    cache.set("pkg", {"k": 2}, {"v": 2}, ttl=10)

    assert cache.cleanup() == 1
    assert cache.get("pkg", {"k": 1}) == (False, None)
    assert cache.get("pkg", {"k": 2})[0] is True


def test_stats_track_hits_misses_and_sets(frozen_time):
    """get_stats() reflects the actual hit/miss/set activity."""
    cache = OPACache()
    cache.set("pkg", {"k": 1}, {"v": 1}, ttl=60)
    cache.get("pkg", {"k": 1})          # hit
    cache.get("pkg", {"k": 2})          # miss
    frozen_time.current += _dt.timedelta(seconds=61)
    cache.get("pkg", {"k": 1})          # expired → miss

    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["sets"] == 1
    assert stats["size"] == 0
    assert stats["hit_rate"] == pytest.approx(100 * 1 / 3)


def test_thread_safety_smoke():
    """Concurrent set/get on a shared OPACache does not corrupt entries."""
    cache = OPACache()
    errors = []

    def worker(i):
        try:
            cache.set("pkg", {"i": i}, {"i": i})
            hit, value = cache.get("pkg", {"i": i})
            assert hit is True
            assert value == {"i": i}
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(64)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert cache.get_stats()["sets"] == 64
    assert cache.get_stats()["size"] == 64


# ────────────────────────────────────────────────────────────────────────
# opa_cache — high-level OPAPolicyCache
# ────────────────────────────────────────────────────────────────────────

def test_policy_cache_unknown_type_falls_back_to_general():
    """Unrecognized policy types resolve to the general cache layer."""
    pc = OPAPolicyCache()
    assert pc.get_cache("unknown") is pc.get_cache("general")
    assert pc.get_cache("pod") is not pc.get_cache("general")


def test_evaluate_with_cache_miss_then_hit():
    """evaluate_with_cache() hits OPA once, stores, then serves from cache."""
    pc = OPAPolicyCache()
    calls = []

    class _FakeClient:
        def evaluate_policy(self, package, input_data, cache_key=None):
            calls.append(package)
            return {"result": {"allow": True}}

    client = _FakeClient()

    cached, result, _key = pc.evaluate_with_cache("pkg", {"k": 1}, "general", client)
    assert cached is False
    assert result == {"result": {"allow": True}}

    cached2, result2, _key2 = pc.evaluate_with_cache("pkg", {"k": 1}, "general", client)
    assert cached2 is True
    assert result2 == {"result": {"allow": True}}
    # OPA was consulted exactly once.
    assert calls == ["pkg"]


def test_opa_policy_cache_singleton_reset():
    """get_opa_policy_cache() returns one instance until reset is called."""
    a = get_opa_policy_cache()
    b = get_opa_policy_cache()
    assert a is b

    reset_opa_policy_cache()
    c = get_opa_policy_cache()
    assert a is not c
