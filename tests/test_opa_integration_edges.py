"""
Edge-case tests for the OPA integration module (predictive_agent.opa_integration).

This file exercises the *real* branch behaviour of ``OPAPolicyEngine`` with the
network boundary stubbed out — it never talks to an OPA server.

Behaviours verified (all read from the actual source, no invented API):

1. Client-unavailable guards: when ``_opa_client is None``, ``check_health``
   returns ``False`` and the ``check_*`` methods hit the engine's internal
   ``RuntimeError("OPA client unavailable")`` guard, which the engine's own
   except-block absorbs into the documented fail-open response
   (``policy_check_failed=True``, allow-by-default). The tests assert that
   observable contract.
2. Each ``check_*`` method's allow and deny branches against a stubbed client.
3. Malformed / empty policy responses fall back safely (no exception).
4. The ``enabled=False`` flag short-circuits before any client call.
5. An exception raised inside the client surfaces as the documented safe
   default (fail-open, error counter bumped).
"""

import logging
from typing import Any, Dict, List

import pytest

from predictive_agent.opa_cache import reset_opa_policy_cache
from predictive_agent.opa_client import OPASecurityError, reset_opa_client
from predictive_agent.opa_integration import OPAPolicyEngine, reset_policy_engine

# Keep the pytest output free of engine/client warning noise.
logging.getLogger("predictive_agent.opa_client").setLevel(logging.ERROR)
logging.getLogger("predictive_agent.opa_integration").setLevel(logging.ERROR)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_opa_globals():
    """Reset every OPA module singleton before and after each test."""
    reset_opa_client()
    reset_opa_policy_cache()
    reset_policy_engine()
    yield
    reset_opa_client()
    reset_opa_policy_cache()
    reset_policy_engine()


class _StubOPAClient:
    """Minimal stand-in for ``OPAClient`` — records calls, returns canned data.

    Each ``*_result`` may be a plain value (returned as-is) or a callable
    (invoked with the forwarded input, so a test can raise on demand).
    """

    def __init__(
        self,
        pod_result: Any = None,
        llm_result: Any = None,
        cache_result: Any = None,
        health: Any = True,
    ) -> None:
        self.pod_result: Any = pod_result
        self.llm_result: Any = llm_result
        self.cache_result: Any = cache_result
        self.health: Any = health
        self.calls: List[str] = []
        self.pod_calls: int = 0
        self.llm_calls: int = 0
        self.cache_calls: int = 0

    @staticmethod
    def _resolve(value: Any, *args: Any) -> Any:
        if callable(value):
            return value(*args)
        return value

    def check_pod_security(self, pod_data: Dict[str, Any]) -> Any:
        self.pod_calls += 1
        self.calls.append("pod")
        return self._resolve(self.pod_result, pod_data)

    def check_llm_usage(self, llm_request: Dict[str, Any]) -> Any:
        self.llm_calls += 1
        self.calls.append("llm")
        return self._resolve(self.llm_result, llm_request)

    def check_cache_config(self, cache_config: Dict[str, Any]) -> Any:
        self.cache_calls += 1
        self.calls.append("cache")
        return self._resolve(self.cache_result, cache_config)

    def health_check(self) -> bool:
        self.calls.append("health")
        if callable(self.health):
            return bool(self.health())
        return bool(self.health)


def _engine_with_stub(
    stub: _StubOPAClient,
    *,
    enabled: bool = True,
    enable_caching: bool = False,
) -> OPAPolicyEngine:
    """Build an engine and swap in the stub at the client boundary."""
    engine = OPAPolicyEngine(enabled=enabled, enable_caching=enable_caching)
    engine._opa_client = stub  # stub the boundary; never touch the network
    return engine


@pytest.fixture
def clientless_engine() -> OPAPolicyEngine:
    """Engine whose client is permanently unavailable (never re-initialized)."""
    engine = OPAPolicyEngine(enabled=True, enable_caching=False)
    engine._opa_client = None

    def _noop_init() -> None:
        return None

    engine._init_clients = _noop_init  # type: ignore[method-assign]
    return engine


# ────────────────────────────────────────────────────────────────────────────
# 1. Client-unavailable guards
# ────────────────────────────────────────────────────────────────────────────
class TestClientUnavailableGuards:
    """When the OPA client is None, the module's None-guards kick in."""

    def test_health_check_returns_false_when_client_unavailable(self, clientless_engine):
        assert clientless_engine.check_health() is False

    def test_pod_check_fail_open_when_client_unavailable(self, clientless_engine):
        result = clientless_engine.check_pod_security({"metadata": {"name": "x"}})
        # The internal RuntimeError("OPA client unavailable") guard fires and is
        # absorbed into the documented fail-open response.
        assert result["result"]["allow"] is True
        assert result.get("policy_check_failed") is True
        assert any("OPA client unavailable" in str(v) for v in result.get("violation", []))
        assert clientless_engine.get_stats()["errors"] == 1

    def test_llm_check_fail_open_when_client_unavailable(self, clientless_engine):
        result = clientless_engine.check_llm_usage({"model": "m", "cost_per_request": 0.01})
        assert result["result"]["allow_llm"] is True
        assert result.get("policy_check_failed") is True
        assert any("OPA client unavailable" in str(v) for v in result.get("violation", []))
        assert clientless_engine.get_stats()["errors"] == 1

    def test_cache_check_fail_open_when_client_unavailable(self, clientless_engine):
        result = clientless_engine.check_cache_config({"layer": "l1", "size": 1000})
        assert result["result"]["allow_cache_action"] is True
        assert result.get("policy_check_failed") is True
        assert any("OPA client unavailable" in str(v) for v in result.get("violation", []))
        assert clientless_engine.get_stats()["errors"] == 1

    def test_client_unavailable_never_touches_network(self, clientless_engine):
        # No real/host client can ever be constructed or called on this engine.
        engine = clientless_engine
        engine.check_pod_security({})
        engine.check_llm_usage({})
        engine.check_cache_config({})
        assert engine._opa_client is None
        assert engine.get_stats()["opa_requests"] == 0


# ────────────────────────────────────────────────────────────────────────────
# 2. Allow / deny branches per check method
# ────────────────────────────────────────────────────────────────────────────
class TestCheckBranches:
    """Each check_* method's allow and deny branches with a stubbed client."""

    def test_pod_security_allow(self):
        stub = _StubOPAClient(pod_result={"result": {"allow": True}, "violation": []})
        engine = _engine_with_stub(stub)
        result = engine.check_pod_security({"metadata": {"name": "app-1"}})
        assert result["result"]["allow"] is True
        assert result["violation"] == []
        assert stub.pod_calls == 1
        stats = engine.get_stats()
        assert stats["total_requests"] == 1
        assert stats["opa_requests"] == 1
        assert stats["allowed"] == 1
        assert stats["denied"] == 0
        assert stats["errors"] == 0

    def test_pod_security_deny(self):
        stub = _StubOPAClient(
            pod_result={"result": {"allow": False}, "violation": ["pod denied by policy"]}
        )
        engine = _engine_with_stub(stub)
        result = engine.check_pod_security({"metadata": {"name": "app-1"}})
        assert result["result"]["allow"] is False
        assert result["violation"] == ["pod denied by policy"]
        assert stub.pod_calls == 1
        stats = engine.get_stats()
        assert stats["denied"] == 1
        assert stats["policy_violations"] == 1
        assert stats["allowed"] == 0

    def test_llm_usage_allow(self):
        stub = _StubOPAClient(llm_result={"result": {"allow_llm": True}, "violation": []})
        engine = _engine_with_stub(stub)
        result = engine.check_llm_usage({"model": "m", "cost_per_request": 0.01})
        assert result["result"]["allow_llm"] is True
        assert stub.llm_calls == 1
        assert engine.get_stats()["opa_requests"] == 1

    def test_llm_usage_deny(self):
        stub = _StubOPAClient(
            llm_result={"result": {"allow_llm": False}, "violation": ["llm usage denied"]}
        )
        engine = _engine_with_stub(stub)
        result = engine.check_llm_usage({"model": "m", "max_tokens": 4096})
        assert result["result"]["allow_llm"] is False
        assert result["violation"] == ["llm usage denied"]
        assert stub.llm_calls == 1
        assert engine.get_stats()["policy_violations"] == 1

    def test_cache_config_allow(self):
        stub = _StubOPAClient(
            cache_result={"result": {"allow_cache_action": True}, "violation": []}
        )
        engine = _engine_with_stub(stub)
        result = engine.check_cache_config({"layer": "l1", "size": 1000, "ttl": 300})
        assert result["result"]["allow_cache_action"] is True
        assert stub.cache_calls == 1
        assert engine.get_stats()["opa_requests"] == 1

    def test_cache_config_deny(self):
        stub = _StubOPAClient(
            cache_result={
                "result": {"allow_cache_action": False},
                "violation": ["cache sizing denied"],
            }
        )
        engine = _engine_with_stub(stub)
        result = engine.check_cache_config({"layer": "l1", "size": 10_000_000})
        assert result["result"]["allow_cache_action"] is False
        assert result["violation"] == ["cache sizing denied"]
        assert stub.cache_calls == 1
        assert engine.get_stats()["policy_violations"] == 1


# ────────────────────────────────────────────────────────────────────────────
# 3. Malformed / empty policy responses
# ────────────────────────────────────────────────────────────────────────────
class TestMalformedResponses:
    """Malformed/empty policy responses must fall back safely (no crash)."""

    def test_empty_dict_response_is_safe(self):
        stub = _StubOPAClient(pod_result={})
        engine = _engine_with_stub(stub)
        result = engine.check_pod_security({})
        assert result == {}
        stats = engine.get_stats()
        assert stats["allowed"] == 0
        assert stats["denied"] == 0
        assert stats["errors"] == 0

    def test_none_response_is_safe(self):
        stub = _StubOPAClient(pod_result=None)
        engine = _engine_with_stub(stub)
        result = engine.check_pod_security({})
        assert result is None
        assert engine.get_stats()["errors"] == 0

    def test_list_response_is_safe(self):
        stub = _StubOPAClient(llm_result=[])
        engine = _engine_with_stub(stub)
        result = engine.check_llm_usage({})
        assert result == []
        assert engine.get_stats()["errors"] == 0

    def test_violation_not_a_list_is_safe(self):
        stub = _StubOPAClient(
            pod_result={"result": {"allow": False}, "violation": "not-a-list"}
        )
        engine = _engine_with_stub(stub)
        result = engine.check_pod_security({})
        assert result["result"]["allow"] is False
        assert result["violation"] == "not-a-list"
        assert engine.get_stats()["policy_violations"] == 1
        assert engine.get_stats()["errors"] == 0


# ────────────────────────────────────────────────────────────────────────────
# 4. Disabled integration short-circuits before any client call
# ────────────────────────────────────────────────────────────────────────────
class TestDisabledShortCircuit:
    """enabled=False must bypass the client entirely and allow by default."""

    def test_disabled_never_calls_the_client(self):
        stub = _StubOPAClient(
            pod_result={"result": {"allow": False}, "violation": ["would deny"]},
            llm_result={"result": {"allow_llm": False}, "violation": ["would deny"]},
            cache_result={"result": {"allow_cache_action": False}, "violation": ["would deny"]},
        )
        engine = _engine_with_stub(stub, enabled=False)

        pod = engine.check_pod_security({"metadata": {"name": "x"}})
        assert pod["result"]["allow"] is True
        assert pod["violation"] == []

        llm = engine.check_llm_usage({"model": "m"})
        assert llm["result"]["allow_llm"] is True
        assert llm["violation"] == []

        cache = engine.check_cache_config({"layer": "l1"})
        assert cache["result"]["allow_cache_action"] is True
        assert cache["violation"] == []

        # The client was never consulted.
        assert stub.pod_calls == 0
        assert stub.llm_calls == 0
        assert stub.cache_calls == 0
        assert stub.calls == []

    def test_disabled_does_not_count_requests(self):
        stub = _StubOPAClient(pod_result={"result": {"allow": True}, "violation": []})
        engine = _engine_with_stub(stub, enabled=False)
        engine.check_pod_security({})
        engine.check_llm_usage({})
        engine.check_cache_config({})
        stats = engine.get_stats()
        assert stats["total_requests"] == 0
        assert stats["opa_requests"] == 0
        assert stats["enabled"] is False


# ────────────────────────────────────────────────────────────────────────────
# 5. Exceptions inside the client surface as the documented safe default
# ────────────────────────────────────────────────────────────────────────────
class TestClientExceptions:
    """A client-side exception must fail open and bump the error counter."""

    def test_pod_client_exception_fails_open(self):
        def _boom(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            raise OPASecurityError("tls handshake failed")

        stub = _StubOPAClient(pod_result=_boom)
        engine = _engine_with_stub(stub)
        result = engine.check_pod_security({"metadata": {"name": "x"}})
        assert result["result"]["allow"] is True
        assert result.get("policy_check_failed") is True
        assert any("tls handshake failed" in str(v) for v in result.get("violation", []))
        assert engine.get_stats()["errors"] == 1

    def test_llm_client_exception_fails_open(self):
        def _boom(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError("llm policy evaluation crashed")

        stub = _StubOPAClient(llm_result=_boom)
        engine = _engine_with_stub(stub)
        result = engine.check_llm_usage({"model": "m"})
        assert result["result"]["allow_llm"] is True
        assert result.get("policy_check_failed") is True
        assert any("llm policy evaluation crashed" in str(v) for v in result.get("violation", []))
        assert engine.get_stats()["errors"] == 1

    def test_cache_client_exception_fails_open(self):
        def _boom(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            raise OPASecurityError("rate limit exceeded")

        stub = _StubOPAClient(cache_result=_boom)
        engine = _engine_with_stub(stub)
        result = engine.check_cache_config({"layer": "l1"})
        assert result["result"]["allow_cache_action"] is True
        assert result.get("policy_check_failed") is True
        assert any("rate limit exceeded" in str(v) for v in result.get("violation", []))
        assert engine.get_stats()["errors"] == 1


# ────────────────────────────────────────────────────────────────────────────
# Health + stats edges
# ────────────────────────────────────────────────────────────────────────────
class TestHealthAndStats:
    """Health reporting and stats plumbing at the edges."""

    def test_health_check_true_when_client_healthy(self):
        stub = _StubOPAClient(health=True)
        engine = _engine_with_stub(stub)
        assert engine.check_health() is True
        assert "health" in stub.calls

    def test_health_check_false_when_client_unhealthy(self):
        stub = _StubOPAClient(health=False)
        engine = _engine_with_stub(stub)
        assert engine.check_health() is False

    def test_get_stats_reports_opa_health_for_stubbed_client(self):
        stub = _StubOPAClient(health=True)
        engine = _engine_with_stub(stub)
        stats = engine.get_stats()
        assert stats["opa_health"] is True
        assert stats["enabled"] is True
        assert stats["total_requests"] == 0

    def test_cached_response_short_circuits_the_client(self):
        stub = _StubOPAClient(pod_result={"result": {"allow": True}, "violation": []})
        engine = _engine_with_stub(stub, enable_caching=True)

        # Pre-populate the pod cache with a decision for the exact input used
        # below so the cache branch (not the client) serves the request.
        cached = {"result": {"allow": False}, "violation": ["cached deny"]}
        pod_input = {"metadata": {"name": "x"}}
        engine._opa_cache.get_cache("pod").set(  # type: ignore[union-attr]
            "predictive_agent.pod", {"kind": pod_input}, cached
        )

        result = engine.check_pod_security(pod_input)
        assert result == cached
        assert stub.pod_calls == 0
        assert engine.get_stats()["cached_requests"] == 1

    def test_reset_stats_clears_counters(self):
        stub = _StubOPAClient(pod_result={"result": {"allow": True}, "violation": []})
        engine = _engine_with_stub(stub)
        engine.check_pod_security({})
        assert engine.get_stats()["total_requests"] == 1
        engine.reset_stats()
        stats = engine.get_stats()
        assert stats["total_requests"] == 0
        assert stats["opa_requests"] == 0
        assert stats["errors"] == 0
