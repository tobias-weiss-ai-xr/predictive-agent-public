"""Tests for the OPA integration wiring.

Covers the compliance requirement that the OPA client is **stdlib-only** (no
third-party deps), the fail-open contract when OPA is disabled/unreachable,
and the new /policy/* server endpoints.

These tests do not require a running OPA server.
"""

import json
import urllib.request

import pytest

from predictive_agent.opa_client import OPASecurityError


@pytest.fixture(autouse=True)
def fast_opa_failures(monkeypatch):
    """Pin small timeout/retries and reset the OPA client singleton so
    unreachable-OPA tests fail in milliseconds instead of 3s of retry
    backoff (see tests/test_opa_integration_simple.py for rationale).
    """
    import predictive_agent.config as _config
    import predictive_agent.opa_client as _opa_client_mod
    import predictive_agent.opa_integration as _opa_integration_mod
    monkeypatch.setattr(_config, "OPA_TIMEOUT", 0.1)
    monkeypatch.setattr(_config, "OPA_MAX_RETRIES", 1)
    _opa_client_mod.reset_opa_client()
    _opa_integration_mod.reset_policy_engine()
    yield
    _opa_client_mod.reset_opa_client()


def test_opa_client_is_stdlib_only():
    # Regression guard: opa_client must not import third-party packages.
    import predictive_agent.opa_client as mod
    src = open(mod.__file__, encoding="utf-8").read()
    assert "import requests" not in src
    assert "from requests" not in src
    assert "import urllib3" not in src
    assert "from urllib3" not in src


def test_opa_client_raises_on_unreachable():
    # stdlib transport: a refused connection surfaces as OPASecurityError.
    from predictive_agent.opa_client import get_opa_client, reset_opa_client
    reset_opa_client()
    client = get_opa_client(
        host="127.0.0.1", port=1, use_tls=False,
        timeout=1.0, max_retries=1, rate_limit=100, rate_window=60,
    )
    with pytest.raises(OPASecurityError):
        client.evaluate_policy("predictive_agent.pod", {"kind": {}})


def test_engine_fail_open_when_disabled():
    # When OPA is disabled, all checks allow by default (fail-open).
    from predictive_agent.opa_integration import OPAPolicyEngine, reset_policy_engine
    reset_policy_engine()
    engine = OPAPolicyEngine(enabled=False)
    pod = engine.check_pod_security({"metadata": {"name": "t"}})
    assert pod["result"]["allow"] is True
    llm = engine.check_llm_usage({"model": "x"})
    assert llm["result"]["allow_llm"] is True
    cache = engine.check_cache_config({"layer": "x"})
    assert cache["result"]["allow_cache_action"] is True


def test_engine_raises_opasecurityerror_when_enabled_but_unreachable():
    # enabled=True but OPA down -> evaluate_policy raises OPASecurityError,
    # which the engine catches and fail-opens to allow=True.
    from predictive_agent.opa_cache import reset_opa_policy_cache
    from predictive_agent.opa_client import reset_opa_client
    from predictive_agent.opa_integration import OPAPolicyEngine, reset_policy_engine
    reset_opa_client()
    reset_opa_policy_cache()
    reset_policy_engine()
    engine = OPAPolicyEngine(enabled=True, opa_host="127.0.0.1",
                              opa_port=1, use_tls=False)
    result = engine.check_pod_security({"metadata": {"name": "t"}})
    # Fail-open: even though OPA is unreachable, the pod is allowed.
    assert result["result"]["allow"] is True
    assert result.get("violation"), "violation list should note the failure"


# ─── Server endpoints (OPA disabled -> disabled responses) ────────────────
@pytest.fixture(scope="function")
def policy_server():
    from predictive_agent.server import _context, start_server
    _context.policy_engine = None
    srv = start_server(18380, 18381)
    yield srv
    srv.shutdown()


def test_policy_health_disabled(policy_server):
    with urllib.request.urlopen("http://localhost:18381/policy/health") as resp:
        data = json.loads(resp.read().decode())
    assert data["status"] == "disabled"


def test_policy_stats_disabled(policy_server):
    with urllib.request.urlopen("http://localhost:18380/policy/stats") as resp:
        data = json.loads(resp.read().decode())
    assert data["status"] == "disabled"


# ─── Server endpoints (stub engine attached) ──────────────────────────────
class _StubEngine:
    def __init__(self):
        self._enabled = True
        self._healthy = True
    def check_health(self):
        return True
    def get_stats(self):
        return {"enabled": True, "total_requests": 7, "allowed": 5, "denied": 0}


@pytest.fixture(scope="function")
def policy_server_with_engine():
    from predictive_agent.server import _context, start_server
    _context.policy_engine = None
    srv = start_server(18382, 18383, policy_engine=_StubEngine())
    yield srv
    srv.shutdown()


def test_policy_health_with_engine(policy_server_with_engine):
    with urllib.request.urlopen("http://localhost:18383/policy/health") as resp:
        data = json.loads(resp.read().decode())
    assert data["opa_healthy"] is True
    assert data["enabled"] is True


def test_policy_stats_with_engine(policy_server_with_engine):
    with urllib.request.urlopen("http://localhost:18382/policy/stats") as resp:
        data = json.loads(resp.read().decode())
    assert data["enabled"] is True
    assert data["total_requests"] == 7
