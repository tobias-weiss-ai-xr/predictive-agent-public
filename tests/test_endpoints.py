"""Test HTTP endpoints (integration tests with real server)."""
import json
import urllib.error
import urllib.request

import pytest

from predictive_agent.server import start_server


@pytest.fixture(scope="module")
def test_server():
    """Start test server for endpoint testing."""
    metrics_server = start_server(18082, 18083)
    yield
    metrics_server.shutdown()


def test_predictions_endpoint(test_server):
    """Test /predictions endpoint."""
    try:
        with urllib.request.urlopen("http://localhost:18082/predictions") as resp:
            data = json.loads(resp.read().decode())
            assert "predictions" in data
            assert "total" in data
    except urllib.error.URLError as e:
        pytest.fail(f"Failed to connect to test server: {e}")


def test_state_endpoint(test_server):
    """Test /state endpoint."""
    try:
        with urllib.request.urlopen("http://localhost:18082/state") as resp:
            data = json.loads(resp.read().decode())
            assert "pods" in data or "markov_chain" in data or "states" in data
    except urllib.error.URLError as e:
        pytest.fail(f"Failed to connect to test server: {e}")


def test_health_endpoints(test_server):
    """Test health endpoints still work."""
    # Test /healthz
    with urllib.request.urlopen("http://localhost:18083/healthz") as resp:
        data = json.loads(resp.read().decode())
        assert "status" in data

    # Test /ready
    with urllib.request.urlopen("http://localhost:18083/ready") as resp:
        data = json.loads(resp.read().decode())
        assert "status" in data

    # Test /metrics
    with urllib.request.urlopen("http://localhost:18082/metrics") as resp:
        text = resp.read().decode()
        assert "opendesk_predictive_agent" in text


def test_plan_endpoint(test_server):
    """Test /plan (POMDP remediation planner) endpoint."""
    from predictive_agent.remediator import RemediationManager
    from predictive_agent.server import _context

    # Wire a remediation manager so /plan has a decision model to plan with.
    _context.remediation_manager = RemediationManager(dry_run=True)

    # Valid state -> 200 with an execution-ready plan.
    with urllib.request.urlopen(
        "http://localhost:18082/plan?state=CRITICAL&horizon=2"
    ) as resp:
        data = json.loads(resp.read().decode())
        assert resp.status == 200
        assert data["status"] == "ok"
        assert data["plan"]["execution_ready"] is True
        assert len(data["plan"]["steps"]) == 2

    # Invalid state -> 400.
    try:
        urllib.request.urlopen("http://localhost:18082/plan?state=BOGUS")
        raise AssertionError("expected HTTP 400 for invalid state")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_serialize_prediction_includes_planned_actions():
    """Planned interventions surface in /predictions via _context.remediation_plans."""
    from predictive_agent.server import _context, _serialize_prediction

    class _R:
        pod_key = "ns/pod"
        risk_score = 0.9

    _context.remediation_plans = {
        "ns/pod": {"execution_ready": True, "horizon": 3,
                   "steps": [{"action": "pod_restart"}]}
    }
    data = _serialize_prediction(_R())
    assert data["planned_actions"] is not None
    assert data["planned_actions"]["steps"][0]["action"] == "pod_restart"
    _context.remediation_plans = {}


def test_world_model_endpoint(test_server):
    """GET /world-model is a read-only what-if simulator."""
    from predictive_agent.remediator import RemediationManager
    from predictive_agent.server import _context

    _context.remediation_manager = RemediationManager(dry_run=True)
    with urllib.request.urlopen(
        "http://localhost:18082/world-model?state=CRITICAL&action=pod_restart&horizon=2"
    ) as resp:
        data = json.loads(resp.read().decode())
        assert resp.status == 200
        assert data["status"] == "ok"
        assert data["start_state"] == "CRITICAL"
        assert len(data["steps"]) == 2
        assert data["steps"][0]["action"] == "pod_restart"
        assert "expected_failed_interval" in data["steps"][0]
    # Invalid state -> 400.
    try:
        urllib.request.urlopen("http://localhost:18082/world-model?state=BOGUS")
        raise AssertionError("expected HTTP 400 for invalid state")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_plan_endpoint_policy_strategy(test_server):
    """GET /plan?strategy=policy returns the value-iteration plan."""
    from predictive_agent.remediator import RemediationManager
    from predictive_agent.server import _context

    _context.remediation_manager = RemediationManager(dry_run=True)
    with urllib.request.urlopen(
        "http://localhost:18082/plan?state=CRITICAL&strategy=policy"
    ) as resp:
        data = json.loads(resp.read().decode())
        assert resp.status == 200
        assert data["status"] == "ok"
        assert data["strategy"] == "policy"
        assert len(data["plan"]["steps"]) >= 1
