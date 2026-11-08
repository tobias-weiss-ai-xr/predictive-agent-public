"""Contract tests — verify HTTP API contract guarantees.

SOTA paradigm: Contract tests verify that the system meets its API contract.
These tests define the expected behavior of every HTTP endpoint: response
codes, content types, JSON structure, and field types. If the contract changes,
these tests break, alerting developers to breaking changes.

Covers:
- /healthz: 200, {"status": "ok"}, always available
- /ready: 200, {"status": "ready"/"not_ready"}, depends on state
- /metrics: 200, text/plain, Prometheus format (opendesk_predictive_agent_* metrics)
- /status: 200, application/json, specific fields
- /predictions: 200, application/json, array of predictions with required fields
- /state: 200, application/json, pods dict with PodTracker fields
- /history: 200, application/json, array of analysis entries
- /reanalyze: 200, {"status": "ok"}, triggers callback
- /cache: 200, application/json, cache dict with total count
- Error handling: invalid endpoints return 404
"""
import json
import time
import urllib.error
import urllib.request

import pytest

from predictive_agent.collector import COLLECTOR_STATE_HEALTHY, set_collector_state
from predictive_agent.predictor import PredictionResult, Predictor
from predictive_agent.server import start_server
from predictive_agent.state_model import StateModel

# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def contract_server():
    """Start a server with populated state for contract testing."""
    sm = StateModel()
    sm.update_pod("contract-ns", "pod-a", memory_mib=500, memory_limit_mib=1024,
                   cpu_m=100, restart_count=0, log_errors=0, node_pressure=False)
    sm.update_pod("contract-ns", "pod-b", memory_mib=900, memory_limit_mib=1024,
                   cpu_m=800, restart_count=3, log_errors=5, node_pressure=True)

    pred = Predictor(risk_threshold=0.3)
    pred.add_prediction("contract-ns/pod-a", PredictionResult(
        "contract-ns/pod-a", 0.05, None, 0.95, "HEALTHY", 0.01, 0.001, 48.8, 10.0
    ))
    pred.add_prediction("contract-ns/pod-b", PredictionResult(
        "contract-ns/pod-b", 0.85, 30, 0.90, "STRESSED", 0.15, 0.05, 87.9, 80.0
    ))

    cache = {"last_analysis": {"pod": "contract-ns/pod-b", "severity": "high"}}
    history = [{"timestamp": "2025-01-01T00:00:00Z", "pod": "contract-ns/pod-b", "action": "investigate"}]

    # P4: pin the collector to HEALTHY for a deterministic /ready contract.
    # The contract asserts the *shape* of the response (200/ready when healthy);
    # degraded/down 503 behaviour is covered by tests/test_degraded_mode.py.
    set_collector_state(COLLECTOR_STATE_HEALTHY, has_prior=True, last_collect=time.time())

    server = start_server(18100, 18101, state_model=sm, predictor=pred,
                          cache=cache, history=history)
    yield server
    server.shutdown()


# ─── /healthz contract ──────────────────────────────────────────────────────

class TestHealthzContract:
    """Contract tests for /healthz."""

    def test_returns_200(self, contract_server):
        with urllib.request.urlopen("http://localhost:18101/healthz") as resp:
            assert resp.status == 200

    def test_returns_json(self, contract_server):
        with urllib.request.urlopen("http://localhost:18101/healthz") as resp:
            data = json.loads(resp.read().decode())
            assert isinstance(data, dict)

    def test_has_status_ok(self, contract_server):
        with urllib.request.urlopen("http://localhost:18101/healthz") as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"

    def test_content_type_json(self, contract_server):
        with urllib.request.urlopen("http://localhost:18101/healthz") as resp:
            content_type = resp.headers.get("Content-Type", "")
            assert "application/json" in content_type


# ─── /ready contract ────────────────────────────────────────────────────────

class TestReadyContract:
    """Contract tests for /ready."""

    def test_returns_200(self, contract_server):
        with urllib.request.urlopen("http://localhost:18101/ready") as resp:
            assert resp.status == 200

    def test_returns_json(self, contract_server):
        with urllib.request.urlopen("http://localhost:18101/ready") as resp:
            data = json.loads(resp.read().decode())
            assert isinstance(data, dict)

    def test_has_status_field(self, contract_server):
        with urllib.request.urlopen("http://localhost:18101/ready") as resp:
            data = json.loads(resp.read().decode())
            assert "status" in data
            assert data["status"] in ("ready", "not_ready")


# ─── /metrics contract ──────────────────────────────────────────────────────

class TestMetricsContract:
    """Contract tests for /metrics."""

    def test_returns_200(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/metrics") as resp:
            assert resp.status == 200

    def test_content_type_text(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/metrics") as resp:
            content_type = resp.headers.get("Content-Type", "")
            assert "text/plain" in content_type

    def test_has_prometheus_metrics(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/metrics") as resp:
            text = resp.read().decode()
            assert "opendesk_predictive_agent_" in text

    def test_has_pods_tracked_metric(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/metrics") as resp:
            text = resp.read().decode()
            assert "opendesk_predictive_agent_pods_tracked" in text

    def test_has_predictions_count_metric(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/metrics") as resp:
            text = resp.read().decode()
            assert "opendesk_predictive_agent_predictions_count" in text

    def test_has_risk_score_metric(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/metrics") as resp:
            text = resp.read().decode()
            assert "opendesk_predictive_agent_pod_risk_score" in text

    def test_metrics_are_numeric(self, contract_server):
        """All metric values should be numeric."""
        with urllib.request.urlopen("http://localhost:18100/metrics") as resp:
            text = resp.read().decode()
            for line in text.strip().split("\n"):
                if line and not line.startswith("#"):
                    parts = line.rsplit(" ", 1)
                    if len(parts) == 2:
                        try:
                            float(parts[1])
                        except ValueError:
                            pytest.fail(f"Non-numeric metric value: {line}")


# ─── /status contract ───────────────────────────────────────────────────────

class TestStatusContract:
    """Contract tests for /status."""

    def test_returns_200(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/status") as resp:
            assert resp.status == 200

    def test_returns_json(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/status") as resp:
            data = json.loads(resp.read().decode())
            assert isinstance(data, dict)

    def test_has_pod_count(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/status") as resp:
            data = json.loads(resp.read().decode())
            assert "pod_count" in data
            assert isinstance(data["pod_count"], int)

    def test_has_predictions_count(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/status") as resp:
            data = json.loads(resp.read().decode())
            assert "predictions_count" in data
            assert isinstance(data["predictions_count"], int)

    def test_has_at_risk_count(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/status") as resp:
            data = json.loads(resp.read().decode())
            assert "at_risk_count" in data
            assert isinstance(data["at_risk_count"], int)

    def test_has_uptime(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/status") as resp:
            data = json.loads(resp.read().decode())
            assert "uptime_seconds" in data
            assert isinstance(data["uptime_seconds"], (int, float))


# ─── /predictions contract ──────────────────────────────────────────────────

class TestPredictionsContract:
    """Contract tests for /predictions."""

    def test_returns_200(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/predictions") as resp:
            assert resp.status == 200

    def test_returns_json(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/predictions") as resp:
            data = json.loads(resp.read().decode())
            assert isinstance(data, dict)

    def test_has_total(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/predictions") as resp:
            data = json.loads(resp.read().decode())
            assert "total" in data
            assert isinstance(data["total"], int)

    def test_has_predictions_array(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/predictions") as resp:
            data = json.loads(resp.read().decode())
            assert "predictions" in data
            assert isinstance(data["predictions"], list)

    def test_prediction_structure(self, contract_server):
        """Each prediction must have required fields with correct types."""
        with urllib.request.urlopen("http://localhost:18100/predictions") as resp:
            data = json.loads(resp.read().decode())
            for pred in data["predictions"]:
                assert "pod_key" in pred and isinstance(pred["pod_key"], str)
                assert "risk_score" in pred and isinstance(pred["risk_score"], (int, float))
                assert 0.0 <= pred["risk_score"] <= 0.99
                assert "confidence" in pred and isinstance(pred["confidence"], (int, float))
                assert 0.0 <= pred["confidence"] <= 1.0
                assert "markov_state" in pred and isinstance(pred["markov_state"], str)
                assert "ttf_minutes" in pred
                assert pred["ttf_minutes"] is None or isinstance(pred["ttf_minutes"], (int, float))

    def test_predictions_sorted_by_risk(self, contract_server):
        """Predictions should be sorted by risk score (descending)."""
        with urllib.request.urlopen("http://localhost:18100/predictions") as resp:
            data = json.loads(resp.read().decode())
            risks = [p["risk_score"] for p in data["predictions"]]
            assert risks == sorted(risks, reverse=True)


# ─── /state contract ────────────────────────────────────────────────────────

class TestStateContract:
    """Contract tests for /state."""

    def test_returns_200(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/state") as resp:
            assert resp.status == 200

    def test_returns_json(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/state") as resp:
            data = json.loads(resp.read().decode())
            assert isinstance(data, dict)

    def test_has_pods(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/state") as resp:
            data = json.loads(resp.read().decode())
            assert "pods" in data
            assert isinstance(data["pods"], dict)

    def test_pod_structure(self, contract_server):
        """Each pod must have required fields with correct types."""
        with urllib.request.urlopen("http://localhost:18100/state") as resp:
            data = json.loads(resp.read().decode())
            for pod_key, pod_data in data["pods"].items():
                assert isinstance(pod_key, str)
                assert "memory_mib" in pod_data
                assert "memory_limit_mib" in pod_data
                assert "cpu_m" in pod_data
                assert "restart_count" in pod_data
                assert "state" in pod_data


# ─── /history contract ──────────────────────────────────────────────────────

class TestHistoryContract:
    """Contract tests for /history."""

    def test_returns_200(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/history") as resp:
            assert resp.status == 200

    def test_returns_json_array(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/history") as resp:
            data = json.loads(resp.read().decode())
            assert isinstance(data, list)

    def test_history_entries_have_timestamp(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/history") as resp:
            data = json.loads(resp.read().decode())
            for entry in data:
                assert "timestamp" in entry


# ─── /reanalyze contract ────────────────────────────────────────────────────

class TestReanalyzeContract:
    """Contract tests for /reanalyze."""

    def test_returns_200(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/reanalyze") as resp:
            assert resp.status == 200

    def test_returns_status_ok(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/reanalyze") as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"


# ─── /cache contract ────────────────────────────────────────────────────────

class TestCacheContract:
    """Contract tests for /cache."""

    def test_returns_200(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/cache") as resp:
            assert resp.status == 200

    def test_returns_json(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/cache") as resp:
            data = json.loads(resp.read().decode())
            assert isinstance(data, dict)

    def test_has_cache_and_total(self, contract_server):
        with urllib.request.urlopen("http://localhost:18100/cache") as resp:
            data = json.loads(resp.read().decode())
            assert "cache" in data
            assert "total" in data
            assert isinstance(data["total"], int)


# ─── Error handling contract ────────────────────────────────────────────────

class TestErrorHandlingContract:
    """Contract tests for error handling."""

    def test_unknown_endpoint_returns_404(self, contract_server):
        """Unknown endpoints should return 404."""
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen("http://localhost:18100/nonexistent")
        assert exc_info.value.code == 404

    def test_unknown_endpoint_health_port_returns_404(self, contract_server):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen("http://localhost:18101/nonexistent")
        assert exc_info.value.code == 404
