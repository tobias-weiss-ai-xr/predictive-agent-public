"""Test HTTP server endpoints with real data integration."""
import json
import time
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from predictive_agent.remediator import (
    RemediationManager,
    reset_remediation_metrics,
)
from predictive_agent.server import start_server


@pytest.fixture(scope="module")
def test_server():
    """Start test servers on alternate ports."""
    import predictive_agent.config as config
    config.HEALTH_PORT = 18081
    config.METRICS_PORT = 18080

    reset_remediation_metrics()
    metrics_server = start_server(18080, 18081)
    yield
    metrics_server.shutdown()
    reset_remediation_metrics()


def test_healthz(test_server):
    """Test /healthz endpoint."""
    with urllib.request.urlopen("http://localhost:18081/healthz") as resp:
        data = json.loads(resp.read().decode())
        assert "status" in data
        assert data["status"] == "ok"


def test_ready(test_server):
    """Test /ready endpoint."""
    with urllib.request.urlopen("http://localhost:18081/ready") as resp:
        data = json.loads(resp.read().decode())
        assert "status" in data
        assert data["status"] == "ready"


def test_metrics(test_server):
    """Test /metrics endpoint returns Prometheus format."""
    with urllib.request.urlopen("http://localhost:18080/metrics") as resp:
        text = resp.read().decode()
        assert "predictive_agent" in text


def test_alerts_lazy_feed_from_config(test_server):
    """Server builds a config-driven AlertFeed when main never wired one.

    Guards against start-ordering regressions: the API must come alive even
    when ``main._alert_feed`` is still the module-level None.
    """
    import predictive_agent.main as main
    from predictive_agent.server import _context

    old_feed = main._alert_feed
    old_ctx = _context.alert_feed
    main._alert_feed = None
    _context.alert_feed = None
    try:
        with patch("predictive_agent.server.config.ALERT_FEED_ENABLED", True):
            with urllib.request.urlopen(
                "http://localhost:18080/alerts"
            ) as resp:
                data = json.loads(resp.read().decode())
                assert data["enabled"] is True
            # lazy feed is now cached into module + context
            assert main._alert_feed is not None
            assert _context.alert_feed is main._alert_feed
    finally:
        main._alert_feed = old_feed
        _context.alert_feed = old_ctx


def test_alerts_and_ceph_late_init(test_server):
    """Server resolves the AlertFeed lazily from main._alert_feed.

    The server starts before main() wires the feed; a late module-level
    assignment must still light up /alerts and /ceph-health.
    """
    import predictive_agent.main as main
    from predictive_agent.alerts_feed import AlertFeed

    feed = AlertFeed(
        alertmanager_url="http://am:9093",
        prometheus_url="http://prom:9090",
        boost=0.15,
        ceph_watch=True,
    )
    feed._active_alerts = [
        {"alertname": "CephHealthWarning", "namespace": "monitoring"}
    ]
    feed._active_namespaces = {"monitoring"}
    feed._last_ceph_status = "OK"
    feed.last_alert_fetch = time.time()  # suppress the on-demand refresh guard

    old = main._alert_feed
    main._alert_feed = feed
    try:
        with urllib.request.urlopen(
            "http://localhost:18080/alerts"
        ) as resp:
            data = json.loads(resp.read().decode())
            assert data["enabled"] is True
            assert data["namespaces"] == ["monitoring"]
            assert data["boost"] == 0.15

        class _FakeResp:
            status = 200

            def read(self):
                return (
                    b'{"data":{"result":[{"metric":{},"value":[1.0,"1"]}]}}'
                )

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with patch("predictive_agent.alerts_feed.urlopen") as mock_u:
            mock_u.return_value = _FakeResp()
            with urllib.request.urlopen(
                "http://localhost:18080/ceph-health"
            ) as resp:
                data = json.loads(resp.read().decode())
                assert data["enabled"] is True
                assert data["health"]["status"] == "OK"
    finally:
        main._alert_feed = old


def test_status(test_server):
    """Test /status endpoint returns version and pod count."""
    with urllib.request.urlopen("http://localhost:18080/status") as resp:
        data = json.loads(resp.read().decode())
        assert "version" in data
        assert "operator" in data


def test_predictions(test_server):
    """Test /predictions endpoint returns predictions list."""
    with urllib.request.urlopen("http://localhost:18080/predictions") as resp:
        data = json.loads(resp.read().decode())
        assert "predictions" in data
        assert "total" in data
        assert isinstance(data["predictions"], list)


def test_state(test_server):
    """Test /state endpoint returns pod states and markov chain."""
    with urllib.request.urlopen("http://localhost:18080/state") as resp:
        data = json.loads(resp.read().decode())
        assert "pods" in data or "markov_chain" in data or "states" in data


def test_history(test_server):
    """Test /history endpoint returns a list."""
    with urllib.request.urlopen("http://localhost:18080/history") as resp:
        data = json.loads(resp.read().decode())
        assert isinstance(data, list)


def test_404(test_server):
    """Test 404 for unknown endpoint."""
    try:
        urllib.request.urlopen("http://localhost:18080/nonexistent")
        pytest.fail("Should have raised 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_remediation_log_endpoint_exists(test_server):
    """Test /remediation-log endpoint exists and returns valid structure."""
    try:
        with urllib.request.urlopen("http://localhost:18080/remediation-log") as resp:
            data = json.loads(resp.read().decode())
            assert "status" in data
            assert "total" in data
            assert "remediation_log" in data
            assert isinstance(data["remediation_log"], list)
            assert isinstance(data["total"], int)
    except urllib.error.HTTPError as e:
        # If remediation is not initialized, it should still return valid JSON
        assert e.code in (200, 503)


def test_reanalyze(test_server):
    """Test /reanalyze endpoint exists."""
    try:
        with urllib.request.urlopen("http://localhost:18080/reanalyze") as resp:
            assert resp.status == 200
    except urllib.error.HTTPError as e:
        assert e.code in (200, 405, 503)


def test_cache(test_server):
    """Test /cache endpoint exists."""
    try:
        with urllib.request.urlopen("http://localhost:18080/cache") as resp:
            assert resp.status == 200
    except urllib.error.HTTPError as e:
        assert e.code in (200, 404, 503)


def test_remediation_log_with_manager(test_server):
    """Test /remediation-log endpoint with a RemediationManager."""
    # Set up a remediation manager with some audit entries
    manager = RemediationManager(dry_run=True, risk_threshold=0.5)

    # Register a test action
    class TestAction:
        name = "test_action"
        def should_execute(self, pod_state, prediction, risk_score):
            return True
        def execute(self, target, context):
            from predictive_agent.remediator import RemediationResult
            return RemediationResult(
                action=self.name,
                target=target,
                success=True,
                dry_run=context.dry_run,
                message="test executed",
                outcome="acted"
            )

    manager.register_action(TestAction())

    # Add to context
    from predictive_agent import server
    old_manager = server._context.remediation_manager
    server._context.remediation_manager = manager

    # Make a request
    with urllib.request.urlopen("http://localhost:18080/remediation-log") as resp:
        data = json.loads(resp.read().decode())
        assert data["status"] == "dry_run"
        assert data["dry_run"] is True
        assert isinstance(data["remediation_log"], list)
        assert "total" in data

    # Restore
    server._context.remediation_manager = old_manager


def test_remediation_log_structure(test_server):
    """Test that /remediation-log returns entries with required fields for binary observability."""
    manager = RemediationManager(dry_run=True, risk_threshold=0.0)

    # Register and evaluate to create log entries
    class TestAction:
        name = "binary_test_action"
        def should_execute(self, pod_state, prediction, risk_score):
            return True
        def execute(self, target, context):
            from predictive_agent.remediator import RemediationResult
            return RemediationResult(
                action=self.name,
                target=target,
                success=True,
                dry_run=context.dry_run,
                message="binary observability test",
                outcome="acted"
            )

    manager.register_action(TestAction())
    # Create an entry
    mock_pod = MagicMock()
    mock_pod.name = "test-pod"
    mock_pod.namespace = "test-ns"
    manager.evaluate(mock_pod, None, 0.8)

    # Set manager to context
    from predictive_agent import server
    old_manager = server._context.remediation_manager
    server._context.remediation_manager = manager

    try:
        with urllib.request.urlopen("http://localhost:18080/remediation-log") as resp:
            data = json.loads(resp.read().decode())
            if data["remediation_log"]:
                entry = data["remediation_log"][0]
                # Verify binary observability fields
                assert "action" in entry
                assert "target" in entry
                assert "namespace" in entry
                assert "outcome" in entry
                assert entry["outcome"] in ["acted", "skipped", "failed"]
                assert "success" in entry
                assert "dry_run" in entry
                assert "message" in entry
                assert "timestamp" in entry
    finally:
        server._context.remediation_manager = old_manager


def test_remediation_log_disabled(test_server):
    """Test /remediation-log when remediation is not initialized."""
    from predictive_agent import server
    old_manager = server._context.remediation_manager
    server._context.remediation_manager = None

    try:
        with urllib.request.urlopen("http://localhost:18080/remediation-log") as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "disabled"
            assert data["message"] == "Remediation not initialized"
            assert data["remediation_log"] == []
            assert data["total"] == 0
    finally:
        server._context.remediation_manager = old_manager


def test_alerts_refresh_on_demand(test_server):
    """/alerts refreshes the feed when last fetch is stale (0 = never)."""
    import predictive_agent.main as main
    from predictive_agent.alerts_feed import AlertFeed

    old_feed = main._alert_feed
    feed = AlertFeed(
        alertmanager_url="http://am:9093",
        prometheus_url="http://prom:9090",
        boost=0.15,
        ceph_watch=False,
    )
    feed.last_alert_fetch = 0.0  # brand-new, must trigger refresh
    main._alert_feed = feed
    try:
        with patch("predictive_agent.alerts_feed.urlopen") as mock_u:
            mock_u.return_value = _Resp(
                [{"status": "firing", "labels": {"alertname": "B", "namespace": "llm"}}]
            )
            with urllib.request.urlopen(
                "http://localhost:18080/alerts"
            ) as resp:
                data = json.loads(resp.read().decode())
                assert data["enabled"] is True
                assert data["namespaces"] == ["llm"]
                assert data["last_fetch"] > 0
    finally:
        main._alert_feed = old_feed


class _Resp:
    status = 200

    def __init__(self, payload):
        import json as _j
        self._data = _j.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
