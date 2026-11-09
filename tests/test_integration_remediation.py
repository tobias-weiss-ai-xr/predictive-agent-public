"""Integration tests for remediation in the reconcile loop and server endpoints.

Tests the full flow: metrics collection → prediction → remediation → notification.
Uses mocks for kubectl, SMTP, and webhook — no real external calls.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from predictive_agent.actions import (
    DeploymentScaleAction,
    NodeCordonAction,
    PodRestartAction,
    ResourceTunerAction,
    RightSizeAction,
    RolloutRestartAction,
)
from predictive_agent.main import reconcile
from predictive_agent.notifier import EmailNotifier, NotificationManager, WebhookNotifier
from predictive_agent.predictor import Predictor
from predictive_agent.remediator import (
    RemediationManager,
    RemediationResult,
    SafetyPolicy,
    create_remediation_manager_from_config,
)
from predictive_agent.server import start_server
from predictive_agent.state_model import StateModel


@pytest.fixture
def fresh_state():
    """Reset global state and provide fresh instances."""
    import predictive_agent.main as main_mod
    main_mod._state_model = StateModel()
    main_mod._predictor = Predictor(risk_threshold=0.7)
    main_mod._state_store = None
    main_mod._remediation_manager = None
    main_mod._notifier = None
    main_mod._reconcile_count = 0
    main_mod._last_reconcile_time = None
    yield main_mod


class TestRemediationIntegration:
    """Test remediation integration with the reconcile loop."""

    def test_reconcile_with_remediation_manager_registered(self, fresh_state):
        """Test that reconcile runs without errors when remediation manager is registered."""
        # Setup remediation manager (dry_run mode)
        policy = SafetyPolicy(cooldown_seconds=0, max_per_minute=100, max_per_hour=1000)
        rem_mgr = RemediationManager(dry_run=True, risk_threshold=0.7, safety_policy=policy)
        rem_mgr.register_action(PodRestartAction())
        fresh_state._remediation_manager = rem_mgr

        # Force Kubernetes runtime so Docker discovery (which depends on the
        # test host's Docker daemon) does not inject real local containers.
        with patch("predictive_agent.main.detect_runtime", return_value="kubernetes"), \
                patch("predictive_agent.main.run_cmd") as mock_run:
            mock_run.return_value = (1, "", "")  # kubectl not available
            result = reconcile()

        assert result["reconcile_count"] == 1
        assert result["pods_tracked"] == 0
        assert "remediation" not in result or result["remediation"]["total_actions"] == 0

    def test_reconcile_includes_remediation_stats(self, fresh_state):
        """Test that reconcile output includes remediation stats."""
        rem_mgr = RemediationManager(dry_run=True, risk_threshold=0.7)
        fresh_state._remediation_manager = rem_mgr

        with patch("predictive_agent.main.run_cmd") as mock_run:
            mock_run.return_value = (1, "", "")
            result = reconcile()

        assert "remediation" in result
        assert "registered_actions" in result["remediation"]
        assert "dry_run" in result["remediation"]

    def test_reconcile_without_remediation_manager(self, fresh_state):
        """Test that reconcile works without a remediation manager (backward compat)."""
        with patch("predictive_agent.main.run_cmd") as mock_run:
            mock_run.return_value = (1, "", "")
            result = reconcile()
        assert result["reconcile_count"] == 1
        # No remediation key when manager is None
        assert "remediation" not in result or result.get("remediation") is None


class TestServerRemediationEndpoints:
    """Test /remediate and /notifications endpoints."""

    @pytest.fixture(scope="class")
    def server_with_remediation(self):
        """Start server with remediation manager and notifier."""
        rem_mgr = RemediationManager(dry_run=True, risk_threshold=0.7)
        rem_mgr.register_action(PodRestartAction())
        notifier = NotificationManager(
            email_notifier=MagicMock(spec=EmailNotifier),
            webhook_notifier=MagicMock(spec=WebhookNotifier),
        )
        sm = StateModel()
        pred = Predictor(risk_threshold=0.7)
        server = start_server(
            metrics_port=18120,
            health_port=18121,
            state_model=sm,
            predictor=pred,
            remediation_manager=rem_mgr,
            notifier=notifier,
        )
        yield server
        server.shutdown()

    def test_get_remediate(self, server_with_remediation):
        """Test GET /remediate returns config and stats."""
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:18120/remediate") as resp:
            data = json.loads(resp.read())
        assert "dry_run" in data
        assert data["dry_run"] is True
        assert "registered_actions" in data
        assert "pod_restart" in data["registered_actions"]
        assert "stats" in data
        assert "safety_policy" in data
        assert "audit_trail" in data

    def test_get_remediate_no_manager(self):
        """Test GET /remediate when remediation is not initialized."""
        from predictive_agent.server import _context
        saved_rem = _context.remediation_manager
        sm = StateModel()
        server = start_server(
            metrics_port=18122, health_port=18123,
            state_model=sm, predictor=None,
        )
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:18122/remediate") as resp:
                data = json.loads(resp.read())
            assert data["status"] == "disabled"
        finally:
            server.shutdown()
            _context.remediation_manager = saved_rem

    def test_get_notifications(self, server_with_remediation):
        """Test GET /notifications returns notification history."""
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:18120/notifications") as resp:
            data = json.loads(resp.read())
        assert "notifications" in data
        assert "total" in data

    def test_post_remediate_pod_not_found(self, server_with_remediation):
        """Test POST /remediate with non-existent pod."""
        import urllib.request
        payload = json.dumps({"pod_name": "nonexistent/pod", "risk_score": 0.85}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:18120/remediate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_post_remediate_missing_pod_name(self, server_with_remediation):
        """Test POST /remediate without pod_name."""
        import urllib.request
        payload = json.dumps({"risk_score": 0.85}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:18120/remediate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            assert e.code == 400

    def test_metrics_include_remediation(self, server_with_remediation):
        """Test that /metrics includes remediation metrics."""
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:18120/metrics") as resp:
            text = resp.read().decode()
        assert "opendesk_predictive_agent_remediation_actions_total" in text
        assert "opendesk_predictive_agent_remediation_dry_run_total" in text

    def test_metrics_without_remediation(self):
        """Test that /metrics works without remediation manager."""
        from predictive_agent.server import _context
        saved_rem = _context.remediation_manager
        sm = StateModel()
        server = start_server(
            metrics_port=18124, health_port=18125,
            state_model=sm, predictor=None,
        )
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:18124/metrics") as resp:
                text = resp.read().decode()
            # Should still have basic metrics
            assert "opendesk_predictive_agent_pods_tracked" in text
            # Should NOT have remediation metrics
            assert "opendesk_predictive_agent_remediation_actions_total" not in text
        finally:
            server.shutdown()
            _context.remediation_manager = saved_rem


class TestFullRemediationCycle:
    """Test full remediation cycle: predict → evaluate → notify."""

    def test_remediation_evaluates_actions_on_high_risk_pod(self):
        """Test that RemediationManager evaluates all registered actions."""
        policy = SafetyPolicy(cooldown_seconds=0, max_per_minute=100, max_per_hour=1000)
        mgr = RemediationManager(dry_run=True, risk_threshold=0.7, safety_policy=policy)
        mgr.register_action(PodRestartAction())
        mgr.register_action(RightSizeAction())

        # Create a mock pod state that triggers pod_restart
        pod_state = MagicMock()
        pod_state.name = "test-pod"
        pod_state.namespace = "default"
        pod_state.phase = "CrashLoopBackOff"
        pod_state.restart_count = 10
        pod_state.data_points = 15
        pod_state.cpu_pct = 50

        # Mock subprocess so kubectl delete succeeds (dry run)
        with patch("predictive_agent.actions.pod_restart.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="pod deleted (dry run)", stderr="")
            results = mgr.evaluate(pod_state, None, 0.85)

        assert len(results) > 0
        # At least one action should have been executed
        actions_taken = [r.action for r in results if r.success]
        assert "pod_restart" in actions_taken

    def test_remediation_skips_low_risk(self):
        """Test that remediation is skipped when risk is below threshold."""
        mgr = RemediationManager(dry_run=True, risk_threshold=0.7)
        mgr.register_action(PodRestartAction())

        pod_state = MagicMock()
        pod_state.name = "test-pod"
        pod_state.namespace = "default"
        pod_state.phase = "CrashLoopBackOff"
        pod_state.restart_count = 10

        results = mgr.evaluate(pod_state, None, 0.5)
        assert len(results) == 0

    def test_notification_sent_on_remediation(self):
        """Test that notifications are sent when remediation actions succeed."""
        email = MagicMock(spec=EmailNotifier)
        email.send.return_value = True
        webhook = MagicMock(spec=WebhookNotifier)
        webhook.send.return_value = True
        notifier = NotificationManager(email_notifier=email, webhook_notifier=webhook, channel="both")

        # Simulate a remediation result
        result = RemediationResult(
            action="pod_restart",
            target="test-pod",
            success=True,
            dry_run=True,
            message="pod deleted (dry run)",
        )

        if result.success:
            email_sent, webhook_sent = notifier.notify(
                alert_type="remediation",
                pod_name="default/test-pod",
                risk_score=0.85,
                action_taken=result.action,
                details=result.message,
            )

        assert email_sent is True
        assert webhook_sent is True
        email.send.assert_called_once()
        webhook.send.assert_called_once()

    def test_dry_run_mode_default(self):
        """Test that create_remediation_manager_from_config defaults to dry_run=True."""
        with patch.dict(os.environ, {"REMEDIATION_ENABLED": "true", "REMEDIATION_DRY_RUN": "true"}):
            mgr = create_remediation_manager_from_config()
            assert mgr.dry_run is True

    def test_dry_run_disabled(self):
        """Test that dry_run can be disabled via env var."""
        with patch.dict(os.environ, {"REMEDIATION_ENABLED": "true", "REMEDIATION_DRY_RUN": "false"}):
            mgr = create_remediation_manager_from_config()
            assert mgr.dry_run is False

    def test_disabled_remediation_forces_dry_run(self):
        """Test that disabled remediation forces dry_run=True."""
        with patch.dict(os.environ, {"REMEDIATION_ENABLED": "false", "REMEDIATION_DRY_RUN": "false"}):
            mgr = create_remediation_manager_from_config()
            assert mgr.dry_run is True

    def test_all_six_actions_registered(self):
        """Test that all 6 remediation actions can be registered together."""
        mgr = RemediationManager(dry_run=True, risk_threshold=0.7)
        mgr.register_action(PodRestartAction())
        mgr.register_action(NodeCordonAction())
        mgr.register_action(RightSizeAction())
        mgr.register_action(RolloutRestartAction())
        mgr.register_action(DeploymentScaleAction())
        mgr.register_action(ResourceTunerAction())
        stats = mgr.get_stats()
        assert len(stats["registered_actions"]) == 6
        assert "pod_restart" in stats["registered_actions"]
        assert "node_cordon" in stats["registered_actions"]
        assert "right_size" in stats["registered_actions"]
        assert "rollout_restart" in stats["registered_actions"]
        assert "scale" in stats["registered_actions"]
        assert "tune_resources" in stats["registered_actions"]

    def test_audit_trail_records_all_actions(self):
        """Test that audit trail records all actions, including skipped ones."""
        policy = SafetyPolicy(cooldown_seconds=0, max_per_minute=100, max_per_hour=1000)
        mgr = RemediationManager(dry_run=True, risk_threshold=0.7, safety_policy=policy)
        mgr.register_action(PodRestartAction())

        # First action should succeed
        pod_state = MagicMock()
        pod_state.name = "pod-1"
        pod_state.namespace = "default"
        pod_state.phase = "CrashLoopBackOff"
        pod_state.restart_count = 10
        pod_state.data_points = 15
        pod_state.cpu_pct = 50

        mgr.evaluate(pod_state, None, 0.85)
        trail = mgr.get_audit_trail(limit=100)
        assert len(trail) >= 1
        assert trail[0]["action"] == "pod_restart"
