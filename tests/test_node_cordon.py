"""Tests for NodeCordonAction (REM-3)."""

from unittest.mock import MagicMock, patch

from predictive_agent.actions.node_cordon import NodeCordonAction
from predictive_agent.remediator import ActionContext


class TestNodeCordonShouldExecute:
    """Test NodeCordonAction.should_execute()."""

    def _make_pod_state(self, node_pressure=False):
        return MagicMock(node_pressure=node_pressure)

    def test_node_pressure_triggers(self):
        action = NodeCordonAction()
        pod = self._make_pod_state(node_pressure=True)
        assert action.should_execute(pod, None, 0.85) is True

    def test_no_node_pressure_does_not_trigger(self):
        action = NodeCordonAction()
        pod = self._make_pod_state(node_pressure=False)
        assert action.should_execute(pod, None, 0.85) is False

    def test_low_risk_does_not_trigger(self):
        action = NodeCordonAction()
        pod = self._make_pod_state(node_pressure=True)
        assert action.should_execute(pod, None, 0.7) is False

    def test_exact_threshold_does_not_trigger(self):
        """Risk exactly at 0.8 should NOT trigger (uses >)."""
        action = NodeCordonAction()
        pod = self._make_pod_state(node_pressure=True)
        assert action.should_execute(pod, None, 0.8) is False

    def test_just_above_threshold_triggers(self):
        """Risk at 0.81 should trigger."""
        action = NodeCordonAction()
        pod = self._make_pod_state(node_pressure=True)
        assert action.should_execute(pod, None, 0.81) is True


class TestNodeCordonExecute:
    """Test NodeCordonAction.execute()."""

    @patch("predictive_agent.actions.node_cordon.subprocess.run")
    def test_execute_dry_run(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="node cordoned (dry run)", stderr="")
        action = NodeCordonAction()
        ctx = ActionContext(namespace="default", dry_run=True, node_name="node-1")
        result = action.execute("pod-xyz", ctx)
        assert result.success is True
        assert result.dry_run is True
        assert result.action == "node_cordon"
        cmd = mock_run.call_args[0][0]
        assert "cordon" in cmd
        assert "node-1" in cmd
        assert "--dry-run=server" in cmd

    @patch("predictive_agent.actions.node_cordon.subprocess.run")
    def test_execute_uses_node_name_not_pod_name(self, mock_run):
        """Regression: cordon target must be the node name, not the pod name."""
        mock_run.return_value = MagicMock(returncode=0, stdout="node/node-1 cordoned", stderr="")
        action = NodeCordonAction()
        ctx = ActionContext(namespace="default", dry_run=False, node_name="node-1")
        result = action.execute("openproject-f79c8ff4b-z5frq", ctx)
        cmd = mock_run.call_args[0][0]
        assert "node-1" in cmd
        assert "openproject-f79c8ff4b-z5frq" not in cmd
        assert result.target == "node-1"

    @patch("predictive_agent.actions.node_cordon.subprocess.run")
    def test_execute_real(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="node/node-1 cordoned", stderr="")
        action = NodeCordonAction()
        ctx = ActionContext(namespace="default", dry_run=False, node_name="node-1")
        result = action.execute("node-1", ctx)
        assert result.success is True
        assert result.dry_run is False
        cmd = mock_run.call_args[0][0]
        assert "--dry-run=server" not in cmd

    @patch("predictive_agent.actions.node_cordon.subprocess.run")
    def test_execute_fallback_to_target(self, mock_run):
        """When node_name is empty, falls back to target (legacy)."""
        mock_run.return_value = MagicMock(returncode=0, stdout="node/node-1 cordoned", stderr="")
        action = NodeCordonAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("node-1", ctx)
        assert result.success is True
        cmd = mock_run.call_args[0][0]
        assert "node-1" in cmd

    @patch("predictive_agent.actions.node_cordon.subprocess.run")
    def test_execute_no_node_name(self, mock_run):
        """Empty node name and empty target → graceful failure."""
        action = NodeCordonAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("", ctx)
        assert result.success is False
        mock_run.assert_not_called()

    @patch("predictive_agent.actions.node_cordon.subprocess.run")
    def test_execute_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error: node not found")
        action = NodeCordonAction()
        ctx = ActionContext(namespace="default", dry_run=False, node_name="node-1")
        result = action.execute("node-1", ctx)
        assert result.success is False
        assert "not found" in result.message

    @patch("predictive_agent.actions.node_cordon.subprocess.run")
    def test_execute_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="kubectl", timeout=30)
        action = NodeCordonAction()
        ctx = ActionContext(namespace="default", dry_run=False, node_name="node-1")
        result = action.execute("node-1", ctx)
        assert result.success is False
        assert "timed out" in result.message

    @patch("predictive_agent.actions.node_cordon.subprocess.run")
    def test_execute_includes_drain_recommendation(self, mock_run):
        """Test that successful cordon includes manual drain recommendation."""
        mock_run.return_value = MagicMock(returncode=0, stdout="node/node-1 cordoned", stderr="")
        action = NodeCordonAction()
        ctx = ActionContext(namespace="default", dry_run=False, node_name="node-1")
        result = action.execute("node-1", ctx)
        assert "drain" in result.message.lower()

    @patch("predictive_agent.actions.node_cordon.subprocess.run")
    def test_execute_exception(self, mock_run):
        mock_run.side_effect = RuntimeError("unexpected")
        action = NodeCordonAction()
        ctx = ActionContext(namespace="default", dry_run=False, node_name="node-1")
        result = action.execute("node-1", ctx)
        assert result.success is False
        assert "unexpected" in result.message

    @patch("predictive_agent.actions.node_cordon.subprocess.run")
    def test_execute_timeout_30s(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="cordoned", stderr="")
        action = NodeCordonAction()
        ctx = ActionContext(namespace="default", dry_run=False, node_name="node-1")
        action.execute("node-1", ctx)
        kwargs = mock_run.call_args[1]
        assert kwargs["timeout"] == 30
