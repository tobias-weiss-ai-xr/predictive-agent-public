"""Tests for RolloutRestartAction (REM-5)."""

from unittest.mock import MagicMock, patch

from predictive_agent.actions.rollout_restart import RolloutRestartAction
from predictive_agent.remediator import ActionContext


class TestRolloutRestartShouldExecute:
    """Test RolloutRestartAction.should_execute()."""

    def _make_pod_state(self, restart_count=5, failing_pods=2, namespace="default"):
        return MagicMock(
            restart_count=restart_count,
            failing_pods_in_deployment=failing_pods,
            namespace=namespace,
        )

    def test_multiple_failing_pods_triggers(self):
        action = RolloutRestartAction()
        pod = self._make_pod_state(restart_count=5, failing_pods=2)
        assert action.should_execute(pod, None, 0.8) is True

    def test_single_failing_pod_does_not_trigger(self):
        action = RolloutRestartAction()
        pod = self._make_pod_state(restart_count=5, failing_pods=1)
        assert action.should_execute(pod, None, 0.8) is False

    def test_low_restart_count_does_not_trigger(self):
        action = RolloutRestartAction()
        pod = self._make_pod_state(restart_count=2, failing_pods=2)
        assert action.should_execute(pod, None, 0.8) is False

    def test_low_risk_does_not_trigger(self):
        action = RolloutRestartAction()
        pod = self._make_pod_state(restart_count=5, failing_pods=2)
        assert action.should_execute(pod, None, 0.7) is False

    def test_protected_namespace_blocks(self):
        action = RolloutRestartAction()
        pod = self._make_pod_state(restart_count=5, failing_pods=2, namespace="kube-system")
        assert action.should_execute(pod, None, 0.8) is False

    def test_exact_restart_threshold(self):
        """Exactly 3 restarts should trigger (>= MIN_RESTART_COUNT)."""
        action = RolloutRestartAction()
        pod = self._make_pod_state(restart_count=3, failing_pods=2)
        assert action.should_execute(pod, None, 0.8) is True


class TestRolloutRestartExecute:
    """Test RolloutRestartAction.execute()."""

    @patch("predictive_agent.actions.rollout_restart.subprocess.run")
    def test_execute_dry_run(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="deployment restarted (dry run)", stderr="")
        action = RolloutRestartAction()
        ctx = ActionContext(namespace="default", dry_run=True)
        result = action.execute("app-deployment", ctx)
        assert result.success is True
        assert result.dry_run is True
        assert result.action == "rollout_restart"
        cmd = mock_run.call_args[0][0]
        assert "rollout" in cmd
        assert "restart" in cmd
        assert "deployment" in cmd
        assert "app-deployment" in cmd
        assert "--dry-run=server" in cmd

    @patch("predictive_agent.actions.rollout_restart.subprocess.run")
    def test_execute_real(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="deployment.apps/app-deployment restarted", stderr="")
        action = RolloutRestartAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("app-deployment", ctx)
        assert result.success is True
        assert result.dry_run is False
        cmd = mock_run.call_args[0][0]
        assert "--dry-run=server" not in cmd

    @patch("predictive_agent.actions.rollout_restart.subprocess.run")
    def test_execute_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error: deployment not found")
        action = RolloutRestartAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("app-deployment", ctx)
        assert result.success is False
        assert "not found" in result.message

    @patch("predictive_agent.actions.rollout_restart.subprocess.run")
    def test_execute_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="kubectl", timeout=60)
        action = RolloutRestartAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("app-deployment", ctx)
        assert result.success is False
        assert "timed out" in result.message

    @patch("predictive_agent.actions.rollout_restart.subprocess.run")
    def test_execute_timeout_60s(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="restarted", stderr="")
        action = RolloutRestartAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        action.execute("app-deployment", ctx)
        kwargs = mock_run.call_args[1]
        assert kwargs["timeout"] == 60

    @patch("predictive_agent.actions.rollout_restart.subprocess.run")
    def test_execute_correct_namespace(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="restarted", stderr="")
        action = RolloutRestartAction()
        ctx = ActionContext(namespace="opendesk", dry_run=False)
        action.execute("app-deployment", ctx)
        cmd = mock_run.call_args[0][0]
        assert "-n" in cmd
        assert "opendesk" in cmd

    @patch("predictive_agent.actions.rollout_restart.subprocess.run")
    def test_execute_exception(self, mock_run):
        mock_run.side_effect = RuntimeError("unexpected")
        action = RolloutRestartAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("app-deployment", ctx)
        assert result.success is False
        assert "unexpected" in result.message

    @patch("predictive_agent.actions.rollout_restart.subprocess.run")
    def test_execute_command_in_result(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="restarted", stderr="")
        action = RolloutRestartAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("app-deployment", ctx)
        assert "kubectl" in result.command
        assert "rollout" in result.command
        assert "restart" in result.command
