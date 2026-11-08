"""Tests for PodRestartAction (REM-2)."""

from unittest.mock import MagicMock, patch

from predictive_agent.actions.pod_restart import PodRestartAction
from predictive_agent.remediator import ActionContext


class TestPodRestartShouldExecute:
    """Test PodRestartAction.should_execute()."""

    def _make_pod_state(self, phase="Running", restart_count=0, namespace="default"):
        return MagicMock(phase=phase, restart_count=restart_count, namespace=namespace)

    def test_crashloopbackoff_triggers(self):
        action = PodRestartAction()
        pod = self._make_pod_state(phase="CrashLoopBackOff", restart_count=10)
        assert action.should_execute(pod, None, 0.8) is True

    def test_failed_phase_triggers(self):
        action = PodRestartAction()
        pod = self._make_pod_state(phase="Failed", restart_count=0)
        assert action.should_execute(pod, None, 0.8) is True

    def test_high_restart_count_triggers(self):
        action = PodRestartAction()
        pod = self._make_pod_state(phase="Running", restart_count=5)
        assert action.should_execute(pod, None, 0.8) is True

    def test_low_restart_count_does_not_trigger(self):
        action = PodRestartAction()
        pod = self._make_pod_state(phase="Running", restart_count=2)
        assert action.should_execute(pod, None, 0.8) is False

    def test_healthy_pod_does_not_trigger(self):
        action = PodRestartAction()
        pod = self._make_pod_state(phase="Running", restart_count=0)
        assert action.should_execute(pod, None, 0.8) is False

    def test_low_risk_does_not_trigger(self):
        action = PodRestartAction()
        pod = self._make_pod_state(phase="Failed", restart_count=10)
        assert action.should_execute(pod, None, 0.5) is False

    def test_protected_namespace_blocks(self):
        action = PodRestartAction()
        pod = self._make_pod_state(phase="CrashLoopBackOff", restart_count=10, namespace="kube-system")
        assert action.should_execute(pod, None, 0.8) is False

    def test_protected_namespace_agent(self):
        action = PodRestartAction()
        pod = self._make_pod_state(phase="CrashLoopBackOff", restart_count=10, namespace="opendesk-predictive-agent")
        assert action.should_execute(pod, None, 0.8) is False


class TestPodRestartExecute:
    """Test PodRestartAction.execute()."""

    @patch("predictive_agent.actions.pod_restart.subprocess.run")
    def test_execute_dry_run(self, mock_run):
        """Test that dry_run mode uses --dry-run=server."""
        mock_run.return_value = MagicMock(returncode=0, stdout="pod deleted (dry run)", stderr="")
        action = PodRestartAction()
        ctx = ActionContext(namespace="default", dry_run=True)
        result = action.execute("pod-1", ctx)
        assert result.success is True
        assert result.dry_run is True
        assert result.action == "pod_restart"
        assert result.target == "pod-1"
        # Check kubectl command includes --dry-run=server
        cmd = mock_run.call_args[0][0]
        assert "delete" in cmd
        assert "pod-1" in cmd
        assert "--dry-run=server" in cmd

    @patch("predictive_agent.actions.pod_restart.subprocess.run")
    def test_execute_real(self, mock_run):
        """Test real (non-dry-run) execution."""
        mock_run.return_value = MagicMock(returncode=0, stdout="pod \"pod-1\" deleted", stderr="")
        action = PodRestartAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("pod-1", ctx)
        assert result.success is True
        assert result.dry_run is False
        cmd = mock_run.call_args[0][0]
        assert "--dry-run=server" not in cmd

    @patch("predictive_agent.actions.pod_restart.subprocess.run")
    def test_execute_failure(self, mock_run):
        """Test handling of kubectl failure."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error from server: NotFound")
        action = PodRestartAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("pod-1", ctx)
        assert result.success is False
        assert "NotFound" in result.message

    @patch("predictive_agent.actions.pod_restart.subprocess.run")
    def test_execute_timeout(self, mock_run):
        """Test handling of subprocess timeout."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="kubectl", timeout=30)
        action = PodRestartAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("pod-1", ctx)
        assert result.success is False
        assert "timed out" in result.message

    @patch("predictive_agent.actions.pod_restart.subprocess.run")
    def test_execute_exception(self, mock_run):
        """Test handling of unexpected exception."""
        mock_run.side_effect = RuntimeError("unexpected")
        action = PodRestartAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("pod-1", ctx)
        assert result.success is False
        assert "unexpected" in result.message

    @patch("predictive_agent.actions.pod_restart.subprocess.run")
    def test_execute_correct_namespace(self, mock_run):
        """Test that kubectl command uses correct namespace."""
        mock_run.return_value = MagicMock(returncode=0, stdout="deleted", stderr="")
        action = PodRestartAction()
        ctx = ActionContext(namespace="opendesk", dry_run=True)
        action.execute("pod-1", ctx)
        cmd = mock_run.call_args[0][0]
        assert "-n" in cmd
        assert "opendesk" in cmd

    @patch("predictive_agent.actions.pod_restart.subprocess.run")
    def test_execute_command_in_result(self, mock_run):
        """Test that the kubectl command is recorded in the result."""
        mock_run.return_value = MagicMock(returncode=0, stdout="deleted", stderr="")
        action = PodRestartAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("pod-1", ctx)
        assert "kubectl" in result.command
        assert "delete" in result.command
        assert "pod-1" in result.command

    @patch("predictive_agent.actions.pod_restart.subprocess.run")
    def test_execute_timeout_30s(self, mock_run):
        """Test that subprocess timeout is 30 seconds."""
        mock_run.return_value = MagicMock(returncode=0, stdout="deleted", stderr="")
        action = PodRestartAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        action.execute("pod-1", ctx)
        kwargs = mock_run.call_args[1]
        assert kwargs["timeout"] == 30
