"""Tests for ContainerRestartAction (docker-mode remediation).

Tests follow the same pattern as test_pod_restart.py, mocking subprocess.run
instead of the removed unix socket Docker client.
"""

from unittest.mock import MagicMock, patch

from predictive_agent.actions.container_restart import ContainerRestartAction
from predictive_agent.remediator import ActionContext, RemediationResult


class TestContainerRestartShouldExecute:
    """Unit tests for the decision logic (no subprocess needed)."""

    def _make_pod_state(self, namespace="monitoring", restart_count=0, running=True, state="Running", status="running"):
        """Create a mock pod/container state object."""
        mock = MagicMock()
        mock.namespace = namespace
        mock.restart_count = restart_count
        mock.running = running
        mock.state = state
        mock.status = status
        return mock

    def _make_action(self):
        return ContainerRestartAction()

    def test_high_risk_and_many_restarts_triggers(self):
        action = self._make_action()
        pod = self._make_pod_state(restart_count=7)
        assert action.should_execute(pod, None, 0.9) is True

    def test_high_risk_running_few_restarts_does_not_trigger(self):
        action = self._make_action()
        pod = self._make_pod_state(restart_count=2, running=True)
        assert action.should_execute(pod, None, 0.9) is False

    def test_low_risk_does_not_trigger_even_with_restarts(self):
        action = self._make_action()
        pod = self._make_pod_state(restart_count=10)
        assert action.should_execute(pod, None, 0.5) is False

    def test_protected_namespace_blocks(self):
        action = self._make_action()
        pod = self._make_pod_state(namespace="kube-system", restart_count=10)
        assert action.should_execute(pod, None, 0.9) is False

    def test_protected_namespace_agent_blocks(self):
        action = self._make_action()
        pod = self._make_pod_state(namespace="opendesk-predictive-agent", restart_count=10)
        assert action.should_execute(pod, None, 0.9) is False

    def test_risk_at_threshold_boundary_does_not_trigger(self):
        import os
        action = self._make_action()
        pod = self._make_pod_state(restart_count=10)
        threshold = float(os.environ.get("REMEDIATION_RISK_THRESHOLD", "0.7"))
        assert action.should_execute(pod, None, threshold) is False

    def test_not_running_triggers(self):
        action = self._make_action()
        pod = self._make_pod_state(restart_count=0, running=False, state="Exited")
        assert action.should_execute(pod, None, 0.8) is True

    def test_exited_state_triggers(self):
        action = self._make_action()
        pod = self._make_pod_state(restart_count=0, state="Exited", status="exited")
        assert action.should_execute(pod, None, 0.8) is True

    def test_dead_state_triggers(self):
        action = self._make_action()
        pod = self._make_pod_state(restart_count=0, state="Dead", status="dead")
        assert action.should_execute(pod, None, 0.8) is True

    def test_min_restart_count_exact_triggers(self):
        action = self._make_action()
        pod = self._make_pod_state(restart_count=5)
        assert action.should_execute(pod, None, 0.8) is True


class TestContainerRestartExecute:
    """Test the execute method with mocked subprocess.run."""

    @patch("predictive_agent.actions.container_restart.subprocess.run")
    def test_dry_run_returns_result_without_executing(self, mock_run):
        """Test that dry_run mode doesn't call subprocess."""
        action = ContainerRestartAction()
        ctx = ActionContext(namespace="monitoring", dry_run=True)
        result = action.execute("predictive-agent", ctx)
        assert result.success is True
        assert result.dry_run is True
        assert "dry run" in result.message
        assert "docker restart predictive-agent" in result.command
        mock_run.assert_not_called()

    @patch("predictive_agent.actions.container_restart.subprocess.run")
    def test_execute_success(self, mock_run):
        """Test successful docker restart execution."""
        mock_run.return_value = MagicMock(returncode=0, stdout="predictive-agent", stderr="")
        action = ContainerRestartAction()
        ctx = ActionContext(namespace="monitoring", dry_run=False)
        result = action.execute("predictive-agent", ctx)
        assert result.success is True
        assert result.dry_run is False
        assert result.action == "container_restart"
        assert result.target == "predictive-agent"
        assert "docker" in result.command
        assert "restart" in result.command
        assert "predictive-agent" in result.command

    @patch("predictive_agent.actions.container_restart.subprocess.run")
    def test_execute_docker_command_uses_target(self, mock_run):
        """Test that the docker command uses the target name."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        action = ContainerRestartAction()
        ctx = ActionContext(namespace="docker", dry_run=False)
        action.execute("my-container-123", ctx)
        cmd = mock_run.call_args[0][0]
        assert cmd == ["docker", "restart", "my-container-123"]

    @patch("predictive_agent.actions.container_restart.subprocess.run")
    def test_execute_timeout(self, mock_run):
        """Test handling of subprocess timeout."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["docker", "restart", "test"], timeout=30)
        action = ContainerRestartAction()
        ctx = ActionContext(namespace="monitoring", dry_run=False)
        result = action.execute("test", ctx)
        assert result.success is False
        assert "timed out" in result.message
        assert "restart test" in result.message

    @patch("predictive_agent.actions.container_restart.subprocess.run")
    def test_execute_failure(self, mock_run):
        """Test handling of docker CLI failure."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error: No such container")
        action = ContainerRestartAction()
        ctx = ActionContext(namespace="monitoring", dry_run=False)
        result = action.execute("missing-container", ctx)
        assert result.success is False
        assert "No such container" in result.message

    @patch("predictive_agent.actions.container_restart.subprocess.run")
    def test_execute_docker_not_found(self, mock_run):
        """Test handling of missing docker CLI."""
        mock_run.side_effect = FileNotFoundError("docker: command not found")
        action = ContainerRestartAction()
        ctx = ActionContext(namespace="monitoring", dry_run=False)
        result = action.execute("container", ctx)
        assert result.success is False
        assert "not found" in result.message

    @patch("predictive_agent.actions.container_restart.subprocess.run")
    def test_execute_permission_denied(self, mock_run):
        """Test handling of permission error."""
        mock_run.side_effect = PermissionError("Permission denied")
        action = ContainerRestartAction()
        ctx = ActionContext(namespace="monitoring", dry_run=False)
        result = action.execute("container", ctx)
        assert result.success is False
        assert "Permission denied" in result.message

    @patch("predictive_agent.actions.container_restart.subprocess.run")
    def test_execute_exception(self, mock_run):
        """Test handling of unexpected exception."""
        mock_run.side_effect = RuntimeError("unexpected error")
        action = ContainerRestartAction()
        ctx = ActionContext(namespace="monitoring", dry_run=False)
        result = action.execute("container", ctx)
        assert result.success is False
        assert "unexpected error" in result.message

    @patch("predictive_agent.actions.container_restart.subprocess.run")
    def test_execute_timeout_value(self, mock_run):
        """Test that subprocess timeout is 30 seconds."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        action = ContainerRestartAction()
        ctx = ActionContext(namespace="monitoring", dry_run=False)
        action.execute("test", ctx)
        kwargs = mock_run.call_args[1]
        assert kwargs["timeout"] == 30

    @patch("predictive_agent.actions.container_restart.subprocess.run")
    def test_execute_uses_text_mode(self, mock_run):
        """Test that subprocess uses text=True."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        action = ContainerRestartAction()
        ctx = ActionContext(namespace="monitoring", dry_run=False)
        action.execute("test", ctx)
        kwargs = mock_run.call_args[1]
        assert kwargs["text"] is True

    @patch("predictive_agent.actions.container_restart.subprocess.run")
    def test_execute_captures_output(self, mock_run):
        """Test that subprocess captures stdout and stderr."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        action = ContainerRestartAction()
        ctx = ActionContext(namespace="monitoring", dry_run=False)
        action.execute("test", ctx)
        kwargs = mock_run.call_args[1]
        assert kwargs["capture_output"] is True

    def test_result_is_remediation_result(self):
        """Test that execute returns a RemediationResult instance."""
        action = ContainerRestartAction()
        ctx = ActionContext(namespace="monitoring", dry_run=True)
        result = action.execute("test", ctx)
        assert isinstance(result, RemediationResult)
