"""Tests for DeploymentScaleAction (REM-6 part 1)."""

from unittest.mock import MagicMock, patch

from predictive_agent.actions.scale import DeploymentScaleAction
from predictive_agent.remediator import ActionContext


class TestScaleShouldExecute:
    """Test DeploymentScaleAction.should_execute()."""

    def _make_pod_state(self, cpu_pct=50, data_points=10, namespace="default"):
        return MagicMock(cpu_pct=cpu_pct, data_points=data_points, namespace=namespace)

    def test_high_cpu_triggers(self):
        action = DeploymentScaleAction()
        pod = self._make_pod_state(cpu_pct=85, data_points=10)
        assert action.should_execute(pod, None, 0.8) is True

    def test_low_cpu_triggers(self):
        action = DeploymentScaleAction()
        pod = self._make_pod_state(cpu_pct=15, data_points=10)
        assert action.should_execute(pod, None, 0.8) is True

    def test_normal_cpu_does_not_trigger(self):
        action = DeploymentScaleAction()
        pod = self._make_pod_state(cpu_pct=50, data_points=10)
        assert action.should_execute(pod, None, 0.8) is False

    def test_insufficient_data_does_not_trigger(self):
        action = DeploymentScaleAction()
        pod = self._make_pod_state(cpu_pct=85, data_points=3)
        assert action.should_execute(pod, None, 0.8) is False

    def test_low_risk_does_not_trigger(self):
        action = DeploymentScaleAction()
        pod = self._make_pod_state(cpu_pct=85, data_points=10)
        assert action.should_execute(pod, None, 0.7) is False

    def test_protected_namespace_blocks(self):
        action = DeploymentScaleAction()
        pod = self._make_pod_state(cpu_pct=85, data_points=10, namespace="kube-system")
        assert action.should_execute(pod, None, 0.8) is False

    def test_protected_namespace_agent(self):
        action = DeploymentScaleAction()
        pod = self._make_pod_state(cpu_pct=85, data_points=10, namespace="opendesk-predictive-agent")
        assert action.should_execute(pod, None, 0.8) is False

    def test_exact_high_threshold(self):
        """CPU exactly at 80% should NOT trigger (uses >)."""
        action = DeploymentScaleAction()
        pod = self._make_pod_state(cpu_pct=80, data_points=10)
        assert action.should_execute(pod, None, 0.8) is False

    def test_just_above_high_threshold(self):
        """CPU at 81% should trigger."""
        action = DeploymentScaleAction()
        pod = self._make_pod_state(cpu_pct=81, data_points=10)
        assert action.should_execute(pod, None, 0.8) is True


class TestScaleExecute:
    """Test DeploymentScaleAction.execute()."""

    @patch("predictive_agent.actions.scale.subprocess.run")
    def test_execute_scale_up_dry_run(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="scaled (dry run)", stderr="")
        action = DeploymentScaleAction()
        ctx = ActionContext(namespace="default", dry_run=True)
        ctx.scale_direction = "up"
        ctx.current_replicas = 3
        result = action.execute("app-deployment", ctx)
        assert result.success is True
        assert result.dry_run is True
        assert result.action == "scale"
        cmd = mock_run.call_args[0][0]
        assert "scale" in cmd
        assert "deployment" in cmd
        assert "app-deployment" in cmd
        assert "--replicas=4" in cmd
        assert "--dry-run=server" in cmd

    @patch("predictive_agent.actions.scale.subprocess.run")
    def test_execute_scale_up_real(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="deployment.apps/app-deployment scaled", stderr="")
        action = DeploymentScaleAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        ctx.scale_direction = "up"
        ctx.current_replicas = 3
        result = action.execute("app-deployment", ctx)
        assert result.success is True
        assert result.dry_run is False
        cmd = mock_run.call_args[0][0]
        assert "--replicas=4" in cmd
        assert "--dry-run=server" not in cmd

    @patch("predictive_agent.actions.scale.subprocess.run")
    def test_execute_scale_down(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="scaled", stderr="")
        action = DeploymentScaleAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        ctx.scale_direction = "down"
        ctx.current_replicas = 5
        result = action.execute("app-deployment", ctx)
        assert result.success is True
        cmd = mock_run.call_args[0][0]
        assert "--replicas=4" in cmd

    @patch("predictive_agent.actions.scale.subprocess.run")
    def test_execute_scale_up_at_max(self, mock_run):
        """Test that scaling up at max replicas is a no-op."""
        action = DeploymentScaleAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        ctx.scale_direction = "up"
        ctx.current_replicas = 10
        result = action.execute("app-deployment", ctx)
        assert result.success is True
        assert "max" in result.message.lower()
        mock_run.assert_not_called()

    @patch("predictive_agent.actions.scale.subprocess.run")
    def test_execute_scale_down_at_min(self, mock_run):
        """Test that scaling down at min replicas is a no-op."""
        action = DeploymentScaleAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        ctx.scale_direction = "down"
        ctx.current_replicas = 1
        result = action.execute("app-deployment", ctx)
        assert result.success is True
        assert "min" in result.message.lower()
        mock_run.assert_not_called()

    @patch("predictive_agent.actions.scale.subprocess.run")
    def test_execute_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error: deployment not found")
        action = DeploymentScaleAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        ctx.scale_direction = "up"
        ctx.current_replicas = 3
        result = action.execute("app-deployment", ctx)
        assert result.success is False
        assert "not found" in result.message

    @patch("predictive_agent.actions.scale.subprocess.run")
    def test_execute_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="kubectl", timeout=30)
        action = DeploymentScaleAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        ctx.scale_direction = "up"
        ctx.current_replicas = 3
        result = action.execute("app-deployment", ctx)
        assert result.success is False
        assert "timed out" in result.message

    @patch("predictive_agent.actions.scale.subprocess.run")
    def test_execute_exception(self, mock_run):
        mock_run.side_effect = RuntimeError("unexpected")
        action = DeploymentScaleAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        ctx.scale_direction = "up"
        ctx.current_replicas = 3
        result = action.execute("app-deployment", ctx)
        assert result.success is False
        assert "unexpected" in result.message

    @patch("predictive_agent.actions.scale.subprocess.run")
    def test_execute_correct_namespace(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="scaled", stderr="")
        action = DeploymentScaleAction()
        ctx = ActionContext(namespace="opendesk", dry_run=False)
        ctx.scale_direction = "up"
        ctx.current_replicas = 2
        action.execute("app-deployment", ctx)
        cmd = mock_run.call_args[0][0]
        assert "-n" in cmd
        assert "opendesk" in cmd

    @patch("predictive_agent.actions.scale.subprocess.run")
    def test_execute_timeout_30s(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="scaled", stderr="")
        action = DeploymentScaleAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        ctx.scale_direction = "up"
        ctx.current_replicas = 2
        action.execute("app-deployment", ctx)
        kwargs = mock_run.call_args[1]
        assert kwargs["timeout"] == 30
