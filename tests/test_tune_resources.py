"""Tests for ResourceTunerAction (REM-6 part 2)."""

import json
from unittest.mock import MagicMock, patch

from predictive_agent.actions.tune_resources import ResourceTunerAction
from predictive_agent.remediator import ActionContext


class TestTuneResourcesShouldExecute:
    """Test ResourceTunerAction.should_execute()."""

    def _make_pod_state(self, data_points=15, trend_confidence=0.9, namespace="default"):
        return MagicMock(
            data_points=data_points,
            trend_confidence=trend_confidence,
            namespace=namespace,
        )

    def test_high_confidence_triggers(self):
        action = ResourceTunerAction()
        pod = self._make_pod_state(data_points=15, trend_confidence=0.9)
        assert action.should_execute(pod, None, 65.0) is True

    def test_low_confidence_does_not_trigger(self):
        action = ResourceTunerAction()
        pod = self._make_pod_state(data_points=15, trend_confidence=0.5)
        assert action.should_execute(pod, None, 65.0) is False

    def test_insufficient_data_does_not_trigger(self):
        action = ResourceTunerAction()
        pod = self._make_pod_state(data_points=5, trend_confidence=0.9)
        assert action.should_execute(pod, None, 65.0) is False

    def test_low_risk_does_not_trigger(self):
        action = ResourceTunerAction()
        pod = self._make_pod_state(data_points=15, trend_confidence=0.9)
        assert action.should_execute(pod, None, 0.5) is False

    def test_protected_namespace_blocks(self):
        action = ResourceTunerAction()
        pod = self._make_pod_state(data_points=15, trend_confidence=0.9, namespace="kube-system")
        assert action.should_execute(pod, None, 65.0) is False

    def test_exact_confidence_threshold(self):
        """Confidence exactly at 0.8 should trigger (>=)."""
        action = ResourceTunerAction()
        pod = self._make_pod_state(data_points=15, trend_confidence=0.8)
        assert action.should_execute(pod, None, 65.0) is True

    def test_exact_data_points_threshold(self):
        """Exactly 10 data points should trigger (>=)."""
        action = ResourceTunerAction()
        pod = self._make_pod_state(data_points=10, trend_confidence=0.9)
        assert action.should_execute(pod, None, 65.0) is True


class TestTuneResourcesExecute:
    """Test ResourceTunerAction.execute()."""

    @patch("predictive_agent.actions.tune_resources.subprocess.run")
    def test_execute_dry_run(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="patched (dry run)", stderr="")
        action = ResourceTunerAction()
        ctx = ActionContext(namespace="default", dry_run=True)
        ctx.cpu_request = "100m"
        ctx.mem_request = "128Mi"
        ctx.cpu_limit = "200m"
        ctx.mem_limit = "256Mi"
        result = action.execute("app-deployment", ctx)
        assert result.success is True
        assert result.dry_run is True
        assert result.action == "tune_resources"
        cmd = mock_run.call_args[0][0]
        assert "patch" in cmd
        assert "deployment" in cmd
        assert "app-deployment" in cmd
        assert "--type=json" in cmd
        assert "--dry-run=server" in cmd

    @patch("predictive_agent.actions.tune_resources.subprocess.run")
    def test_execute_real(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="deployment.apps/app-deployment patched", stderr="")
        action = ResourceTunerAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        ctx.cpu_request = "100m"
        ctx.mem_request = "128Mi"
        ctx.cpu_limit = "200m"
        ctx.mem_limit = "256Mi"
        result = action.execute("app-deployment", ctx)
        assert result.success is True
        assert result.dry_run is False
        cmd = mock_run.call_args[0][0]
        assert "--dry-run=server" not in cmd

    @patch("predictive_agent.actions.tune_resources.subprocess.run")
    def test_execute_json_patch_content(self, mock_run):
        """Test that the JSON patch is correctly formatted."""
        mock_run.return_value = MagicMock(returncode=0, stdout="patched", stderr="")
        action = ResourceTunerAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        ctx.cpu_request = "150m"
        ctx.mem_request = "256Mi"
        ctx.cpu_limit = "300m"
        ctx.mem_limit = "512Mi"
        action.execute("app-deployment", ctx)
        cmd = mock_run.call_args[0][0]
        # Find the -p= argument
        patch_arg = [c for c in cmd if c.startswith("-p=")][0]
        patch_json = patch_arg[3:]
        patch = json.loads(patch_json)
        assert len(patch) == 4
        assert patch[0]["op"] == "replace"
        assert patch[0]["path"] == "/spec/template/spec/containers/0/resources/requests/cpu"
        assert patch[0]["value"] == "150m"
        assert patch[1]["value"] == "256Mi"
        assert patch[2]["value"] == "300m"
        assert patch[3]["value"] == "512Mi"

    @patch("predictive_agent.actions.tune_resources.subprocess.run")
    def test_execute_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error: deployment not found")
        action = ResourceTunerAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("app-deployment", ctx)
        assert result.success is False
        assert "not found" in result.message

    @patch("predictive_agent.actions.tune_resources.subprocess.run")
    def test_execute_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="kubectl", timeout=30)
        action = ResourceTunerAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("app-deployment", ctx)
        assert result.success is False
        assert "timed out" in result.message

    @patch("predictive_agent.actions.tune_resources.subprocess.run")
    def test_execute_exception(self, mock_run):
        mock_run.side_effect = RuntimeError("unexpected")
        action = ResourceTunerAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("app-deployment", ctx)
        assert result.success is False
        assert "unexpected" in result.message

    @patch("predictive_agent.actions.tune_resources.subprocess.run")
    def test_execute_correct_namespace(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="patched", stderr="")
        action = ResourceTunerAction()
        ctx = ActionContext(namespace="opendesk", dry_run=False)
        action.execute("app-deployment", ctx)
        cmd = mock_run.call_args[0][0]
        assert "-n" in cmd
        assert "opendesk" in cmd

    @patch("predictive_agent.actions.tune_resources.subprocess.run")
    def test_execute_timeout_30s(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="patched", stderr="")
        action = ResourceTunerAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        action.execute("app-deployment", ctx)
        kwargs = mock_run.call_args[1]
        assert kwargs["timeout"] == 30

    @patch("predictive_agent.actions.tune_resources.subprocess.run")
    def test_execute_default_resources(self, mock_run):
        """Test that default resources are used when context doesn't provide them."""
        mock_run.return_value = MagicMock(returncode=0, stdout="patched", stderr="")
        action = ResourceTunerAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        # Don't set cpu_request, mem_request, etc. — should use defaults
        action.execute("app-deployment", ctx)
        cmd = mock_run.call_args[0][0]
        patch_arg = [c for c in cmd if c.startswith("-p=")][0]
        patch_json = patch_arg[3:]
        patch = json.loads(patch_json)
        assert patch[0]["value"] == "100m"  # default cpu_request
        assert patch[1]["value"] == "128Mi"  # default mem_request
