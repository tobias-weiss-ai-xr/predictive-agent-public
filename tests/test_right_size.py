"""Tests for RightSizeAction (REM-4)."""

import json
from unittest.mock import MagicMock

from predictive_agent.actions.right_size import RightSizeAction
from predictive_agent.remediator import ActionContext


class TestRightSizeShouldExecute:
    """Test RightSizeAction.should_execute()."""

    def _make_pod_state(self, restart_count=0, data_points=15):
        return MagicMock(restart_count=restart_count, data_points=data_points, namespace="default")

    def test_stable_pod_triggers(self):
        action = RightSizeAction()
        pod = self._make_pod_state(restart_count=0, data_points=15)
        assert action.should_execute(pod, None, 0.55) is True

    def test_pod_with_restarts_does_not_trigger(self):
        action = RightSizeAction()
        pod = self._make_pod_state(restart_count=3, data_points=15)
        assert action.should_execute(pod, None, 0.55) is False

    def test_insufficient_data_does_not_trigger(self):
        action = RightSizeAction()
        pod = self._make_pod_state(restart_count=0, data_points=5)
        assert action.should_execute(pod, None, 0.55) is False

    def test_low_risk_does_not_trigger(self):
        action = RightSizeAction()
        pod = self._make_pod_state(restart_count=0, data_points=15)
        assert action.should_execute(pod, None, 0.4) is False

    def test_exact_data_points_threshold(self):
        """Exactly 10 data points should trigger (>= MIN_DATA_POINTS)."""
        action = RightSizeAction()
        pod = self._make_pod_state(restart_count=0, data_points=10)
        assert action.should_execute(pod, None, 0.55) is True


class TestRightSizeExecute:
    """Test RightSizeAction.execute()."""

    def test_execute_returns_recommendation(self):
        action = RightSizeAction()
        ctx = ActionContext(namespace="default", dry_run=True)
        result = action.execute("pod-1", ctx)
        assert result.success is True
        assert result.dry_run is True
        assert result.action == "right_size"
        assert result.target == "pod-1"

    def test_execute_returns_json_message(self):
        action = RightSizeAction()
        ctx = ActionContext(namespace="default", dry_run=True)
        result = action.execute("pod-1", ctx)
        recommendation = json.loads(result.message)
        assert recommendation["pod"] == "pod-1"
        assert recommendation["action"] == "right_size"
        assert recommendation["advisory"] is True

    def test_execute_does_not_call_kubectl(self):
        """RightSizeAction is advisory only — should NOT call subprocess."""
        # right_size.py doesn't import subprocess, so there's nothing to mock.
        # Just verify the action produces a result without any subprocess call.
        action = RightSizeAction()
        ctx = ActionContext(namespace="default", dry_run=True)
        result = action.execute("pod-1", ctx)
        assert result.success is True
        assert result.command == ""  # No kubectl command

    def test_execute_with_trends(self):
        action = RightSizeAction()
        ctx = ActionContext(namespace="default", dry_run=True)
        ctx.cpu_trend = 150  # 150m CPU
        ctx.mem_trend = 200  # 200Mi memory
        result = action.execute("pod-1", ctx)
        recommendation = json.loads(result.message)
        assert "recommended_requests" in recommendation
        assert recommendation["confidence"] == 0.75

    def test_execute_without_trends(self):
        action = RightSizeAction()
        ctx = ActionContext(namespace="default", dry_run=True)
        result = action.execute("pod-1", ctx)
        recommendation = json.loads(result.message)
        assert recommendation["confidence"] == 0.5
        assert "Insufficient" in recommendation["reason"]

    def test_execute_always_dry_run(self):
        """RightSizeAction is advisory only — always dry_run=True."""
        action = RightSizeAction()
        ctx = ActionContext(namespace="default", dry_run=False)
        result = action.execute("pod-1", ctx)
        assert result.dry_run is True  # Always dry_run for advisory action

    def test_execute_empty_command(self):
        """RightSizeAction has no kubectl command."""
        action = RightSizeAction()
        ctx = ActionContext(namespace="default", dry_run=True)
        result = action.execute("pod-1", ctx)
        assert result.command == ""
