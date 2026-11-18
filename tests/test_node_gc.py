"""Tests for NodeDiskGCAction — bounded-autonomy node disk-pressure GC (REM-NODEGC).

Covers:
- should_execute gating (enabled, method, risk, disk pressure, node allowlist)
- execute method=docker -> docker system prune
- execute method=kubectl -> cordon + operator instruction (no host access)
- docker-prune in k8s runtime (no docker on host agent) degrades to kubectl
- failure / timeout handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from predictive_agent.actions.node_gc import NodeDiskGCAction
from predictive_agent.remediator import ActionContext


def _pod(node_disk_pressure=True, node_name="clrz14-06"):
    return MagicMock(node_disk_pressure=node_disk_pressure, node_name=node_name,
                     state="CRITICAL", name="prometheus-x", namespace="monitoring")


class TestShouldExecute:
    def test_disabled(self):
        action = NodeDiskGCAction(method="kubectl", enabled=False)
        assert action.should_execute(_pod(), None, 0.9) is False

    def test_no_method(self):
        action = NodeDiskGCAction(method="none", enabled=True)
        assert action.should_execute(_pod(), None, 0.9) is False

    def test_low_risk(self):
        action = NodeDiskGCAction(method="kubectl", enabled=True)
        assert action.should_execute(_pod(), None, 0.5) is False

    def test_no_disk_pressure(self):
        action = NodeDiskGCAction(method="kubectl", enabled=True)
        assert action.should_execute(_pod(node_disk_pressure=False), None, 0.9) is False

    def test_node_allowlist_blocks(self):
        action = NodeDiskGCAction(method="kubectl", enabled=True, nodes="clrz14-07")
        assert action.should_execute(_pod(node_name="clrz14-06"), None, 0.9) is False

    def test_node_allowlist_allows(self):
        action = NodeDiskGCAction(method="kubectl", enabled=True, nodes="clrz14-06")
        assert action.should_execute(_pod(node_name="clrz14-06"), None, 0.9) is True

    def test_empty_allowlist_allows_all(self):
        action = NodeDiskGCAction(method="kubectl", enabled=True, nodes="")
        assert action.should_execute(_pod(node_name="clrz14-06"), None, 0.9) is True

    def test_exact_risk_threshold_not_enough(self):
        action = NodeDiskGCAction(method="kubectl", enabled=True)
        assert action.should_execute(_pod(), None, 0.7) is False


class TestExecuteKubectl:
    @patch("predictive_agent.actions.node_gc.subprocess.run")
    def test_cordons_and_instructs(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="node/clrz14-06 cordoned", stderr="")
        action = NodeDiskGCAction(method="kubectl", enabled=True)
        ctx = ActionContext(namespace="monitoring", dry_run=False)
        ctx.node_name = "clrz14-06"
        res = action.execute("prometheus-x", ctx)
        assert res.success is True
        assert "cordon" in res.command
        assert "clrz14-06" in res.command
        assert "prune" in res.message.lower()

    @patch("predictive_agent.actions.node_gc.subprocess.run")
    def test_dry_run(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        action = NodeDiskGCAction(method="kubectl", enabled=True)
        res = action.execute("prometheus-x", ActionContext(namespace="monitoring", dry_run=True))
        assert res.dry_run is True
        assert "--dry-run" in res.command


class TestExecuteDocker:
    @patch("predictive_agent.actions.node_gc.subprocess.run")
    def test_docker_prune(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Total reclaimed space: 19.69GB", stderr="")
        action = NodeDiskGCAction(method="docker", enabled=True)
        res = action.execute("prometheus-x", ActionContext(namespace="monitoring", dry_run=False))
        assert res.success is True
        assert "docker" in res.command
        assert "prune" in res.command
        assert "19.69GB" in res.message

    @patch("predictive_agent.actions.node_gc.subprocess.run")
    def test_docker_gc_not_available_on_host_falls_back(self, mock_run):
        """In k8s runtime the agent has no docker socket -> degrade to cordon."""
        mock_run.side_effect = [
            FileNotFoundError("no docker binary"),
            MagicMock(returncode=0, stdout="node/clrz14-06 cordoned", stderr=""),
        ]
        action = NodeDiskGCAction(method="docker", enabled=True)
        ctx = ActionContext(namespace="monitoring", dry_run=False)
        ctx.node_name = "clrz14-06"
        res = action.execute("prometheus-x", ctx)
        # Degraded path: still succeeds as a bounded cordon + instruction
        assert res.success is True
        assert "cordon" in res.command

    @patch("predictive_agent.actions.node_gc.subprocess.run")
    def test_docker_prune_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="permission denied")
        action = NodeDiskGCAction(method="docker", enabled=True)
        res = action.execute("prometheus-x", ActionContext(namespace="monitoring", dry_run=False))
        assert res.success is False
        assert "permission denied" in res.message


class TestExecuteSafety:
    def test_no_node_name(self):
        action = NodeDiskGCAction(method="kubectl", enabled=True)
        ctx = ActionContext(namespace="monitoring", dry_run=False)
        ctx.node_name = ""
        res = action.execute("prometheus-x", ctx)
        assert res.success is False
