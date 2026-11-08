"""Node cordon action: cordon nodes with sustained pressure.

Cordons nodes that have sustained high CPU/memory or disk pressure.
Does NOT drain — cordon only (draining is too risky for automated action).
Logs a recommendation to drain manually.
"""

from __future__ import annotations

import logging
import subprocess

from predictive_agent.remediator import ActionContext, RemediationAction, RemediationResult

logger = logging.getLogger(__name__)

# Thresholds
RISK_THRESHOLD = 0.8


class NodeCordonAction(RemediationAction):
    """Cordon a node with sustained pressure (marks it unschedulable).

    Unlike pod-level actions, this action checks the pod's node for memory
    pressure (set by the reconcile loop from K8s node conditions) rather
    than the pod's own CPU/memory metrics. The cordon target is the node
    name (from ``context.node_name``), NOT the pod name.
    """

    name = "node_cordon"
    # Cordoned nodes stay cordoned — re-running is safe (P3 idempotent remediation).
    idempotent: bool = True
    # P6 bounded autonomy: risk tier of this action
    risk_tier: str = "destructive"

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        """Check if the pod's node should be cordoned.

        Returns True when:
        - The pod's node is under memory pressure (node_pressure flag set
          by the reconcile loop from K8s node conditions)
        - risk_score > 0.8 (higher than pod restart — cordon is more disruptive)
        """
        if risk_score <= RISK_THRESHOLD:
            return False

        # Check node pressure (set by reconcile loop from node conditions)
        node_pressure = getattr(pod_state, "node_pressure", False)
        if not node_pressure:
            return False

        return True

    def execute(self, target: str, context: ActionContext) -> RemediationResult:
        """Cordon the node via kubectl.

        Uses ``context.node_name`` (set by RemediationManager.evaluate from
        ``pod_state.node_name``) as the cordon target. Falls back to the
        ``target`` argument only if node_name is empty (legacy behaviour).

        In dry_run mode, uses --dry-run=server.
        Does NOT drain — only marks unschedulable.
        """
        # Use node_name from context, not the pod name (target)
        node_name = getattr(context, "node_name", "") or target
        if not node_name:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=context.dry_run,
                message="no node name available for cordon",
                command="",
            )

        cmd = ["kubectl", "cordon", node_name]
        if context.dry_run:
            cmd.append("--dry-run=server")

        cmd_str = " ".join(cmd)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            success = result.returncode == 0
            message = result.stdout.strip() if success else result.stderr.strip()
            if not message:
                message = "node cordoned" if success else "kubectl cordon failed"
            if success:
                message += " — recommend manual drain: kubectl drain " + node_name + " --ignore-daemonsets --delete-emptydir-data"

            return RemediationResult(
                action=self.name,
                target=node_name,
                success=success,
                dry_run=context.dry_run,
                message=message,
                command=cmd_str,
            )
        except subprocess.TimeoutExpired:
            return RemediationResult(
                action=self.name,
                target=node_name,
                success=False,
                dry_run=context.dry_run,
                message="kubectl cordon timed out after 30s",
                command=cmd_str,
            )
        except Exception as e:
            return RemediationResult(
                action=self.name,
                target=node_name,
                success=False,
                dry_run=context.dry_run,
                message=f"error: {e}",
                command=cmd_str,
            )
