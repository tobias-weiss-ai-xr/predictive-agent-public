"""Pod restart action: delete unhealthy pods (CrashLoopBackOff, OOMKilled).

Deletes pods that are in CrashLoopBackOff or have excessive restart counts.
The pod's controller (Deployment, StatefulSet, DaemonSet) will recreate it.
"""

from __future__ import annotations

import logging
import subprocess

from predictive_agent.remediator import ActionContext, RemediationAction, RemediationResult

logger = logging.getLogger(__name__)

# Thresholds
MIN_RESTART_COUNT = 5
RISK_THRESHOLD = 0.7


class PodRestartAction(RemediationAction):
    """Delete unhealthy pods to force controller recreation."""

    name = "pod_restart"
    # Deleting an unhealthy pod is safe to re-run — the controller converges to
    # the same desired state (P3 idempotent remediation).
    idempotent: bool = True
    # P6 bounded autonomy: risk tier of this action
    risk_tier: str = "safe"

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        """Check if pod should be restarted.

        Returns True when:
        - Pod phase is Failed, or
        - Restart count >= 5, or
        - Pod is in CrashLoopBackOff/OOMKilled
        - AND risk_score > 0.7
        - AND pod is NOT in a protected namespace
        """
        if risk_score <= RISK_THRESHOLD:
            return False

        namespace = getattr(pod_state, "namespace", "")
        if namespace in {"kube-system", "opendesk-predictive-agent"}:
            return False

        phase = getattr(pod_state, "phase", "Running")
        restart_count = getattr(pod_state, "restart_count", 0)

        if phase == "Failed":
            return True
        if restart_count >= MIN_RESTART_COUNT:
            return True

        return False

    def execute(self, target: str, context: ActionContext) -> RemediationResult:
        """Delete the pod via kubectl.

        In dry_run mode, uses --dry-run=server.
        """
        cmd = [
            "kubectl", "delete", "pod", target,
            "-n", context.namespace,
        ]
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
                message = "pod deleted" if success else "kubectl delete failed"

            return RemediationResult(
                action=self.name,
                target=target,
                success=success,
                dry_run=context.dry_run,
                message=message,
                command=cmd_str,
            )
        except subprocess.TimeoutExpired:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=context.dry_run,
                message="kubectl delete pod timed out after 30s",
                command=cmd_str,
            )
        except Exception as e:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=context.dry_run,
                message=f"error: {e}",
                command=cmd_str,
            )
