"""Rollout restart action: restart deployments with chronic pod failures.

Triggers `kubectl rollout restart` for deployments that have multiple pods
with high restart counts. This performs a rolling restart, which is safe
because Kubernetes gradually terminates old pods and creates new ones.
"""

from __future__ import annotations

import logging
import subprocess

from predictive_agent.remediator import ActionContext, RemediationAction, RemediationResult

logger = logging.getLogger(__name__)

# Thresholds
MIN_FAILING_PODS = 2       # need 2+ failing pods to warrant rollout restart
MIN_RESTART_COUNT = 3      # pods must have 3+ restarts
RISK_THRESHOLD = 0.75
ROLLOUT_COOLDOWN = 600     # 10 minutes between rollout restarts of same deployment


class RolloutRestartAction(RemediationAction):
    """Restart a deployment via kubectl rollout restart."""

    name = "rollout_restart"
    # Re-running a rollout restart could double-restart — dedup guards it (P3).
    idempotent: bool = False
    # P6 bounded autonomy: risk tier of this action
    risk_tier: str = "safe"
    # Closed loop: rollout restart is canary-CAPABLE but disabled by default
    # (kubectl rollout restart cannot restart a subset; a fractional "leg"
    # needs a rollout_pause/maxUnavailable strategy, so operators opt in by
    # subclassing with canary_fraction > 0 — clamped to CANARY_MAX_FRACTION).
    # Not reversible: a restart cannot be undone; a failed verification holds
    # and requires operator approval instead of auto-revert.
    canary_fraction: float = 0.0
    reversible: bool = False

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        """Check if rollout restart should be triggered.

        Returns True when:
        - Multiple pods from the same deployment are failing (2+ with restart_count > 3)
        - risk_score > 0.75
        - Deployment is NOT in a protected namespace
        """
        if risk_score <= RISK_THRESHOLD:
            return False

        namespace = getattr(pod_state, "namespace", "")
        if namespace in {"kube-system", "opendesk-predictive-agent"}:
            return False

        restart_count = getattr(pod_state, "restart_count", 0)
        if restart_count < MIN_RESTART_COUNT:
            return False

        # Check if multiple pods are failing (from pod_state or context)
        failing_pods = getattr(pod_state, "failing_pods_in_deployment", 1)
        if failing_pods < MIN_FAILING_PODS:
            return False

        return True

    def execute(self, target: str, context: ActionContext) -> RemediationResult:
        """Restart the deployment via kubectl rollout restart.

        In dry_run mode, uses --dry-run=server.
        """
        # target may be namespace/name or just name; kubectl needs just the name
        dep_name = target.split("/")[-1]
        cmd = [
            "kubectl", "rollout", "restart", "deployment", dep_name,
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
                timeout=60,
            )
            success = result.returncode == 0
            message = result.stdout.strip() if success else result.stderr.strip()
            if not message:
                message = "deployment restarted" if success else "kubectl rollout restart failed"

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
                message="kubectl rollout restart timed out after 60s",
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
