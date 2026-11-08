"""Deployment scale action: scale deployments up/down based on resource pressure.

Scales deployments when CPU is consistently high (scale up) or low (scale down).
Uses kubectl scale with min/max bounds to prevent runaway scaling.
"""

from __future__ import annotations

import logging
import subprocess

from predictive_agent.remediator import ActionContext, RemediationAction, RemediationResult

logger = logging.getLogger(__name__)

# Thresholds
HIGH_CPU_THRESHOLD = 80.0    # scale up if CPU > 80% for sustained period
LOW_CPU_THRESHOLD = 20.0     # scale down if CPU < 20% for sustained period
MIN_REPLICAS = 1
MAX_REPLICAS = 10
SCALE_COOLDOWN = 600          # 10 minutes between scaling actions
SUSTAINED_CYCLES = 5          # need 5+ cycles of high/low CPU
RISK_THRESHOLD = 0.75


class DeploymentScaleAction(RemediationAction):
    """Scale a deployment up or down based on resource pressure."""

    name = "scale"
    # Scaling is NOT idempotent — a re-run could double-scale. The dedup window
    # plus the recorded desired_state (replicas) make re-runs converge (P3).
    idempotent: bool = False
    # P6 bounded autonomy: risk tier of this action
    risk_tier: str = "moderate"
    # Closed loop: scale is reversible (scale back to the recorded replica
    # count) and canary-capable: with canary_fraction > 0 the manager runs a
    # bounded first leg (clamped to CANARY_MAX_FRACTION) and only widens after
    # that leg verifies. Enable canary by raising canary_fraction above 0.
    reversible: bool = True
    canary_fraction: float = 0.15
    resolution_signals: tuple = ("state", "memory_pct", "cpu_pct", "restart_count")

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        """Check if deployment should be scaled.

        Returns True when:
        - CPU consistently high (>80% for 5+ cycles) → scale up
        - CPU consistently low (<20% for 5+ cycles) → scale down
        - risk_score > 0.75
        - NOT in protected namespace
        """
        if risk_score <= RISK_THRESHOLD:
            return False

        namespace = getattr(pod_state, "namespace", "")
        if namespace in {"kube-system", "opendesk-predictive-agent"}:
            return False

        cpu_pct = getattr(pod_state, "cpu_pct", 0)
        sustained_cycles = getattr(pod_state, "data_points", 0)

        if sustained_cycles < SUSTAINED_CYCLES:
            return False

        if cpu_pct > HIGH_CPU_THRESHOLD:
            return True
        if cpu_pct < LOW_CPU_THRESHOLD:
            return True

        return False

    def execute(self, target: str, context: ActionContext) -> RemediationResult:
        """Scale the deployment via kubectl scale.

        Determines scale up or down based on context.
        In dry_run mode, uses --dry-run=server.
        Canary legs (context.canary_mode == "canary") bound the step to the
        effective canary fraction of the current replica count, so a verified
        widen can follow; the "widen" leg takes the normal step.
        """
        # Determine scale direction
        scale_direction = getattr(context, "scale_direction", "up")
        current_replicas = getattr(context, "current_replicas", 1)

        if scale_direction == "up":
            new_replicas = min(current_replicas + 1, MAX_REPLICAS)
        else:
            new_replicas = max(current_replicas - 1, MIN_REPLICAS)

        # Closed loop: bound the canary (first) leg to a fraction of capacity.
        if getattr(context, "canary_mode", None) == "canary":
            frac = max(0.0, float(getattr(context, "canary_fraction", 0.0) or 0.0))
            if frac > 0.0 and scale_direction == "up":
                bounded = current_replicas + max(1, int(current_replicas * frac))
                new_replicas = min(new_replicas, bounded)
            # A scale-down step (−1) is already a minimal, bounded change.

        if new_replicas == current_replicas:
            return RemediationResult(
                action=self.name,
                target=target,
                success=True,
                dry_run=context.dry_run,
                message=f"already at {'max' if scale_direction == 'up' else 'min'} replicas ({current_replicas})",
                command="",
                desired_state={"replicas": current_replicas},
            )

        # target may be namespace/name or just name; kubectl needs just the name
        dep_name = target.split("/")[-1]
        cmd = [
            "kubectl", "scale", "deployment", dep_name,
            "-n", context.namespace,
            f"--replicas={new_replicas}",
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
                message = f"scaled {target} to {new_replicas} replicas"

            return RemediationResult(
                action=self.name,
                target=target,
                success=success,
                dry_run=context.dry_run,
                message=message,
                command=cmd_str,
                # P3: record the desired state (absolute replica count) so a
                # re-run within the dedup window converges instead of stacking.
                # Closed loop: prev_replicas lets revert() restore the exact
                # prior state after a failed verification.
                desired_state={
                    "replicas": new_replicas,
                    "prev_replicas": current_replicas,
                },
            )
        except subprocess.TimeoutExpired:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=context.dry_run,
                message="kubectl scale timed out after 30s",
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

    def revert(self, target: str, prev_state: dict | None, context: ActionContext) -> RemediationResult | None:
        """Scale back to the replica count recorded before the action ran.

        ``prev_state`` is the snapshot the manager captured at execution time;
        the prior replica count comes from ``desired_state.prev_replicas``
        stored alongside it. Honors dry_run exactly like execute().
        """
        desired = (prev_state or {}).get("desired_state") or {}
        prev_replicas = desired.get("prev_replicas")
        if prev_replicas is None:
            logger.warning(
                "revert(scale): no prev_replicas recorded for %s — cannot undo",
                target,
            )
            return None
        # target may be namespace/name or just name; kubectl needs just the name
        dep_name = target.split("/")[-1]
        cmd = [
            "kubectl", "scale", "deployment", dep_name,
            "-n", context.namespace,
            f"--replicas={int(prev_replicas)}",
        ]
        if context.dry_run:
            cmd.append("--dry-run=server")
        cmd_str = " ".join(cmd)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            success = result.returncode == 0
            message = result.stdout.strip() if success else result.stderr.strip()
            if not message:
                message = f"reverted {target} to {prev_replicas} replicas"
            return RemediationResult(
                action=self.name,
                target=target,
                success=success,
                dry_run=context.dry_run,
                message=message,
                command=cmd_str,
                desired_state={"replicas": int(prev_replicas)},
            )
        except Exception as e:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=context.dry_run,
                message=f"revert error: {e}",
                command=cmd_str,
            )
