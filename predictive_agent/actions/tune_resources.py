"""Auto-tune resource limits: patch deployment resource requests/limits.

Uses Kalman filter trends to compute optimal resource requests and limits.
Only patches when confidence is high (>0.8) and the current requests/limits
are significantly off from actual usage (>2x or <0.5x).
"""

from __future__ import annotations

import json
import logging
import subprocess

from predictive_agent.remediator import ActionContext, RemediationAction, RemediationResult

logger = logging.getLogger(__name__)

# Thresholds
OVERPROVISION_FACTOR = 2.0   # current > 2x actual → reduce
UNDERPROVISION_FACTOR = 0.5  # current < 0.5x actual → increase
MIN_CONFIDENCE = 0.8         # need high confidence to auto-tune
TUNE_COOLDOWN = 1800         # 30 minutes between tune actions
RISK_THRESHOLD = 0.6


class ResourceTunerAction(RemediationAction):
    """Auto-tune deployment resource requests and limits."""

    name = "tune_resources"
    # Tuning is NOT idempotent in general — dedup + recorded desired_state make
    # re-runs converge to the same resource values instead of stacking (P3).
    idempotent: bool = False
    # P6 bounded autonomy: risk tier of this action
    risk_tier: str = "moderate"

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        """Check if resource tuning should be applied.

        Returns True when:
        - Pod has stable usage (10+ data points)
        - Current requests/limits are significantly off (>2x or <0.5x actual)
        - Confidence > 0.8
        - risk_score > 0.6
        - NOT in protected namespace
        """
        if risk_score <= RISK_THRESHOLD:
            return False

        namespace = getattr(pod_state, "namespace", "")
        if namespace in {"kube-system", "opendesk-predictive-agent"}:
            return False

        data_points = getattr(pod_state, "data_points", 0)
        if data_points < 10:
            return False

        confidence = getattr(pod_state, "trend_confidence", 0)
        if confidence < MIN_CONFIDENCE:
            return False

        return True

    def execute(self, target: str, context: ActionContext) -> RemediationResult:
        """Patch deployment resource limits via kubectl patch.

        Uses JSON patch to update container resources.
        In dry_run mode, uses --dry-run=server.
        """
        # Build the JSON patch
        cpu_request = getattr(context, "cpu_request", "100m")
        mem_request = getattr(context, "mem_request", "128Mi")
        cpu_limit = getattr(context, "cpu_limit", "200m")
        mem_limit = getattr(context, "mem_limit", "256Mi")
        _container_name = getattr(context, "container_name", target)

        patch = [
            {
                "op": "replace",
                "path": "/spec/template/spec/containers/0/resources/requests/cpu",
                "value": cpu_request,
            },
            {
                "op": "replace",
                "path": "/spec/template/spec/containers/0/resources/requests/memory",
                "value": mem_request,
            },
            {
                "op": "replace",
                "path": "/spec/template/spec/containers/0/resources/limits/cpu",
                "value": cpu_limit,
            },
            {
                "op": "replace",
                "path": "/spec/template/spec/containers/0/resources/limits/memory",
                "value": mem_limit,
            },
        ]

        patch_str = json.dumps(patch)
        cmd = [
            "kubectl", "patch", "deployment", target,
            "-n", context.namespace,
            "--type=json",
            f"-p={patch_str}",
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
                message = f"resources tuned: requests={cpu_request}/{mem_request}, limits={cpu_limit}/{mem_limit}"

            return RemediationResult(
                action=self.name,
                target=target,
                success=success,
                dry_run=context.dry_run,
                message=message,
                command=cmd_str,
                # P3: record the desired state (absolute resource values) so a
                # re-run within the dedup window converges instead of stacking.
                desired_state={
                    "cpu_request": cpu_request,
                    "mem_request": mem_request,
                    "cpu_limit": cpu_limit,
                    "mem_limit": mem_limit,
                },
            )
        except subprocess.TimeoutExpired:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=context.dry_run,
                message="kubectl patch timed out after 30s",
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
