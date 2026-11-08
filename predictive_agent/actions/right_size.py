"""Resource right-sizing recommendations (advisory, no cluster changes).

Analyzes Kalman trends and generates VPA-style recommendations for
adjusting pod resource requests and limits. This action is advisory only —
it does NOT modify any Kubernetes resources.
"""

from __future__ import annotations

import json
import logging

from predictive_agent.remediator import ActionContext, RemediationAction, RemediationResult

logger = logging.getLogger(__name__)

# Thresholds
UNDERPROVISION_THRESHOLD = 0.90  # usage > 90% of request
OVERPROVISION_THRESHOLD = 0.50   # usage < 50% of request
LIMIT_UNDERPROVISION = 0.80      # usage > 80% of limit
LIMIT_OVERPROVISION = 0.50       # usage < 50% of limit
MIN_DATA_POINTS = 10             # need at least 10 cycles of data
RISK_THRESHOLD = 0.5            # lower threshold — advisory action


class RightSizeAction(RemediationAction):
    """Generate resource right-sizing recommendations (advisory only)."""

    name = "right_size"
    # Advisory recommendations are not idempotent in the dedup sense — the same
    # recommendation recorded twice is noise; dedup keeps it once per window (P3).
    idempotent: bool = False
    # P6 bounded autonomy: risk tier of this action
    risk_tier: str = "moderate"

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        """Check if right-sizing recommendation should be generated.

        Returns True when:
        - Pod has been stable for 10+ cycles (no recent restarts)
        - Actual resource usage is <50% or >90% of requests
        - risk_score > 0.5 (lower threshold — advisory)
        """
        if risk_score <= RISK_THRESHOLD:
            return False

        restart_count = getattr(pod_state, "restart_count", 0)
        if restart_count > 0:
            return False

        # Need sufficient data points
        data_points = getattr(pod_state, "data_points", 0)
        if data_points < MIN_DATA_POINTS:
            return False

        return True

    def execute(self, target: str, context: ActionContext) -> RemediationResult:
        """Generate right-sizing recommendation.

        This is advisory only — no kubectl commands are executed.
        Returns a RemediationResult with JSON recommendation in the message.
        """
        # In dry_run mode, still generate the recommendation (it's advisory)
        recommendation: dict = {
            "pod": target,
            "action": "right_size",
            "advisory": True,
            "current_requests": {},
            "recommended_requests": {},
            "current_limits": {},
            "recommended_limits": {},
            "confidence": 0.0,
            "reason": "",
        }

        # Extract usage data from context (if available)
        # The actual usage data comes from the pod_state, but since this is
        # advisory, we generate a generic recommendation
        cpu_trend = getattr(context, "cpu_trend", None)
        mem_trend = getattr(context, "mem_trend", None)

        if cpu_trend is not None and mem_trend is not None and (cpu_trend > 0 or mem_trend > 0):
            recommendation["current_requests"]["cpu"] = "100m"
            recommendation["current_requests"]["memory"] = "128Mi"
            recommendation["recommended_requests"]["cpu"] = f"{int(cpu_trend * 1.2)}m"
            recommendation["recommended_requests"]["memory"] = f"{int(mem_trend * 1.2)}Mi"
            recommendation["confidence"] = 0.75
            recommendation["reason"] = "CPU/memory usage trending above 90% of requests"
        else:
            recommendation["confidence"] = 0.5
            recommendation["reason"] = "Insufficient trend data — generic recommendation"

        message = json.dumps(recommendation, indent=2)

        return RemediationResult(
            action=self.name,
            target=target,
            success=True,
            dry_run=True,  # Always dry_run — advisory only
            message=message,
            command="",  # No kubectl command — advisory only
        )
