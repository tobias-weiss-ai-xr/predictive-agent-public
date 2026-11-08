"""Remediation action modules for predictive-agent.

Each module implements one remediation action:
- pod_restart: Delete unhealthy pods (CrashLoopBackOff, OOMKilled)
- node_cordon: Cordon nodes with sustained pressure
- right_size: Generate resource right-sizing recommendations (advisory)
- rollout_restart: Restart deployments with chronic failures
- scale: Scale deployments up/down based on resource pressure
- tune_resources: Auto-tune deployment resource limits
- container_restart: Restart Docker containers (Docker runtime only)
"""

from predictive_agent.actions.container_restart import ContainerRestartAction
from predictive_agent.actions.node_cordon import NodeCordonAction
from predictive_agent.actions.pod_restart import PodRestartAction
from predictive_agent.actions.right_size import RightSizeAction
from predictive_agent.actions.rollout_restart import RolloutRestartAction
from predictive_agent.actions.scale import DeploymentScaleAction
from predictive_agent.actions.tune_resources import ResourceTunerAction

__all__ = [
    "PodRestartAction",
    "NodeCordonAction",
    "RightSizeAction",
    "RolloutRestartAction",
    "DeploymentScaleAction",
    "ResourceTunerAction",
    "ContainerRestartAction",
]
