"""NodeDiskGCAction — bounded-autonomy node disk-pressure remediation.

Addresses the 2026-11-18 incident class: a node enters DiskPressure (local
ephemeral/image storage exhaustion), the scheduler taints it, and workloads
pinned to it (e.g. a nodeSelector-constrained Prometheus) get evicted and
cannot reschedule — monitoring goes blind.

Bounded-autonomy design:
  * ``NODE_GC_METHOD=kubectl`` (default) — the agent has no host access in the
    cluster. It cordons the pressuring node (stops further scheduling) and
    emits a precise operator instruction (the exact prune command), so the
    blast radius never grows and a human has a ready-made fix.
  * ``NODE_GC_METHOD=docker`` — only when the agent itself runs on the node
    with the docker socket (standalone docker runtime / monitoring_agents
    role). Executes ``docker system prune -af`` (configurable args) with a
    timeout and records reclaimed space. On any failure (no binary, no socket)
    it degrades safely to the kubectl cordon + instruction path.
  * ``NODE_GC_NODES`` allowlist (comma list; empty = all nodes).
  * NOT reversible (a prune cannot be undone) — the manager therefore requires
    operator approval before re-attempts after failed verification.

Uses the pod's node (``context.node_name`` / ``pod_state.node_name``) as the
target — the same resolution as NodeCordonAction.
"""

from __future__ import annotations

import logging
import subprocess

from predictive_agent import config
from predictive_agent.remediator import (
    ActionContext,
    RemediationAction,
    RemediationResult,
)

logger = logging.getLogger(__name__)

RISK_THRESHOLD = 0.7


class NodeDiskGCAction(RemediationAction):
    """Reduce node disk pressure (prune or cordon + instruct)."""

    name = "node_gc"
    idempotent: bool = True
    risk_tier: str = "destructive"
    reversible: bool = False
    resolution_signals: tuple = ("node_disk_pressure", "state")

    def __init__(
        self,
        method=None,
        nodes=None,
        enabled=None,
        docker_prune_args=None,
        timeout_s=None,
        threshold=None,
    ):
        self.enabled = config.NODE_GC_ENABLED if enabled is None else bool(enabled)
        self.method = (
            config.NODE_GC_METHOD if method is None else (method or "none").lower()
        )
        raw_nodes = config.NODE_GC_NODES if nodes is None else (nodes or "")
        self.nodes = {n.strip() for n in str(raw_nodes).split(",") if n.strip()}
        self.docker_prune_args = (
            config.NODE_GC_DOCKER_PRUNE_ARGS if docker_prune_args is None
            else (docker_prune_args or "-af")
        )
        self.timeout_s = (
            config.NODE_GC_TIMEOUT_S if timeout_s is None else float(timeout_s)
        )
        self.threshold = (
            config.NODE_GC_RISK_THRESHOLD if threshold is None else float(threshold)
        )

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        if not self.enabled:
            return False
        if self.method not in ("kubectl", "docker"):
            return False
        if risk_score <= self.threshold:
            return False
        if not getattr(pod_state, "node_disk_pressure", False):
            return False
        node_name = getattr(pod_state, "node_name", "") or ""
        if self.nodes and node_name not in self.nodes:
            return False
        return True

    def execute(self, target: str, context: ActionContext) -> RemediationResult:
        node_name = getattr(context, "node_name", "") or target
        if not node_name:
            return RemediationResult(
                action=self.name, target=target, success=False,
                dry_run=context.dry_run,
                message="node_gc: no node name available",
            )
        if not self.enabled:
            return RemediationResult(
                action=self.name, target=node_name, success=False,
                dry_run=context.dry_run,
                message="node_gc not enabled",
            )
        if self.method == "docker":
            result = self._docker_prune(node_name, context)
            if result.success or result.dry_run:
                return result
            # degrade: docker GC unavailable -> cordon + instruct (bounded)
            logger.warning(
                "node_gc: docker GC failed on %s — degrading to cordon+instruction (%s)",
                node_name, result.message,
            )
        return self._cordon_and_instruct(node_name, context)

    def _docker_prune(self, node_name: str, context) -> RemediationResult:
        cmd = ["docker", "system", "prune"] + self.docker_prune_args.split()
        if context.dry_run:
            cmd.append("-f")  # -af already implies -f
            cmd = [c for c in cmd if c != "-f"] or ["docker", "system", "prune", "-af"]
        cmd_str = " ".join(cmd)
        # NOTE: never pass --volumes — registry/blob volumes carry data.
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_s
            )
        except subprocess.TimeoutExpired:
            return RemediationResult(
                action=self.name, target=node_name, success=False,
                dry_run=context.dry_run,
                message=f"docker prune timed out after {self.timeout_s}s",
                command=cmd_str,
            )
        except Exception as e:  # FileNotFoundError: no docker on this host
            return RemediationResult(
                action=self.name, target=node_name, success=False,
                dry_run=context.dry_run,
                message=f"docker GC unavailable: {e}",
                command=cmd_str,
            )
        success = res.returncode == 0
        message = res.stdout.strip() if success else res.stderr.strip()
        if not message:
            message = "docker system prune done" if success else "docker prune failed"
        return RemediationResult(
            action=self.name, target=node_name, success=success,
            dry_run=context.dry_run, message=message, command=cmd_str,
        )

    def _cordon_and_instruct(self, node_name: str, context) -> RemediationResult:
        cmd = ["kubectl", "cordon", node_name]
        if context.dry_run:
            cmd.append("--dry-run=server")
        cmd_str = " ".join(cmd)
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return RemediationResult(
                action=self.name, target=node_name, success=False,
                dry_run=context.dry_run,
                message="kubectl cordon timed out after 30s",
                command=cmd_str,
            )
        except Exception as e:  # noqa: BLE001
            return RemediationResult(
                action=self.name, target=node_name, success=False,
                dry_run=context.dry_run,
                message=f"error: {e}", command=cmd_str,
            )
        success = res.returncode == 0
        message = res.stdout.strip() if success else res.stderr.strip()
        if success:
            message += (
                " — disk pressure: prune node storage first, e.g. "
                "'docker system prune -af' (no volumes) or containerd image "
                "GC, then 'kubectl uncordon " + node_name + "'"
            )
        else:
            message = message or "kubectl cordon failed"
        return RemediationResult(
            action=self.name, target=node_name, success=success,
            dry_run=context.dry_run, message=message, command=cmd_str,
        )
