"""Dec-POMDP / multi-agent coordination across pods (#9).

The single-agent planner (:class:`~predictive_agent.pomdp.Planner`) optimises
one pod in isolation. On a real cluster, pods share nodes and deployments, so
independent plans can conflict (e.g. cordoning a node while another pod on it is
being restarted, or scaling the same deployment twice). This module coordinates
the per-pod plans into a single *joint* plan that is safe to execute
concurrently:

* **Deployment-level actions are deduplicated** — ``scale`` / ``rollout_restart``
  act on a Deployment, so only one is kept per (deployment, action).
* **Node-cordon conflicts are resolved** — if any pod on a node is cordoned,
  pod-level mutating actions on that node are deferred in this wave (recorded as
  a conflict) to avoid acting on a draining node.
* **Per-node concurrency cap** — actions are packed into waves so no node
  exceeds ``max_actions_per_node`` simultaneous remediations, and higher-risk
  pods (CRITICAL/FAILED first) get priority.

This is the decentralised layer: each pod still uses its own planner, but the
coordinator resolves cross-pod dependencies before execution.
"""

from collections import defaultdict
from typing import Dict, List

# Node-level actions that conflict with pod-level mutations on the same node.
NODE_LEVEL_ACTIONS = {"node_cordon"}
# Pod-level mutating actions that must not run on a node being cordoned.
NODE_CONFLICTING_ACTIONS = {
    "pod_restart", "rollout_restart", "scale", "right_size", "tune_resources",
}
# Deployment-level actions: one instance suffices per deployment.
DEPLOYMENT_ACTIONS = {"scale", "rollout_restart"}

_STATE_PRIORITY = {
    "FAILED": 5,
    "CRITICAL": 5,
    "STRESSED": 4,
    "DEGRADED": 3,
    "RECOVERED": 1,
    "HEALTHY": 0,
}


def _priority(state: str) -> int:
    return _STATE_PRIORITY.get(state, 0)


class MultiAgentCoordinator:
    """Coordinate independent per-pod plans into one safe joint plan."""

    def __init__(self, manager, max_actions_per_node: int = 2):
        self.manager = manager
        self.max_actions_per_node = max_actions_per_node

    # ─── public API ──────────────────────────────────────────────────────────
    def coordinate(
        self,
        pods: Dict[str, dict],
        horizon: int = 3,
        strategy: str = "planner",
    ) -> dict:
        """Build a coordinated joint plan for ``pods``.

        ``pods`` maps a pod key to either a state string or a dict with at least
        ``state`` plus optional ``node``, ``namespace``, ``deployment``.
        Returns a dict with ``global`` (ordered waves), ``per_pod`` plans, and
        ``conflicts_resolved``.
        """
        pod_info: Dict[str, dict] = {}
        for key, val in pods.items():
            info = dict(val) if isinstance(val, dict) else {"state": val}
            info.setdefault("state", "HEALTHY")
            pod_info[key] = info

        # 1. Per-pod base plans (each pod optimises itself, ignoring others).
        per_pod: Dict[str, dict] = {}
        for key, info in pod_info.items():
            actions = self._base_plan(key, info, horizon, strategy)
            per_pod[key] = {"state": info["state"], "actions": actions}

        conflicts: List[dict] = []

        # 2. Deduplicate deployment-level actions across pods.
        seen_deployment: dict = {}
        for key in sorted(pod_info, key=lambda k: _priority(pod_info[k].get("state") or ""), reverse=True):
            info = pod_info[key]
            dep = info.get("deployment")
            kept = []
            for a in per_pod[key]["actions"]:
                if a in DEPLOYMENT_ACTIONS and dep:
                    if (dep, a) in seen_deployment:
                        conflicts.append({
                            "type": "deployment_dedup",
                            "pod": key,
                            "action": a,
                            "deployment": dep,
                            "kept_on": seen_deployment[(dep, a)],
                        })
                        continue
                    seen_deployment[(dep, a)] = key
                kept.append(a)
            per_pod[key]["actions"] = kept

        # 3. Resolve node-cordon conflicts: defer pod mutations on cordoned nodes.
        cordoned_nodes = {
            pod_info[k].get("node")
            for k in pod_info
            if "node_cordon" in per_pod[k]["actions"] and pod_info[k].get("node")
        }
        if cordoned_nodes:
            for key, info in pod_info.items():
                node = info.get("node")
                if node in cordoned_nodes:
                    new_actions = []
                    for a in per_pod[key]["actions"]:
                        if a in NODE_CONFLICTING_ACTIONS:
                            conflicts.append({
                                "type": "node_cordon_conflict",
                                "pod": key,
                                "action": a,
                                "node": node,
                            })
                            continue
                        new_actions.append(a)
                    per_pod[key]["actions"] = new_actions

        # 4. Sequence into waves (priority + per-node concurrency cap).
        waves = self._sequence(pod_info, per_pod)

        return {
            "global": {"waves": waves, "execution_ready": True},
            "per_pod": per_pod,
            "conflicts_resolved": conflicts,
            "execution_ready": True,
            "rationale": (
                "Joint Dec-POMDP plan: deployment actions deduplicated, "
                "node-cordon conflicts deferred, actions sequenced per-node to "
                "avoid overload."
            ),
        }

    # ─── internals ────────────────────────────────────────────────────────────
    def _base_plan(self, pod_key: str, info: dict, horizon: int, strategy: str) -> List[str]:
        """Per-pod action list from the underlying manager's planner."""
        plan = self.manager.plan_remediation(info["state"], horizon)
        if isinstance(plan, dict):
            steps = plan.get("steps", [])
            actions = [s.get("action") for s in steps]
        else:
            steps = getattr(plan, "steps", [])
            actions = [getattr(s, "action", None) for s in steps]
        return [a for a in actions if a and a != "none"]

    def _sequence(self, pod_info: Dict[str, dict], per_pod: Dict[str, dict]) -> List[list]:
        items = []
        for key, info in pod_info.items():
            state = info.get("state") or ""
            node = info.get("node")
            for a in per_pod[key]["actions"]:
                items.append({
                    "pod": key,
                    "action": a,
                    "node": node,
                    "priority": _priority(state),
                })
        # CRITICAL / FAILED first; stable tie-break by pod key.
        items.sort(key=lambda it: (-it["priority"], it["pod"]))
        waves: List[list] = []
        remaining = list(items)
        while remaining:
            wave = []
            used: defaultdict = defaultdict(int)
            still = []
            for it in remaining:
                n = it["node"]
                if n is None or used[n] < self.max_actions_per_node:
                    wave.append(it)
                    if n is not None:
                        used[n] += 1
                else:
                    still.append(it)
            if not wave:
                # Safety: never loop forever if nothing could be packed.
                break
            waves.append(wave)
            remaining = still
        return waves
