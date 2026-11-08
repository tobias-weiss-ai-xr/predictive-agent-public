"""Tests for Dec-POMDP / multi-agent coordination across pods (#9).

Coordination resolves cross-pod conflicts the single-agent planner ignores:
deployment-level action deduplication, node-cordon conflicts, and per-node
concurrency caps with risk-priority sequencing.
"""

import json
import urllib.request

from predictive_agent.coordinator import MultiAgentCoordinator
from predictive_agent.pomdp import ActionAwareTransitionModel, MarkovChain
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationResult,
)
from predictive_agent.server import _context, start_server

ACTION_NAMES = [
    "pod_restart", "rollout_restart", "node_cordon",
    "scale", "right_size", "tune_resources",
]


class _FakeAction(RemediationAction):
    def __init__(self, name):
        self.name = name

    def should_execute(self, pod_state, prediction, risk_score):
        return True

    def execute(self, target, context):
        return RemediationResult(action=self.name, target=target, success=True,
                                  dry_run=context.dry_run, message="ok (fake)")


def _build_model():
    model = ActionAwareTransitionModel(base_chain=MarkovChain())
    for _ in range(40):
        model.record_transition("CRITICAL", "RECOVERED", "pod_restart")
        model.record_transition("CRITICAL", "FAILED", "none")
        model.record_transition("STRESSED", "RECOVERED", "scale")
        model.record_transition("STRESSED", "FAILED", "none")
    return model


def _build_manager():
    rem = RemediationManager(dry_run=False)
    for name in ACTION_NAMES:
        rem.register_action(_FakeAction(name))
    rem.decision_model = _build_model()
    return rem


class _ScriptedCoordinator(MultiAgentCoordinator):
    """Coordinator with per-pod base plans scripted for deterministic tests."""

    def __init__(self, manager, plans, **kw):
        super().__init__(manager, **kw)
        self._scripted = plans

    def _base_plan(self, pod_key, info, horizon, strategy):
        return list(self._scripted.get(pod_key, []))


def test_dedup_deployment_actions_real_manager():
    rem = _build_manager()
    coord = MultiAgentCoordinator(rem)
    pods = {
        "ns/a": {"state": "STRESSED", "deployment": "dep1", "node": "n1"},
        "ns/b": {"state": "STRESSED", "deployment": "dep1", "node": "n1"},
    }
    res = coord.coordinate(pods, horizon=3)
    total_scale = sum(
        1 for p in res["per_pod"].values() for a in p["actions"] if a == "scale"
    )
    assert total_scale == 1
    assert any(c["type"] == "deployment_dedup" for c in res["conflicts_resolved"])


def test_node_cordon_conflict_defers_pod_actions():
    rem = _build_manager()
    plans = {"ns/a": ["node_cordon"], "ns/b": ["pod_restart"]}
    coord = _ScriptedCoordinator(rem, plans, max_actions_per_node=2)
    pods = {
        "ns/a": {"state": "CRITICAL", "node": "n1"},
        "ns/b": {"state": "CRITICAL", "node": "n1"},
    }
    res = coord.coordinate(pods, horizon=3)
    # The cordon on n1 defers the pod-level restart in this wave.
    assert res["per_pod"]["ns/b"]["actions"] == []
    assert "node_cordon" in res["per_pod"]["ns/a"]["actions"]
    assert any(c["type"] == "node_cordon_conflict" for c in res["conflicts_resolved"])


def test_priority_and_per_node_cap():
    rem = _build_manager()
    plans = {f"ns/p{i}": ["pod_restart"] for i in range(3)}
    coord = _ScriptedCoordinator(rem, plans, max_actions_per_node=2)
    pods = {f"ns/p{i}": {"state": "CRITICAL", "node": "n1"} for i in range(3)}
    res = coord.coordinate(pods, horizon=3)
    waves = res["global"]["waves"]
    assert len(waves) == 2
    assert len(waves[0]) == 2
    assert len(waves[1]) == 1
    assert all(w["action"] == "pod_restart" for w in waves[0] + waves[1])


def test_coordinate_endpoint():
    rem = _build_manager()
    ports = (18230, 18231)
    srv = start_server(*ports)
    _context.remediation_manager = rem
    body = {
        "pods": {
            "ns/a": {"state": "STRESSED", "deployment": "dep1", "node": "n1"},
            "ns/b": {"state": "STRESSED", "deployment": "dep1", "node": "n1"},
        },
        "horizon": 3,
    }
    try:
        req = urllib.request.Request(
            f"http://localhost:{ports[0]}/coordinate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            data = json.loads(resp.read().decode())
    finally:
        _context.remediation_manager = None
        srv.shutdown()
        srv.server_close()

    assert status == 200
    assert data["status"] == "ok"
    assert data["execution_ready"] is True
    assert "waves" in data["global"]
    assert any(c["type"] == "deployment_dedup" for c in data["conflicts_resolved"])
