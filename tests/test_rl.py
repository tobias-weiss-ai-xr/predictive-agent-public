"""Tests for sampled RL: model-free Q-learning from outcomes (#8).

The Q-learning policy learns Q(state, action) from the *same* real outcomes
that train the transition model (via record_observations), and is exposed as a
5th benchmark strategy and a /plan?strategy=q preview. It must converge to
acting when acting reduces failure cost.
"""

import json as _json
import tempfile
import urllib.request

import pytest

from predictive_agent.benchmark import run_benchmark
from predictive_agent.persistence import StateStore
from predictive_agent.pomdp import ActionAwareTransitionModel, MarkovChain
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationResult,
)
from predictive_agent.rl import QLearningPolicy
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


class _Pod:
    def __init__(self, state):
        self.state = state
        self.name = "ns/pod"
        self.namespace = "ns"
        self.node_name = "n"
        self.pod_phase = "Running"
        self.cpu_trend = 0.0
        self.memory_trend = 0.0
        self.memory_pct = 0
        self.cpu_pct = 0
        self.memory_anomaly_score = 0
        self.cpu_anomaly_score = 0


class _Pred:
    def __init__(self, risk_score=0.9, confidence=0.8):
        self.risk_score = risk_score
        self.confidence = confidence


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


def test_q_learning_learns_to_act():
    model = _build_model()
    q = QLearningPolicy(model=model)
    # Pod restart recovers CRITICAL; doing nothing fails. After enough outcomes
    # the Q-value of acting must exceed the Q-value of doing nothing.
    for _ in range(50):
        q.update("CRITICAL", "pod_restart", "RECOVERED")
        q.update("CRITICAL", "none", "FAILED")
    assert q.Q["CRITICAL"]["pod_restart"] > q.Q["CRITICAL"]["none"]
    # And the greedy policy should choose to act.
    assert q.action("CRITICAL") == "pod_restart"


def test_q_policy_plan_reduces_failure():
    model = _build_model()
    q = QLearningPolicy(model=model)
    for _ in range(50):
        q.update("CRITICAL", "pod_restart", "RECOVERED")
        q.update("CRITICAL", "none", "FAILED")
    actions, q_pf = q.plan("CRITICAL", 3, model)
    none_pf = _none_baseline(model, "CRITICAL", 3)
    assert "pod_restart" in actions
    assert q_pf <= none_pf + 1e-9


def _none_baseline(model, state, horizon):
    dist = {s: 0.0 for s in MarkovChain.STATES}
    dist[state] = 1.0
    for _ in range(horizon):
        dist = model.step_distribution(dist, "none")
    return model.failed_probability(dist)


def test_record_observations_feeds_rl_policy():
    rem = _build_manager()
    q = QLearningPolicy(model=rem.decision_model)
    rem.rl_policy = q
    before = q.Q["CRITICAL"]["pod_restart"]
    pod = _Pod("CRITICAL")
    rem.plan_and_execute("CRITICAL", "ns/pod", pod, _Pred(), 0.9, horizon=3)
    rem.record_observations({"ns/pod": "RECOVERED"})
    # The real outcome (CRITICAL, pod_restart -> RECOVERED) updated the Q-table.
    assert q.Q["CRITICAL"]["pod_restart"] != before


def test_benchmark_includes_q_strategy():
    rem = _build_manager()
    report = run_benchmark(rem, [("CRITICAL", 0.9), ("STRESSED", 0.9)], horizon=3)
    for s in report.scenarios:
        assert "q_actions" in s.to_dict()
        assert "q_p_failed" in s.to_dict()
        assert isinstance(s.q_p_failed, float)
    assert "mean_q_p_failed" in report.summary()
    assert "q_intervenes" in report.summary()


def test_plan_endpoint_q_strategy():
    rem = _build_manager()
    ports = (18220, 18221)
    srv = start_server(*ports)
    _context.remediation_manager = rem
    try:
        with urllib.request.urlopen(f"http://localhost:{ports[0]}/plan?state=CRITICAL&strategy=q") as resp:
            status = resp.status
            data = _json.loads(resp.read().decode())
    finally:
        _context.remediation_manager = None
        srv.shutdown()
        srv.server_close()

    assert status == 200
    assert data["strategy"] == "q"
    assert data["plan"]["execution_ready"] is True
    assert len(data["plan"]["steps"]) >= 1


def test_q_table_persistence_roundtrip():
    model = _build_model()
    q = QLearningPolicy(model=model)
    for _ in range(30):
        q.update("CRITICAL", "pod_restart", "RECOVERED")
        q.update("CRITICAL", "none", "FAILED")
    tmp = tempfile.mkdtemp(prefix="predagent-q-")
    store = StateStore(state_model_file=f"{tmp}/state.json")
    store.save_q_table(q.to_dict())
    loaded = store.load_q_table()
    assert loaded is not None
    q2 = QLearningPolicy(model=model)
    q2.load(loaded)
    assert q2.Q["CRITICAL"]["pod_restart"] == pytest.approx(q.Q["CRITICAL"]["pod_restart"])
