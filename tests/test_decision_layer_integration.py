"""End-to-end integration test for the POMDP decision layer.

This is the "spin it up locally" test: it boots the real HTTP server together
with a fully wired :class:`RemediationManager` (using no-op fake actions so no
Kubernetes cluster is required) and a learned decision model, then drives the
whole loop the operator would see in production:

    predict -> plan -> (dry-run) execute -> learn -> persist

and verifies the operator-facing endpoints (/plan, /plan?strategy=policy,
/world-model, /backtest), the closed learning loop, model persistence, and the
four-strategy benchmark. It runs entirely on the local machine.

The HTTP server is started once per test session (session-scoped fixture) to
avoid macOS socket-reuse races from repeatedly binding/unbinding ports; each
test then wires its own decision layer into the shared server context.
"""

import json
import tempfile
import urllib.request

import pytest

from predictive_agent.backtester import Backtester
from predictive_agent.benchmark import run_benchmark
from predictive_agent.markov import MarkovChain
from predictive_agent.persistence import StateStore
from predictive_agent.pomdp import ActionAwareTransitionModel
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationResult,
)
from predictive_agent.server import _context, start_server

METRICS_PORT = 18150
HEALTH_PORT = 18151

ACTION_NAMES = [
    "pod_restart",
    "rollout_restart",
    "node_cordon",
    "scale",
    "right_size",
    "tune_resources",
]


class _FakeAction(RemediationAction):
    """No-op action so the loop runs without a real Kubernetes cluster."""

    def __init__(self, name, should=True):
        self.name = name
        self._should = should

    def should_execute(self, pod_state, prediction, risk_score):
        return self._should

    def execute(self, target, context):
        return RemediationResult(
            action=self.name,
            target=target,
            success=True,
            dry_run=context.dry_run,
            message="ok (fake, no kubectl)",
        )


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


def _build_manager() -> RemediationManager:
    rem = RemediationManager(dry_run=False)
    for name in ACTION_NAMES:
        rem.register_action(_FakeAction(name))
    model = ActionAwareTransitionModel(base_chain=MarkovChain())
    # Seed some learning so the planner has a reason to act.
    for _ in range(40):
        model.record_transition("CRITICAL", "RECOVERED", "pod_restart")
        model.record_transition("CRITICAL", "FAILED", "none")
        model.record_transition("STRESSED", "RECOVERED", "scale")
        model.record_transition("STRESSED", "FAILED", "none")
    rem.decision_model = model
    return rem


@pytest.fixture(scope="session")
def server_session():
    """Start the real metrics/API server once for the whole module."""
    srv = start_server(METRICS_PORT, HEALTH_PORT)
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def spun_up(server_session):
    """Wire a fresh, learned decision layer into the shared server context."""
    tmp = tempfile.mkdtemp(prefix="predagent-it-")
    rem = _build_manager()
    bt = Backtester(history_file=f"{tmp}/backtester.json")
    _context.remediation_manager = rem
    _context.backtester = bt
    yield rem, bt
    _context.remediation_manager = None
    _context.backtester = None


def _get(path):
    with urllib.request.urlopen(f"http://localhost:{METRICS_PORT}{path}") as resp:
        return resp.status, json.loads(resp.read().decode())


# ─── Operator-facing endpoints ──────────────────────────────────────────────
def test_plan_endpoint_responds(spun_up):
    _, _ = spun_up
    status, data = _get("/plan?state=CRITICAL&horizon=3")
    assert status == 200
    assert data["status"] == "ok"
    assert data["plan"]["execution_ready"] is True
    assert len(data["plan"]["steps"]) == 3
    assert any(s["action"] != "none" for s in data["plan"]["steps"])


def test_plan_policy_strategy_responds(spun_up):
    _, _ = spun_up
    status, data = _get("/plan?state=CRITICAL&strategy=policy")
    assert status == 200
    assert data["strategy"] == "policy"
    assert len(data["plan"]["steps"]) >= 1


def test_world_model_endpoint_responds(spun_up):
    _, _ = spun_up
    status, data = _get("/world-model?state=CRITICAL&action=pod_restart&horizon=2")
    assert status == 200
    assert data["status"] == "ok"
    assert len(data["steps"]) == 2
    assert "expected_failed_interval" in data["steps"][0]
    # The interval brackets the displayed point estimate.
    mean, lo, hi = data["steps"][0]["expected_failed_interval"]
    pf = data["steps"][0]["expected_failed_prob"]
    assert lo <= pf <= hi or abs(mean - pf) < 0.02


def test_backtest_endpoint_responds(spun_up):
    _, _ = spun_up
    status, data = _get("/backtest")
    assert status == 200
    # The backtester returns a populated evaluation report.
    assert isinstance(data, dict) and "brier_score" in data


# ─── Closed learning loop ───────────────────────────────────────────────────
def test_execution_and_learning_loop(spun_up):
    rem, _ = spun_up
    pod = _Pod("CRITICAL")
    results = rem.plan_and_execute(
        "CRITICAL", "ns/pod", pod, None, 0.95, horizon=3
    )
    assert results, "planner should have executed at least one action"
    assert all(r.success for r in results)

    # Simulate the observed next state and feed it back to the model.
    before = sum(
        sum(r) for r in rem.decision_model.action_counts.get("pod_restart", [])
    )
    rem.feed_transition("CRITICAL", "pod_restart", "RECOVERED")
    after = sum(
        sum(r) for r in rem.decision_model.action_counts.get("pod_restart", [])
    )
    assert after == before + 1, "learning loop must record the observed outcome"


def test_persistence_roundtrip(spun_up, tmp_path):
    rem, _ = spun_up
    store = StateStore(state_model_file=str(tmp_path / "state.json"))
    store.save_decision_model(rem.decision_model.to_action_dict())
    loaded = store.load_decision_model()
    assert loaded is not None
    restored = ActionAwareTransitionModel(base_chain=MarkovChain())
    restored.load_action_counts(loaded)
    assert restored.action_counts == rem.decision_model.action_counts


def test_benchmark_four_strategies(spun_up):
    rem, _ = spun_up
    report = run_benchmark(
        rem, [("CRITICAL", 0.9), ("STRESSED", 0.9)], horizon=3
    )
    for s in report.scenarios:
        # Both the myopic planner and the RL/value-iteration policy must beat
        # doing nothing.
        assert s.planner_p_failed <= s.none_p_failed + 1e-9
        assert s.policy_p_failed <= s.none_p_failed + 1e-9
