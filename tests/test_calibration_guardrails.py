"""Tests for calibration monitoring + safety guardrails (#3).

Covers:
  * CalibrationMonitor drift detection (Brier / over-prediction rate).
  * The reconcile learning path feeding predicted P(FAILED) vs realised outcome
    into the monitor (record_observations -> calibration.record).
  * Confidence gate: the planner refuses to act when prediction confidence is
    below config.MIN_ACTION_CONFIDENCE.
  * Circuit breaker: max_actions caps actions executed per call (blast radius).
  * Degraded fallback: on detected drift the planner falls back to reactive.
  * The read-only /calibration endpoint.
"""

import json as _json
import urllib.request

from predictive_agent import config
from predictive_agent.calibration import CalibrationMonitor
from predictive_agent.pomdp import ActionAwareTransitionModel, MarkovChain
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationResult,
)
from predictive_agent.server import _context, start_server

ACTION_NAMES = [
    "pod_restart",
    "rollout_restart",
    "node_cordon",
    "scale",
    "right_size",
    "tune_resources",
]


class _FakeAction(RemediationAction):
    def __init__(self, name):
        self.name = name

    def should_execute(self, pod_state, prediction, risk_score):
        return True

    def execute(self, target, context):
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="ok (fake)",
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


class _Pred:
    def __init__(self, risk_score=0.9, confidence=0.8):
        self.risk_score = risk_score
        self.confidence = confidence


def _build_manager(states=None):
    """Build a manager whose seeded model yields a multi-step plan from CRITICAL."""
    rem = RemediationManager(dry_run=False)
    for name in ACTION_NAMES:
        rem.register_action(_FakeAction(name))
    model = ActionAwareTransitionModel(base_chain=MarkovChain())
    for _ in range(40):
        # CRITICAL -> (pod_restart) STRESSED -> (scale) RECOVERED: a 2-step plan.
        model.record_transition("CRITICAL", "STRESSED", "pod_restart")
        model.record_transition("CRITICAL", "FAILED", "none")
        model.record_transition("STRESSED", "RECOVERED", "scale")
        model.record_transition("STRESSED", "FAILED", "none")
    rem.decision_model = model
    return rem


# ─── CalibrationMonitor ────────────────────────────────────────────────────
def test_calibration_monitor_reports_healthy():
    mon = CalibrationMonitor(window=50, min_samples=20, brier_alert=0.25, over_rate_alert=0.5)
    for _ in range(30):
        mon.record(0.1, False)
        mon.record(0.9, True)
    assert mon.drift_detected() is False
    assert mon.status() == "healthy"
    assert mon.brier() < 0.05


def test_calibration_monitor_detects_over_prediction_drift():
    mon = CalibrationMonitor(window=50, min_samples=20, brier_alert=0.25, over_rate_alert=0.5)
    # Predict failure but it never happens -> over-prediction drift.
    for _ in range(30):
        mon.record(0.9, False)
    assert mon.drift_detected() is True
    assert mon.status() == "drifted"
    assert mon.over_rate() == 1.0


def test_record_observations_feeds_calibration():
    rem = _build_manager()
    mon = CalibrationMonitor(window=50, min_samples=1)
    rem.calibration_monitor = mon
    pod = _Pod("CRITICAL")
    pred = _Pred(risk_score=0.9, confidence=0.8)

    results = rem.plan_and_execute("CRITICAL", "ns/pod", pod, pred, 0.9, horizon=3)
    assert results, "planner should act from CRITICAL"

    # Observed state recovers -> calibration pairs (predicted=0.9, actual=False),
    # one per executed action in the plan.
    rem.record_observations({"ns/pod": "RECOVERED"})
    assert mon.n() >= 1
    for predicted, actual in mon._pairs:
        assert abs(predicted - 0.9) < 1e-9
        assert actual is False


# ─── Confidence gate ───────────────────────────────────────────────────────
def test_confidence_gate_skips_action(monkeypatch):
    rem = _build_manager()
    monkeypatch.setattr(config, "MIN_ACTION_CONFIDENCE", 0.99)
    pod = _Pod("CRITICAL")
    pred = _Pred(risk_score=0.9, confidence=0.5)  # below gate
    results = rem.plan_and_execute("CRITICAL", "ns/pod", pod, pred, 0.9, horizon=3)
    assert results == []


# ─── Circuit breaker (max_actions) ────────────────────────────────────────
def test_max_actions_circuit_breaker():
    rem = _build_manager()
    pod = _Pod("CRITICAL")
    pred = _Pred(risk_score=0.9, confidence=0.8)

    full_plan = rem.plan_remediation("CRITICAL", 3)["steps"]
    full_actions = [s for s in full_plan if s["action"] != "none"]

    capped = rem.plan_and_execute(
        "CRITICAL", "ns/pod", pod, pred, 0.9, horizon=3, max_actions=1
    )
    assert len(capped) <= 1
    if len(full_actions) > 1:
        # The cap must visibly truncate a multi-step plan.
        assert len(capped) == 1

    none = rem.plan_and_execute(
        "CRITICAL", "ns/pod", pod, pred, 0.9, horizon=3, max_actions=0
    )
    assert none == []


# ─── Degraded fallback ─────────────────────────────────────────────────────
def test_degraded_falls_back_to_reactive():
    rem = _build_manager()
    pod = _Pod("CRITICAL")
    pred = _Pred(risk_score=0.9, confidence=0.8)

    # Degraded: planner must call the reactive evaluate() path.
    rem.degraded = True
    called = []
    orig_evaluate = rem.evaluate
    rem.evaluate = lambda *a, **k: (called.append(1), [])[1]
    try:
        rem.plan_and_execute("CRITICAL", "ns/pod", pod, pred, 0.9, horizon=3)
    finally:
        rem.evaluate = orig_evaluate
    assert called, "degraded planner should fall back to evaluate()"

    # Healthy: planner path, evaluate() not invoked.
    rem.degraded = False
    called.clear()
    rem.evaluate = lambda *a, **k: (called.append(1), [])[1]
    try:
        rem.plan_and_execute("CRITICAL", "ns/pod", pod, pred, 0.9, horizon=3)
    finally:
        rem.evaluate = orig_evaluate
    assert not called, "healthy planner should not use evaluate()"


# ─── /calibration endpoint ────────────────────────────────────────────────
def test_calibration_endpoint_serves_summary():
    rem = _build_manager()
    mon = CalibrationMonitor(window=50, min_samples=5)
    mon.record(0.9, True)
    rem.calibration_monitor = mon

    ports = (18170, 18171)
    srv = start_server(*ports)
    _context.remediation_manager = rem
    try:
        with urllib.request.urlopen(f"http://localhost:{ports[0]}/calibration") as resp:
            status = resp.status
            data = _json.loads(resp.read().decode())
    finally:
        _context.remediation_manager = None
        srv.shutdown()
        srv.server_close()

    assert status == 200
    assert data["enabled"] is True
    assert data["degraded"] is False
    assert data["summary"]["n"] == 1
    assert data["summary"]["status"] == "warmup"
