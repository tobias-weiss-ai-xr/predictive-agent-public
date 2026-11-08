"""Tests for decision-layer Prometheus metrics (#7).

Covers the intervention / learning / calibration counters on the
RemediationManager, the metric collection helper, and the /metrics (+ JSON
/decision-metrics) server endpoints.
"""

import json as _json
import urllib.request

from predictive_agent.calibration import CalibrationMonitor
from predictive_agent.metrics import collect_decision_metrics, render_prometheus
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


class _FakePredictor:
    risk_threshold = 0.7
    _predictions = {"ns/pod": _Pred(0.9, 0.8)}


def _build_manager():
    rem = RemediationManager(dry_run=False)
    for name in ACTION_NAMES:
        rem.register_action(_FakeAction(name))
    model = ActionAwareTransitionModel(base_chain=MarkovChain())
    for _ in range(40):
        model.record_transition("CRITICAL", "RECOVERED", "pod_restart")
        model.record_transition("CRITICAL", "FAILED", "none")
        model.record_transition("STRESSED", "RECOVERED", "scale")
        model.record_transition("STRESSED", "FAILED", "none")
    rem.decision_model = model
    return rem


def test_manager_accumulates_intervention_counters():
    rem = _build_manager()
    pod = _Pod("CRITICAL")
    pred = _Pred(0.9, 0.8)
    rem.plan_and_execute("CRITICAL", "ns/pod", pod, pred, 0.9, horizon=3)
    assert rem.interventions_executed >= 1
    assert rem.cumulative_p_failed_reduction > 0.0
    mean = rem.cumulative_p_failed_reduction / rem.interventions_executed
    assert 0.0 <= mean <= 1.0


def test_collect_decision_metrics_includes_core_and_calibration():
    rem = _build_manager()
    pod = _Pod("CRITICAL")
    pred = _Pred(0.9, 0.8)
    rem.plan_and_execute("CRITICAL", "ns/pod", pod, pred, 0.9, horizon=3)

    # With a calibration monitor attached, calibration metrics appear.
    mon = CalibrationMonitor(window=50, min_samples=1)
    mon.record(0.9, True)
    rem.calibration_monitor = mon
    rem.degraded = True

    metrics = {name: value for (name, _t, _h, value) in collect_decision_metrics(rem, _FakePredictor())}
    assert "decision_layer_interventions_total" in metrics
    assert metrics["decision_layer_interventions_total"] >= 1
    assert "decision_layer_learning_transitions_total" in metrics
    assert metrics["decision_layer_learning_transitions_total"] > 0
    assert "decision_layer_mean_p_failed_reduction" in metrics
    assert metrics["decision_layer_degraded"] == 1
    assert metrics["decision_layer_calibration_brier"] >= 0.0
    assert metrics["decision_layer_calibration_drift"] == 0
    assert metrics["decision_layer_calibration_samples"] == 1
    # Predictor confidence is surfaced when a predictor is supplied.
    assert "decision_layer_mean_confidence" in metrics
    assert abs(metrics["decision_layer_mean_confidence"] - 0.8) < 1e-9


def test_render_prometheus_format():
    rem = _build_manager()
    rem.plan_and_execute("CRITICAL", "ns/pod", _Pod("CRITICAL"), _Pred(), 0.9, horizon=3)
    text = render_prometheus(collect_decision_metrics(rem))
    assert "opendesk_predictive_agent_decision_layer_interventions_total" in text
    assert "# TYPE" in text and "# HELP" in text


def test_decision_metrics_json_endpoint():
    rem = _build_manager()
    mon = CalibrationMonitor(window=50, min_samples=5)
    mon.record(0.9, True)
    rem.calibration_monitor = mon
    rem.degraded = True

    ports = (18210, 18211)
    srv = start_server(*ports)
    _context.remediation_manager = rem
    try:
        with urllib.request.urlopen(f"http://localhost:{ports[0]}/decision-metrics") as resp:
            status = resp.status
            data = _json.loads(resp.read().decode())
    finally:
        _context.remediation_manager = None
        srv.shutdown()
        srv.server_close()

    assert status == 200
    assert data["decision_layer_interventions_total"] == 0  # no execution yet
    assert data["decision_layer_degraded"] == 1
    assert data["decision_layer_calibration_samples"] == 1


def test_metrics_endpoint_includes_decision_metrics():
    rem = _build_manager()
    rem.plan_and_execute("CRITICAL", "ns/pod", _Pod("CRITICAL"), _Pred(), 0.9, horizon=3)

    ports = (18212, 18213)
    srv = start_server(*ports)
    _context.remediation_manager = rem
    try:
        with urllib.request.urlopen(f"http://localhost:{ports[0]}/metrics") as resp:
            status = resp.status
            text = resp.read().decode()
    finally:
        _context.remediation_manager = None
        srv.shutdown()
        srv.server_close()

    assert status == 200
    assert "opendesk_predictive_agent_decision_layer_interventions_total" in text
    assert "opendesk_predictive_agent_decision_layer_learning_transitions_total" in text
