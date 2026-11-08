"""Tests for cost-model calibration from incident data (#5).

Verifies that the decision-theoretic costs (action_costs / failure_cost) can be
derived from real incident history via predictive_agent.cost_model, that the
live RemediationManager picks them up, and that the planner's intervention
decisions actually respond to the calibrated costs. Also covers the
GET/POST /cost-model endpoint.
"""

import json as _json
import urllib.request

import pytest

from predictive_agent import config
from predictive_agent.cost_model import calibrate_costs
from predictive_agent.pomdp import ActionAwareTransitionModel, MarkovChain, Planner
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


def _seeded_model():
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
    rem.decision_model = _seeded_model()
    return rem


# ─── calibrate_costs ───────────────────────────────────────────────────────
def test_calibrate_costs_failure_cost_from_slo():
    incidents = [{"action": "pod_restart", "downtime_minutes": 2}]
    _, failure_cost = calibrate_costs(
        incidents, slo_cost_per_minute=2.0, typical_outage_minutes=15.0
    )
    assert failure_cost == 2.0 * 15.0  # 30.0


def test_calibrate_costs_scales_with_downtime():
    incidents = [
        {"action": "pod_restart", "downtime_minutes": 2},
        {"action": "node_cordon", "downtime_minutes": 20},
    ]
    action_costs, _ = calibrate_costs(
        incidents, slo_cost_per_minute=1.0, typical_outage_minutes=30.0,
        disruption_fraction=0.1,
    )
    # node_cordon disrupts 10x longer -> ~10x the cost of pod_restart.
    assert action_costs["node_cordon"] > action_costs["pod_restart"]
    assert abs(action_costs["pod_restart"] - (1.0 * 2 * 0.1)) < 1e-9
    assert abs(action_costs["node_cordon"] - (1.0 * 20 * 0.1)) < 1e-9


def test_calibrate_costs_unseen_action_keeps_default():
    incidents = [{"action": "pod_restart", "downtime_minutes": 2}]
    action_costs, _ = calibrate_costs(
        incidents, default_action_cost=0.05, action_names=ACTION_NAMES
    )
    assert action_costs["scale"] == 0.05  # never observed -> default


# ─── RemediationManager wiring ─────────────────────────────────────────────
def test_manager_calibrate_from_incidents():
    rem = _build_manager()
    assert rem.cost_source == "defaults"
    incidents = [
        {"action": "pod_restart", "downtime_minutes": 2},
        {"action": "scale", "downtime_minutes": 10},
    ]
    costs = rem.calibrate_from_incidents(incidents)
    assert rem.cost_source == "calibrated"
    assert rem.failure_cost == config.SLO_COST_PER_MINUTE * config.INCIDENT_TYPICAL_OUTAGE_MIN
    assert "pod_restart" in rem.action_costs
    assert costs["action_costs"] == rem.action_costs


# ─── Planner responds to calibrated costs ──────────────────────────────────
def test_planner_respects_calibrated_costs():
    model = _seeded_model()
    # Default costs: planner intervenes on CRITICAL with pod_restart.
    default_plan = Planner(model=model).plan("CRITICAL", 3)
    assert any(s.action == "pod_restart" for s in default_plan.steps)

    # Calibrated: pod_restart is ruinously expensive -> planner should not use it.
    expensive = {"pod_restart": 100.0}
    calibrated_plan = Planner(model=model, action_costs=expensive, failure_cost=1.0).plan("CRITICAL", 3)
    assert all(s.action != "pod_restart" for s in calibrated_plan.steps)


# ─── /cost-model endpoint ──────────────────────────────────────────────────
def test_cost_model_endpoint_get_and_post():
    rem = _build_manager()
    ports = (18190, 18191)
    srv = start_server(*ports)
    _context.remediation_manager = rem
    base = f"http://localhost:{ports[0]}"
    try:
        # Defaults before calibration.
        with urllib.request.urlopen(f"{base}/cost-model") as resp:
            assert resp.status == 200
            data = _json.loads(resp.read().decode())
        assert data["source"] == "defaults"
        assert data["failure_cost"] == 1.0

        # POST real incident data -> recalibrate live.
        incidents = [
            {"action": "pod_restart", "downtime_minutes": 2},
            {"action": "scale", "downtime_minutes": 10},
        ]
        body = _json.dumps({"incidents": incidents}).encode()
        req = urllib.request.Request(
            f"{base}/cost-model", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            posted = _json.loads(resp.read().decode())
        assert posted["source"] == "calibrated"
        assert posted["failure_cost"] == config.SLO_COST_PER_MINUTE * config.INCIDENT_TYPICAL_OUTAGE_MIN

        # GET now reflects the calibrated costs.
        with urllib.request.urlopen(f"{base}/cost-model") as resp:
            updated = _json.loads(resp.read().decode())
        assert updated["source"] == "calibrated"
        assert updated["action_costs"]["pod_restart"] == pytest.approx(1.0 * 2 * 0.1)
    finally:
        _context.remediation_manager = None
        srv.shutdown()
        srv.server_close()
